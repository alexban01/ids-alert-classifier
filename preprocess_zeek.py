"""
preprocess_zeek.py — Build training JSONL from Zeek-native / Zeek-compatible datasets.

Datasets supported:
  - IoT-23        : native Zeek conn.log.labeled files
  - CTU-13        : binetflow (Argus bidirectional flows)
  - UNSW-NB15     : CSV/parquet (Bro/Zeek-generated, good column overlap)
  - CICIDS2017    : CICFlowMeter CSVs (already present) — mapped to Zeek schema
  - UWF-ZeekData24: real Zeek conn.log from university cyber range (MITRE labeled)
  - CTU-Normal    : benign-only Zeek conn.log from real user browsing

Outputs:
  zeek_dataset.jsonl       — training samples (~90% per source)
  zeek_dataset_eval.jsonl  — held-out eval samples (~10% per source, stratified)

Zeek conn.log feature set used across all sources:
  proto, service, duration, orig_pkts, resp_pkts, orig_bytes, resp_bytes,
  conn_state, bytes_per_sec, orig_bytes_per_pkt, resp_bytes_per_pkt
"""

import os
import json
import random
import tarfile
import glob
import math

import pandas as pd

from prompt_utils import SYSTEM_PROMPT, build_prompt

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_FILE    = "zeek_dataset.jsonl"
EVAL_FILE     = "zeek_dataset_eval.jsonl"
RANDOM_SEED   = 42
EVAL_FRAC     = 0.10            # fraction of each source held out for eval

# This is for fast training on my 3070
TRAINING_FACTOR = 0.1

MAX_PER_SOURCE_CLASS = int(80_000 * TRAINING_FACTOR)   # default cap per (source, class)
IOT23_BENIGN_CAP     = int(20_000 * TRAINING_FACTOR)   # IoT-23 benign is 89% S0-dominated; reduce to avoid
                                   # "S0 = benign" bias that hurts real-world SF traffic
CTU_NORMAL_CAP       = int(100_000 * TRAINING_FACTOR)  # increase — only significant SF benign source

# v7: 2:1 benign:attack ratio — real networks are overwhelmingly benign,
# training balanced (1:1) makes the model trigger-happy on real traffic.
FINAL_BENIGN  = int(240_000 * TRAINING_FACTOR)
FINAL_ATTACK  = int(120_000 * TRAINING_FACTOR)

CONN_STATE_MASK_PROB = 0.20   # fraction of samples where conn_state is blanked to "-"
                              # forces model to learn from numeric features when state is absent

DATASETS = {
    "iot23":      "datasets/iot-23/iot_23_datasets_small.tar.gz",
    "ctu13":      "datasets/ctu-13/CTU-13-Dataset.tar.bz2",
    "unsw":       "datasets/unsw-nb15/",
    "cicids":     ".",   # looks for *.pcap_ISCX.csv in cwd
    "uwf":        "datasets/uwf-zeekdata24/",
    "ctu_normal": "datasets/ctu-normal/",
}

ATTACK_REASONS = [
    # ── Generic flow-level anomalies ──────────────────────────────────────────
    "Traffic pattern matches known malicious behavior with anomalous packet ratios.",
    "Connection characteristics are inconsistent with normal user activity.",
    "Flow statistics deviate significantly from baseline benign traffic.",
    "Short-duration high-volume flow typical of automated attack tooling.",
    "Asymmetric byte distribution and connection state indicate hostile intent.",
    "Packet timing and volume match scanning or exploitation patterns.",
    "Connection state and byte counts are inconsistent with legitimate use.",
    "Flow metrics match botnet C2 communication profiles.",
    "Unidirectional or near-unidirectional flow with anomalous size is suspicious.",
    "Connection terminated abnormally with byte ratios typical of attack traffic.",

    # ── Port scanning (IoT-23 PartOfAHorizontalPortScan, UNSW Reconnaissance/Analysis) ──
    "S0 TCP connection to a common service port with zero resp_bytes indicates the remote host did not respond, consistent with a port scan probe against a closed or filtered port.",
    "S0 connection with zero resp_bytes and sub-millisecond duration is characteristic of a single port scan probe — the originator sent a SYN but received no reply.",
    "Short SYN-only flow (S0 state) with zero resp_bytes and near-zero duration is consistent with automated reconnaissance; a legitimate connection attempt would progress to SF or RSTO.",
    "SYN probe to a high-numbered port with S0 state and zero resp_bytes is consistent with a port sweep looking for open services on non-standard ports.",

    # ── DDoS / DoS (IoT-23 DDoS, Ngrbot on-command, UNSW DoS) ──────────────
    "High packet count with minimal response bytes is consistent with a SYN flood or amplification attack.",
    "Very high UDP packet rate to a single destination with near-zero resp_bytes and large orig_bytes is consistent with a UDP flood DDoS attack.",
    "Large number of short TCP connections to the same destination port in rapid succession with no sustained data exchange indicates an HTTP flood or Layer-7 DDoS.",
    "Many long-duration TCP connections to a web port (80/443) with near-zero data exchange and S1 state suggest a Slowloris-style connection exhaustion attack.",
    "Small UDP orig_bytes with disproportionately large resp_bytes on port 53 or 123 indicates DNS or NTP amplification/reflection abuse.",
    "ICMP flows with very high packet count and large orig_bytes but no application payload indicate an ICMP flood attack.",

    # ── Brute force (SSH, RDP, VNC, Telnet, DB) ──────────────────────────────
    "Destination port 22 with many short RSTO or S0 connections in rapid succession suggests automated SSH brute force.",
    "Destination port 3389 (RDP) with repeated short SYN-only or RSTO connections indicates RDP brute force or password spraying.",
    "Destination port 23 (Telnet) with many S0 flows to many unique IPs and zero resp_bytes matches Mirai-style Telnet scanning.",
    "Multiple short RSTO connections to port 5900 (VNC) from the same source indicate remote desktop brute force.",
    "Rapid short connections to database ports (1433 MSSQL, 3306 MySQL, 5432 PostgreSQL) with RSTO state suggest credential brute force against a database service.",
    "Many S0 or RSTO flows to port 23 (Telnet) from IoT-class IPs with zero response bytes indicate Mirai or Mirai-variant botnet scanning.",

    # ── Exploitation (SMB, GlassFish, Tomcat, EternalBlue) ───────────────────
    "Destination port 445 is associated with SMB exploitation (EternalBlue/MS17-010, WannaCry, NotPetya lateral movement).",
    "Connection to port 4848 (GlassFish admin console) or 8080/8443/9090 (Tomcat, JBoss, WildFly) from an external or untrusted host suggests admin interface exploitation.",
    "RSTO state with asymmetric orig_bytes on a service port (21, 22, 80, 443) is consistent with an exploitation attempt that triggered a server-side reset.",
    "Short TCP connection to a service port that terminates in RSTO state after non-trivial orig_bytes may indicate a failed exploit or shellcode delivery attempt.",
    "Established connection to port 445 with large orig_bytes followed by RSTO suggests SMB exploitation payload delivery.",

    # ── Worm / self-propagation (UNSW Worms, Virut via SMB, Mirai spreading) ─
    "S0 TCP connection originating from a workstation-class host to port 445 (SMB) on a remote IP with zero resp_bytes is consistent with worm-style lateral movement probing (WannaCry/NotPetya pattern).",
    "S0 TCP probe to port 445 from an internal host with zero resp_bytes suggests unsolicited SMB connection attempt consistent with worm propagation or lateral movement scanning.",

    # ── C2: IRC-based (Ngrbot Botnet-44, Neris, Rbot CTU-13) ─────────────────
    "Persistent established TCP connection to port 6667 (IRC) from a non-server host is consistent with IRC-based botnet C2.",
    "Long-duration S1 or SF TCP connection to port 6667 with low-rate symmetric byte exchange matches an IRC C2 keep-alive channel.",

    # ── C2: HTTP/HTTPS-based (Zeus Botnet-78-2, Ramnit Botnet-42, Htbot Botnet-52, Pushdo Botnet-90, Virut) ──
    "Short established HTTP connection with disproportionately small response bytes is consistent with C2 gate polling or command retrieval (Zeus, Ramnit, Virut pattern).",
    "Repeated fixed-size flows to a single remote IP at regular short intervals indicate automated C2 beaconing or heartbeat.",
    "Periodic short HTTPS connections to a single remote IP with consistent minimal byte counts suggest encrypted C2 beaconing (Pushdo pattern).",
    "Short SF TCP connection to port 80 with near-zero resp_bytes and tiny orig_bytes is consistent with an HTTP C2 check-in with no pending command.",
    "Established TCP connection to a non-standard high port with periodic low-volume bidirectional data and no clear application-layer service indicates a custom C2 channel.",

    # ── C2: P2P-based (Kelihos Botnet-3, NSIS.ay) ────────────────────────────
    "Established connections to many unique IPs on non-standard high ports with balanced bidirectional byte exchange indicate P2P botnet communication.",
    "Many short bidirectional UDP flows to random high-numbered ports on many different IPs with uniform packet sizes suggest a P2P botnet overlay.",

    # ── Spam (Kelihos Botnet-3, Neris CTU-13) ────────────────────────────────
    "Outbound connections to port 25 (SMTP) from a non-mail-server with large orig_bytes and high connection rate suggest spam bot activity.",
    "Many short SF TCP connections to port 25 from a desktop-class IP with consistent orig_bytes size indicate a spam campaign delivering email payloads.",

    # ── Credential / admin interface attacks (UWF Credential Access) ─────────
    "Connection to port 4848 (GlassFish admin) or similar admin-only port with SSL service from an untrusted source indicates a credential access attack.",
    "Multiple SF TCP connections to an admin-only port with consistent moderate orig_bytes indicate automated credential stuffing or brute force against a management interface.",

    # ── Data exfiltration ─────────────────────────────────────────────────────
    "Large orig_bytes relative to resp_bytes on a high-numbered port suggests an outbound data exfiltration channel.",
    "Unusually large orig_bytes on an established connection to a non-standard port indicates potential C2 exfiltration.",
    "Connection to port 21 (FTP) with large orig_bytes from a workstation-class host suggests credential exfiltration or C2 file staging (Ramnit pattern).",
    "Large outbound byte count on a non-HTTP port to an external IP with no prior inbound connection is consistent with staged data exfiltration.",

    # ── DNS / tunneling ────────────────────────────────────────────────────────
    "Oversized traffic on port 53 with high byte-to-packet ratio suggests DNS tunneling for data exfiltration or C2.",
    "Very high rate of short UDP flows to port 53 with varying payload sizes from a single host is consistent with DNS-based C2 or data exfiltration.",

    # ── Backdoor / persistent access (UNSW Backdoors) ────────────────────────
    "Long-duration established TCP connection to an unusual port with periodic low-volume bidirectional data and no known application-layer service indicates a backdoor channel.",
    "Established TCP session to a high-numbered non-standard port originating from an external IP with large resp_bytes suggests a reverse shell or backdoor listener.",

    # ── Industrial / SCADA (Siemens Botnet-54) ────────────────────────────────
    "Connection to industrial control protocol ports (Modbus 502, Siemens S7 102, OPC-UA 4840/4843) from an IT-network or external host is highly suspicious.",
    "TCP connection to port 102 (Siemens S7) or 502 (Modbus) from a host outside the OT network segment indicates a potential ICS/SCADA attack.",

    # ── Telnet exploitation / IoT payload delivery (Mirai) ───────────────────
    "Established Telnet (port 23) connection with large resp_bytes after a scanning phase is consistent with Mirai-style malware binary download to an IoT device.",

    # ── Fuzzing / UNSW-NB15 specific ─────────────────────────────────────────
    "Many connections with irregular orig_bytes patterns and varied states across many ports from a single source indicate network fuzzing or vulnerability probing.",
    "Connections with unusual flag combinations reflected in RSTO or OTH states across many destination ports suggest packet-level manipulation or fuzzing.",

    # ── Pushdo fake-SSL flood: short RSTO TCP/443 with near-zero bytes ──────
    # Observable per-flow: RSTO state + port 443 + near-zero bytes — distinct from legitimate
    # HTTPS which completes (SF) with non-trivial byte exchange.
    "Short RSTO TCP connection to port 443 with near-zero orig_bytes and resp_bytes is consistent with a fake or malformed SSL handshake used by malware to blend C2 traffic among legitimate HTTPS connections.",

    # ── Ramnit port-443 non-SSL: established TCP/443, service absent, asymmetric bytes ──
    # Observable: port 443 + service="-" + established flow with atypical bytes.
    # Softened to avoid flagging every Zeek SSL-parse failure as ATTACK.
    "Established TCP connection to port 443 with no identified SSL service and atypical byte ratios may indicate malware using a custom encrypted protocol on the HTTPS port rather than standard TLS.",

    # ── IRC C2 on non-standard ports (Botnet-90 uses 2081) ───────────────────
    "Persistent established TCP connection to an uncommon port (e.g. 2081, 194, 531) with low-rate symmetric byte exchange resembling IRC framing may indicate C2 using IRC on an alternate port to evade port-based filters.",

    # ── DNS anomaly per-flow (Kelihos DNS flood — observable signal is large DNS payload) ──
    # Reworded to focus on what IS visible in one flow: UDP/53 with atypically large bytes.
    "UDP flow to port 53 with atypically large orig_bytes or resp_bytes relative to a normal DNS query/response may indicate DNS tunneling, oversized TXT record abuse, or botnet DNS flooding behavior.",

    # ── Zeus config-download pattern: small symmetric HTTP GET, immediate close ──
    "Short SF HTTP connection to port 80 with very small orig_bytes and similarly small resp_bytes followed by immediate close is consistent with a malware config or command file fetch (Zeus-style gate check-in).",

    # ── S0 SYN flow as part of scanning / worm discovery ─────────────────────
    # Observable per-flow: S0 state + zero resp_bytes + short or zero duration.
    "S0 TCP connection with zero resp_bytes and zero or near-zero duration is consistent with a SYN probe used in port scanning or worm host-discovery; the scanner moved on without completing a handshake.",

    # ── Proxy/relay traffic (Htbot, Botnet-52) ────────────────────────────────
    "Established TCP connection to port 8080 or 3128 with moderate bidirectional byte exchange from a host that is not a designated proxy server may indicate HTTP proxy abuse or a proxy-relay botnet node.",
]

BENIGN_REASONS = [
    # ── Generic ───────────────────────────────────────────────────────────────
    "Symmetric flow with normal byte and packet ratios consistent with legitimate traffic.",
    "Connection established and closed cleanly; no anomalous volume or timing.",
    "Packet sizes and counts match expected web or file-transfer activity.",
    "Flow duration and byte counts are within normal operational range.",
    "Bidirectional exchange with balanced originator/responder bytes; likely benign.",
    "Connection state indicates normal handshake and teardown sequence.",
    "Traffic volume and protocol use are consistent with standard enterprise behavior.",
    "Flow characteristics match DNS, HTTP, or routine background communication.",
    "Short-lived connection with small byte counts; consistent with keep-alives or health checks.",
    "No anomalous ratios detected; flow matches baseline for this protocol.",

    # ── Web / HTTP / HTTPS ────────────────────────────────────────────────────
    "Completed HTTPS connection on port 443 with symmetric byte exchange is consistent with standard web browsing.",
    "Short TCP session to port 80 with low byte count is consistent with a web API health check or lightweight GET request.",
    "High resp_bytes on port 443 with clean SF completion is consistent with a software update or media download over HTTPS.",
    "Large orig_bytes on port 443 with clean SF completion is consistent with a cloud storage upload (Google Drive, OneDrive, S3).",
    "Frequent short HTTP connections to the same endpoint with minimal payload are consistent with API polling, monitoring agents, or load balancer health checks.",
    "Short periodic HTTPS connections to a CDN or vendor endpoint with stable byte counts are consistent with software update checks or telemetry.",

    # ── DNS / NTP / ICMP ──────────────────────────────────────────────────────
    "DNS query on port 53 with small payload and immediate response is normal resolver traffic.",
    "UDP flow to port 123 (NTP) with tiny symmetric byte counts and sub-second duration is normal time synchronization traffic.",
    "Small bidirectional ICMP flow with short duration and minimal bytes is consistent with a network ping or reachability check.",

    # ── SSH / Admin ───────────────────────────────────────────────────────────
    "SSH session on port 22 with moderate byte exchange and clean SF completion is consistent with legitimate interactive admin access.",
    "High orig_bytes on port 22 with long duration and SF completion is consistent with a bulk file transfer via SCP or rsync over SSH.",
    "Established TCP connection to an admin port (8080, 8443) from an internal management IP with normal byte exchange indicates authorized web application administration.",

    # ── SMTP / FTP ────────────────────────────────────────────────────────────
    "SMTP connection from a known mail server IP with large orig_bytes and SF completion indicates normal outbound email delivery.",
    "FTP data transfer with large resp_bytes, clean SF state, and normal packet ratios is consistent with authorized bulk file download.",
    "FTP control session on port 21 with small symmetric byte exchange and SF state is consistent with a routine directory listing or authentication.",

    # ── IRC / Chat (complement to IRC C2 reason) ──────────────────────────────
    "IRC or chat application connection on port 6667 with low-rate symmetric exchange and clean SF completion is consistent with legitimate messaging.",

    # ── Streaming / VoIP / P2P ────────────────────────────────────────────────
    "High UDP packet rate with large resp_bytes and long duration on a high-numbered port is consistent with video streaming or media delivery.",
    "Bidirectional UDP flows to multiple IPs on high-numbered ports with consistent small packet sizes and long duration are consistent with VoIP or real-time communication.",
    "Many TCP or UDP connections to many IPs on high ports with moderate bidirectional byte exchange are consistent with legitimate BitTorrent or P2P file-sharing activity.",

    # ── Database (complement to DB brute force reason) ────────────────────────
    "Multiple short SF TCP connections to a database port (1433, 3306, 5432) from an application server IP with moderate bytes are consistent with normal ORM connection pool traffic.",

    # ── Industrial / internal (complement to ICS attack reason) ─────────────
    "TCP connection to Modbus (502) or OPC-UA (4840) from a known engineering workstation IP within the OT network segment is consistent with authorized SCADA operations.",

    # ── Background / misc ─────────────────────────────────────────────────────
    "Short S0 or RSTO connection to a common port with zero bytes is consistent with a firewall probe, NAT keepalive, or connection refused response.",
    "UDP flows with small symmetric byte counts to well-known service ports are consistent with routine background protocol traffic.",
    "Long-duration established TCP connection with very low byte rate is consistent with a persistent application-layer session such as a database connection pool or XMPP stream.",

    # ── Complements to new attack reasons ─────────────────────────────────────
    # Complement to fake-SSL flood: legitimate CDN/load-balancer resets
    "Short RSTO TCP connections to port 443 from a load balancer health check or CDN probe are normal and expected for infrastructure monitoring.",
    # Complement to TCP/443 non-SSL: some applications use custom TLS or QUIC on 443
    "An established TCP connection to port 443 with an unrecognized service field may reflect a non-standard TLS implementation, HTTP/3 negotiation, or proprietary application protocol over the HTTPS port.",
    # Complement to DNS flood: resolver doing NXDOMAIN lookups or CDN health checks
    "High-volume DNS queries from an internal resolver or CDN edge node are expected behavior for recursive resolution under load.",
    # Complement to proxy ports: legitimate forward proxy or dev environment
    "TCP connections to port 8080 or 3128 from clients to a designated internal proxy or development web server are normal in enterprise and developer environments.",
]

random.seed(RANDOM_SEED)

def pick_reason(verdict):
    pool = ATTACK_REASONS if verdict == "ATTACK" else BENIGN_REASONS
    return random.choice(pool)

# ── Sample builder ─────────────────────────────────────────────────────────────
def make_sample(proto, duration, orig_pkts, resp_pkts,
                orig_bytes, resp_bytes, conn_state, verdict, source,
                service="-", resp_port="-", orig_port="-"):
    # Mask conn_state with CONN_STATE_MASK_PROB — forces model to use numeric
    # features (bytes, packets, port) when state is unavailable or ambiguous.
    prompt_conn_state = "-" if random.random() < CONN_STATE_MASK_PROB else conn_state
    prompt  = build_prompt(proto, duration, orig_pkts, resp_pkts,
                           orig_bytes, resp_bytes, prompt_conn_state, service,
                           resp_port=resp_port, orig_port=orig_port)
    reason  = pick_reason(verdict)
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": f"VERDICT: {verdict}\nREASON: {reason}"},
        ],
        "source":     source,
        "verdict":    verdict,
        "conn_state": conn_state,  # original (pre-mask) — used for SF oversampling
    }

# ── Loaders ────────────────────────────────────────────────────────────────────

def load_iot23(archive_path):
    """Read IoT-23 conn.log.labeled files from tar.gz archive."""
    if not os.path.isfile(archive_path):
        print(f"[SKIP] IoT-23 archive not found: {archive_path}")
        return []

    samples = {"ATTACK": [], "FALSE POSITIVE": []}
    print(f"[IoT-23] Opening archive {archive_path} ...")

    with tarfile.open(archive_path, "r:gz") as tf:
        members = [m for m in tf.getmembers()
                   if m.name.endswith("conn.log.labeled") and m.isfile()]
        print(f"  Found {len(members)} conn.log.labeled files")

        for member in members:
            f = tf.extractfile(member)
            if f is None:
                continue
            lines_read = attacks = benign = 0
            for raw in f:
                line = raw.decode("utf-8", errors="replace").strip()
                # Skip Zeek header comment lines
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                # Standard Zeek conn.log has 21 tab-separated fields.
                # The last tab field bundles "tunnel_parents label detailed-label"
                # as space-separated sub-tokens (IoT-23 specific formatting).
                if len(parts) < 21:
                    continue
                try:
                    orig_port  = parts[3]
                    resp_port  = parts[5]
                    proto      = parts[6]
                    service    = parts[7]
                    duration   = parts[8]
                    orig_bytes = parts[9]
                    resp_bytes = parts[10]
                    conn_state = parts[11]
                    orig_pkts  = parts[16]
                    resp_pkts  = parts[18]
                    # Last field: "tunnel_parents   label   detailed-label"
                    # Label is always "Malicious" or "Benign" — search directly.
                    last = parts[-1]
                    if "Malicious" in last:
                        label = "Malicious"
                    elif "Benign" in last:
                        label = "Benign"
                    else:
                        continue
                except IndexError:
                    continue

                if label == "Malicious":
                    verdict = "ATTACK"
                elif label == "Benign":
                    verdict = "FALSE POSITIVE"
                else:
                    continue

                cap    = IOT23_BENIGN_CAP if verdict == "FALSE POSITIVE" else MAX_PER_SOURCE_CLASS
                bucket = samples[verdict]
                if len(bucket) < cap:
                    bucket.append(make_sample(
                        proto, duration, orig_pkts, resp_pkts,
                        orig_bytes, resp_bytes, conn_state, verdict, "iot23",
                        service=service, resp_port=resp_port, orig_port=orig_port,
                    ))
                lines_read += 1
                if verdict == "ATTACK":  attacks += 1
                else:                    benign  += 1

            print(f"    {member.name}: {attacks} attacks, {benign} benign")

    total = len(samples["ATTACK"]) + len(samples["FALSE POSITIVE"])
    print(f"  IoT-23 total: {len(samples['ATTACK'])} attacks, "
          f"{len(samples['FALSE POSITIVE'])} benign")
    return samples["ATTACK"] + samples["FALSE POSITIVE"]


def load_ctu13(archive_path):
    """Read CTU-13 binetflow files from tar.bz2 archive.

    Binetflow uses Argus state notation — mapped to Zeek conn_state equivalents
    so the model only ever sees Zeek states during training.
    """
    # Argus binetflow state → Zeek conn_state
    CTU_STATE_MAP = {
        "INT":        "S1",    # mid-flow established, no FIN seen
        "CON":        "SF",    # completed connection
        "FIN":        "SF",    # completed with FIN
        "FSPA_FSPA":  "SF",    # FIN bidirectional = completed
        "FSA_FSA":    "SF",
        "SPA_FSPA":   "SF",
        "PA_PA":      "OTH",   # PSH-ACK only, no SYN seen
        "EST":        "S1",    # established
        "S_":         "S0",    # SYN only, no response
        "REQ":        "S0",
        "SRPA_SPA":   "RSTO",  # RST from originator
        "SRST":       "RSTO",
    }

    if not os.path.isfile(archive_path):
        print(f"[SKIP] CTU-13 archive not found: {archive_path}")
        return []

    samples = {"ATTACK": [], "FALSE POSITIVE": []}
    print(f"[CTU-13] Opening archive {archive_path} ...")

    with tarfile.open(archive_path, "r:bz2") as tf:
        members = [m for m in tf.getmembers()
                   if m.name.endswith(".binetflow") and m.isfile()]
        print(f"  Found {len(members)} binetflow files")

        for member in members:
            f = tf.extractfile(member)
            if f is None:
                continue
            attacks = benign = 0
            header = None
            for raw in f:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if header is None:
                    header = [c.strip() for c in line.split(",")]
                    continue
                parts = line.split(",")
                if len(parts) < len(header):
                    continue
                row = dict(zip(header, parts))

                label = row.get("Label", "").strip()
                if "Botnet" in label:
                    verdict = "ATTACK"
                elif "Normal" in label:
                    verdict = "FALSE POSITIVE"
                else:
                    continue   # skip Background (unlabeled)

                proto      = row.get("Proto", "unknown").strip().lower()
                duration   = row.get("Dur",   "0").strip()
                raw_state  = row.get("State", "-").strip().upper()
                conn_state = CTU_STATE_MAP.get(raw_state, "-")
                tot_pkts   = row.get("TotPkts",  "0").strip()
                src_bytes  = row.get("SrcBytes",  "0").strip()
                tot_bytes  = row.get("TotBytes",  "0").strip()
                orig_port  = row.get("Sport", "-").strip()
                resp_port  = row.get("Dport", "-").strip()
                try:
                    dst_bytes = str(float(tot_bytes) - float(src_bytes))
                except ValueError:
                    dst_bytes = "0"
                # binetflow has TotPkts only; split evenly as approximation
                try:
                    half = str(int(float(tot_pkts)) // 2)
                except ValueError:
                    half = "0"

                bucket = samples[verdict]
                if len(bucket) < MAX_PER_SOURCE_CLASS:
                    bucket.append(make_sample(
                        proto, duration, half, half,
                        src_bytes, dst_bytes, conn_state, verdict, "ctu13",
                        service="-",  # binetflow has no app-layer service field
                        resp_port=resp_port, orig_port=orig_port,
                    ))
                if verdict == "ATTACK":  attacks += 1
                else:                    benign  += 1

            print(f"    {member.name}: {attacks} attacks, {benign} benign")

    print(f"  CTU-13 total: {len(samples['ATTACK'])} attacks, "
          f"{len(samples['FALSE POSITIVE'])} benign")
    return samples["ATTACK"] + samples["FALSE POSITIVE"]


def load_unsw(dataset_dir):
    """Read UNSW-NB15 from CSV or parquet files."""
    if not os.path.isdir(dataset_dir):
        print(f"[SKIP] UNSW-NB15 directory not found: {dataset_dir}")
        return []

    # Try parquet first (Network-Flows subdir), then CSV
    parquet_files = glob.glob(os.path.join(dataset_dir, "**/*.parquet"), recursive=True)
    csv_files     = glob.glob(os.path.join(dataset_dir, "**/*.csv"),     recursive=True)
    # exclude HuggingFace cache metadata and index files
    csv_files = [f for f in csv_files
                 if ".cache" not in f and "metadata" not in f.lower()
                 and not os.path.basename(f).startswith(".")]

    files = parquet_files if parquet_files else csv_files
    if not files:
        print(f"[SKIP] No parquet or CSV files found in {dataset_dir}")
        return []

    print(f"[UNSW-NB15] Loading {len(files)} file(s) from {dataset_dir}")
    samples = {"ATTACK": [], "FALSE POSITIVE": []}

    for fpath in files:
        print(f"  Reading {os.path.basename(fpath)} ...")
        try:
            if fpath.endswith(".parquet"):
                df = pd.read_parquet(fpath)
            else:
                df = pd.read_csv(fpath, low_memory=False)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        df.columns = [c.strip().lower() for c in df.columns]

        # UNSW-NB15 column names: proto, state, dur, sbytes, dbytes, spkts, dpkts, label, attack_cat
        proto_col  = next((c for c in ["proto", "protocol"]            if c in df.columns), None)
        dur_col    = next((c for c in ["dur", "duration"]             if c in df.columns), None)
        spkts_col  = next((c for c in ["spkts", "orig_pkts"]         if c in df.columns), None)
        dpkts_col  = next((c for c in ["dpkts", "resp_pkts"]         if c in df.columns), None)
        sbytes_col = next((c for c in ["sbytes", "orig_bytes"]       if c in df.columns), None)
        dbytes_col = next((c for c in ["dbytes", "resp_bytes"]       if c in df.columns), None)
        state_col  = next((c for c in ["state", "conn_state"]        if c in df.columns), None)
        svc_col    = next((c for c in ["service"]                    if c in df.columns), None)
        sport_col  = next((c for c in ["sport", "source_port", "orig_p"]       if c in df.columns), None)
        dport_col  = next((c for c in ["dport", "destination_port", "resp_p"]  if c in df.columns), None)
        label_col  = next((c for c in ["binary_label", "label"]      if c in df.columns), None)

        if label_col is None:
            print(f"    No label column found, skipping.")
            continue

        df = df.replace([float("inf"), float("-inf")], float("nan")).dropna(subset=[label_col])

        attacks = benign = 0
        for _, row in df.iterrows():
            lv = row[label_col]
            try:
                verdict = "ATTACK" if int(float(lv)) == 1 else "FALSE POSITIVE"
            except (ValueError, TypeError):
                verdict = "ATTACK" if str(lv).strip() not in ("0", "Normal", "BENIGN") else "FALSE POSITIVE"

            bucket = samples[verdict]
            if len(bucket) >= MAX_PER_SOURCE_CLASS:
                continue

            bucket.append(make_sample(
                proto      = str(row[proto_col])  if proto_col  else "unknown",
                duration   = str(row[dur_col])    if dur_col    else "0",
                orig_pkts  = str(row[spkts_col])  if spkts_col  else "0",
                resp_pkts  = str(row[dpkts_col])  if dpkts_col  else "0",
                orig_bytes = str(row[sbytes_col]) if sbytes_col else "0",
                resp_bytes = str(row[dbytes_col]) if dbytes_col else "0",
                conn_state = str(row[state_col])  if state_col  else "-",
                verdict    = verdict,
                source     = "unsw",
                service    = str(row[svc_col]).strip()  if svc_col    else "-",
                orig_port  = str(row[sport_col])        if sport_col  else "-",
                resp_port  = str(row[dport_col])        if dport_col  else "-",
            ))
            if verdict == "ATTACK": attacks += 1
            else:                   benign  += 1

        print(f"    {attacks} attacks, {benign} benign")

    print(f"  UNSW-NB15 total: {len(samples['ATTACK'])} attacks, "
          f"{len(samples['FALSE POSITIVE'])} benign")
    return samples["ATTACK"] + samples["FALSE POSITIVE"]


def load_cicids(base_dir):
    """Read CICIDS2017 CSVs — map CICFlowMeter columns to Zeek schema."""
    csv_files = glob.glob(os.path.join(base_dir, "*.pcap_ISCX.csv"))
    if not csv_files:
        print(f"[SKIP] No CICIDS2017 CSVs found in {base_dir}")
        return []

    # Cap per file so no single CSV (e.g. DDoS) dominates the whole source budget.
    # 8 files × 10k = 80k max per class across CICIDS2017.
    PER_FILE_CAP = 10_000

    print(f"[CICIDS2017] Loading {len(csv_files)} CSV(s) ...")
    samples = {"ATTACK": [], "FALSE POSITIVE": []}

    # CICFlowMeter → Zeek mapping
    COLMAP = {
        "Protocol":                   "proto",
        "Flow Duration":              "duration_us",   # microseconds — convert
        "Total Fwd Packets":          "orig_pkts",
        "Total Backward Packets":     "resp_pkts",
        "Total Length of Fwd Packets":"orig_bytes",
        "Total Length of Bwd Packets":"resp_bytes",
        "Label":                      "label",
    }

    for fpath in csv_files:
        print(f"  Reading {os.path.basename(fpath)} ...")
        try:
            df = pd.read_csv(fpath, low_memory=False)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
        df.columns = df.columns.str.strip()
        df = df.replace([float("inf"), float("-inf")], float("nan")).dropna()

        avail = {v: k for k, v in COLMAP.items() if k in df.columns}
        if "label" not in avail:
            continue

        file_counts = {"ATTACK": 0, "FALSE POSITIVE": 0}
        attacks = benign = 0
        for _, row in df.iterrows():
            raw_label = str(row[avail["label"]]).strip()
            verdict   = "FALSE POSITIVE" if raw_label == "BENIGN" else "ATTACK"
            if file_counts[verdict] >= PER_FILE_CAP:
                continue
            bucket    = samples[verdict]
            if len(bucket) >= MAX_PER_SOURCE_CLASS:
                continue

            proto     = str(int(float(row[avail["proto"]]))) if "proto" in avail else "unknown"
            dur_us    = row[avail["duration_us"]] if "duration_us" in avail else 0
            try:
                duration = str(float(dur_us) / 1e6)  # µs → seconds
            except (ValueError, TypeError):
                duration = "0"

            bucket.append(make_sample(
                proto      = proto,
                duration   = duration,
                orig_pkts  = str(row[avail["orig_pkts"]])  if "orig_pkts"  in avail else "0",
                resp_pkts  = str(row[avail["resp_pkts"]])  if "resp_pkts"  in avail else "0",
                orig_bytes = str(row[avail["orig_bytes"]]) if "orig_bytes" in avail else "0",
                resp_bytes = str(row[avail["resp_bytes"]]) if "resp_bytes" in avail else "0",
                conn_state = "-",   # CICFlowMeter has no conn_state equivalent
                verdict    = verdict,
                source     = "cicids",
            ))
            file_counts[verdict] += 1
            if verdict == "ATTACK": attacks += 1
            else:                   benign  += 1

        print(f"    {attacks} attacks, {benign} benign")

    print(f"  CICIDS2017 total: {len(samples['ATTACK'])} attacks, "
          f"{len(samples['FALSE POSITIVE'])} benign")
    return samples["ATTACK"] + samples["FALSE POSITIVE"]


def load_uwf(dataset_dir):
    """Read UWF-ZeekData24 CSV files (Spark output, one dir per MITRE tactic)."""
    if not os.path.isdir(dataset_dir):
        print(f"[SKIP] UWF-ZeekData24 directory not found: {dataset_dir}")
        return []

    csv_files = glob.glob(os.path.join(dataset_dir, "**/*.csv"), recursive=True)
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith(".")]
    if not csv_files:
        print(f"[SKIP] No CSV files found in {dataset_dir}")
        return []

    # v8.1: Re-enable selected UWF attack tactics now that port data is in the prompt.
    # Credential Access (port 4848/ssl) and Defense Evasion (port 445/smb) are
    # distinguishable by dest port — the v7 "indistinguishable" conclusion was wrong
    # because port extraction was broken. Skip Initial Access (port 80 SF = ambiguous
    # web traffic) and Exfiltration (23 rows, too small).
    UWF_ALLOWED_TACTICS = {"Credential Access", "Defense Evasion"}
    print(f"[UWF-ZeekData24] Loading {len(csv_files)} CSV(s) from {dataset_dir}")
    samples = {"ATTACK": [], "FALSE POSITIVE": []}

    for fpath in csv_files:
        print(f"  Reading {os.path.relpath(fpath, dataset_dir)} ...")
        try:
            df = pd.read_csv(fpath, low_memory=False)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        df.columns = [c.strip() for c in df.columns]

        label_col = next((c for c in ["label_binary", "label_tactic"] if c in df.columns), None)
        if label_col is None:
            print(f"    No label column found, skipping.")
            continue

        attacks = benign = 0
        for _, row in df.iterrows():
            if label_col == "label_binary":
                verdict = "ATTACK" if str(row[label_col]).strip() == "True" else "FALSE POSITIVE"
            else:
                verdict = "ATTACK" if str(row[label_col]).strip() != "none" else "FALSE POSITIVE"

            # Only include attacks from tactics with a real port-based signal.
            if verdict == "ATTACK":
                tactic = str(row.get("label_tactic", "")).strip()
                if tactic not in UWF_ALLOWED_TACTICS:
                    continue

            bucket = samples[verdict]
            if len(bucket) >= MAX_PER_SOURCE_CLASS:
                continue

            # UWF uses empty strings for missing values — pass through as-is
            # (build_prompt / _safe will convert to N/A)
            proto      = str(row.get("proto", "unknown")).strip()
            service    = str(row.get("service", "-")).strip()
            duration   = str(row.get("duration", "")).strip()
            orig_pkts  = str(row.get("orig_pkts", "")).strip()
            resp_pkts  = str(row.get("resp_pkts", "")).strip()
            orig_bytes = str(row.get("orig_bytes", "")).strip()
            resp_bytes = str(row.get("resp_bytes", "")).strip()
            conn_state = str(row.get("conn_state", "-")).strip()
            orig_port  = str(row.get("id.orig_p", row.get("orig_p", row.get("src_port_zeek", "-")))).strip()
            resp_port  = str(row.get("id.resp_p", row.get("resp_p", row.get("dest_port_zeek", "-")))).strip()

            # pandas converts empty CSV cells to nan
            if service    in ("nan", "None"): service    = "-"
            if duration   in ("nan", "None"): duration   = ""
            if orig_pkts  in ("nan", "None"): orig_pkts  = ""
            if resp_pkts  in ("nan", "None"): resp_pkts  = ""
            if orig_bytes in ("nan", "None"): orig_bytes = ""
            if resp_bytes in ("nan", "None"): resp_bytes = ""
            if orig_port  in ("nan", "None"): orig_port  = "-"
            if resp_port  in ("nan", "None"): resp_port  = "-"

            bucket.append(make_sample(
                proto, duration, orig_pkts, resp_pkts,
                orig_bytes, resp_bytes, conn_state, verdict, "uwf",
                service=service, resp_port=resp_port, orig_port=orig_port,
            ))
            if verdict == "ATTACK": attacks += 1
            else:                   benign  += 1

        print(f"    {attacks} attacks, {benign} benign")

    print(f"  UWF-ZeekData24 total: {len(samples['ATTACK'])} attacks, "
          f"{len(samples['FALSE POSITIVE'])} benign")
    return samples["ATTACK"] + samples["FALSE POSITIVE"]


def load_ctu_normal(dataset_dir):
    """Read CTU-Normal benign Zeek conn.log files (standard 21-field TSV)."""
    if not os.path.isdir(dataset_dir):
        print(f"[SKIP] CTU-Normal directory not found: {dataset_dir}")
        return []

    log_files = sorted(glob.glob(os.path.join(dataset_dir, "*.log")))
    if not log_files:
        print(f"[SKIP] No .log files found in {dataset_dir}")
        return []

    print(f"[CTU-Normal] Loading {len(log_files)} conn.log file(s) from {dataset_dir}")
    samples = []
    total = 0

    for fpath in log_files:
        count = 0
        with open(fpath) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 21:
                    continue

                if len(samples) >= CTU_NORMAL_CAP:
                    break

                orig_port  = parts[3]
                resp_port  = parts[5]
                proto      = parts[6]
                service    = parts[7]
                duration   = parts[8]
                orig_bytes = parts[9]
                resp_bytes = parts[10]
                conn_state = parts[11]
                orig_pkts  = parts[16]
                resp_pkts  = parts[18]

                # All CTU-Normal traffic is benign — pass - values through as-is
                samples.append(make_sample(
                    proto, duration, orig_pkts, resp_pkts,
                    orig_bytes, resp_bytes, conn_state,
                    "FALSE POSITIVE", "ctu_normal",
                    service=service, resp_port=resp_port, orig_port=orig_port,
                ))
                count += 1

        total += count
        print(f"  {os.path.basename(fpath)}: {count} benign entries")

    print(f"  CTU-Normal total: {len(samples)} benign")
    return samples


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from collections import Counter, defaultdict
    random.seed(RANDOM_SEED)

    all_samples = []
    all_samples += load_iot23(DATASETS["iot23"])
    all_samples += load_ctu13(DATASETS["ctu13"])
    all_samples += load_unsw(DATASETS["unsw"])
    # CICIDS2017 dropped in v7: CICFlowMeter produces proto="unknown" and
    # conn_state="-" for every flow — both fields are always missing, so the
    # model can't learn Zeek-discriminative patterns from this source.
    # Also has documented label quality issues (~10-15% mislabeled).
    # all_samples += load_cicids(DATASETS["cicids"])
    all_samples += load_uwf(DATASETS["uwf"])
    all_samples += load_ctu_normal(DATASETS["ctu_normal"])

    # ── Source-stratified train/eval split ────────────────────────────────────
    # Hold out EVAL_FRAC from each (source, verdict) bucket so eval distribution
    # matches the real-world variety of all sources — not just whatever ends up
    # in a random 10% slice of the merged pool.
    by_bucket = defaultdict(list)
    for s in all_samples:
        by_bucket[(s["source"], s["verdict"])].append(s)

    train_pool, eval_pool = [], []
    for bucket_samples in by_bucket.values():
        random.shuffle(bucket_samples)
        n_eval = max(1, int(len(bucket_samples) * EVAL_FRAC))
        eval_pool.extend(bucket_samples[:n_eval])
        train_pool.extend(bucket_samples[n_eval:])

    # ── Subsample train pool to target ratio ──────────────────────────────────
    attacks = [s for s in train_pool if s["verdict"] == "ATTACK"]
    benign  = [s for s in train_pool if s["verdict"] == "FALSE POSITIVE"]

    print(f"\nRaw train pool: {len(attacks)} attacks, {len(benign)} benign")
    print(f"Eval pool     : {sum(1 for s in eval_pool if s['verdict']=='ATTACK')} attacks, "
          f"{sum(1 for s in eval_pool if s['verdict']=='FALSE POSITIVE')} benign")

    random.shuffle(benign)
    benign = benign[:FINAL_BENIGN]

    # SF-state attack oversampling (Change 5): give completed/established attacks
    # 2× weight to address near-zero recall on SF-state attacks (Credential Access,
    # novel exfil). S0/SYN attacks keep 1× weight. Uses random.choices (with
    # replacement) — duplicates are rare since pool >> FINAL_ATTACK.
    sf_attacks    = [s for s in attacks if s.get("conn_state", "-") in ("SF", "S1", "OTH")]
    other_attacks = [s for s in attacks if s.get("conn_state", "-") not in ("SF", "S1", "OTH")]
    weights       = [2.0] * len(sf_attacks) + [1.0] * len(other_attacks)
    print(f"  SF/S1/OTH attacks (2× weight): {len(sf_attacks):,} | other: {len(other_attacks):,}")
    attacks       = random.choices(sf_attacks + other_attacks, weights=weights, k=FINAL_ATTACK)

    final_train = attacks + benign
    random.shuffle(final_train)
    random.shuffle(eval_pool)

    def write_jsonl(path, samples):
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps({"messages": s["messages"]}) + "\n")

    write_jsonl(TRAIN_FILE, final_train)
    write_jsonl(EVAL_FILE,  eval_pool)

    print(f"\n✅ {len(final_train)} train samples → {TRAIN_FILE}")
    print(f"   Attacks: {len(attacks):>7,}  |  Benign: {len(benign):>7,}  "
          f"(ratio 1:{len(benign)/max(len(attacks),1):.1f})")
    print(f"✅ {len(eval_pool)} eval samples  → {EVAL_FILE}")

    print(f"\n   Train source breakdown:")
    sources = Counter(s["source"] for s in final_train)
    for src, n in sorted(sources.items()):
        a = sum(1 for s in final_train if s["source"] == src and s["verdict"] == "ATTACK")
        b = sum(1 for s in final_train if s["source"] == src and s["verdict"] == "FALSE POSITIVE")
        print(f"   {src:12s}: {n:>7,}  (atk {a:>6,} / ben {b:>6,})")
