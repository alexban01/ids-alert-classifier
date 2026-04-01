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
import csv
import json
import re
import random
import tarfile
import glob
import math
import urllib.request

import pandas as pd

from behavior_features import build_behavior_contexts
from prompt_utils import SYSTEM_PROMPT, build_prompt

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_FILE    = "zeek_dataset.jsonl"
EVAL_FILE     = "zeek_dataset_eval.jsonl"
RANDOM_SEED   = 42
EVAL_FRAC     = 0.10            # fraction of each source held out for eval

# This is for fast training on my 3070
TRAINING_FACTOR = 0.03

MAX_PER_SOURCE_CLASS = int(80_000 * TRAINING_FACTOR)   # default cap per (source, class)
IOT23_BENIGN_CAP     = int(20_000 * TRAINING_FACTOR)   # IoT-23 benign is 89% S0-dominated; reduce to avoid
                                   # "S0 = benign" bias that hurts real-world SF traffic
CTU_NORMAL_CAP       = int(100_000 * TRAINING_FACTOR)  # increase — only significant SF benign source

# v7: 2:1 benign:attack ratio — real networks are overwhelmingly benign,
# training balanced (1:1) makes the model trigger-happy on real traffic.
# These are fixed targets for the full-scale run. TRAINING_FACTOR only controls
# per-source caps (how much is loaded). When the pool is smaller than the target
# (fast local runs), all available samples are used — no artificial discard.
FINAL_BENIGN  = 240_000
FINAL_ATTACK  = 120_000

CONN_STATE_MASK_PROB = 0.20   # fraction of samples where conn_state is blanked to "-"
                              # forces model to learn from numeric features when state is absent

CONTEXT_MASK_PROB    = 0.50   # per-section probability of dropping http/dns/ssl context in training
                              # prevents "has http section → ATTACK" shortcut; forces model to
                              # classify correctly from conn.log alone half the time

HARD_BENIGN_MIN_SCORE   = 3     # score threshold to count a benign sample as "hard"
HARD_BENIGN_TARGET_FRAC = 0.35  # when subsampling benigns, reserve up to this fraction
                                # for the hardest benign negatives first

CTU_MALWARE_DIR = "datasets/ctu-malware/"        # download cache for bro logs + binetflow

# CTU-Malware-Capture scenarios to include in training.
# Botnet-3 (Kelihos) is held out as permanent OOD test in benchmark_realworld.py.
# URL pattern: https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-{ID}/
CTU_MALWARE_SCENARIOS = [
    # (scenario_id,  family,    base_url)
    ("Botnet-42",  "Ramnit",
     "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42"),
    ("Botnet-44",  "Ngrbot",
     "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-44"),
    ("Botnet-52",  "Htbot",
     "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-52"),
    ("Botnet-54",  "Siemens",
     "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-54"),
    ("Botnet-78-2", "Zeus",
     "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-78-2"),
    ("Botnet-90",  "Pushdo",
     "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-90"),
    ("Botnet-91",  "Ballpit",
     "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-91"),
]

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

    # ── Multi-log context: HTTP (Phase 3 — used when http.log is available) ───
    "HTTP POST to a .php endpoint with minimal response body is consistent with a C2 gate check-in or command retrieval (Zeus/Ramnit pattern).",
    "Obsolete or spoofed User-Agent string (e.g. MSIE 6.0 on a modern OS) in an HTTP request is commonly used by malware to mimic old browsers while communicating with C2 infrastructure.",
    "HTTP request to an uncommon URI path (e.g. /gate.php, /config.bin, /panel/) with a small symmetric byte exchange is consistent with malware polling a C2 server for commands.",
    "HTTP response body of zero or very few bytes after a POST request suggests C2 acknowledgement with no payload — the server confirmed receipt but issued no command.",

    # ── Multi-log context: DNS (Phase 3 — used when dns.log is available) ─────
    "DNS query to a domain with high lexical entropy or random-looking subdomain is consistent with domain generation algorithm (DGA) C2 or DNS tunneling.",
    "NXDOMAIN response to a DNS query may indicate an active DGA botnet client iterating through generated domains looking for an active C2 server.",
    "DNS query to a recently registered or low-reputation domain with a very short TTL (< 60 seconds) is consistent with fast-flux C2 infrastructure.",

    # ── Multi-log context: SSL/TLS (Phase 3 — used when ssl.log is available) ─
    "SSL/TLS connection with a self-signed certificate or failed certificate validation is consistent with malware C2 over HTTPS without a legitimate PKI chain.",
    "Use of an obsolete TLS version (SSLv3, TLSv1.0) or weak cipher suite (RC4, NULL, EXPORT) in an established connection may indicate legacy malware C2 or an adversary-in-the-middle setup.",
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

    # ── Multi-log context complements (Phase 3) ────────────────────────────────
    # HTTP complements
    "Standard browser User-Agent with an expected HTTP 200 response to a known CDN or vendor endpoint is consistent with legitimate web browsing or software telemetry.",
    "HTTP GET request to a well-known API endpoint with a normal-sized JSON response is consistent with routine application data retrieval.",
    # DNS complements
    "DNS query resolving to a well-known IP range (CDN, cloud provider, major vendor) with a stable TTL is consistent with normal application name resolution.",
    "DNS query for a recognizable public hostname with a NOERROR response and standard TTL is consistent with routine DNS resolution.",
    # SSL complements
    "TLS connection to a well-known hostname using a modern cipher suite and a certificate from a trusted CA is consistent with standard secure web traffic.",
    "SSL/TLS session with successful certificate validation and a current TLS version (1.2 or 1.3) is consistent with normal HTTPS browsing or API communication.",
]

random.seed(RANDOM_SEED)

def pick_reason(verdict):
    pool = ATTACK_REASONS if verdict == "ATTACK" else BENIGN_REASONS
    return random.choice(pool)


def score_hard_benign(proto, conn_state, service="-", resp_port="-", orig_port="-",
                      http_ctx=None, dns_ctx=None, ssl_ctx=None, behavior_ctx=None):
    """Return (score, flags) for benign samples that look attack-like.

    Higher scores mean the benign flow is more useful as a hard negative.
    """
    score = 0
    flags = []

    proto_s = str(proto or "").strip().lower()
    state_s = str(conn_state or "").strip().upper()
    service_s = str(service or "").strip().lower()
    resp_port_s = str(resp_port or "").strip()

    try:
        resp_port_i = int(float(resp_port_s))
    except (ValueError, TypeError):
        resp_port_i = None

    if http_ctx is not None:
        score += 2
        flags.append("http_ctx")
        method = str(http_ctx.get("method") or "").upper()
        host = str(http_ctx.get("host") or "")
        uri = str(http_ctx.get("uri") or "")
        if method in ("GET", "POST"):
            score += 1
            flags.append(f"http_{method.lower()}")
        if host or uri:
            score += 1
            flags.append("http_named_endpoint")

    if dns_ctx is not None:
        score += 2
        flags.append("dns_ctx")
        rcode = str(dns_ctx.get("rcode_name") or "").upper()
        ttl = str(dns_ctx.get("ttl") or "").strip()
        if rcode in ("NXDOMAIN", "NXERROR"):
            score += 2
            flags.append("dns_nxdomain")
        try:
            if ttl and float(ttl) <= 60:
                score += 1
                flags.append("dns_short_ttl")
        except (ValueError, TypeError):
            pass

    if ssl_ctx is not None:
        score += 2
        flags.append("ssl_ctx")
        version = str(ssl_ctx.get("version") or "").upper()
        cipher = str(ssl_ctx.get("cipher") or "").upper()
        validation = str(ssl_ctx.get("validation_status") or "").upper()
        issuer = str(ssl_ctx.get("issuer") or "")
        if "FAILED" in validation or "SELF-SIGNED" in issuer.upper():
            score += 2
            flags.append("ssl_untrusted")
        if version in ("SSLV3", "TLSV1", "TLSV1.0") or "RC4" in cipher:
            score += 1
            flags.append("ssl_legacy")

    if resp_port_i in {22, 23, 25, 53, 80, 123, 443, 445, 502, 6667, 8080, 8443, 3128, 3389}:
        score += 1
        flags.append(f"port_{resp_port_i}")
    if resp_port_i in {23, 445, 6667, 3389, 4848}:
        score += 1
        flags.append("high_risk_service_port")

    if state_s in {"S0", "RSTO", "S1"}:
        score += 1
        flags.append(f"state_{state_s.lower()}")
    if resp_port_i == 443 and service_s in {"", "-", "unknown"}:
        score += 1
        flags.append("port443_no_service")
    if proto_s == "udp" and resp_port_i == 53:
        score += 1
        flags.append("udp_dns")

    if behavior_ctx:
        score += 1
        flags.append("behavior_ctx")
        src_conn_60s = int(behavior_ctx.get("src_conn_60s") or 0)
        unique_dst_60s = int(behavior_ctx.get("src_unique_dst_60s") or 0)
        unique_ports_60s = int(behavior_ctx.get("src_unique_ports_60s") or 0)
        same_port_60s = int(behavior_ctx.get("same_resp_port_60s") or 0)
        repeats_300s = int(behavior_ctx.get("same_flow_size_repeats_300s") or 0)
        periodic = str(behavior_ctx.get("pair_periodic_score") or "").lower()
        if src_conn_60s >= 10:
            score += 2
            flags.append("burst_60s")
        if unique_dst_60s >= 5 or unique_ports_60s >= 5:
            score += 2
            flags.append("fanout_60s")
        if same_port_60s >= 5:
            score += 1
            flags.append("same_port_repeat")
        if repeats_300s >= 3:
            score += 1
            flags.append("same_size_repeat")
        if periodic in {"high", "medium"}:
            score += 2 if periodic == "high" else 1
            flags.append(f"periodic_{periodic}")

    return score, flags


# ── Sample builder ─────────────────────────────────────────────────────────────
def make_sample(proto, duration, orig_pkts, resp_pkts,
                orig_bytes, resp_bytes, conn_state, verdict, source,
                service="-", resp_port="-", orig_port="-",
                http_ctx=None, dns_ctx=None, ssl_ctx=None, behavior_ctx=None):
    hard_benign_score = 0
    hard_benign_flags = []
    if verdict == "FALSE POSITIVE":
        hard_benign_score, hard_benign_flags = score_hard_benign(
            proto, conn_state, service=service, resp_port=resp_port, orig_port=orig_port,
            http_ctx=http_ctx, dns_ctx=dns_ctx, ssl_ctx=ssl_ctx, behavior_ctx=behavior_ctx,
        )

    # Mask conn_state with CONN_STATE_MASK_PROB — forces model to use numeric
    # features (bytes, packets, port) when state is unavailable or ambiguous.
    prompt_conn_state = "-" if random.random() < CONN_STATE_MASK_PROB else conn_state

    # Per-section context masking (50% chance each section is dropped).
    # Prevents "has http section → ATTACK" shortcut; forces conn.log-only correctness.
    if http_ctx is not None and random.random() < CONTEXT_MASK_PROB:
        http_ctx = None
    if dns_ctx is not None and random.random() < CONTEXT_MASK_PROB:
        dns_ctx = None
    if ssl_ctx is not None and random.random() < CONTEXT_MASK_PROB:
        ssl_ctx = None

    prompt  = build_prompt(proto, duration, orig_pkts, resp_pkts,
                           orig_bytes, resp_bytes, prompt_conn_state, service,
                           resp_port=resp_port, orig_port=orig_port,
                           http_ctx=http_ctx, dns_ctx=dns_ctx, ssl_ctx=ssl_ctx,
                           behavior_ctx=behavior_ctx)
    reason  = pick_reason(verdict)
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": f"VERDICT: {verdict}\nREASON: {reason}"},
        ],
        "source":            source,
        "verdict":           verdict,
        "conn_state":        conn_state,  # original (pre-mask) — used for SF oversampling
        "hard_benign_score": hard_benign_score,
        "hard_benign_flags": hard_benign_flags,
        "is_hard_benign":    hard_benign_score >= HARD_BENIGN_MIN_SCORE,
    }

# ── CTU-Malware-Capture helpers ────────────────────────────────────────────────

def _norm_key(proto, ip_a, port_a, ip_b, port_b):
    """Normalise 5-tuple so (A→B) and (B→A) produce the same lookup key."""
    pair_a = (str(ip_a).strip(), str(port_a).strip())
    pair_b = (str(ip_b).strip(), str(port_b).strip())
    lo, hi = (pair_a, pair_b) if pair_a <= pair_b else (pair_b, pair_a)
    return (str(proto).strip().lower(), lo[0], lo[1], hi[0], hi[1])


def _ctu_download(url, scenario_id, filename=None, optional=False):
    """Download url to CTU_MALWARE_DIR/{scenario_id}_{filename}. Returns path or None."""
    if filename is None:
        filename = url.rstrip("/").split("/")[-1]
    local = os.path.join(CTU_MALWARE_DIR, f"{scenario_id}_{filename}")
    if os.path.isfile(local):
        print(f"    [cache] {os.path.basename(local)}")
        return local
    try:
        os.makedirs(CTU_MALWARE_DIR, exist_ok=True)
        urllib.request.urlretrieve(url, local)
        print(f"    Downloaded {os.path.basename(local)} "
              f"({os.path.getsize(local) // 1024} KB)")
        return local
    except Exception as e:
        if optional:
            print(f"    [SKIP] {filename}: {e}")
            if os.path.isfile(local):
                os.remove(local)
            return None
        raise


def _find_binetflow_url(base_url):
    """Fetch the Stratosphere capture directory index and return the binetflow URL.

    Priority order:
    1. .binetflow.labeled in root (e.g. Botnet-78-2/Zeus)
    2. .binetflow in detailed-bidirectional-flow-labels/ subdir (e.g. Botnet-42/44/52/54)
    3. Plain .binetflow in root (fallback — may be unlabeled, yields 0 samples)
    """
    def _list_dir(url):
        req = urllib.request.Request(
            url.rstrip("/") + "/", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        return re.findall(r'href="([^"/][^"]*?)"', html)

    try:
        links = _list_dir(base_url)

        # Priority 1: .binetflow.labeled in root
        labeled = [l for l in links if l.endswith(".binetflow.labeled")]
        if labeled:
            return base_url.rstrip("/") + "/" + labeled[0].lstrip("/")

        # Priority 2: labeled binetflow in detailed-bidirectional-flow-labels/ subdir
        if any("detailed-bidirectional-flow-labels" in l for l in links):
            try:
                sub = _list_dir(
                    base_url.rstrip("/") + "/detailed-bidirectional-flow-labels/")
                bf = [l for l in sub if l.endswith(".binetflow")]
                if bf:
                    return (base_url.rstrip("/")
                            + "/detailed-bidirectional-flow-labels/"
                            + bf[0].lstrip("/"))
            except Exception:
                pass

        # Priority 3: plain .binetflow in root (may be unlabeled)
        plain = [l for l in links if l.endswith(".binetflow")]
        if plain:
            return base_url.rstrip("/") + "/" + plain[0].lstrip("/")

    except Exception as e:
        print(f"    [WARN] Could not fetch index for {base_url}: {e}")
    return None


def _parse_zeek_log(path):
    """Parse a Zeek TSV log (#fields header), return list of row dicts."""
    rows   = []
    fields = None
    sep    = "\t"
    unset  = {"-", "(empty)"}
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#fields"):
                fields = line[len("#fields"):].strip().split(sep)
            elif line.startswith("#separator"):
                raw = line.split()[-1]
                sep = (bytes(raw, "utf-8").decode("unicode_escape")
                       if "\\x" in raw else raw)
            elif line.startswith("#empty_field"):
                unset.add(line.split(sep)[-1])
            elif line.startswith("#unset_field"):
                unset.add(line.split(sep)[-1])
            elif line.startswith("#") or not line.strip():
                continue
            elif fields is not None:
                parts = line.split(sep)
                row = {k: (None if (parts[i] if i < len(parts) else "-") in unset
                           else (parts[i] if i < len(parts) else None))
                       for i, k in enumerate(fields)}
                rows.append(row)
    return rows


def _build_binetflow_lookup(path):
    """Parse binetflow CSV → {norm_5tuple: label} dict.

    Label mapping for training: Botnet → ATTACK, Normal → FALSE POSITIVE,
    Background → skip (label noise; CTU-Normal covers benign diversity).
    """
    lookup = {}
    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            if header is None:
                header = [h.strip() for h in row]
                idx    = {h: i for i, h in enumerate(header)}
                needed = {"Proto", "SrcAddr", "Sport", "DstAddr", "Dport", "Label"}
                missing = needed - set(header)
                if missing:
                    raise ValueError(f"binetflow missing columns: {missing}\nGot: {header}")
                continue
            if len(row) <= max(idx["Label"], idx["Proto"],
                               idx["SrcAddr"], idx["Sport"],
                               idx["DstAddr"], idx["Dport"]):
                continue
            raw_label = row[idx["Label"]].strip().lower()
            if "botnet" in raw_label or "malware" in raw_label:
                label = "ATTACK"
            elif "normal" in raw_label:
                label = "FALSE POSITIVE"
            else:
                continue  # Background → skip
            key = _norm_key(
                row[idx["Proto"]],
                row[idx["SrcAddr"]], row[idx["Sport"]],
                row[idx["DstAddr"]], row[idx["Dport"]],
            )
            if key not in lookup or label == "ATTACK":  # ATTACK wins on conflict
                lookup[key] = label
    return lookup


def _build_http_lookup(path):
    """Build uid → http_ctx dict from Zeek http.log (first request per uid)."""
    lookup = {}
    for row in _parse_zeek_log(path):
        uid = row.get("uid")
        if not uid or uid in lookup:
            continue
        lookup[uid] = {
            "method":        row.get("method"),
            "host":          row.get("host"),
            "uri":           row.get("uri"),
            "user_agent":    row.get("user_agent"),
            "status_code":   row.get("status_code"),
            "resp_body_len": row.get("response_body_len"),
        }
    return lookup


def _build_dns_lookup(path):
    """Build uid → dns_ctx dict from Zeek dns.log (first response per uid)."""
    lookup = {}
    for row in _parse_zeek_log(path):
        uid = row.get("uid")
        if not uid or uid in lookup:
            continue
        answers_raw = row.get("answers") or ""
        ttls_raw    = row.get("TTLs") or row.get("ttls") or ""
        lookup[uid] = {
            "query":      row.get("query"),
            "answers":    answers_raw.split(",")[0].strip() or None,
            "qtype_name": row.get("qtype_name"),
            "ttl":        ttls_raw.split(",")[0].strip() or None,
            "rcode_name": row.get("rcode_name"),
        }
    return lookup


def _build_ssl_lookup(path):
    """Build uid → ssl_ctx dict from Zeek ssl.log (first session per uid)."""
    lookup = {}
    for row in _parse_zeek_log(path):
        uid = row.get("uid")
        if not uid or uid in lookup:
            continue
        # Simplify issuer: if same as subject it's self-signed; otherwise show CN
        issuer_raw  = row.get("issuer")  or ""
        subject_raw = row.get("subject") or ""
        if not issuer_raw or issuer_raw == subject_raw:
            issuer = "Self-Signed"
        else:
            cn = next((p.replace("CN=", "").strip()
                       for p in issuer_raw.split(",")
                       if p.strip().startswith("CN=")), issuer_raw)
            issuer = cn[:48]
        lookup[uid] = {
            "version":           row.get("version"),
            "cipher":            row.get("cipher"),
            "issuer":            issuer,
            "validation_status": row.get("validation_status"),
        }
    return lookup


def load_ctu_malware_captures():
    """Download and parse CTU-Malware-Capture scenarios from Stratosphere Lab.

    For each scenario:
      1. Downloads bro/conn.log + binetflow (labeled) for flow-level labelling.
      2. Downloads bro/dns.log, http.log, ssl.log for application-layer context.
      3. Matches conn.log flows to binetflow labels via direction-agnostic 5-tuple.
      4. Builds uid → {http_ctx, dns_ctx, ssl_ctx} lookup from auxiliary logs.
      5. Returns training samples with randomly masked context (CONTEXT_MASK_PROB).

    Botnet-3 (Kelihos) is NOT included here — it is the permanent OOD hold-out
    tracked in benchmark_realworld.py.

    Label mapping: Botnet → ATTACK, Normal → FALSE POSITIVE, Background → skip.
    Cap: MAX_PER_SOURCE_CLASS (same as CTU-13/IoT-23/UNSW) — same label quality,
    same research group methodology, no reason to under-cap this source.
    """
    all_samples = []

    for scenario_id, family, base_url in CTU_MALWARE_SCENARIOS:
        print(f"\n[CTU-Malware {scenario_id} / {family}]")

        # ── 1. Find and download binetflow ────────────────────────────────────
        binetflow_url = _find_binetflow_url(base_url)
        if not binetflow_url:
            print(f"  [SKIP] No binetflow found for {scenario_id}")
            continue
        try:
            binetflow_path = _ctu_download(binetflow_url, scenario_id)
        except Exception as e:
            print(f"  [SKIP] binetflow download failed: {e}")
            continue

        # ── 2. Download conn.log ──────────────────────────────────────────────
        try:
            conn_path = _ctu_download(f"{base_url}/bro/conn.log", scenario_id, "conn.log")
        except Exception as e:
            print(f"  [SKIP] conn.log download failed: {e}")
            continue

        # ── 3. Download optional logs (non-fatal if missing) ──────────────────
        http_path = _ctu_download(f"{base_url}/bro/http.log", scenario_id,
                                  "http.log", optional=True)
        dns_path  = _ctu_download(f"{base_url}/bro/dns.log",  scenario_id,
                                  "dns.log",  optional=True)
        ssl_path  = _ctu_download(f"{base_url}/bro/ssl.log",  scenario_id,
                                  "ssl.log",  optional=True)

        # ── 4. Build label and context lookups ────────────────────────────────
        try:
            flow_labels = _build_binetflow_lookup(binetflow_path)
        except Exception as e:
            print(f"  [SKIP] binetflow parse failed: {e}")
            continue

        uid_http = _build_http_lookup(http_path) if http_path else {}
        uid_dns  = _build_dns_lookup(dns_path)   if dns_path  else {}
        uid_ssl  = _build_ssl_lookup(ssl_path)   if ssl_path  else {}

        print(f"  Labels: {sum(1 for v in flow_labels.values() if v=='ATTACK')} attacks, "
              f"{sum(1 for v in flow_labels.values() if v=='FALSE POSITIVE')} benign  "
              f"| http={len(uid_http)} dns={len(uid_dns)} ssl={len(uid_ssl)} uids")

        # ── 5. Process conn.log ───────────────────────────────────────────────
        buckets = {"ATTACK": [], "FALSE POSITIVE": []}
        conn_rows = []
        with open(conn_path, errors="replace") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 19:
                    continue
                conn_rows.append({
                    "ts":         parts[0],
                    "uid":        parts[1],
                    "orig_h":     parts[2],
                    "orig_p":     parts[3],
                    "resp_h":     parts[4],
                    "resp_p":     parts[5],
                    "proto":      parts[6],
                    "service":    parts[7],
                    "duration":   parts[8],
                    "orig_bytes": parts[9],
                    "resp_bytes": parts[10],
                    "conn_state": parts[11],
                    "orig_pkts":  parts[16] if len(parts) > 16 else "-",
                    "resp_pkts":  parts[18] if len(parts) > 18 else "-",
                })

        behavior_ctxs = build_behavior_contexts(conn_rows)
        for row, behavior_ctx in zip(conn_rows, behavior_ctxs):
            key = _norm_key(
                row["proto"], row["orig_h"], row["orig_p"], row["resp_h"], row["resp_p"]
            )
            label = flow_labels.get(key)
            if label is None:
                continue

            bucket = buckets[label]
            if len(bucket) >= MAX_PER_SOURCE_CLASS:
                continue

            bucket.append(make_sample(
                proto      = row["proto"],
                duration   = row["duration"],
                orig_pkts  = row["orig_pkts"],
                resp_pkts  = row["resp_pkts"],
                orig_bytes = row["orig_bytes"],
                resp_bytes = row["resp_bytes"],
                conn_state = row["conn_state"],
                verdict    = label,
                source     = "ctu_malware",
                service    = row["service"],
                orig_port  = row["orig_p"],
                resp_port  = row["resp_p"],
                http_ctx   = uid_http.get(row["uid"]),
                dns_ctx    = uid_dns.get(row["uid"]),
                ssl_ctx    = uid_ssl.get(row["uid"]),
                behavior_ctx=behavior_ctx,
            ))

        atk = len(buckets["ATTACK"])
        ben = len(buckets["FALSE POSITIVE"])
        print(f"  {scenario_id}: {atk} attacks, {ben} benign sampled")
        all_samples.extend(buckets["ATTACK"] + buckets["FALSE POSITIVE"])

    total_atk = sum(1 for s in all_samples if s["verdict"] == "ATTACK")
    total_ben = sum(1 for s in all_samples if s["verdict"] == "FALSE POSITIVE")
    print(f"\n  CTU-Malware total: {total_atk} attacks, {total_ben} benign "
          f"across {len(CTU_MALWARE_SCENARIOS)} scenarios")
    return all_samples


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
            rows = []
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

                rows.append({
                    "ts":         parts[0],
                    "orig_h":     parts[2],
                    "orig_p":     orig_port,
                    "resp_h":     parts[4],
                    "resp_p":     resp_port,
                    "proto":      proto,
                    "service":    service,
                    "duration":   duration,
                    "orig_bytes": orig_bytes,
                    "resp_bytes": resp_bytes,
                    "conn_state": conn_state,
                    "orig_pkts":  orig_pkts,
                    "resp_pkts":  resp_pkts,
                    "verdict":    verdict,
                })
                cap    = IOT23_BENIGN_CAP if verdict == "FALSE POSITIVE" else MAX_PER_SOURCE_CLASS
                lines_read += 1
                if verdict == "ATTACK":  attacks += 1
                else:                    benign  += 1

            behavior_ctxs = build_behavior_contexts(rows)
            for row, behavior_ctx in zip(rows, behavior_ctxs):
                cap = IOT23_BENIGN_CAP if row["verdict"] == "FALSE POSITIVE" else MAX_PER_SOURCE_CLASS
                bucket = samples[row["verdict"]]
                if len(bucket) < cap:
                    bucket.append(make_sample(
                        row["proto"], row["duration"], row["orig_pkts"], row["resp_pkts"],
                        row["orig_bytes"], row["resp_bytes"], row["conn_state"],
                        row["verdict"], "iot23", service=row["service"],
                        resp_port=row["resp_p"], orig_port=row["orig_p"],
                        behavior_ctx=behavior_ctx,
                    ))

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
        rows = []
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

                rows.append({
                    "ts":         parts[0],
                    "orig_h":     parts[2],
                    "orig_p":     orig_port,
                    "resp_h":     parts[4],
                    "resp_p":     resp_port,
                    "proto":      proto,
                    "service":    service,
                    "duration":   duration,
                    "orig_bytes": orig_bytes,
                    "resp_bytes": resp_bytes,
                    "conn_state": conn_state,
                    "orig_pkts":  orig_pkts,
                    "resp_pkts":  resp_pkts,
                })
                count += 1

        behavior_ctxs = build_behavior_contexts(rows)
        for row, behavior_ctx in zip(rows, behavior_ctxs):
            samples.append(make_sample(
                row["proto"], row["duration"], row["orig_pkts"], row["resp_pkts"],
                row["orig_bytes"], row["resp_bytes"], row["conn_state"],
                "FALSE POSITIVE", "ctu_normal", service=row["service"],
                resp_port=row["resp_p"], orig_port=row["orig_p"],
                behavior_ctx=behavior_ctx,
            ))

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
    # v9.0: CTU-Malware-Capture series — multi-log enriched training samples.
    # Downloads conn.log + binetflow (for labels) + dns/http/ssl.log (for context).
    # Botnet-3 (Kelihos) held out as OOD test — not included here.
    all_samples += load_ctu_malware_captures()

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
    print(f"Hard benigns  : {sum(1 for s in benign if s.get('is_hard_benign'))} "
          f"(score >= {HARD_BENIGN_MIN_SCORE})")

    # SF-state attack oversampling: give completed/established attacks 2× weight to
    # address near-zero recall on SF-state attacks (Credential Access, HTTP C2, exfil).
    # S0/SYN attacks keep 1× weight. Uses random.choices (with replacement) at full
    # scale (pool >> FINAL_ATTACK). At small TRAINING_FACTOR the pool may be smaller
    # than FINAL_ATTACK — k is capped at pool size to avoid pure duplication.
    sf_attacks    = [s for s in attacks if s.get("conn_state", "-") in ("SF", "S1", "OTH")]
    other_attacks = [s for s in attacks if s.get("conn_state", "-") not in ("SF", "S1", "OTH")]
    weights       = [2.0] * len(sf_attacks) + [1.0] * len(other_attacks)
    k_attacks     = min(FINAL_ATTACK, len(attacks))
    print(f"  SF/S1/OTH attacks (2× weight): {len(sf_attacks):,} | other: {len(other_attacks):,}")
    attacks       = random.choices(sf_attacks + other_attacks, weights=weights, k=k_attacks)

    # Enforce 2:1 ratio: benign target is 2× actual attacks taken, capped at pool size.
    # v9.1: reserve part of the benign budget for hard negatives that look
    # attack-like by state/port/context/behavior, then fill the rest randomly.
    k_benign      = min(FINAL_BENIGN, 2 * k_attacks, len(benign))
    hard_benign   = [s for s in benign if s.get("is_hard_benign")]
    other_benign  = [s for s in benign if not s.get("is_hard_benign")]

    if len(benign) <= k_benign:
        random.shuffle(benign)
    else:
        random.shuffle(hard_benign)
        hard_benign.sort(key=lambda s: s.get("hard_benign_score", 0), reverse=True)
        hard_keep = min(len(hard_benign), int(k_benign * HARD_BENIGN_TARGET_FRAC))
        selected_hard = hard_benign[:hard_keep]

        remaining = k_benign - len(selected_hard)
        random.shuffle(other_benign)
        benign = selected_hard + other_benign[:remaining]
        random.shuffle(benign)

    final_train = attacks + benign
    random.shuffle(final_train)
    random.shuffle(eval_pool)

    def write_jsonl(path, samples):
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps({"messages": s["messages"]}) + "\n")

    write_jsonl(TRAIN_FILE, final_train)
    write_jsonl(EVAL_FILE,  eval_pool)

    # Context coverage check — warns if attack:http_ctx ratio >> benign:http_ctx ratio
    def _ctx_pct(pool, key):
        has_ctx = sum(1 for s in pool
                      if any(key in msg["content"]
                             for msg in s["messages"] if msg["role"] == "user"))
        return 100 * has_ctx / max(len(pool), 1)

    atk_pool = [s for s in final_train if s["verdict"] == "ATTACK"]
    ben_pool = [s for s in final_train if s["verdict"] == "FALSE POSITIVE"]
    for section in ("[HTTP]", "[DNS]", "[SSL]", "[BEHAVIOR]"):
        ap = _ctx_pct(atk_pool, section)
        bp = _ctx_pct(ben_pool, section)
        flag = " ⚠ imbalanced" if ap > 0 and bp == 0 else ""
        print(f"   Context {section}: atk {ap:.1f}% / ben {bp:.1f}%{flag}")

    print(f"\n✅ {len(final_train)} train samples → {TRAIN_FILE}")
    print(f"   Attacks: {len(attacks):>7,}  |  Benign: {len(benign):>7,}  "
          f"(ratio 1:{len(benign)/max(len(attacks),1):.1f})")
    train_hard_benign = [s for s in final_train if s["verdict"] == "FALSE POSITIVE"
                         and s.get("is_hard_benign")]
    print(f"   Hard benign kept: {len(train_hard_benign):>7,}  "
          f"({100*len(train_hard_benign)/max(len(benign),1):.1f}% of benign train)")
    print(f"✅ {len(eval_pool)} eval samples  → {EVAL_FILE}")

    print(f"\n   Train source breakdown:")
    sources = Counter(s["source"] for s in final_train)
    for src, n in sorted(sources.items()):
        a = sum(1 for s in final_train if s["source"] == src and s["verdict"] == "ATTACK")
        b = sum(1 for s in final_train if s["source"] == src and s["verdict"] == "FALSE POSITIVE")
        print(f"   {src:12s}: {n:>7,}  (atk {a:>6,} / ben {b:>6,})")

    print(f"\n   Hard benign source breakdown:")
    hb_sources = Counter(s["source"] for s in train_hard_benign)
    for src, n in sorted(hb_sources.items()):
        avg_score = (sum(s.get("hard_benign_score", 0) for s in train_hard_benign
                         if s["source"] == src) / max(n, 1))
        print(f"   {src:12s}: {n:>7,}  (avg score {avg_score:.1f})")
