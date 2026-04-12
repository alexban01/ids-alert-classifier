"""
preprocess_config.py — Shared configuration, caps, and reason pools for preprocess_zeek.

All loader modules and the main entry point import from here.
"""

# ── Output files ───────────────────────────────────────────────────────────────
TRAIN_FILE  = "zeek_dataset.jsonl"
EVAL_FILE   = "zeek_dataset_eval.jsonl"
RANDOM_SEED = 42
EVAL_FRAC   = 0.10            # fraction of each (source, class) bucket held out for eval

# ── Scale factor ───────────────────────────────────────────────────────────────
# Set to 1.0 for full RunPod runs (~360k samples).
# Set to 0.03–0.1 for fast local validation on RTX 3070.
TRAINING_FACTOR = 1.0

# ── Per-source caps ────────────────────────────────────────────────────────────
MAX_PER_SOURCE_CLASS     = int(80_000 * TRAINING_FACTOR)   # default cap per (source, class)
IOT23_BENIGN_CAP         = int(20_000 * TRAINING_FACTOR)   # IoT-23 benign is 89% S0-dominated;
                                                            # reduced to avoid "S0 = benign" bias
# Per-file caps for IoT-23 parallel dispatch.
# Attack cap matches CTU13_FILE_CAP so IoT-23 (~50k total) competes fairly with
# CTU-13 (~52k) and UNSW (~40k) in the weighted draw — prevents S0-flood dominance.
IOT23_FILE_ATTACK_CAP    = int(3_000 * TRAINING_FACTOR)
IOT23_FILE_BENIGN_CAP    = max(50, int(1_000 * TRAINING_FACTOR))
CTU_NORMAL_CAP           = int(100_000 * TRAINING_FACTOR)  # only significant SF benign source
UWF_ATTACK_CAP           = int(25_000 * TRAINING_FACTOR)   # Credential Access + Defense Evasion
                                                            # re-enabled in v8.1 — cap prevents UWF
                                                            # from dominating the attack pool

# CTU balance knobs — tune these to control CTU-13 vs CTU-Malware contribution.
# CTU-13 has 13 binetflow files; CTU-Malware has 20 scenarios (v11).
# Total ceiling: CTU-13 = 13 × CTU13_FILE_CAP, CTU-Malware = 20 × CTU_MALWARE_SCENARIO_CAP.
# Raise CTU_MALWARE_SCENARIO_CAP and lower CTU13_FILE_CAP to shift weight toward
# diverse botnet families (the OOD-relevant data) and away from 2011 CTU-13 captures.
CTU13_FILE_CAP           = int(4_000 * TRAINING_FACTOR)    # per binetflow file  → 52k max total
CTU_MALWARE_SCENARIO_CAP = int(8_000 * TRAINING_FACTOR)    # per scenario        → 160k max total

# ── Final ratio targets ────────────────────────────────────────────────────────
# v7: 2:1 benign:attack — real networks are overwhelmingly benign; 1:1 training
# makes the model trigger-happy on real traffic.  TRAINING_FACTOR only controls
# per-source caps.  When the pool is smaller (fast local runs), all available
# samples are used with no artificial discard.
FINAL_ATTACK = 120_000
FINAL_BENIGN = 240_000

# Guaranteed CTU-Malware attack budget within FINAL_ATTACK.
# Without this, CTU-Malware ends up at ~19% of attack samples after the
# weighted random.choices draw because it competes with larger IoT-23/CTU-13 pools.
# v11 actual pool: ~16k attacks (14 of 19 scenarios have data; 5 return 0).
# Budget set just above actual pool so all CTU-Malware attacks are always taken.
# Raise if more scenarios get fixed and pool grows.
# Not scaled by TRAINING_FACTOR (same pattern as FINAL_ATTACK/FINAL_BENIGN):
# on small local runs, min(budget, pool_size) naturally falls back to pool_size.
CTU_MALWARE_ATTACK_BUDGET = 24_000

# ── Training-time masking probabilities ───────────────────────────────────────
CONN_STATE_MASK_PROB = 0.20   # blank conn_state to "-" for this fraction of samples;
                              # forces model to use numeric features when state is absent
CONTEXT_MASK_PROB    = 0.50   # per-section probability of dropping http/dns/ssl context;
                              # prevents "has http section → ATTACK" shortcut

# ── Hard-benign sampling ──────────────────────────────────────────────────────
HARD_BENIGN_MIN_SCORE   = 3     # score threshold to count a benign sample as "hard"
HARD_BENIGN_TARGET_FRAC = 0.35  # reserve up to this fraction of the benign budget for
                                # hard negatives (score >= HARD_BENIGN_MIN_SCORE) first

# ── Dataset paths ──────────────────────────────────────────────────────────────
CTU_MALWARE_DIR = "datasets/ctu-malware/"   # download cache for bro logs + binetflow

DATASETS = {
    "iot23":      "datasets/iot-23/",
    "ctu13":      "datasets/ctu-13/CTU-13-Dataset/",
    "unsw":       "datasets/unsw-nb15/",
    "cicids":     ".",          # looks for *.pcap_ISCX.csv in cwd
    "uwf":        "datasets/uwf-zeekdata24/",
    "ctu_normal": "datasets/ctu-normal/",
}

# CTU-Malware-Capture scenarios to include in training.
# Botnet-3 (Kelihos) is held out as the permanent OOD test in benchmark_realworld.py.
# URL pattern: https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-{ID}/
CTU_MALWARE_SCENARIOS = [
    # (scenario_id,  family,    base_url)
    # All scenarios verified to have bro/conn.log + labeled binetflow in
    # detailed-bidirectional-flow-labels/ (or .binetflow.labeled in root).
    # Botnet-3 (Kelihos) is held out as OOD probe in benchmark_realworld.py.
    ("Botnet-42",  "Ramnit",
     "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42"),
    ("Botnet-43",  "Neris",
     "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-43"),
    ("Botnet-44",  "Ngrbot",
     "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-44"),
    ("Botnet-45",  "Rbot",
     "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-45"),
    ("Botnet-46",  "Virut",
     "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-46"),
    ("Botnet-48",  "Sogou",
     "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-48"),
    ("Botnet-52",  "Htbot",
     "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-52"),
    ("Botnet-53",  "NSIS.ay",
     "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-53"),
    ("Botnet-54",  "Siemens",
     "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-54"),
    ("Botnet-78-2", "Zeus",
     "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-78-2"),
    # v11 additions — 10 new scenarios (4 new families + 6 additional captures)
    # All verified to have bro/conn.log + labeled binetflow.
    ("Botnet-25-1", "Zbot",
     "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-25-1"),
    # Botnet-25-2: binetflow downloaded but Label column entirely empty — unlabeled capture
    ("Botnet-47",   "DonBot",
     "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-47"),
    ("Botnet-49",   "Murlo",
     "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-49"),
    ("Botnet-50",   "Neris-v2",
     "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-50"),
    ("Botnet-51",   "Rbot-v2",
     "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-51"),
    # Botnet-55: binetflow Label column entirely empty — unlabeled capture
    # Botnet-61-1: binetflow Label column entirely empty — unlabeled capture
    # Botnet-64: binetflow Label column entirely empty — unlabeled capture
]

# ── Reason pools ───────────────────────────────────────────────────────────────

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
    "Short RSTO TCP connection to port 443 with near-zero orig_bytes and resp_bytes is consistent with a fake or malformed SSL handshake used by malware to blend C2 traffic among legitimate HTTPS connections.",

    # ── Ramnit port-443 non-SSL: established TCP/443, service absent, asymmetric bytes ──
    "Established TCP connection to port 443 with no identified SSL service and atypical byte ratios may indicate malware using a custom encrypted protocol on the HTTPS port rather than standard TLS.",

    # ── IRC C2 on non-standard ports (Botnet-90 uses 2081) ───────────────────
    "Persistent established TCP connection to an uncommon port (e.g. 2081, 194, 531) with low-rate symmetric byte exchange resembling IRC framing may indicate C2 using IRC on an alternate port to evade port-based filters.",

    # ── DNS anomaly per-flow (Kelihos DNS flood) ──────────────────────────────
    "UDP flow to port 53 with atypically large orig_bytes or resp_bytes relative to a normal DNS query/response may indicate DNS tunneling, oversized TXT record abuse, or botnet DNS flooding behavior.",

    # ── Zeus config-download pattern: small symmetric HTTP GET, immediate close ──
    "Short SF HTTP connection to port 80 with very small orig_bytes and similarly small resp_bytes followed by immediate close is consistent with a malware config or command file fetch (Zeus-style gate check-in).",

    # ── S0 SYN flow as part of scanning / worm discovery ─────────────────────
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
    "Short RSTO TCP connections to port 443 from a load balancer health check or CDN probe are normal and expected for infrastructure monitoring.",
    "An established TCP connection to port 443 with an unrecognized service field may reflect a non-standard TLS implementation, HTTP/3 negotiation, or proprietary application protocol over the HTTPS port.",
    "High-volume DNS queries from an internal resolver or CDN edge node are expected behavior for recursive resolution under load.",
    "TCP connections to port 8080 or 3128 from clients to a designated internal proxy or development web server are normal in enterprise and developer environments.",

    # ── Multi-log context complements (Phase 3) ────────────────────────────────
    "Standard browser User-Agent with an expected HTTP 200 response to a known CDN or vendor endpoint is consistent with legitimate web browsing or software telemetry.",
    "HTTP GET request to a well-known API endpoint with a normal-sized JSON response is consistent with routine application data retrieval.",
    "DNS query resolving to a well-known IP range (CDN, cloud provider, major vendor) with a stable TTL is consistent with normal application name resolution.",
    "DNS query for a recognizable public hostname with a NOERROR response and standard TTL is consistent with routine DNS resolution.",
    "TLS connection to a well-known hostname using a modern cipher suite and a certificate from a trusted CA is consistent with standard secure web traffic.",
    "SSL/TLS session with successful certificate validation and a current TLS version (1.2 or 1.3) is consistent with normal HTTPS browsing or API communication.",
]
