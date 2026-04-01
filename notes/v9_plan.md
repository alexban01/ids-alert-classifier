# v9.0 Plan: Multi-Log Enrichment via CTU-Malware-Capture Series

## Confirmed v8.1 State
- Training-distribution MCC: **+0.9942** (regenerated cache, 4 sources)
- OOD (CTU-Malware-Capture-Botnet-3 / Kelihos): **+0.06** — near-random

---

## Key Insight: CTU-Malware-Capture Series Already Has Full Zeek Output

Individual CTU captures have a `bro/` subdirectory with ALL Zeek logs:
```
https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-78-2/bro/
  conn.log   dns.log   http.log   ssl.log   files.log   weird.log   ...
```
No pcap reprocessing needed. Download the logs directly.

The binetflow labels can be matched to Zeek conn.log flows using the exact same
5-tuple normalization already implemented in `compare_binetflow.py` (`_norm_key()`).
This is already proven to work (used for Zeus OOD evaluation).

---

## Label Strategy

| Binetflow label | Training label | Notes |
|----------------|---------------|-------|
| `flow=From-Botnet-*` | **ATTACK** | Core attack signal |
| `From-Normal-*` | **FALSE POSITIVE** | Clean benign |
| `flow=Background-*` | **SKIP** | Unknown — not labeled by researchers |

Background is skipped (not treated as benign) to avoid label noise in training.
CTU-Normal (already in training at 100k cap) covers benign diversity.

---

## Data Pipeline: compare_binetflow.py → preprocess_zeek.py

`compare_binetflow.py` already implements:
1. Download conn.log (URL or local)
2. Download binetflow (URL or local)
3. Match by `_norm_key(proto, ip_a, port_a, ip_b, port_b)` — direction-agnostic 5-tuple
4. Build labeled flow list with Zeek-native fields

For v9.0, extend this matching to also:
5. Download dns.log, http.log, ssl.log from the same bro/ directory
6. Build a `uid → {dns_rows, http_rows, ssl_rows}` lookup
7. Enrich matched flows with application-layer context via uid

---

## New Multi-Log Prompt Format

Add conditional sections — omitted entirely when no matching log entries exist.
Random masking during training (50% probability per section) prevents the model
from learning "presence of http context = ATTACK".

```
Analyze this network connection and classify it as ATTACK or FALSE POSITIVE.

  Proto:          tcp          Dest Port:      4444
  Duration (s):   0.823456     Src Port:       54321
  Orig Packets:   6            Bytes/sec:      892.4
  Resp Packets:   4            Orig Bytes/Pkt: 102.4
  Orig Bytes:     614          Resp Bytes/Pkt: 89.0
  Resp Bytes:     356
  Conn State:     SF
  Service:        N/A

[HTTP]
  Method:     POST
  Host:       updates.corp-internal.net
  URI:        /gate.php
  User-Agent: Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)
  Status:     200    Response: 48 bytes

[DNS]
  Query:    updates.corp-internal.net
  Answer:   185.220.101.50    Type: A
  TTL:      60s    NXDOMAIN: No

[SSL]
  Version:  TLSv1.0    Cipher: RC4-MD5
  Issuer:   Self-Signed    Validated: FAILED
```

Sections present only when: a uid match exists in that log AND not randomly masked.
The [HTTP], [DNS], [SSL] labels are short single-word headers to save tokens.

---

## CTU-Malware-Capture Scenarios to Download

URL pattern: `https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-{N}/`

Target diverse botnet families — prioritize those NOT already in training (CTU-13 covers
scenarios 1-13 via binetflow; individual captures are separate and OOD):

| Scenario | Botnet | C2 Type | Why Useful |
|----------|--------|---------|-----------|
| Botnet-3 | Kelihos v1 | Spam / P2P | Currently 0.06 MCC — must improve |
| Botnet-42 | Ramnit | HTTP C2 | Banking trojan |
| Botnet-44 | Ngrbot | IRC C2 | Different C2 protocol |
| Botnet-52 | Htbot | HTTP C2 | HTTP-based C2 |
| Botnet-54 | Siemens | HTTP C2 | Industrial malware |
| Botnet-78-2 | Zeus | HTTP C2 | Already used for OOD eval; add to training |
| Botnet-90 | Pushdo | HTTPS C2 | TLS-based C2 — tests SSL context |
| Botnet-91 | Ballpit | HTTP C2 | Diverse HTTP botnet |

Download per scenario: `bro/conn.log`, `bro/dns.log`, `bro/http.log`, `bro/ssl.log`,
and the `.binetflow` labeled file. ~100-500 MB per scenario.

---

## Training Samples: Random Context Masking

To prevent the model from learning "http section present = ATTACK":

```python
CONTEXT_MASK_PROB = 0.50   # 50% chance each context section is dropped in training
```

This forces the model to:
1. Classify conn.log-only samples correctly (backwards compat with current sources)
2. Use context when available as supporting evidence, not as primary signal
3. Generalize: if it sees Zeus-like http.log at inference, it can reason about it
   even if training examples were sometimes masked

---

## Implementation Plan

### Step 1: New data loader — `load_ctu_malware_captures()`
New function in `preprocess_zeek.py` (or separate `load_ctu_malware.py`):
- Downloads bro/conn.log + dns.log + http.log + ssl.log for each configured scenario URL
- Downloads the binetflow labeled CSV
- Matches flows by 5-tuple (reuse `_norm_key` from compare_binetflow.py)
- Builds uid → {dns, http, ssl} lookup
- Returns list of `(flow_dict, label, context_dict)` tuples

### Step 2: Multi-log prompt — extend `prompt_utils.py`
```python
def build_prompt(proto, duration, orig_pkts, resp_pkts, orig_bytes, resp_bytes,
                 conn_state, orig_p=None, resp_p=None, service=None,
                 http_ctx=None, dns_ctx=None, ssl_ctx=None):
```
Each `*_ctx` is a dict of fields, or `None`. Section is appended only when non-None.
Context fields: minimal — only the most discriminative per log type (see format above).

### Step 3: Training context masking — in `preprocess_zeek.py`
When building training samples from CTU-Malware captures:
```python
if random.random() < CONTEXT_MASK_PROB:
    http_ctx = None   # drop this section for this sample
```
Applied independently per section per sample.

### Step 4: Increase `max_length` in `train.py`
- Current: 512 tokens
- Multi-log prompt (all sections): ~400-500 tokens
- Budget: 512 → 1024 to accommodate enriched prompts
- Impact: increases VRAM usage — may need RunPod for training

### Step 5: Update `classify_conn_log.py`
- Accept optional `--dns-log`, `--http-log`, `--ssl-log` arguments
- Build uid → context lookup from additional logs
- Pass context to `build_prompt()` for enriched inference

### Step 6: Add CTU-Malware OOD eval to `benchmark_realworld.py`
- Add CTU-Malware-Capture-Botnet-3 as a permanent eval source (the failing one)
- Track MCC on this OOD source across model versions

---

## Files to Create/Modify

| File | Type | Change |
|------|------|--------|
| `prompt_utils.py` | Modify | Extend `build_prompt()` with `http_ctx`, `dns_ctx`, `ssl_ctx` params |
| `preprocess_zeek.py` | Modify | Add `load_ctu_malware_captures()`; add context masking |
| `train.py` | Modify | `max_length`: 512 → 1024 |
| `classify_conn_log.py` | Modify | `--dns-log`, `--http-log`, `--ssl-log` args; uid-based context join |
| `benchmark_realworld.py` | Modify | Add CTU-Malware-Botnet-3 as permanent OOD regression test |
| `load_ctu_malware.py` | New (optional) | Separate module for CTU-Malware-Capture download + label matching |

`compare_binetflow.py` — no changes needed; `_norm_key()` and `load_binetflow()`
are reused as-is by the new loader.

---

## Overfitting Risks and Mitigations

### Risk 1: Context presence as a shortcut (biggest risk)
If only ATTACK samples have [HTTP]/[DNS]/[SSL] sections (because benign flows are
pure TCP or come from sources without those logs), the model learns "has http section
= ATTACK" rather than reasoning about the content.

**Mitigation A — 50% context masking** (already planned): forces the model to classify
conn.log-only samples correctly half the time.

**Mitigation B — benign traffic WITH context**: source benign HTTP flows that also have
http.log entries. Options:
- CTU-Normal bro/ output: download and check if any flows have http.log entries
- IoT-23 benign flows: some IoT devices make HTTP calls — check for http.log in bro/
- UWF-ZeekData24 benign: real university traffic, likely has HTTP context

If benign:http_ctx ratio is near 0% while attack:http_ctx ratio is high, masking
alone is not enough — the model will still learn the shortcut on unmasked samples.
**Verify after preprocessing: check context coverage % by label.**

### Risk 2: URI/UA string memorization
Training on one Zeus capture means the model sees `/gate.php` + `MSIE 6.0` exactly
once. It may memorize those strings rather than learning the general pattern
(unexpected UA + .php endpoint + small POST response = C2).

**Mitigation — multiple captures per botnet family**: find additional Zeus and Kelihos
captures with different C2 URIs/domains. The Stratosphere corpus has many individual
captures per family. More URI diversity = model learns the pattern, not the string.

### Risk 3: Source memorization (already present in v8.1)
v8.1 is MCC +0.9942 on training sources and +0.06 OOD — it has already memorized.
Adding more CTU-Malware captures risks the same: perfect recall on those captures,
no improvement on truly novel ones.

**Mitigation — held-out OOD validation**: keep at least one CTU-Malware scenario
completely out of training. Track its MCC at each checkpoint. Use it alongside
eval_loss for model selection — don't just optimize training-distribution eval_loss.

Recommended hold-out: CTU-Malware-Capture-Botnet-3 (Kelihos). It's the current
OOD test case (MCC +0.06) and should stay out of training to serve as the honest
generalization metric.

### Risk 4: Imbalanced context in reasons pool
The existing ATTACK_REASONS and BENIGN_REASONS pools don't mention http/dns/ssl.
If multi-log training samples have generic conn.log-style reasons, the model won't
learn to cite the context it was given.

**Done (v9 prep): reason pools expanded to 66 attack / 38 benign entries** in
preprocess_zeek.py, organized by category with web-research-verified patterns:

| Category | Patterns covered |
|----------|-----------------|
| Port scanning | Horizontal (S0 many IPs), vertical (many ports), SYN scan, full-connect scan |
| DDoS/DoS | UDP flood, SYN flood, HTTP flood, Slowloris, DNS/NTP amplification, ICMP flood |
| Brute force | SSH (22), RDP (3389), Telnet (23/Mirai), VNC (5900), DB ports (1433/3306/5432) |
| Exploitation | SMB/445, GlassFish/4848, Tomcat/8080, RSTO-after-payload pattern |
| Worms | Sequential IP scan, SMB/445 worm spread |
| IRC C2 | Port 6667 (Ngrbot/Neris/Rbot), non-standard ports like 2081 (Botnet-90) |
| HTTP C2 | Zeus GET config (port 80, tiny GET+response), Ramnit/Htbot gate polling |
| HTTPS C2 | Pushdo periodic beaconing, Pushdo fake-SSL flood (many RSTO/443 to many IPs) |
| Ramnit non-SSL/443 | TCP/443 with service="-" (own encrypted protocol on HTTPS port) |
| P2P C2 | Kelihos P2P (many IPs, high ports, bidirectional UDP/TCP) |
| Spam | SMTP/25 (Kelihos/Neris): high orig_bytes from desktop, many connections |
| DNS flood | Kelihos: 1.1M DNS queries pattern (extreme UDP/53 volume from single host) |
| Exfiltration | Large orig_bytes non-standard port, FTP upload from workstation (Ramnit) |
| DNS tunneling | Oversized DNS payload, high query rate |
| Backdoors | Long-duration non-standard port, reverse shell pattern |
| Industrial | Modbus/502, Siemens S7/102, OPC-UA/4840 |
| Proxy abuse | Htbot: port 8080/3128 from non-proxy endpoint |
| Fuzzing | Irregular bytes/states across many ports (UNSW-NB15) |
| Each attack reason has a complementary benign reason | |

**Still needed for Phase 3 (multi-log context)**: extend with http/dns/ssl-aware reasons:
- ATTACK: "MSIE 6.0 user-agent is obsolete and commonly spoofed by malware"
- ATTACK: "POST to .php endpoint with minimal response is consistent with C2 gate"
- ATTACK: "Self-signed certificate with failed validation indicates C2 over TLS"
- BENIGN: "Standard browser UA with expected HTTP 200 response to known CDN"
- BENIGN: "DNS query resolves to well-known IP range with stable TTL"
These will be added to preprocess_zeek.py when the multi-log loader is implemented.

### Risk 5: max_length increase inflates VRAM and reduces effective batch size
Going from 512 → 1024 tokens roughly doubles attention memory. On RTX 3070 (8 GB),
this may require halving the batch size, effectively halving training speed.

**Check before committing to 1024**: estimate the 95th percentile prompt length
after preprocessing. If most enriched prompts are ≤ 640 tokens, use 768 instead.
Saves significant VRAM without losing coverage.

---

## Verification

| Test | v8.1 baseline | v9.0 target |
|------|--------------|-------------|
| Training-distribution (4 sources, --regen) | MCC +0.9942 | ≥ +0.99 (must not regress) |
| OOD: CTU-Botnet-3 (Kelihos) | MCC +0.06 | MCC > +0.50 |
| OOD: CTU-Botnet-78-2 (Zeus) | MCC ~+0.04 | MCC > +0.50 (if http.log context present) |
| Format failure rate | 0% | 0% |

Note: conn.log-only inference on Zeus/Kelihos will still be near-random (the stealth C2
problem is fundamental at the flow level). The OOD improvement requires http.log context
to be present at inference time — Zeek provides this automatically from live traffic.
