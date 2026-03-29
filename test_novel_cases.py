"""
test_novel_cases.py — Hand-crafted novel test cases for ids-classifier.

Scenarios are designed to be:
  - NOT in any training source (no IoT-23 / CTU-13 / UWF / CTU-Normal flows)
  - Representative of real-world attack and benign patterns
  - Diverse across protocol, conn_state, and traffic profile

Usage:
    .venv/bin/python test_novel_cases.py [MODEL]
"""

import sys
import json
import urllib.request
import urllib.error

from prompt_utils import SYSTEM_PROMPT, build_prompt, extract_verdict

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = sys.argv[1] if len(sys.argv) > 1 else "ids-classifier"

# ── Test cases ────────────────────────────────────────────────────────────────
# Each entry: (scenario_name, expected, proto, duration, orig_pkts, resp_pkts,
#              orig_bytes, resp_bytes, conn_state, service, resp_port, orig_port)

CASES = [
    # ── ATTACKS ──────────────────────────────────────────────────────────────

    # UDP flood: 5000 packets in 0.5s, no response — classic volumetric DDoS
    ("UDP flood (DDoS)",
     "ATTACK",
     "udp", "0.5", "5000", "0", "320000", "0", "S0", "-", "9999", "54321"),

    # SSH brute force: rapid RSTO, tiny symmetric packets, ssh service
    ("SSH brute force",
     "ATTACK",
     "tcp", "0.12", "4", "3", "540", "480", "RSTO", "ssh", "22", "61200"),

    # DNS tunneling: DNS packets 10x normal size (normal DNS < 100 bytes)
    ("DNS tunneling (oversized queries)",
     "ATTACK",
     "udp", "0.002", "1", "1", "512", "512", "SF", "dns", "53", "55312"),

    # SMB exploit probe: RSTO after sending large payload to SMB port
    ("SMB exploit attempt (EternalBlue-style)",
     "ATTACK",
     "tcp", "0.08", "8", "2", "2048", "156", "RSTO", "smb", "445", "49800"),

    # Slow loris: connection held open 5 min, almost no data — HTTP DoS
    ("Slow Loris HTTP DoS",
     "ATTACK",
     "tcp", "300.0", "15", "8", "960", "1200", "S1", "http", "80", "58432"),

    # ICMP flood: 1000 packets in 1s, no replies — ping flood
    ("ICMP ping flood",
     "ATTACK",
     "icmp", "1.0", "1000", "0", "64000", "0", "OTH", "-", "-", "-"),

    # Data exfiltration: 12 MB upload vs 18 KB response — highly asymmetric
    ("Data exfiltration (large upload)",
     "ATTACK",
     "tcp", "45.2", "8000", "120", "12000000", "18000", "SF", "-", "4444", "52100"),

    # SYN scan: single SYN, no response — port scanning
    ("SYN port scan",
     "ATTACK",
     "tcp", "-", "1", "0", "60", "0", "S0", "-", "8080", "63000"),

    # ── BENIGN ───────────────────────────────────────────────────────────────

    # Video streaming: large asymmetric download, long duration, SSL
    ("Video streaming (Netflix/YouTube)",
     "FALSE POSITIVE",
     "tcp", "120.0", "200", "5000", "15000", "18000000", "SF", "ssl", "443", "51200"),

    # NTP sync: tiny symmetric UDP, sub-millisecond
    ("NTP clock sync",
     "FALSE POSITIVE",
     "udp", "0.003", "1", "1", "48", "48", "SF", "ntp", "123", "58900"),

    # Normal HTTPS page load: moderate download, short duration
    ("Normal HTTPS browsing",
     "FALSE POSITIVE",
     "tcp", "0.45", "12", "18", "1200", "45000", "SF", "ssl", "443", "54876"),

    # Interactive SSH admin session: long, bidirectional, moderate bytes
    ("SSH admin session (legitimate)",
     "FALSE POSITIVE",
     "tcp", "420.3", "580", "620", "85000", "120000", "SF", "ssh", "22", "62001"),

    # Software update: large one-way download over HTTPS
    ("Software update download",
     "FALSE POSITIVE",
     "tcp", "8.2", "55", "4200", "4200", "6300000", "SF", "ssl", "443", "50123"),

    # Routine DNS lookup: tiny, fast, SF
    ("Normal DNS lookup",
     "FALSE POSITIVE",
     "udp", "0.001", "1", "1", "65", "120", "SF", "dns", "53", "57777"),

    # Internal DB query: short, tiny, bidirectional SF
    ("Database health check",
     "FALSE POSITIVE",
     "tcp", "0.004", "5", "6", "480", "1200", "SF", "-", "5432", "60100"),

    # SMTP email delivery: moderate send, short-ish duration
    ("SMTP email delivery",
     "FALSE POSITIVE",
     "tcp", "2.1", "35", "28", "28000", "5400", "SF", "smtp", "25", "49200"),
]

# ── Ollama call ───────────────────────────────────────────────────────────────

def build_qwen_prompt(system, user):
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def classify(prompt_text):
    payload = json.dumps({
        "model":  MODEL,
        "prompt": build_qwen_prompt(SYSTEM_PROMPT, prompt_text),
        "stream": False,
        "raw":    True,
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["response"]

# ── Run ───────────────────────────────────────────────────────────────────────

print(f"\nModel : {MODEL}")
print(f"Cases : {len(CASES)}\n")
print(f"{'#':<3} {'Scenario':<42} {'Expected':<16} {'Got':<16} {'OK'}")
print(f"{'─'*3} {'─'*42} {'─'*16} {'─'*16} {'─'*4}")

correct = 0
results = []

for i, (name, expected, proto, dur, op, rp, ob, rb, state, svc, resp_p, orig_p) in enumerate(CASES, 1):
    prompt_text = build_prompt(proto, dur, op, rp, ob, rb, state, svc,
                               resp_port=resp_p, orig_port=orig_p)
    raw         = classify(prompt_text)
    verdict     = extract_verdict(raw)
    ok          = verdict == expected
    if ok:
        correct += 1
    mark = "✓" if ok else "✗"
    print(f"{i:<3} {name:<42} {expected:<16} {verdict:<16} {mark}")
    results.append((name, expected, verdict, raw))

# ── Summary ───────────────────────────────────────────────────────────────────

atk_cases = [(n, e, g) for n, e, g, _ in results if e == "ATTACK"]
ben_cases = [(n, e, g) for n, e, g, _ in results if e == "FALSE POSITIVE"]
atk_correct = sum(1 for _, e, g in atk_cases if e == g)
ben_correct = sum(1 for _, e, g in ben_cases if e == g)

print(f"\n{'─'*80}")
print(f"Overall  : {correct}/{len(CASES)} correct  ({100*correct/len(CASES):.0f}%)")
print(f"Attacks  : {atk_correct}/{len(atk_cases)} correct  ({100*atk_correct/max(len(atk_cases),1):.0f}%)")
print(f"Benign   : {ben_correct}/{len(ben_cases)} correct  ({100*ben_correct/max(len(ben_cases),1):.0f}%)")

# Print raw model output for wrong cases
wrong = [(n, e, g, r) for n, e, g, r in results if g != e]
if wrong:
    print(f"\n── Wrong predictions ──────────────────────────────────────────────────────")
    for name, expected, got, raw in wrong:
        print(f"\n  [{name}]  expected={expected}  got={got}")
        print(f"  Model said: {raw.strip()[:200]}")
