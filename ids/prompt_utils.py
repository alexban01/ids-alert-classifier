"""Shared prompt utilities for IDS classifier training and inference."""

SYSTEM_PROMPT = (
    "You are a network security analyst. "
    "Always respond with VERDICT: <ATTACK or FALSE POSITIVE> on the first line, "
    "followed by REASON: <brief explanation>."
)

# Verdict-only variant: used by the --no-reason preprocessing ablation, which
# drops the (randomly-picked, non-grounded) REASON line from training targets.
# A model trained on verdict-only data must be benchmarked/deployed with THIS
# system prompt, not SYSTEM_PROMPT — otherwise the system instruction would ask
# for a REASON the model was never trained to emit.
SYSTEM_PROMPT_VERDICT_ONLY = (
    "You are a network security analyst. "
    "Respond with exactly one line: VERDICT: <ATTACK or FALSE POSITIVE>."
)

_NA_VALUES = (None, "", "-", "?", "None", "nan", "NaN")


def _sctx(d, key, default="N/A"):
    """Get a value from a context dict, returning default if absent/None/empty."""
    if not d:
        return default
    v = d.get(key)
    if v is None or str(v).strip() in ("", "-", "None", "nan"):
        return default
    return str(v).strip()


def _safe(v, fmt=".1f"):
    """Format a value as a float string, or 'N/A' if unset."""
    try:
        return format(float(v), fmt) if v not in _NA_VALUES else "N/A"
    except (ValueError, TypeError):
        return "N/A"


def _is_na(v):
    """Check whether a value represents an unset/missing field."""
    if v in _NA_VALUES:
        return True
    try:
        float(v)
        return False
    except (ValueError, TypeError):
        return True


def _fmt_port(v):
    """Format a port number as integer string, or 'N/A' if unset."""
    if v in _NA_VALUES:
        return "N/A"
    try:
        return str(int(float(v)))
    except (ValueError, TypeError):
        try:
            return str(int(str(v), 0))  # handles hex strings like "0x50"
        except (ValueError, TypeError):
            return "N/A"


def _fmt_int(v):
    """Format an integer-like value, or N/A if unset."""
    if v in _NA_VALUES:
        return "N/A"
    try:
        return str(int(float(v)))
    except (ValueError, TypeError):
        return "N/A"


def build_prompt(proto, duration, orig_pkts, resp_pkts,
                 orig_bytes, resp_bytes, conn_state, service="-",
                 resp_port="-", orig_port="-",
                 http_ctx=None, dns_ctx=None, ssl_ctx=None,
                 behavior_ctx=None):
    """Convert Zeek-native features to model prompt text.

    When base fields are unset (-, empty, etc.), derived fields
    (Bytes/sec, Orig Bytes/Pkt, Resp Bytes/Pkt) propagate as N/A
    rather than falling through to 0.0.

    Optional context dicts add application-layer sections to the prompt:
      http_ctx  — keys: method, host, uri, user_agent, status_code, resp_body_len
      dns_ctx   — keys: query, answers, qtype_name, ttl, rcode_name
      ssl_ctx   — keys: version, cipher, issuer, validation_status
      behavior_ctx — compact sliding-window behavior features from nearby conn.log rows
    Pass None to omit a section entirely (used for context masking in training).
    """
    dur_na = _is_na(duration)
    ob_na  = _is_na(orig_bytes)
    rb_na  = _is_na(resp_bytes)
    op_na  = _is_na(orig_pkts)
    rp_na  = _is_na(resp_pkts)

    # Parse numeric values (None if unset)
    try:
        dur_f = None if dur_na else float(duration)
        ob_f  = None if ob_na  else float(orig_bytes)
        rb_f  = None if rb_na  else float(resp_bytes)
        op_f  = None if op_na  else float(orig_pkts)
        rp_f  = None if rp_na  else float(resp_pkts)
    except (ValueError, TypeError):
        dur_f = ob_f = rb_f = op_f = rp_f = None

    # Derived: Bytes/sec — needs duration + both byte counts
    if dur_f is not None and ob_f is not None and rb_f is not None:
        bps = (ob_f + rb_f) / dur_f if dur_f > 0 else 0.0
    elif dur_na or ob_na or rb_na:
        bps = None  # N/A
    else:
        bps = 0.0

    # Derived: Orig Bytes/Pkt
    if ob_f is not None and op_f is not None:
        op_sz = ob_f / op_f if op_f > 0 else 0.0
    elif ob_na or op_na:
        op_sz = None  # N/A
    else:
        op_sz = 0.0

    # Derived: Resp Bytes/Pkt
    if rb_f is not None and rp_f is not None:
        rp_sz = rb_f / rp_f if rp_f > 0 else 0.0
    elif rb_na or rp_na:
        rp_sz = None  # N/A
    else:
        rp_sz = 0.0

    def _fmt(val, fmt):
        return "N/A" if val is None else format(val, fmt)

    svc = service if service not in _NA_VALUES else "N/A"

    lines = [
        "Analyze this network connection and classify it as ATTACK or FALSE POSITIVE.\n",
        f"  Proto:              {proto}",
        f"  Service:            {svc}",
        f"  Dest Port:          {_fmt_port(resp_port)}",
        f"  Src Port:           {_fmt_port(orig_port)}",
        f"  Duration (s):       {_safe(duration, '.6f')}",
        f"  Orig Packets:       {_safe(orig_pkts, '.0f')}",
        f"  Resp Packets:       {_safe(resp_pkts, '.0f')}",
        f"  Orig Bytes:         {_safe(orig_bytes, '.0f')}",
        f"  Resp Bytes:         {_safe(resp_bytes, '.0f')}",
        f"  Conn State:         {conn_state}",
        f"  Bytes/sec:          {_fmt(bps, '.1f')}",
        f"  Orig Bytes/Pkt:     {_fmt(op_sz, '.1f')}",
        f"  Resp Bytes/Pkt:     {_fmt(rp_sz, '.1f')}",
    ]

    # ── Optional application-layer context sections ────────────────────────────
    if http_ctx is not None:
        status  = _sctx(http_ctx, "status_code")
        rbl     = _sctx(http_ctx, "resp_body_len")
        lines += [
            "\n[HTTP]",
            f"  Method:     {_sctx(http_ctx, 'method')}",
            f"  Host:       {_sctx(http_ctx, 'host')}",
            f"  URI:        {_sctx(http_ctx, 'uri')}",
            f"  User-Agent: {_sctx(http_ctx, 'user_agent')}",
            f"  Status:     {status}    Response: {rbl} bytes",
        ]

    if dns_ctx is not None:
        rcode   = _sctx(dns_ctx, "rcode_name", "")
        nxdomain = "Yes" if rcode.upper() in ("NXDOMAIN", "NXERROR") else "No"
        lines += [
            "\n[DNS]",
            f"  Query:    {_sctx(dns_ctx, 'query')}",
            f"  Answer:   {_sctx(dns_ctx, 'answers')}    Type: {_sctx(dns_ctx, 'qtype_name')}",
            f"  TTL:      {_sctx(dns_ctx, 'ttl')}s    NXDOMAIN: {nxdomain}",
        ]

    if ssl_ctx is not None:
        lines += [
            "\n[SSL]",
            f"  Version:  {_sctx(ssl_ctx, 'version')}    Cipher: {_sctx(ssl_ctx, 'cipher')}",
            f"  Issuer:   {_sctx(ssl_ctx, 'issuer')}    Validated: {_sctx(ssl_ctx, 'validation_status', 'UNKNOWN')}",
        ]

    if behavior_ctx is not None:
        lines += [
            "\n[BEHAVIOR]",
            f"  Src Conns 60s:        {_fmt_int(behavior_ctx.get('src_conn_60s'))}",
            f"  Src Conns 300s:       {_fmt_int(behavior_ctx.get('src_conn_300s'))}",
            f"  Unique Dst IPs 60s:   {_fmt_int(behavior_ctx.get('src_unique_dst_60s'))}",
            f"  Unique Dst Ports 60s: {_fmt_int(behavior_ctx.get('src_unique_ports_60s'))}",
            f"  S0 / RSTO / SF 60s:   "
            f"{_fmt_int(behavior_ctx.get('src_s0_60s'))} / "
            f"{_fmt_int(behavior_ctx.get('src_rsto_60s'))} / "
            f"{_fmt_int(behavior_ctx.get('src_sf_60s'))}",
            f"  Pair Conns 300s:      {_fmt_int(behavior_ctx.get('pair_conn_300s'))}",
            f"  Mean Gap:             {_safe(behavior_ctx.get('pair_mean_gap_s'), '.1f')}s",
            f"  Periodic:             {_sctx(behavior_ctx, 'pair_periodic_score', 'Low')}",
            f"  Same Resp Port 60s:   {_fmt_int(behavior_ctx.get('same_resp_port_60s'))}",
            f"  Same-Size Repeats:    {_fmt_int(behavior_ctx.get('same_flow_size_repeats_300s'))}",
        ]

    return "\n".join(lines)


def build_host_prompt(host, summary):
    """Build a second-pass host-level prompt from aggregated flow behavior."""
    lines = [
        "Analyze this source host's recent Zeek activity and classify it as ATTACK or FALSE POSITIVE.\n",
        f"  Source Host:         {host}",
        f"  Total Flows:         {_fmt_int(summary.get('total_flows'))}",
        f"  Pass-1 ATTACK:       {_fmt_int(summary.get('pred_attack'))}",
        f"  Pass-1 Benign:       {_fmt_int(summary.get('pred_benign'))}",
        f"  Pass-1 Unknown:      {_fmt_int(summary.get('pred_unknown'))}",
        f"  Attack Ratio:        {_safe(100 * float(summary.get('attack_ratio', 0.0)), '.1f')}%",
        f"  Unique Dst IPs:      {_fmt_int(summary.get('unique_dst_ips'))}",
        f"  Unique Dst Ports:    {_fmt_int(summary.get('unique_dst_ports'))}",
        f"  S0 / RSTO / SF:      "
        f"{_fmt_int(summary.get('state_s0'))} / "
        f"{_fmt_int(summary.get('state_rsto'))} / "
        f"{_fmt_int(summary.get('state_sf'))}",
        f"  HTTP / DNS / SSL:    "
        f"{_fmt_int(summary.get('http_flows'))} / "
        f"{_fmt_int(summary.get('dns_flows'))} / "
        f"{_fmt_int(summary.get('ssl_flows'))}",
        f"  Periodic High:       {_fmt_int(summary.get('periodic_high'))}",
        f"  Periodic Medium:     {_fmt_int(summary.get('periodic_medium'))}",
        f"  Bursty Flows:        {_fmt_int(summary.get('bursty_flows'))}",
        f"  Fan-out Flows:       {_fmt_int(summary.get('fanout_flows'))}",
        f"  Same-Size Repeats:   {_fmt_int(summary.get('same_size_repeat_flows'))}",
        f"  Top Ports:           {_sctx(summary, 'top_ports')}",
        f"  Top Services:        {_sctx(summary, 'top_services')}",
        f"  Attack Ports:        {_sctx(summary, 'top_attack_ports')}",
        f"  Attack Destinations: {_sctx(summary, 'top_attack_dsts')}",
    ]
    return "\n".join(lines)


def extract_verdict(output):
    """Parse model output for VERDICT: line. Returns ATTACK, FALSE POSITIVE, or UNKNOWN."""
    for line in output.upper().splitlines():
        if "VERDICT:" in line:
            if "FALSE POSITIVE" in line:
                return "FALSE POSITIVE"
            if "ATTACK" in line:
                return "ATTACK"
    return "UNKNOWN"


def extract_reason(output):
    """Parse model output for REASON: line. Returns the reason string, or empty string."""
    for line in output.splitlines():
        if line.upper().startswith("REASON:"):
            return line[7:].strip()
    return ""
