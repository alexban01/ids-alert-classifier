"""Shared prompt utilities for IDS classifier training and inference."""

SYSTEM_PROMPT = (
    "You are a network security analyst. "
    "Always respond with VERDICT: <ATTACK or FALSE POSITIVE> on the first line, "
    "followed by REASON: <brief explanation>."
)

_NA_VALUES = (None, "", "-", "?")


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


def build_prompt(proto, duration, orig_pkts, resp_pkts,
                 orig_bytes, resp_bytes, conn_state):
    """Convert Zeek-native features to model prompt text.

    When base fields are unset (-, empty, etc.), derived fields
    (Bytes/sec, Orig Bytes/Pkt, Resp Bytes/Pkt) propagate as N/A
    rather than falling through to 0.0.
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

    lines = [
        "Analyze this network connection and classify it as ATTACK or FALSE POSITIVE.\n",
        f"  Proto:              {proto}",
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
