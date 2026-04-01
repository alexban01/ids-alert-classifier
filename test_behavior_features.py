from behavior_features import _periodic_label, build_behavior_contexts


def _row(ts, orig_h="10.0.0.1", resp_h="1.1.1.1", resp_p="443", proto="tcp",
         conn_state="SF", orig_bytes="100", resp_bytes="200"):
    return {
        "ts": ts,
        "orig_h": orig_h,
        "resp_h": resp_h,
        "resp_p": resp_p,
        "proto": proto,
        "conn_state": conn_state,
        "orig_bytes": orig_bytes,
        "resp_bytes": resp_bytes,
    }


def test_empty_input_returns_empty_list():
    assert build_behavior_contexts([]) == []


def test_single_row_has_zero_window_counts():
    contexts = build_behavior_contexts([_row("1710000000.0")])
    assert len(contexts) == 1
    ctx = contexts[0]
    assert ctx is not None
    assert ctx["src_conn_60s"] == 0
    assert ctx["src_conn_300s"] == 0
    assert ctx["src_unique_dst_60s"] == 0
    assert ctx["src_unique_ports_60s"] == 0
    assert ctx["src_s0_60s"] == 0
    assert ctx["src_rsto_60s"] == 0
    assert ctx["src_sf_60s"] == 0
    assert ctx["pair_conn_300s"] == 0
    assert ctx["same_resp_port_60s"] == 0
    assert ctx["same_flow_size_repeats_300s"] == 0
    assert ctx["pair_periodic_score"] == "Low"
    assert ctx["pair_mean_gap_s"] is None


def test_two_rows_30s_apart_sets_src_conn_60s_to_one_on_second():
    rows = [_row("1710000000.0"), _row("1710000030.0")]
    contexts = build_behavior_contexts(rows)
    assert contexts[1] is not None
    assert contexts[1]["src_conn_60s"] == 1


def test_missing_ts_rows_return_none_context_entries():
    rows = [_row(None), _row("-"), _row("1710000000.0")]
    contexts = build_behavior_contexts(rows)
    assert contexts[0] is None
    assert contexts[1] is None
    assert contexts[2] is not None


def test_periodic_label_three_identical_gaps_is_high():
    assert _periodic_label([30.0, 30.0, 30.0]) == "High"


def test_periodic_label_fewer_than_three_gaps_is_low():
    assert _periodic_label([30.0, 30.0]) == "Low"

