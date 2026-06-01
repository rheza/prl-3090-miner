from miner.protocol_hashrate import DEFAULT_STRATUM_SHARE_UNIT, parse_lines


def test_share_equivalent_hashrate_uses_accepted_shares_only():
    stats = parse_lines([
        "authorized diff=32768",
        "share accepted diff=32768",
        "share accepted diff=32768",
        "share rejected diff=32768",
        "share stale diff=32768",
    ])

    summary = stats.summary(elapsed_sec=60, share_unit=DEFAULT_STRATUM_SHARE_UNIT, power_avg_w=320)

    assert summary["accepted_shares"] == 2
    assert summary["rejected"] == 1
    assert summary["stale"] == 1
    assert summary["pool_credited_th_s"] == round(2 * 32768 * DEFAULT_STRATUM_SHARE_UNIT / 60 / 1e12, 6)
    assert summary["efficiency_th_w"] == round(summary["pool_credited_th_s"] / 320, 6)


def test_reported_th_s_samples_are_averaged_separately_from_share_rate():
    stats = parse_lines([
        "hashrate 100.0 TH/s",
        "speed: 110.0 TH/s",
    ])

    summary = stats.summary(elapsed_sec=10, share_unit=DEFAULT_STRATUM_SHARE_UNIT)

    assert summary["raw_th_s"] == 105.0
    assert summary["pool_credited_th_s"] == 0.0
    assert summary["reported_samples"] == 2
