import base64

import pytest

from .alphapool_client import AlphaPoolEndpoint, AlphaPoolJob, alpha_submit_message


def test_parse_alphapool_endpoint_defaults_scheme():
    ep = AlphaPoolEndpoint.parse("us2.alphapool.tech:5566")
    assert ep.host == "us2.alphapool.tech"
    assert ep.port == 5566
    assert ep.tls is False


def test_parse_alphapool_endpoint_tls_flag():
    ep = AlphaPoolEndpoint.parse("stratum+ssl://us2.alphapool.tech:5567")
    assert ep.host == "us2.alphapool.tech"
    assert ep.port == 5567
    assert ep.tls is True


def test_notify_params_parse_to_job():
    job = AlphaPoolJob.from_notify_params([
        "job-1", "aa", "bb", 65612, "6a1d3f02", "1b01fffe", True,
    ])
    assert job.job_id == "job-1"
    assert job.height == 65612
    assert job.clean_jobs is True


def test_notify_params_reject_wrong_arity():
    with pytest.raises(ValueError, match="expected 7"):
        AlphaPoolJob.from_notify_params(["job-1"])


def test_submit_message_uses_base64_plainproof_shape():
    msg = alpha_submit_message(5, "prl1paddr.worker", "job-1", b"plain-proof")
    assert msg == {
        "id": 5,
        "method": "mining.submit",
        "params": [
            "prl1paddr.worker",
            "job-1",
            base64.b64encode(b"plain-proof").decode("ascii"),
        ],
    }
