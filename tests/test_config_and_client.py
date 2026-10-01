import time

import pytest

from prefagent.config import load_config
from prefagent.llm.cache import ResponseCache, request_key
from prefagent.llm.client import RateLimiter


def test_default_config_is_valid():
    cfg = load_config()
    assert cfg["win_rate"]["jury"]


def test_juror_sharing_family_with_label_judges_is_rejected():
    with pytest.raises(ValueError, match="share a family"):
        load_config(overrides=["win_rate.jury=[gemini,kimi_k2,command]"])


def test_judging_must_be_compute_matched():
    with pytest.raises(ValueError, match="compute-matched"):
        load_config(overrides=["judging.single_judge.samples=3"])


def test_cache_roundtrip_and_usage(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    key = request_key({"model": "m", "messages": [], "sample_idx": 0})
    assert cache.get(key) is None
    cache.put(key, "m", {"x": 1}, {"text": "hi",
                                   "usage": {"prompt_tokens": 3, "completion_tokens": 2}})
    assert cache.get(key)["text"] == "hi"
    assert cache.usage_by_model() == {"m": {"calls": 1, "prompt_tokens": 3,
                                            "completion_tokens": 2}}


def test_sample_index_changes_cache_key():
    a = request_key({"model": "m", "sample_idx": 0})
    b = request_key({"model": "m", "sample_idx": 1})
    assert a != b


def test_rate_limiter_spaces_requests():
    limiter = RateLimiter(rpm=1200)          # 50 ms spacing
    start = time.monotonic()
    for _ in range(4):
        limiter.wait()
    assert time.monotonic() - start >= 0.14  # 3 gaps × 50 ms, with slack
