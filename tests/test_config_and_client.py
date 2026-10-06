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
        load_config(overrides=["win_rate.jury=[gemini_flash,gptoss20b,command]"])


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


def test_token_limiter_blocks_when_window_full(monkeypatch):
    from prefagent.llm import client as c
    clock = {"t": 0.0}
    monkeypatch.setattr(c.time, "monotonic", lambda: clock["t"])
    slept = []
    monkeypatch.setattr(c.time, "sleep", lambda s: (slept.append(s), clock.update(t=clock["t"] + s)))
    lim = c.TokenLimiter(tpm=1000, headroom=1.0)
    e = lim.acquire(600)
    lim.settle(e, 700)                 # real usage replaces the estimate
    lim.acquire(300)                   # 700 + 300 = 1000: fits
    assert not slept
    lim.acquire(100)                   # would exceed → waits for the window to roll over
    assert slept and slept[0] > 59


def test_config_files_do_not_share_top_level_keys():
    """Files are deep-merged, so a shared top-level key silently mixes two sections
    (this happened twice: `models`, then `generation`)."""
    import yaml
    from prefagent.config import CONFIG_DIR, DEFAULT_FILES
    seen = {}
    for name in DEFAULT_FILES:
        for key in yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text()):
            assert key not in seen, f"'{key}' defined in both {seen[key]} and {name}"
            seen[key] = name
