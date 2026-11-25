import pytest

from promptquery.llm import (
    CachingLLMClient,
    LLMClient,
    LLMError,
    is_retriable,
    with_retries,
)


class Boom(Exception):
    def __init__(self, status_code=None):
        super().__init__(f"boom {status_code}")
        self.status_code = status_code


def test_retriable_classification():
    assert is_retriable(Boom(429)) and is_retriable(Boom(529)) and is_retriable(Boom(500))
    assert not is_retriable(Boom(400)) and not is_retriable(Boom(401)) and not is_retriable(ValueError())

    class APIConnectionError(Exception): ...

    assert is_retriable(APIConnectionError())


def test_retries_then_succeeds_with_exponential_backoff():
    calls, sleeps = [], []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise Boom(429)
        return "ok"

    assert with_retries(fn, sleep=sleeps.append) == "ok"
    assert len(calls) == 3 and sleeps == [1.0, 2.0]


def test_gives_up_after_attempts_with_llm_error():
    sleeps = []
    with pytest.raises(LLMError):
        with_retries(lambda: (_ for _ in ()).throw(Boom(503)), attempts=3, sleep=sleeps.append)
    assert len(sleeps) == 2  # no sleep after the final attempt


def test_non_retriable_error_is_raised_immediately():
    calls = []

    def fn():
        calls.append(1)
        raise Boom(400)

    with pytest.raises(Boom):
        with_retries(fn, sleep=lambda s: None)
    assert len(calls) == 1


class Counting(LLMClient):
    name = "fake"
    model = "m1"

    def __init__(self):
        self.calls = 0

    def generate(self, system, user):
        self.calls += 1
        return f"SQL for {user}"


def test_cache_hits_skip_the_inner_client(tmp_path):
    inner = Counting()
    c = CachingLLMClient(inner, tmp_path)
    assert c.generate("sys", "q1") == "SQL for q1"
    assert c.generate("sys", "q1") == "SQL for q1"
    assert inner.calls == 1 and c.hits == 1 and c.misses == 1


def test_cache_key_covers_system_user_and_model(tmp_path):
    inner = Counting()
    c = CachingLLMClient(inner, tmp_path)
    c.generate("sysA", "q")
    c.generate("sysB", "q")
    c.generate("sysA", "q2")
    assert inner.calls == 3
    inner.model = "m2"
    c.generate("sysA", "q")
    assert inner.calls == 4


def test_cache_persists_across_instances_and_refresh_overwrites(tmp_path):
    a = Counting()
    CachingLLMClient(a, tmp_path).generate("s", "q")
    b = Counting()
    assert CachingLLMClient(b, tmp_path).generate("s", "q") == "SQL for q" and b.calls == 0
    r = Counting()
    CachingLLMClient(r, tmp_path, refresh=True).generate("s", "q")
    assert r.calls == 1


def test_multi_message_cache_distinguishes_history(tmp_path):
    inner = Counting()
    c = CachingLLMClient(inner, tmp_path)
    m1 = [{"role": "user", "content": "a"}]
    m2 = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "x"},
          {"role": "user", "content": "b"}]
    c.generate_messages("s", m1)
    c.generate_messages("s", m2)
    c.generate_messages("s", m2)
    assert inner.calls == 2
