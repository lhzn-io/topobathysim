from typing import cast

from topobathysim.utils.cache import MemoizeWithLocks, concurrent_lru_cache


class MockResource:
    def __init__(self, val: int) -> None:
        self.val = val
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_memoize_lru_eviction() -> None:
    """
    Verify that the implementation of MemoizeWithLocks supports LRU eviction.
    """
    # Use a small maxsize
    cache_decorator = concurrent_lru_cache(maxsize=10)

    # Access the underlying cache instance
    memoizer = cast(MemoizeWithLocks, cache_decorator)

    resources: list[MockResource] = []

    @memoizer
    def dummy_func(x: int) -> MockResource:
        res = MockResource(x)
        resources.append(res)
        return res

    # Add 20 items (limit 10)
    for i in range(20):
        dummy_func(i)

    # Check cache size is capped at 10
    assert len(memoizer.cache) == 10, f"Cache size should be 10, but is {len(memoizer.cache)}"

    # Check that the first 10 items (0-9) are evicted and closed
    # The last 10 items (10-19) should remain in cache
    for i in range(10):
        # Evicted items should be closed
        assert resources[i].closed is True, f"Resource {i} should be closed"

    for i in range(10, 20):
        # Recent items should be open
        assert resources[i].closed is False, f"Resource {i} should be open"

    print("LRU eviction and resource cleanup verified.")
