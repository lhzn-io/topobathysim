import logging
import threading
from collections.abc import Callable, Hashable
from functools import wraps
from typing import Any, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MemoizeWithLocks:
    """
    A thread-safe cache dictionary that ensures only one computation runs
    for a given input key, preventing "cache stampedes" where multiple threads
    fetch the same network resource concurrently because the first thread hasn't
    finished saving its result to the cache yet.
    """

    def __init__(self, ttl: int | None = None) -> None:
        self.cache: dict[Hashable, Any] = {}
        self.locks: dict[Hashable, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Create a simple hashable key. Warning: Not perfect for nested dicts.
            try:
                # Discard 'cls' or 'self' if they are the first arguments and not useful for hashing
                # For safety, we just stringify the arguments.
                key = str(args) + str(frozenset(kwargs.items()))
            except Exception:
                logger.warning("MemoizeWithLocks failed to hash args, skipping cache.")
                return func(*args, **kwargs)

            # 1. Fast check
            with self._global_lock:
                if key in self.cache:
                    return cast(T, self.cache[key])
                if key not in self.locks:
                    self.locks[key] = threading.Lock()
                lock = self.locks[key]

            # 2. Acquire specific lock and compute if still missing
            with lock:
                if key in self.cache:
                    return cast(T, self.cache[key])

                # Compute
                result = func(*args, **kwargs)

                # Cache and release lock
                self.cache[key] = result

                # We could delete the lock here, but keeping it is fine for memory until cache clears
            return result

        return wrapper


def concurrent_lru_cache() -> Callable[..., Any]:
    """
    Decorator for preventing cache stampedes in multithreaded fastAPI calls.
    """
    # Note: we use an instance per function, not a single global one
    decorator_instance = MemoizeWithLocks()
    return decorator_instance
