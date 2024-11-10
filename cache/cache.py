from abc import ABC, abstractmethod
import sqlite3
import pickle
import hashlib
import asyncio
from typing import Optional, Callable, Tuple, Any
import os

BUST_CACHE = os.getenv("BUST_CACHE", "")
CACHE_ONLY = os.getenv("CACHE_ONLY", "false").lower() != "false"
PRINT_CACHE_MISSES = os.getenv("PRINT_CACHE_MISSES", "false").lower() != "false"

bust_cache_ids = set(BUST_CACHE.split(","))

if "all" in bust_cache_ids:
    print("Busting all caches, good luck")
elif bust_cache_ids != {""}:
    print(f"Busting cache for {bust_cache_ids}")

if CACHE_ONLY:
    print("CACHE_ONLY is set to True. Cache will be used exclusively.")


class CacheBackend(ABC):
    @abstractmethod
    async def setup(self) -> None:
        pass

    @abstractmethod
    async def get(self, fn_id: str, arg_hash: str) -> Tuple[bool, Any]:
        """Returns (cache_hit, result)"""
        pass

    @abstractmethod
    async def set(self, fn_id: str, arg_hash: str, result: Any) -> None:
        pass

    @abstractmethod
    async def delete(self, fn_id: str, arg_hash: str) -> None:
        """Deletes a cache entry"""
        pass

    @abstractmethod
    async def delete_all(self) -> None:
        """Deletes all cache entries"""
        pass

    @abstractmethod
    async def delete_by_fn_id(self, fn_id: str) -> None:
        """Deletes all cache entries for a specific function ID"""
        pass


class SQLiteBackend(CacheBackend):
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    async def setup(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    fn_id TEXT,
                    arg_hash TEXT,
                    result BLOB,
                    chunk_index INTEGER,
                    is_chunked BOOLEAN,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (fn_id, arg_hash, chunk_index)
                )
            """
            )
            conn.commit()

    async def get(self, fn_id: str, arg_hash: str) -> Tuple[bool, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT result, is_chunked, chunk_index FROM cache 
                WHERE fn_id = ? AND arg_hash = ?
                ORDER BY chunk_index
            """,
                (fn_id, arg_hash),
            )
            rows = cursor.fetchall()

            if not rows:
                return False, None

            if not rows[0][1]:  # not chunked
                return True, pickle.loads(rows[0][0])

            # Combine chunks
            result_data = b"".join(row[0] for row in rows)
            return True, pickle.loads(result_data)

    async def set(self, fn_id: str, arg_hash: str, result: Any) -> None:
        pickled_data = pickle.dumps(result)
        chunk_size = 1024 * 1024  # 1MB chunks
        is_chunked = len(pickled_data) > chunk_size

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM cache WHERE fn_id = ? AND arg_hash = ?",
                (fn_id, arg_hash),
            )

            if not is_chunked:
                cursor.execute(
                    """
                    INSERT INTO cache (fn_id, arg_hash, result, chunk_index, is_chunked)
                    VALUES (?, ?, ?, 0, 0)
                """,
                    (fn_id, arg_hash, pickled_data),
                )
            else:
                chunks = [
                    pickled_data[i : i + chunk_size]
                    for i in range(0, len(pickled_data), chunk_size)
                ]
                for i, chunk in enumerate(chunks):
                    cursor.execute(
                        """
                        INSERT INTO cache (fn_id, arg_hash, result, chunk_index, is_chunked)
                        VALUES (?, ?, ?, ?, 1)
                    """,
                        (fn_id, arg_hash, chunk, i),
                    )

            conn.commit()

    async def delete(self, fn_id: str, arg_hash: str) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM cache WHERE fn_id = ? AND arg_hash = ?",
                (fn_id, arg_hash),
            )
            conn.commit()

    async def delete_all(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM cache")
            conn.commit()

    async def delete_by_fn_id(self, fn_id: str) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM cache WHERE fn_id = ?", (fn_id,))
            conn.commit()


class Cache:
    def __init__(self, backend: CacheBackend):
        self.backend = backend
        self._setup_done = False

    async def ensure_setup(self):
        if not self._setup_done:
            await self.backend.setup()
            self._setup_done = True

    def cache(
        self,
        id: Optional[str] = None,
        hash_func: Optional[Callable[..., str]] = None,
        debug: bool = False,
    ):
        def decorator(func):
            if not asyncio.iscoroutinefunction(func):
                raise ValueError(
                    "Cache decorator can only be used with async functions"
                )

            async def async_wrapper(*args, **kwargs):
                await self.ensure_setup()
                fn_id = id if id else func.__name__

                try:
                    if hash_func:
                        arg_hash = hashlib.sha256(
                            hash_func(*args, **kwargs).encode()
                        ).hexdigest()
                    else:
                        arg_hash = hashlib.sha256(
                            pickle.dumps((args, kwargs))
                        ).hexdigest()
                except Exception as e:
                    print(
                        f"Error computing arg_hash for fn_id={fn_id} with inputs args={args}, kwargs={kwargs}"
                    )
                    raise e

                if debug:
                    print(f"Debug: fn_id={fn_id}, arg_hash={arg_hash}")

                should_bust_cache = fn_id in bust_cache_ids or "all" in bust_cache_ids
                if debug:
                    print(f"should_bust_cache={should_bust_cache}")

                if not should_bust_cache:
                    cache_hit, cached_result = await self.backend.get(fn_id, arg_hash)
                    if cache_hit:
                        return cached_result

                if CACHE_ONLY:
                    raise Exception(
                        f"Cache miss for fn_id={fn_id} with arg_hash={arg_hash}. CACHE_ONLY is set to True."
                    )

                if PRINT_CACHE_MISSES:
                    print(
                        f"Cache miss for fn_id={fn_id} with arg_hash={arg_hash}. Executing function."
                    )

                result = await func(*args, **kwargs)
                await self.backend.set(fn_id, arg_hash, result)
                return result

            async def bust_cache(*args, **kwargs):
                await self.ensure_setup()
                fn_id = id if id else func.__name__

                if not args and not kwargs:
                    await self.backend.delete_by_fn_id(fn_id)
                    return

                try:
                    if hash_func:
                        arg_hash = hashlib.sha256(
                            hash_func(*args, **kwargs).encode()
                        ).hexdigest()
                    else:
                        arg_hash = hashlib.sha256(
                            pickle.dumps((args, kwargs))
                        ).hexdigest()
                    await self.backend.delete(fn_id, arg_hash)
                except Exception as e:
                    print(
                        f"Error computing arg_hash for fn_id={fn_id} with inputs args={args}, kwargs={kwargs}"
                    )
                    raise e

            async def read_cache(*args, **kwargs):
                await self.ensure_setup()
                fn_id = id if id else func.__name__

                try:
                    if hash_func:
                        arg_hash = hashlib.sha256(
                            hash_func(*args, **kwargs).encode()
                        ).hexdigest()
                    else:
                        arg_hash = hashlib.sha256(
                            pickle.dumps((args, kwargs))
                        ).hexdigest()
                except Exception as e:
                    print(
                        f"Error computing arg_hash for fn_id={fn_id} with inputs args={args}, kwargs={kwargs}"
                    )
                    raise e

                return await self.backend.get(fn_id, arg_hash)

            async_wrapper.bust_cache = bust_cache
            async_wrapper.read_cache = read_cache
            return async_wrapper

        return decorator

    async def bust_all(self) -> None:
        """Busts the entire cache"""
        await self.ensure_setup()
        await self.backend.delete_all()
