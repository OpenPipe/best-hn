import pytest
import asyncio
import os
from cache import Cache, SQLiteBackend, S3Backend
from moto import mock_aws
import boto3


# Test fixtures
@pytest.fixture(
    params=[
        # ("sqlite", lambda: SQLiteBackend("test_cache.db")),
        (
            "s3",
            lambda: S3Backend("test-bucket", prefix="test/", region_name="us-east-1"),
        ),
    ]
)
def cache(request):
    backend_type, backend_factory = request.param

    if backend_type == "sqlite":
        db_path = "test_cache.db"
        backend = backend_factory()
        cache = Cache(backend)

        yield cache

        # Cleanup
        asyncio.run(cache.bust_all())
        if os.path.exists(db_path):
            os.remove(db_path)
    elif backend_type == "s3":
        mock = mock_aws()
        mock.start()

        print("CREATING BUCKET")
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")

        backend = backend_factory()
        cache = Cache(backend)

        yield cache

        mock.stop()


# Test functions to be cached
async def async_add(a: int, b: int) -> int:
    return a + b


async def async_multiply(a: int, b: int) -> int:
    return a * b


@pytest.mark.asyncio
async def test_async_cache_hit(cache):
    cached_add = cache.cache()(async_add)

    # First call - should miss cache
    result1 = await cached_add(2, 3)
    assert result1 == 5

    # Verify cache hit
    cache_hit, cached_result = await cached_add.read_cache(2, 3)
    assert cache_hit
    assert cached_result == 5

    # Verify cached result is returned quickly
    result2 = await cached_add(2, 3)
    assert result2 == 5


@pytest.mark.asyncio
async def test_custom_cache_id(cache):
    cached_add = cache.cache(id="custom_add")(async_add)

    # Initial call and verify result is cached
    result = await cached_add(2, 3)
    assert result == 5
    cache_hit, cached_result = await cached_add.read_cache(2, 3)
    assert cache_hit
    assert cached_result == 5

    # Bust cache and verify it's gone
    await cached_add.bust_cache()
    cache_hit, _ = await cached_add.read_cache(2, 3)
    assert not cache_hit


@pytest.mark.asyncio
async def test_custom_hash_function(cache):
    def hash_func(a, b):
        return f"{a}+{b}"

    cached_add = cache.cache(hash_func=hash_func)(async_add)

    # Initial call and verify result is cached
    result = await cached_add(2, 3)
    assert result == 5
    cache_hit, cached_result = await cached_add.read_cache(2, 3)
    assert cache_hit
    assert cached_result == 5


@pytest.mark.asyncio
async def test_bust_specific_args(cache):
    cached_add = cache.cache()(async_add)

    # Cache multiple different calls
    result1 = await cached_add(2, 3)
    result2 = await cached_add(4, 5)
    assert result1 == 5
    assert result2 == 9

    # Bust specific args and verify only that entry is removed
    await cached_add.bust_cache(2, 3)

    cache_hit1, _ = await cached_add.read_cache(2, 3)
    cache_hit2, cached_result2 = await cached_add.read_cache(4, 5)
    assert not cache_hit1  # Should be busted
    assert cache_hit2  # Should still exist
    assert cached_result2 == 9


@pytest.mark.asyncio
async def test_bust_entire_function(cache):
    cached_add = cache.cache()(async_add)
    cached_multiply = cache.cache()(async_multiply)

    await cached_add(2, 3)
    await cached_multiply(4, 5)

    await cached_add.bust_cache()

    # Verify first cache is cleared but second remains
    add_hit, _ = await cached_add.read_cache(2, 3)
    mult_hit, mult_result = await cached_multiply.read_cache(4, 5)
    assert not add_hit
    assert mult_hit
    assert mult_result == 20


@pytest.mark.asyncio
async def test_bust_entire_cache(cache):
    cached_add1 = cache.cache()(async_add)
    cached_add2 = cache.cache()(async_add)

    # Cache results for both functions
    await cached_add1(2, 3)
    await cached_add2(4, 5)

    # Bust entire cache
    await cache.bust_all()

    # Verify all caches are cleared
    add1_hit, _ = await cached_add1.read_cache(2, 3)
    add2_hit, _ = await cached_add2.read_cache(4, 5)
    assert not add1_hit
    assert not add2_hit


@pytest.mark.asyncio
async def test_direct_cache_operations(cache):
    # Test setting and getting a value
    await cache.set("test_key", "test_value")
    result = await cache.get("test_key")
    assert result == "test_value"

    # Test getting a non-existent key
    with pytest.raises(KeyError):
        await cache.get("nonexistent_key")

    # Test overwriting an existing key
    await cache.set("test_key", "new_value")
    result = await cache.get("test_key")
    assert result == "new_value"

    # Test that bust_all clears direct cache entries
    await cache.bust_all()
    with pytest.raises(KeyError):
        await cache.get("test_key")
