"""Tests for the chunked key fetch module."""

import asyncio
from collections.abc import Sequence
from typing import Any, Union

import pytest

from soildb.chunked import fetch_chunked
from soildb.query import Query
from soildb.response import SDAResponse


class DummyQuery:
    """Dummy query object for testing."""

    def __init__(self, keys: Sequence[Any]):
        self.keys = list(keys)

    def __str__(self) -> str:
        return f"SELECT * FROM dummy WHERE id IN ({len(self.keys)} keys)"


class ChunkTracker:
    """Tracks calls to build_query and execute for testing."""

    def __init__(self):
        self.call_count = 0
        self.calls: list[tuple[tuple[Any, ...], str]] = []
        self.concurrent_count = 0
        self.max_concurrent_count = 0
        self.fail_on_keys: set[Any] = set()

    def build_query(self, keys: Sequence[Any]) -> Union[str, DummyQuery]:
        """Build a dummy query."""
        return DummyQuery(keys)

    async def execute(self, query: Union[str, Query]) -> SDAResponse:
        """Execute a fake query, tracking concurrency."""
        # Track concurrency
        self.concurrent_count += 1
        self.max_concurrent_count = max(
            self.max_concurrent_count, self.concurrent_count
        )

        try:
            # Extract keys from the test query
            keys = []
            if hasattr(query, "keys"):
                keys = query.keys  # type: ignore

            call_num = self.call_count
            self.call_count += 1

            # Minimal delay to allow concurrent tasks to interleave
            await asyncio.sleep(0.00001)

            # Check if this batch should fail
            if any(k in self.fail_on_keys for k in keys):
                raise RuntimeError(f"Simulated failure for keys: {keys}")

            # Return an SDAResponse with the keys as data
            rows = [[k] for k in keys]
            sda_type = "int" if keys and isinstance(keys[0], int) else "varchar"
            response = SDAResponse.from_rows(
                rows,
                ["key"],
                [sda_type],
            )
            self.calls.append((tuple(keys), f"call_{call_num}"))
            return response
        finally:
            self.concurrent_count -= 1


@pytest.mark.asyncio
async def test_empty_keys():
    """Test that empty keys returns empty response without calling execute."""
    tracker = ChunkTracker()

    result = await fetch_chunked(
        [],
        tracker.build_query,
        tracker.execute,
        chunk_size=10,
        max_concurrency=2,
    )

    assert result.is_empty()
    assert tracker.call_count == 0


@pytest.mark.asyncio
async def test_single_chunk():
    """Test keys that fit in a single chunk."""
    tracker = ChunkTracker()

    keys = [1, 2, 3, 4, 5]
    result = await fetch_chunked(
        keys,
        tracker.build_query,
        tracker.execute,
        chunk_size=10,
        max_concurrency=2,
    )

    assert len(result.data) == 5
    assert tracker.call_count == 1
    assert tracker.max_concurrent_count == 1


@pytest.mark.asyncio
async def test_multiple_chunks():
    """Test keys split into multiple chunks."""
    tracker = ChunkTracker()

    keys = list(range(25))
    result = await fetch_chunked(
        keys,
        tracker.build_query,
        tracker.execute,
        chunk_size=10,
        max_concurrency=4,
    )

    assert len(result.data) == 25
    # 25 keys with chunk_size=10 -> 3 chunks
    assert tracker.call_count == 3
    assert tracker.max_concurrent_count <= 4


@pytest.mark.asyncio
async def test_concurrency_limit_respected():
    """Test that concurrency never exceeds max_concurrency."""
    tracker = ChunkTracker()

    keys = list(range(100))
    result = await fetch_chunked(
        keys,
        tracker.build_query,
        tracker.execute,
        chunk_size=10,
        max_concurrency=3,
    )

    assert len(result.data) == 100
    # 100 keys with chunk_size=10 -> 10 chunks
    assert tracker.call_count == 10
    # Peak concurrency should not exceed 3
    assert tracker.max_concurrent_count <= 3


@pytest.mark.asyncio
async def test_order_preserved():
    """Test that output rows are in input key order."""
    tracker = ChunkTracker()

    keys = [10, 20, 30, 40, 50]
    result = await fetch_chunked(
        keys,
        tracker.build_query,
        tracker.execute,
        chunk_size=2,
        max_concurrency=2,
    )

    # Extract keys from result
    result_keys = [row[0] for row in result.data]
    assert result_keys == keys


@pytest.mark.asyncio
async def test_failed_chunk_with_retries():
    """Test that a failed chunk triggers retries via splitting."""
    tracker = ChunkTracker()

    # Make a single key fail
    tracker.fail_on_keys = {0}
    # Just one key that will fail on first attempt
    keys = [0]

    # This should fail because key 0 always fails
    with pytest.raises(RuntimeError, match="Simulated failure"):
        await fetch_chunked(
            keys,
            tracker.build_query,
            tracker.execute,
            chunk_size=10,
            max_concurrency=2,
            max_retries=2,
        )

    # Single key fails immediately on retry_level 0, so just 1 call
    assert tracker.call_count == 1


@pytest.mark.asyncio
async def test_retries_exhausted():
    """Test that retries exhausted raises after max_retries."""
    tracker = ChunkTracker()

    # Make a specific key fail
    tracker.fail_on_keys = {1}
    keys = [1, 2, 3, 4, 5]

    # With chunk_size=10 and max_retries=0 (only 1 attempt allowed)
    # The single chunk [1,2,3,4,5] will fail, split into [1,2] and [3,4,5]
    # But with max_retries=0, we can't retry the halves, so re-raise
    with pytest.raises(RuntimeError):
        await fetch_chunked(
            keys,
            tracker.build_query,
            tracker.execute,
            chunk_size=10,
            max_concurrency=2,
            max_retries=0,
        )


@pytest.mark.asyncio
async def test_chunk_size_validation():
    """Test that invalid chunk_size raises ValueError."""
    tracker = ChunkTracker()

    with pytest.raises(ValueError, match="chunk_size must be > 0"):
        await fetch_chunked(
            [1, 2, 3],
            tracker.build_query,
            tracker.execute,
            chunk_size=0,
        )

    with pytest.raises(ValueError, match="chunk_size must be > 0"):
        await fetch_chunked(
            [1, 2, 3],
            tracker.build_query,
            tracker.execute,
            chunk_size=-1,
        )


@pytest.mark.asyncio
async def test_max_concurrency_validation():
    """Test that invalid max_concurrency raises ValueError."""
    tracker = ChunkTracker()

    with pytest.raises(ValueError, match="max_concurrency must be > 0"):
        await fetch_chunked(
            [1, 2, 3],
            tracker.build_query,
            tracker.execute,
            max_concurrency=0,
        )

    with pytest.raises(ValueError, match="max_concurrency must be > 0"):
        await fetch_chunked(
            [1, 2, 3],
            tracker.build_query,
            tracker.execute,
            max_concurrency=-1,
        )


@pytest.mark.asyncio
async def test_max_retries_validation():
    """Test that invalid max_retries raises ValueError."""
    tracker = ChunkTracker()

    with pytest.raises(ValueError, match="max_retries must be >= 0"):
        await fetch_chunked(
            [1, 2, 3],
            tracker.build_query,
            tracker.execute,
            max_retries=-1,
        )


@pytest.mark.asyncio
async def test_response_merge_order():
    """Test that responses are merged in chunk order."""
    tracker = ChunkTracker()

    # Use string keys to make order more obvious
    keys = ["a", "b", "c", "d", "e", "f"]
    result = await fetch_chunked(
        keys,
        tracker.build_query,
        tracker.execute,
        chunk_size=2,
        max_concurrency=2,
    )

    # Extract keys from result
    result_keys = [row[0] for row in result.data]
    assert result_keys == keys


@pytest.mark.asyncio
async def test_only_failed_chunks_retried():
    """Test that only failed chunks are retried."""
    tracker = ChunkTracker()

    # Make only the middle chunk fail
    # With chunk_size=10, keys [0-9] succeed, [10-19] fail, [20-29] succeed
    tracker.fail_on_keys = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
    keys = list(range(30))

    # This will fail because the middle chunk fails
    with pytest.raises(RuntimeError):
        await fetch_chunked(
            keys,
            tracker.build_query,
            tracker.execute,
            chunk_size=10,
            max_concurrency=3,
            max_retries=1,
        )

    # The first chunk [0-9] should be called once
    # The second chunk [10-19] should be called, fail, split, and retried
    # The third chunk [20-29] should be called once
    # So we should see at least 4 calls (1 + 2 + 1)
    assert tracker.call_count >= 4


@pytest.mark.asyncio
async def test_deadlock_fix_with_max_concurrency_one():
    """Test that a failing chunk succeeds when split, with max_concurrency=1.

    Deadlock would occur if the semaphore were held during recursion.
    This test verifies the fix: semaphore is held only around execute(),
    released before splits. Ensures completion within 2 seconds.
    """
    # Track which call attempts have been made to fail only the first attempt
    call_attempts: set[frozenset[Any]] = set()

    async def execute_fail_first_attempt(
        query: Union[str, "DummyQuery"],
    ) -> SDAResponse:
        """Fail only on the first attempt for multi-key chunks, allow splits."""
        keys = []
        if hasattr(query, "keys"):
            keys = query.keys  # type: ignore

        # Only fail multi-key chunks on first attempt (avoid cascading failures)
        key_set = frozenset(keys)
        should_fail = len(keys) > 1 and key_set not in call_attempts

        if should_fail:
            call_attempts.add(key_set)
            await asyncio.sleep(0.00001)
            raise RuntimeError(f"Simulated failure for keys: {keys}")

        await asyncio.sleep(0.00001)
        rows = [[k] for k in keys]
        sda_type = "int" if keys and isinstance(keys[0], int) else "varchar"
        response = SDAResponse.from_rows(rows, ["key"], [sda_type])
        return response

    keys = [0, 1, 2, 3]

    # This should not deadlock and should finish within 2 seconds
    try:
        result = await asyncio.wait_for(
            fetch_chunked(
                keys,
                ChunkTracker().build_query,
                execute_fail_first_attempt,
                chunk_size=10,
                max_concurrency=1,
                max_retries=2,
            ),
            timeout=2.0,
        )
        # The chunk [0,1,2,3] fails, splits to [0,1] and [2,3], both succeed
        assert len(result.data) == 4
    except asyncio.TimeoutError:
        pytest.fail("Test deadlocked or took too long (>2 seconds)")


@pytest.mark.asyncio
async def test_peak_concurrency_with_splits():
    """Test that peak concurrency stays <= max_concurrency during splits.

    With 4 failing chunks that succeed when split, verifies that peak
    concurrent execution never exceeds max_concurrency=2.
    """
    # Track which call attempts have been made
    call_attempts: set[frozenset[Any]] = set()
    max_concurrent_seen = 0
    concurrent_count = 0
    lock = asyncio.Lock()

    async def execute_fail_first_attempt(
        query: Union[str, "DummyQuery"],
    ) -> SDAResponse:
        """Fail only on the first attempt for multi-key chunks, allow splits."""
        nonlocal concurrent_count, max_concurrent_seen

        async with lock:
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)

        try:
            keys = []
            if hasattr(query, "keys"):
                keys = query.keys  # type: ignore

            # Only fail multi-key chunks on first attempt (avoid cascading failures)
            key_set = frozenset(keys)
            should_fail = len(keys) > 1 and key_set not in call_attempts

            if should_fail:
                call_attempts.add(key_set)
                await asyncio.sleep(0.0001)
                raise RuntimeError(f"Simulated failure for keys: {keys}")

            await asyncio.sleep(0.0001)
            rows = [[k] for k in keys]
            sda_type = "int" if keys and isinstance(keys[0], int) else "varchar"
            response = SDAResponse.from_rows(rows, ["key"], [sda_type])
            return response
        finally:
            async with lock:
                concurrent_count -= 1

    keys = [0, 1, 2, 3]

    result = await fetch_chunked(
        keys,
        ChunkTracker().build_query,
        execute_fail_first_attempt,
        chunk_size=2,
        max_concurrency=2,
        max_retries=2,
    )

    assert len(result.data) == 4
    # Peak concurrency should respect the limit
    assert max_concurrent_seen <= 2
