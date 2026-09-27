"""Chunked key fetch module for bounded concurrent execution with recursive retry.

This module provides `fetch_chunked()`, an internal reusable coroutine that
splits a sequence of keys into chunks and executes them concurrently with
configurable limits. Failed chunks are recursively split in half and retried
up to a maximum depth, with single-key failures re-raised immediately.

Internal use only: not in `soildb.__all__` and has no `.sync` method. Called by
`fetch.py` and `ldm/client.py`.

Used by both SSURGO (via `fetch.py`) and LDM workflows to fetch data for many
keys without overwhelming the backend or exceeding memory limits.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

from soildb.response import SDAResponse

Q = TypeVar("Q")

logger = logging.getLogger(__name__)


async def fetch_chunked(
    keys: Sequence[Any],
    build_query: Callable[[list[Any]], Q],
    execute: Callable[[Q], Awaitable[SDAResponse]],
    *,
    chunk_size: int = 1000,
    max_concurrency: int = 4,
    max_retries: int = 2,
) -> SDAResponse:
    """Fetch data for keys in chunks with bounded concurrency and recursive retry.

    Splits `keys` into chunks of `chunk_size`, executes them concurrently with
    at most `max_concurrency` in-flight tasks. A chunk that fails is
    recursively split in half and retried, up to `max_retries` levels. A
    single-key chunk that still fails re-raises.

    Results are merged in input order using `SDAResponse.concat()`, preserving
    the original key sequence.

    Args:
        keys: The keys to fetch. Can be any sequence type.
        build_query: Function that builds a query for a chunk of keys.
            Called once per attempted chunk with a list of keys.
        execute: Async function that executes a query and returns an
            SDAResponse. Called once per attempted chunk.
        chunk_size: Number of keys per chunk (default: 1000). Must be > 0.
        max_concurrency: Maximum number of concurrent tasks (default: 4).
            Must be > 0.
        max_retries: Maximum retry depth for failed chunks (default: 2).
            A value of 2 means: try full chunk, halve and retry, halve
            again and retry (3 attempts max per original chunk). Must be >= 0.

    Returns:
        Merged response with rows from all chunks in input order.
        If `keys` is empty, returns an empty SDAResponse without calling
        `execute()`.

    Raises:
        ValueError: If `chunk_size <= 0`, `max_concurrency <= 0`, or
            `max_retries < 0`.
        Exception: Any exception raised by `execute()` that cannot be
            recovered by retry. For single-key chunks, re-raised immediately
            after exhausting retries.

    Example:
        >>> async def fake_executor(query):
        ...     # Return one row per key in the query
        ...     return SDAResponse.from_rows(
        ...         [[k] for k in [1, 2, 3, 4, 5]],
        ...         ["key"],
        ...         ["int"]
        ...     )
        >>> keys = [1, 2, 3, 4, 5]
        >>> result = await fetch_chunked(
        ...     keys,
        ...     lambda chunk: f"SELECT * WHERE id IN ({chunk})",
        ...     fake_executor,
        ...     chunk_size=2,
        ... )
        >>> len(result)
        5
    """
    # Validate parameters
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if max_concurrency <= 0:
        raise ValueError(f"max_concurrency must be > 0, got {max_concurrency}")
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}")

    # Empty keys: return empty response without calling execute
    if not keys:
        logger.debug("fetch_chunked: empty keys, returning empty response")
        return SDAResponse({"Table": [[], []]})

    # Create semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(max_concurrency)

    # Helper to split a sequence in half
    def split_in_half(seq: Sequence[Any]) -> tuple[Sequence[Any], Sequence[Any]]:
        """Split sequence into two halves."""
        mid = len(seq) // 2
        return seq[:mid], seq[mid:]

    async def execute_with_retries(
        chunk: Sequence[Any], retry_level: int
    ) -> SDAResponse:
        """Execute a chunk with recursive retry by splitting on failure.

        Internal helper that holds the semaphore only during the execute()
        call, releasing it before any retry/split logic to prevent deadlock.

        Args:
            chunk: Keys for this chunk.
            retry_level: Current retry depth (0 = first attempt, max_retries
                = last).

        Returns:
            Response for this chunk.

        Raises:
            Exception: If chunk fails and cannot be split (single key) or
                retries exhausted.
        """
        # Acquire semaphore, build and execute, then release before retry logic
        try:
            async with semaphore:
                query = build_query(list(chunk))
                logger.debug(
                    f"fetch_chunked: executing chunk of {len(chunk)} keys "
                    f"(retry_level={retry_level})"
                )
                response = await execute(query)
            logger.debug(f"fetch_chunked: chunk succeeded ({len(response.data)} rows)")
            return response
        except Exception as e:
            # Semaphore is released here; retry logic runs outside the lock
            # Check if we can retry
            if retry_level >= max_retries:
                logger.error(
                    f"fetch_chunked: chunk of {len(chunk)} keys failed "
                    f"after {retry_level + 1} attempts: {e}"
                )
                raise

            # Single-key chunks that fail are re-raised immediately
            if len(chunk) == 1:
                logger.error(
                    f"fetch_chunked: single-key chunk failed "
                    f"(retry_level={retry_level}): {e}"
                )
                raise

            # Split and retry both halves (outside the semaphore)
            logger.warning(
                f"fetch_chunked: chunk of {len(chunk)} keys failed "
                f"(retry_level={retry_level}), splitting and retrying: {e}"
            )
            first_half, second_half = split_in_half(chunk)
            first_response = await execute_with_retries(first_half, retry_level + 1)
            second_response = await execute_with_retries(second_half, retry_level + 1)
            # Merge the two halves' responses
            return SDAResponse.concat([first_response, second_response])

    # Split keys into initial chunks
    chunks = [keys[i : (i + chunk_size)] for i in range(0, len(keys), chunk_size)]

    logger.info(
        f"fetch_chunked: fetching {len(keys)} keys in {len(chunks)} chunks "
        f"(chunk_size={chunk_size}, max_concurrency={max_concurrency}, "
        f"max_retries={max_retries})"
    )

    # Execute all chunks with retry, preserving order
    chunk_tasks = [execute_with_retries(chunk, retry_level=0) for chunk in chunks]
    chunk_responses = await asyncio.gather(*chunk_tasks)

    # Merge results in input order
    merged = SDAResponse.concat(chunk_responses)
    logger.info(f"fetch_chunked: completed, merged {len(merged.data)} total rows")
    return merged
