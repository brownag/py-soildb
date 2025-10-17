"""Test simple network connectivity to SDA."""

import asyncio
import httpx
import time


async def test_direct_http():
    """Test direct HTTP call without soildb."""
    print("Testing direct HTTP call to SDA...", flush=True)

    # Build the request
    url = "https://sdmdataaccess.sc.egov.usda.gov/tabular/post.rest"
    query = """
        SELECT m.mukey, m.muname, m.musym, l.areasymbol AS lkey, sa.areaname
        FROM mupolygon AS mu
        INNER JOIN mapunit AS m ON mu.mukey = m.mukey
        INNER JOIN legend AS l ON m.lkey = l.lkey
        INNER JOIN laoverlap AS lo ON l.lkey = lo.lkey
        INNER JOIN sacatalog AS sa ON lo.areasymbol = sa.areasymbol
        WHERE mu.mupolygongeo.STIntersects(
            geometry::STGeomFromText('POINT(-93.5 41.5)', 4326)
        ) = 1
    """
    request_body = {"query": query.strip(), "format": "json+columnname+metadata"}

    start = time.time()

    async with httpx.AsyncClient(timeout=10.0) as client:
        print(f"Sending POST request... (timeout=10s)", flush=True)
        try:
            response = await client.post(url, json=request_body)
            elapsed = time.time() - start
            print(f"Response received in {elapsed:.2f}s", flush=True)
            print(f"Status: {response.status_code}", flush=True)
            print(f"Response length: {len(response.text)} chars", flush=True)

            # Try to parse
            data = response.json()
            if "Table" in data:
                print(f"Table has {len(data['Table'])} rows", flush=True)
            return True
        except httpx.TimeoutException:
            elapsed = time.time() - start
            print(f"Request timed out after {elapsed:.2f}s", flush=True)
            return False
        except Exception as e:
            elapsed = time.time() - start
            print(f"Request failed after {elapsed:.2f}s: {e}", flush=True)
            return False


async def test_soildb_client():
    """Test using soildb client."""
    print("\nTesting soildb SDAClient...", flush=True)

    from soildb.client import SDAClient
    from soildb.query import Query

    start = time.time()

    # Create client with shorter timeout
    client = SDAClient(timeout=10.0)

    try:
        print("Creating query...", flush=True)
        query = Query().select("mukey").from_("mapunit").limit(1)

        print(f"Executing query: {query.to_sql()}", flush=True)
        response = await client.execute(query)

        elapsed = time.time() - start
        print(f"Query completed in {elapsed:.2f}s", flush=True)
        print(f"Response: {response}", flush=True)
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"Query failed after {elapsed:.2f}s: {e}", flush=True)
        return False
    finally:
        await client.close()


async def test_spatial_query():
    """Test soildb spatial_query function."""
    print("\nTesting soildb spatial_query...", flush=True)

    from soildb.spatial import spatial_query

    start = time.time()

    try:
        print("Calling spatial_query...", flush=True)
        wkt_point = "POINT(-93.5 41.5)"
        response = await spatial_query(wkt_point, table="mupolygon", what="mukey")

        elapsed = time.time() - start
        print(f"spatial_query completed in {elapsed:.2f}s", flush=True)
        print(f"Response: {response}", flush=True)
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"spatial_query failed after {elapsed:.2f}s: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("=" * 60, flush=True)
    print("Network Connectivity Tests", flush=True)
    print("=" * 60, flush=True)

    # Test 1: Direct HTTP
    result1 = await test_direct_http()
    print(f"\nTest 1 (Direct HTTP): {'PASS' if result1 else 'FAIL'}", flush=True)

    # Test 2: SDAClient
    result2 = await test_soildb_client()
    print(f"\nTest 2 (SDAClient): {'PASS' if result2 else 'FAIL'}", flush=True)

    # Test 3: spatial_query
    result3 = await test_spatial_query()
    print(f"\nTest 3 (spatial_query): {'PASS' if result3 else 'FAIL'}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(f"Summary: {sum([result1, result2, result3])}/3 tests passed", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
