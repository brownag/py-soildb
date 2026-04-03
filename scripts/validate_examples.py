#!/usr/bin/env python3
"""Validate that example scripts have correct syntax and can import."""

import sys
import py_compile
import importlib.util
from pathlib import Path

examples = [
    'docs/examples/01_basic.py',
    'docs/examples/02_spatial.py',
    'docs/examples/03_metadata.py',
    'docs/examples/04_schema.py',
    'docs/examples/05_awdb.py',
    'docs/examples/06_awdb_availability.py',
    'docs/examples/07_querybuilder.py',
    'docs/examples/08_fetch.py',
    'docs/examples/09_wss_download.py',
]

print("Validating example scripts...")
errors = []

# Phase 1: Quick syntax check (fail fast)
print("\n1. Checking syntax...")
for example in examples:
    path = Path(example)
    if not path.exists():
        errors.append(f"{example}: file not found")
        print(f"  FAIL: {example} (not found)")
        continue

    try:
        py_compile.compile(str(example), doraise=True)
        print(f"  OK: {example}")
    except py_compile.PyCompileError as e:
        errors.append(f"{example}: {e}")
        print(f"  FAIL: {example}")

if errors:
    print(f"\nSyntax errors found:")
    for err in errors:
        print(f"  {err}")
    sys.exit(1)

# Phase 2: Import check (ensures all dependencies resolve)
print("\n2. Checking imports...")
errors = []
for example in examples:
    try:
        spec = importlib.util.spec_from_file_location("example", example)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec for {example}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"  OK: {example}")
    except Exception as e:
        error_msg = f"{example}: {type(e).__name__}: {str(e)[:60]}"
        errors.append(error_msg)
        print(f"  FAIL: {example} ({type(e).__name__})")

if errors:
    print(f"\nImport errors found:")
    for err in errors:
        print(f"  {err}")
    sys.exit(1)

print(f"\n✓ All {len(examples)} examples validated")
