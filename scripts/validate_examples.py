#!/usr/bin/env python3
"""Validate that example scripts have correct syntax and can import."""

import importlib.util
import py_compile
import sys
from pathlib import Path

examples = [
    'docs/examples/01_basic.py',
    'docs/examples/02_spatial.py',
    'docs/examples/03_metadata.py',
    'docs/examples/04_schema.py',
    'docs/examples/05_awdb.py',
    'docs/examples/06_awdb_availability.py',
    'docs/examples/07_query_templates.py',
    'docs/examples/08_fetch.py',
    'docs/examples/09_wss_download.py',
    'docs/examples/soilprofilecollection/01_basic_conversion.py',
    'docs/examples/soilprofilecollection/02_with_site_metadata.py',
    'docs/examples/soilprofilecollection/03_lab_pedon_workflow.py',
    'docs/examples/soilprofilecollection/04_custom_columns.py',
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
    print("\nSyntax errors found:")
    for err in errors:
        print(f"  {err}")
    sys.exit(1)

# Phase 2: Import check (ensures all dependencies resolve)
print("\n2. Checking imports...")
has_spc = importlib.util.find_spec("soilprofilecollection") is not None
errors = []
skipped = 0

for example in examples:
    if "soilprofilecollection" in example and not has_spc:
        print(f"  SKIP: {example} (soilprofilecollection not installed)")
        skipped += 1
        continue

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
    print("\nImport errors found:")
    for err in errors:
        print(f"  {err}")
    sys.exit(1)

if skipped:
    print(f"\n✓ All {len(examples)} examples validated ({skipped} skipped import check: missing optional dependency)")
else:
    print(f"\n✓ All {len(examples)} examples validated")
