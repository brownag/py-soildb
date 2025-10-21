#!/usr/bin/env python3
"""
AWDB API Schema Change Detection Script.

Monitors for changes in AWDB API response schemas and data quality flags.
Useful for detecting when historical data quality changes or API schema updates occur.
"""

import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, 'src')

from soildb.awdb.client import AWDBClient


class SchemaMonitor:
    """Monitor AWDB API schema changes and data quality."""

    def __init__(self, db_path: str = "awdb_schema_monitor.db"):
        self.db_path = Path(db_path)
        self.client = AWDBClient(timeout=30)
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for storing schema snapshots."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS schema_snapshots (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    endpoint TEXT,
                    response_hash TEXT,
                    response_sample TEXT,
                    field_count INTEGER,
                    record_count INTEGER
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS quality_flag_history (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    station_triplet TEXT,
                    element TEXT,
                    date_range TEXT,
                    qc_flags TEXT,
                    qa_flags TEXT,
                    suspect_data_count INTEGER,
                    total_data_count INTEGER
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS schema_changes (
                    id INTEGER PRIMARY KEY,
                    detected_at TEXT,
                    endpoint TEXT,
                    change_type TEXT,
                    description TEXT,
                    old_hash TEXT,
                    new_hash TEXT
                )
            ''')

    def _hash_response(self, response: Any) -> str:
        """Create hash of response for change detection."""
        # Convert to normalized JSON string for consistent hashing
        if isinstance(response, (list, dict)):
            normalized = json.dumps(response, sort_keys=True, default=str)
            return hashlib.sha256(normalized.encode()).hexdigest()
        return hashlib.sha256(str(response).encode()).hexdigest()

    def _get_latest_snapshot(self, endpoint: str) -> Optional[Dict]:
        """Get the most recent snapshot for an endpoint."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM schema_snapshots
                WHERE endpoint = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (endpoint,))

            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'timestamp': row[1],
                    'endpoint': row[2],
                    'response_hash': row[3],
                    'response_sample': row[4],
                    'field_count': row[5],
                    'record_count': row[6]
                }
        return None

    def check_schema_changes(self) -> List[Dict]:
        """Check for schema changes in key API endpoints."""
        print(" Checking for schema changes...")

        endpoints_to_monitor = [
            ('stations', lambda: self.client.get_stations(network_codes=['SCAN'], active_only=True)),
            ('data', lambda: self.client.get_station_data(
                "301:CA:SNTL", "TAVG", "2023-01-01", "2023-01-31", duration="DAILY"
            )),
            ('reference', lambda: self.client.get_reference_data(['elements', 'networks'])),
        ]

        changes_detected = []

        for endpoint_name, endpoint_func in endpoints_to_monitor:
            try:
                # Get current response
                response = endpoint_func()

                # Calculate metrics
                current_hash = self._hash_response(response)
                field_count = len(response) if isinstance(response, dict) else len(str(response))
                record_count = len(response) if isinstance(response, list) else 1

                # Get previous snapshot
                prev_snapshot = self._get_latest_snapshot(endpoint_name)

                # Check for changes
                if prev_snapshot:
                    if prev_snapshot['response_hash'] != current_hash:
                        change_desc = f"Schema change detected in {endpoint_name}"
                        if prev_snapshot['field_count'] != field_count:
                            change_desc += f" (field count: {prev_snapshot['field_count']} -> {field_count})"
                        if prev_snapshot['record_count'] != record_count:
                            change_desc += f" (record count: {prev_snapshot['record_count']} -> {record_count})"

                        changes_detected.append({
                            'endpoint': endpoint_name,
                            'change_type': 'schema_change',
                            'description': change_desc,
                            'old_hash': prev_snapshot['response_hash'],
                            'new_hash': current_hash
                        })

                        # Log change
                        self._log_schema_change(endpoint_name, 'schema_change', change_desc,
                                              prev_snapshot['response_hash'], current_hash)

                # Store current snapshot
                self._store_snapshot(endpoint_name, current_hash, response, field_count, record_count)

            except Exception as e:
                print(f" Failed to check {endpoint_name}: {e}")
                changes_detected.append({
                    'endpoint': endpoint_name,
                    'change_type': 'error',
                    'description': f"Failed to check endpoint: {e}",
                    'old_hash': None,
                    'new_hash': None
                })

        return changes_detected

    def _store_snapshot(self, endpoint: str, response_hash: str, response: Any,
                       field_count: int, record_count: int):
        """Store a schema snapshot."""
        with sqlite3.connect(self.db_path) as conn:
            # Store sample of response (first few items only for large responses)
            if isinstance(response, list) and len(response) > 3:
                sample = response[:3]  # First 3 items
            else:
                sample = response

            conn.execute('''
                INSERT INTO schema_snapshots
                (timestamp, endpoint, response_hash, response_sample, field_count, record_count)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                endpoint,
                response_hash,
                json.dumps(sample, default=str),
                field_count,
                record_count
            ))

    def _log_schema_change(self, endpoint: str, change_type: str, description: str,
                          old_hash: str, new_hash: str):
        """Log a detected schema change."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO schema_changes
                (detected_at, endpoint, change_type, description, old_hash, new_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                endpoint,
                change_type,
                description,
                old_hash,
                new_hash
            ))

    def monitor_data_quality_flags(self, stations: List[str] = None) -> Dict[str, Any]:
        """Monitor data quality flags that might change over time."""
        print("  Monitoring data quality flags...")

        if stations is None:
            # Use a sample of stations from different networks
            stations = [
                "301:CA:SNTL",    # SNTL - Snow Telemetry
                "2235:CA:SCAN",   # SCAN - Soil Climate Analysis Network
                "AGP:CA:MSNT",    # MSNT - Manual Snow Telemetry
            ]

        quality_summary = {
            'stations_checked': len(stations),
            'total_data_points': 0,
            'quality_flag_distribution': {},
            'suspect_data_changes': [],
            'timestamp': datetime.now().isoformat()
        }

        for station in stations:
            try:
                # Get recent data (last 6 months)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=180)

                data = self.client.get_station_data(
                    station_triplet=station,
                    elements="TAVG",  # Air temperature - commonly has quality flags
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    duration="DAILY",
                    return_flags=True
                )

                if not data:
                    continue

                # Analyze quality flags
                qc_flags = {}
                qa_flags = {}
                suspect_count = 0
                total_count = len(data)

                for point in data:
                    # Count QC flags
                    if point.qc_flag:
                        qc_flags[point.qc_flag] = qc_flags.get(point.qc_flag, 0) + 1

                    # Count QA flags
                    if point.qa_flag:
                        qa_flags[point.qa_flag] = qa_flags.get(point.qa_flag, 0) + 1

                    # Check for suspect data (this is what can change over time)
                    if point.qc_flag in ['E', 'Q'] or point.qa_flag in ['S', 'R']:
                        suspect_count += 1

                # Store in database for historical comparison
                self._store_quality_snapshot(
                    station, "TAVG",
                    f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                    qc_flags, qa_flags, suspect_count, total_count
                )

                # Check for significant changes in suspect data ratio
                prev_snapshot = self._get_previous_quality_snapshot(station, "TAVG")
                if prev_snapshot:
                    old_suspect_ratio = prev_snapshot['suspect_data_count'] / prev_snapshot['total_data_count']
                    new_suspect_ratio = suspect_count / total_count
                    ratio_change = abs(new_suspect_ratio - old_suspect_ratio)

                    if ratio_change > 0.05:  # 5% change threshold
                        quality_summary['suspect_data_changes'].append({
                            'station': station,
                            'old_ratio': old_suspect_ratio,
                            'new_ratio': new_suspect_ratio,
                            'change': ratio_change
                        })

                # Update summary
                quality_summary['total_data_points'] += total_count

                # Merge flag distributions
                for flag, count in qc_flags.items():
                    key = f"QC:{flag}"
                    quality_summary['quality_flag_distribution'][key] = \
                        quality_summary['quality_flag_distribution'].get(key, 0) + count

                for flag, count in qa_flags.items():
                    key = f"QA:{flag}"
                    quality_summary['quality_flag_distribution'][key] = \
                        quality_summary['quality_flag_distribution'].get(key, 0) + count

                print(f"  {station}: {total_count} points, {suspect_count} suspect")

            except Exception as e:
                print(f"   {station}: Failed - {e}")

        return quality_summary

    def _store_quality_snapshot(self, station: str, element: str, date_range: str,
                               qc_flags: Dict, qa_flags: Dict, suspect_count: int, total_count: int):
        """Store quality flag snapshot."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO quality_flag_history
                (timestamp, station_triplet, element, date_range, qc_flags, qa_flags,
                 suspect_data_count, total_data_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                station,
                element,
                date_range,
                json.dumps(qc_flags),
                json.dumps(qa_flags),
                suspect_count,
                total_count
            ))

    def _get_previous_quality_snapshot(self, station: str, element: str) -> Optional[Dict]:
        """Get the most recent quality snapshot for comparison."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM quality_flag_history
                WHERE station_triplet = ? AND element = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (station, element))

            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'timestamp': row[1],
                    'station_triplet': row[2],
                    'element': row[3],
                    'date_range': row[4],
                    'qc_flags': json.loads(row[5] or '{}'),
                    'qa_flags': json.loads(row[6] or '{}'),
                    'suspect_data_count': row[7],
                    'total_data_count': row[8]
                }
        return None

    def get_change_history(self, days: int = 30) -> List[Dict]:
        """Get recent schema and quality changes."""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Get schema changes
            schema_changes = conn.execute('''
                SELECT * FROM schema_changes
                WHERE detected_at > ?
                ORDER BY detected_at DESC
            ''', (cutoff_date,)).fetchall()

            # Get quality changes (simplified)
            quality_changes = conn.execute('''
                SELECT station_triplet, COUNT(*) as snapshots
                FROM quality_flag_history
                WHERE timestamp > ?
                GROUP BY station_triplet
                ORDER BY snapshots DESC
            ''', (cutoff_date,)).fetchall()

        return {
            'schema_changes': [
                {
                    'detected_at': row[1],
                    'endpoint': row[2],
                    'change_type': row[3],
                    'description': row[4]
                }
                for row in schema_changes
            ],
            'quality_monitoring': [
                {
                    'station': row[0],
                    'snapshots': row[1]
                }
                for row in quality_changes
            ]
        }


def main():
    """Run schema and quality monitoring."""
    print(" AWDB Schema & Quality Monitor")
    print("=" * 50)

    monitor = SchemaMonitor()

    # Check for schema changes
    print("\n1. Checking for API schema changes...")
    schema_changes = monitor.check_schema_changes()

    if schema_changes:
        print(f"  {len(schema_changes)} schema changes detected:")
        for change in schema_changes:
            if change['change_type'] != 'error':
                print(f"    {change['endpoint']}: {change['description']}")
            else:
                print(f"    {change['endpoint']}: ERROR - {change['description']}")
    else:
        print(" No schema changes detected")

    # Monitor data quality flags
    print("\n2. Monitoring data quality flags...")
    quality_summary = monitor.monitor_data_quality_flags()

    print(f"   Checked {quality_summary['stations_checked']} stations")
    print(f"   Total data points: {quality_summary['total_data_points']}")

    if quality_summary['suspect_data_changes']:
        print(f"     {len(quality_summary['suspect_data_changes'])} stations with suspect data changes:")
        for change in quality_summary['suspect_data_changes']:
            print(".3f")
    else:
        print("    No significant suspect data changes detected")

    # Show flag distribution
    if quality_summary['quality_flag_distribution']:
        print("   Quality flag distribution:")
        for flag, count in sorted(quality_summary['quality_flag_distribution'].items()):
            print(f"     {flag}: {count}")

    # Get recent change history
    print("\n3. Recent change history (last 30 days):")
    history = monitor.get_change_history(days=30)

    if history['schema_changes']:
        print(f"   Schema changes: {len(history['schema_changes'])}")
        for change in history['schema_changes'][:3]:  # Show last 3
            print(f"     {change['detected_at'][:10]}: {change['endpoint']} - {change['change_type']}")
    else:
        print("   Schema changes: None")

    print(f"   Quality monitoring: {len(history['quality_monitoring'])} stations tracked")

    print("\n Monitoring complete. Database updated at:", monitor.db_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Monitoring interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        sys.exit(1)
