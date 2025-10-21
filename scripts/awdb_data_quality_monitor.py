#!/usr/bin/env python3
"""
AWDB Data Quality Monitoring Script.

Monitors data quality changes over time, particularly focusing on suspect data flags
that may change as QA processes are applied to historical data. Perfect for maintaining
data availability indices that need periodic re-indexing.
"""

import asyncio
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sqlite3

# Add src to path for imports
sys.path.insert(0, 'src')

from soildb.awdb.client import AWDBClient
from soildb.awdb.exceptions import AWDBError


class DataQualityMonitor:
    """Monitor AWDB data quality changes over time."""

    def __init__(self, db_path: str = "awdb_quality_monitor.db"):
        self.db_path = Path(db_path)
        self.client = AWDBClient(timeout=60)  # Longer timeout for bulk operations
        self._init_db()

    def _init_db(self):
        """Initialize database for quality monitoring."""
        with sqlite3.connect(self.db_path) as conn:
            # Main quality snapshots table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS quality_snapshots (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    station_triplet TEXT,
                    element TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    duration TEXT,
                    total_points INTEGER,
                    valid_points INTEGER,
                    suspect_points INTEGER,
                    missing_points INTEGER,
                    qc_flag_distribution TEXT,
                    qa_flag_distribution TEXT,
                    data_completeness REAL,
                    quality_score REAL
                )
            ''')

            # Quality change alerts
            conn.execute('''
                CREATE TABLE IF NOT EXISTS quality_alerts (
                    id INTEGER PRIMARY KEY,
                    detected_at TEXT,
                    station_triplet TEXT,
                    element TEXT,
                    alert_type TEXT,
                    description TEXT,
                    old_value REAL,
                    new_value REAL,
                    severity TEXT
                )
            ''')

            # Data availability index (for your use case)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS availability_index (
                    station_triplet TEXT,
                    element TEXT,
                    year INTEGER,
                    month INTEGER,
                    data_points INTEGER,
                    completeness REAL,
                    last_updated TEXT,
                    PRIMARY KEY (station_triplet, element, year, month)
                )
            ''')

            # SMS depth-specific completeness tracking
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sms_depth_completeness (
                    station_triplet TEXT,
                    depth_inches REAL,
                    ordinal INTEGER,
                    year INTEGER,
                    month INTEGER,
                    data_points INTEGER,
                    completeness REAL,
                    last_updated TEXT,
                    PRIMARY KEY (station_triplet, depth_inches, ordinal, year, month)
                )
            ''')

            # SMS sensor configuration tracking
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sms_sensor_config (
                    station_triplet TEXT,
                    depth_inches REAL,
                    ordinal INTEGER,
                    begin_date TEXT,
                    end_date TEXT,
                    data_precision INTEGER,
                    stored_unit_code TEXT,
                    last_updated TEXT,
                    PRIMARY KEY (station_triplet, depth_inches, ordinal)
                )
            ''')

    async def scan_station_quality(self, station_triplet: str, elements: List[str] = None,
                             years_back: int = 5) -> Dict[str, Any]:
        """Scan data quality for a specific station."""
        if elements is None:
            # Use semantic property names instead of cryptic element codes
            elements = ['air_temp_avg', 'precipitation', 'snow_water_equivalent', 'snow_depth', 'soil_moisture']

        print(f" Scanning quality for {station_triplet}...")

        results = {
            'station': station_triplet,
            'scan_timestamp': datetime.now().isoformat(),
            'elements_scanned': [],
            'alerts_generated': [],
            'availability_index_updates': 0,
            'sms_depth_analysis': [],
            'sensor_metadata': {}
        }

        # Get comprehensive sensor metadata for the station
        try:
            from soildb.awdb.convenience import get_station_sensor_metadata
            sensor_metadata = await get_station_sensor_metadata(station_triplet)
            results['sensor_metadata'] = sensor_metadata

            # Count total sensors by property type
            sensor_counts = {}
            for prop_name, sensors in sensor_metadata.get('sensors', {}).items():
                sensor_counts[prop_name] = len(sensors)

            print(f"   Station has {sum(sensor_counts.values())} sensors across {len(sensor_counts)} property types:")
            for prop_name, count in sorted(sensor_counts.items()):
                print(f"     {prop_name}: {count} sensors")

        except Exception as e:
            print(f"   Warning: Could not retrieve sensor metadata: {e}")
            results['sensor_metadata'] = {'error': str(e)}

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years_back)

        for element in elements:
            try:
                # Convert semantic property name to element code
                from soildb.awdb.convenience import PROPERTY_ELEMENT_MAP
                element_code = PROPERTY_ELEMENT_MAP.get(element, element)  # Fallback to original if not found

                if element == 'soil_moisture':
                    # Special handling for soil_moisture - analyze by depth
                    sms_results = await self._analyze_sms_by_depth(
                        station_triplet, start_date, end_date
                    )
                    results['sms_depth_analysis'].extend(sms_results)
                    results['elements_scanned'].append({
                        'property': element,
                        'element_code': element_code,
                        'depths_analyzed': len(sms_results),
                        'total_data_points': sum(r.get('total_points', 0) for r in sms_results)
                    })
                    continue

                # Get data for the entire period
                print(f"    Fetching {element} ({element_code}) data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                async with self.client:
                    data = await self.client.get_station_data(
                        station_triplet=station_triplet,
                        elements=element_code,
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d"),
                        duration="DAILY",
                        return_flags=True
                    )
                print(f"    Retrieved {len(data)} data points")

                # Ensure timestamps are datetime objects for processing
                print(f"    Processing {len(data)} timestamp objects...")
                for i, point in enumerate(data):
                    if hasattr(point, 'timestamp'):
                        if isinstance(point.timestamp, str):
                            # Convert string timestamp to datetime if needed
                            try:
                                point.timestamp = datetime.fromisoformat(point.timestamp.replace('Z', '+00:00'))
                            except Exception as e:
                                print(f"      Timestamp conversion failed for point {i}: {e}, value: {point.timestamp}")
                                # If conversion fails, try to create a datetime from the string
                                # This handles cases where timestamp might be malformed
                                try:
                                    # Assume it's already a valid datetime string
                                    point.timestamp = datetime.fromisoformat(point.timestamp)
                                except:
                                    # Last resort: create a dummy datetime to avoid crashes
                                    point.timestamp = datetime.now()
                        elif not isinstance(point.timestamp, datetime):
                            # If it's not a string or datetime, convert it
                            point.timestamp = datetime.now()
                        # Ensure timestamp has year, month, day attributes for grouping
                        if not hasattr(point.timestamp, 'year'):
                            point.timestamp = datetime.now()
                    else:
                        # If no timestamp attribute, create one
                        point.timestamp = datetime.now()
                print(f"    Timestamp processing complete")

                if not data:
                    print(f"  {element}: No data available")
                    continue

                # Analyze quality metrics
                print(f"    Analyzing quality metrics for {len(data)} points...")
                quality_metrics = self._analyze_quality_metrics(data, element_code)
                print(f"    Quality analysis complete")

                # Store snapshot
                self._store_quality_snapshot(
                    station_triplet, element_code, start_date, end_date,
                    quality_metrics
                )

                # Check for quality changes
                print(f"    Checking for quality changes...")
                try:
                    alerts = self._check_quality_changes(
                        station_triplet, element_code, quality_metrics
                    )
                    print(f"    Found {len(alerts)} quality alerts")
                except Exception as e:
                    print(f"    Quality change check failed: {e}")
                    alerts = []
                results['alerts_generated'].extend(alerts)

                # Update availability index
                index_updates = self._update_availability_index(
                    station_triplet, element_code, data
                )
                results['availability_index_updates'] += index_updates

                results['elements_scanned'].append({
                    'property': element,
                    'element_code': element_code,
                    'data_points': len(data),
                    'quality_metrics': quality_metrics,
                    'alerts': len(alerts)
                })

                print(f"  {element} ({element_code}): {len(data)} points, completeness: {quality_metrics['data_completeness']:.1%}")

            except Exception as e:
                print(f"   {element}: Failed - {e}")
                results['elements_scanned'].append({
                    'element': element,
                    'error': str(e)
                })

        return results

    def _analyze_quality_metrics(self, data: List, element: str) -> Dict[str, Any]:
        """Analyze quality metrics for a dataset."""
        total_points = len(data)
        valid_points = 0
        suspect_points = 0
        missing_points = 0

        qc_flags = {}
        qa_flags = {}

        for point in data:
            # Count valid vs missing data
            if point.value is None:
                missing_points += 1
            else:
                valid_points += 1

                # Check for suspect quality flags
                if (point.qc_flag in ['E', 'Q', 'S'] or
                    point.qa_flag in ['S', 'R', 'Q']):
                    suspect_points += 1

            # Count flag distributions
            if point.qc_flag:
                qc_flags[point.qc_flag] = qc_flags.get(point.qc_flag, 0) + 1
            if point.qa_flag:
                qa_flags[point.qa_flag] = qa_flags.get(point.qa_flag, 0) + 1

        # Calculate metrics
        data_completeness = valid_points / total_points if total_points > 0 else 0
        suspect_ratio = suspect_points / valid_points if valid_points > 0 else 0

        # Quality score (higher is better)
        quality_score = data_completeness * (1 - suspect_ratio)

        return {
            'total_points': total_points,
            'valid_points': valid_points,
            'suspect_points': suspect_points,
            'missing_points': missing_points,
            'data_completeness': data_completeness,
            'suspect_ratio': suspect_ratio,
            'quality_score': quality_score,
            'qc_flag_distribution': qc_flags,
            'qa_flag_distribution': qa_flags
        }

    def _store_quality_snapshot(self, station: str, element: str, start_date: datetime,
                               end_date: datetime, metrics: Dict[str, Any]):
        """Store quality snapshot in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO quality_snapshots
                (timestamp, station_triplet, element, start_date, end_date, duration,
                 total_points, valid_points, suspect_points, missing_points,
                 qc_flag_distribution, qa_flag_distribution, data_completeness, quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                station,
                element,
                start_date.isoformat(),
                end_date.isoformat(),
                'DAILY',
                metrics['total_points'],
                metrics['valid_points'],
                metrics['suspect_points'],
                metrics['missing_points'],
                json.dumps(metrics['qc_flag_distribution']),
                json.dumps(metrics['qa_flag_distribution']),
                metrics['data_completeness'],
                metrics['quality_score']
            ))

    def _check_quality_changes(self, station: str, element: str, current_metrics: Dict) -> List[Dict]:
        """Check for significant quality changes compared to previous snapshots."""
        alerts = []

        # Get previous snapshot
        prev_metrics = self._get_previous_quality_snapshot(station, element)

        if not prev_metrics:
            return alerts  # No previous data to compare

        # Check for significant changes
        thresholds = {
            'data_completeness': 0.05,  # 5% change in completeness
            'suspect_ratio': 0.10,      # 10% change in suspect ratio
            'quality_score': 0.10       # 10% change in overall quality
        }

        for metric, threshold in thresholds.items():
            if metric in current_metrics and metric in prev_metrics:
                current_val = current_metrics[metric]
                prev_val = prev_metrics[metric]

                # Ensure both values are numeric
                try:
                    current_val = float(current_val)
                    prev_val = float(prev_val)
                    change = abs(current_val - prev_val)

                    if change > threshold:
                        alert = {
                            'station': station,
                            'element': element,
                            'alert_type': f'{metric}_change',
                            'description': f'{metric} changed by {change:.1%} ({prev_val:.1%} -> {current_val:.1%})',
                            'old_value': prev_val,
                            'new_value': current_val,
                            'severity': 'high' if change > threshold * 2 else 'medium'
                        }
                        alerts.append(alert)

                        # Store alert
                        self._store_quality_alert(alert)
                except (ValueError, TypeError) as e:
                    print(f"      Warning: Could not compare {metric} values: {prev_val} vs {current_val} - {e}")
                    continue

        return alerts

    def _get_previous_quality_snapshot(self, station: str, element: str) -> Optional[Dict]:
        """Get the most recent quality snapshot for comparison."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT data_completeness, suspect_points, valid_points, quality_score
                FROM quality_snapshots
                WHERE station_triplet = ? AND element = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (station, element))

            row = cursor.fetchone()
            if row:
                return {
                    'data_completeness': row[0],  # Already a float
                    'suspect_ratio': row[1] / row[2] if row[2] > 0 else 0,  # suspect/valid
                    'quality_score': row[3]  # Already a float
                }
        return None

    def _store_quality_alert(self, alert: Dict):
        """Store quality alert in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO quality_alerts
                (detected_at, station_triplet, element, alert_type, description,
                 old_value, new_value, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                alert['station'],
                alert['element'],
                alert['alert_type'],
                alert['description'],
                alert['old_value'],
                alert['new_value'],
                alert['severity']
            ))

    def _update_availability_index(self, station: str, element: str, data: List) -> int:
        """Update the availability index with monthly data completeness."""
        updates = 0

        # Group data by year/month
        monthly_data = {}
        for point in data:
            if point.value is not None:  # Only count days with data
                year = point.timestamp.year
                month = point.timestamp.month

                key = (year, month)
                if key not in monthly_data:
                    monthly_data[key] = []
                monthly_data[key].append(point)

        # Calculate monthly completeness and update index
        with sqlite3.connect(self.db_path) as conn:
            for (year, month), month_data in monthly_data.items():
                # Calculate expected days in month
                if month == 2:  # February
                    expected_days = 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28
                elif month in [4, 6, 9, 11]:  # 30-day months
                    expected_days = 30
                else:  # 31-day months
                    expected_days = 31

                completeness = len(month_data) / expected_days

                # Update or insert availability record
                conn.execute('''
                    INSERT OR REPLACE INTO availability_index
                    (station_triplet, element, year, month, data_points, completeness, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    station,
                    element,
                    year,
                    month,
                    len(month_data),
                    completeness,
                    datetime.now().isoformat()
                ))

                updates += 1

        return updates

    async def _analyze_sms_by_depth(self, station_triplet: str, start_date: datetime,
                             end_date: datetime) -> List[Dict[str, Any]]:
        """Analyze SMS data by depth, handling multiple sensors at different depths."""
        print(f"    Analyzing SMS data by depth for {station_triplet}...")

        results = []

        # First, try to get station metadata to understand sensor configuration
        try:
            async with self.client:
                stations = await self.client.get_stations(
                    station_triplets=[station_triplet],
                    return_station_elements=True
                )

            if stations and stations[0].station_elements:
                # Extract SMS sensor configurations
                sms_elements = [
                    elem for elem in stations[0].station_elements
                    if elem.get('elementCode') == 'SMS'
                ]

                if sms_elements:
                    print(f"    Found {len(sms_elements)} SMS sensors configured")

                    # Update sensor configuration in database
                    for elem in sms_elements:
                        depth_inches = elem.get('heightDepth', 0) or 0
                        ordinal = elem.get('ordinal', 1)

                        with sqlite3.connect(self.db_path) as conn:
                            conn.execute('''
                                INSERT OR REPLACE INTO sms_sensor_config
                                (station_triplet, depth_inches, ordinal, begin_date, end_date,
                                 data_precision, stored_unit_code, last_updated)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                station_triplet,
                                depth_inches,
                                ordinal,
                                elem.get('beginDate'),
                                elem.get('endDate'),
                                elem.get('dataPrecision'),
                                elem.get('storedUnitCode'),
                                datetime.now().isoformat()
                            ))
        except Exception as e:
            print(f"    Warning: Could not retrieve station metadata: {e}")

        # Now analyze data for each configured SMS sensor
        if sms_elements:
            print(f"    Analyzing data for {len(sms_elements)} configured SMS sensors...")

            for elem in sms_elements:
                depth_inches = elem.get('heightDepth', 0) or 0
                ordinal = elem.get('ordinal', 1)

                try:
                    # Get SMS data for this specific sensor configuration
                    # Format: SMS:heightDepth:ordinal (heightDepth in inches)
                    element_string = f"SMS:{int(depth_inches)}:{ordinal}"
                    async with self.client:
                        data = await self.client.get_station_data(
                            station_triplet=station_triplet,
                            elements=element_string,
                            start_date=start_date.strftime("%Y-%m-%d"),
                            end_date=end_date.strftime("%Y-%m-%d"),
                            duration="DAILY",
                            return_flags=True
                        )

                    if not data:
                        continue

                    # Process timestamps
                    for point in data:
                        if isinstance(point.timestamp, str):
                            try:
                                point.timestamp = datetime.fromisoformat(point.timestamp.replace('Z', '+00:00'))
                            except:
                                point.timestamp = datetime.now()

                    # Analyze quality metrics for this depth
                    quality_metrics = self._analyze_quality_metrics(data, 'SMS')

                    # Group by month and calculate monthly completeness
                    monthly_completeness = self._calculate_monthly_completeness(data)

                    # Store depth-specific completeness data
                    for (year, month), completeness_data in monthly_completeness.items():
                        with sqlite3.connect(self.db_path) as conn:
                            conn.execute('''
                                INSERT OR REPLACE INTO sms_depth_completeness
                                (station_triplet, depth_inches, ordinal, year, month,
                                 data_points, completeness, last_updated)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                station_triplet,
                                depth_inches,
                                ordinal,
                                year,
                                month,
                                completeness_data['data_points'],
                                completeness_data['completeness'],
                                datetime.now().isoformat()
                            ))

                    depth_result = {
                        'station_triplet': station_triplet,
                        'depth_inches': depth_inches,
                        'ordinal': ordinal,
                        'total_points': len(data),
                        'quality_metrics': quality_metrics,
                        'monthly_completeness': monthly_completeness,
                        'date_range': {
                            'start': start_date.isoformat(),
                            'end': end_date.isoformat()
                        }
                    }

                    results.append(depth_result)

                    print(f"    SMS {depth_inches}\": {len(data)} points, completeness: {quality_metrics['data_completeness']:.1%}")

                except Exception as e:
                    # Skip combinations that don't exist
                    continue
        else:
            # Fallback: try typical depths if no sensor config available
            print(f"    No sensor configuration found, trying typical SCAN depths...")
            typical_depths = [2, 4, 8, 20, 40]  # inches
            max_ordinals = 3  # Allow for duplicates/triplicates

            for depth_inches in typical_depths:
                for ordinal in range(1, max_ordinals + 1):
                    try:
                        # Get SMS data for this specific depth/ordinal
                        # Format: SMS:heightDepth:ordinal (heightDepth in inches)
                        element_string = f"SMS:{int(depth_inches)}:{ordinal}"
                        async with self.client:
                            data = await self.client.get_station_data(
                                station_triplet=station_triplet,
                                elements=element_string,
                                start_date=start_date.strftime("%Y-%m-%d"),
                                end_date=end_date.strftime("%Y-%m-%d"),
                                duration="DAILY",
                                return_flags=True
                            )

                        if not data:
                            continue

                        # Process timestamps
                        for point in data:
                            if isinstance(point.timestamp, str):
                                try:
                                    point.timestamp = datetime.fromisoformat(point.timestamp.replace('Z', '+00:00'))
                                except:
                                    point.timestamp = datetime.now()

                        # Analyze quality metrics for this depth
                        quality_metrics = self._analyze_quality_metrics(data, 'SMS')

                        # Group by month and calculate monthly completeness
                        monthly_completeness = self._calculate_monthly_completeness(data)

                        # Store depth-specific completeness data
                        for (year, month), completeness_data in monthly_completeness.items():
                            with sqlite3.connect(self.db_path) as conn:
                                conn.execute('''
                                    INSERT OR REPLACE INTO sms_depth_completeness
                                    (station_triplet, depth_inches, ordinal, year, month,
                                     data_points, completeness, last_updated)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                ''', (
                                    station_triplet,
                                    depth_inches,
                                    ordinal,
                                    year,
                                    month,
                                    completeness_data['data_points'],
                                    completeness_data['completeness'],
                                    datetime.now().isoformat()
                                ))

                        depth_result = {
                            'station_triplet': station_triplet,
                            'depth_inches': depth_inches,
                            'ordinal': ordinal,
                            'total_points': len(data),
                            'quality_metrics': quality_metrics,
                            'monthly_completeness': monthly_completeness,
                            'date_range': {
                                'start': start_date.isoformat(),
                                'end': end_date.isoformat()
                            }
                        }

                        results.append(depth_result)

                        print(f"    SMS {depth_inches}\": {len(data)} points, completeness: {quality_metrics['data_completeness']:.1%}")

                    except Exception as e:
                        # Skip combinations that don't exist
                        continue

        print(f"    SMS depth analysis complete: {len(results)} depth/ordinal combinations found")
        return results

    def _calculate_monthly_completeness(self, data: List) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """Calculate monthly completeness for SMS data."""
        monthly_data = {}

        # Group data by year/month
        for point in data:
            if point.value is not None:  # Only count days with data
                year = point.timestamp.year
                month = point.timestamp.month

                key = (year, month)
                if key not in monthly_data:
                    monthly_data[key] = []
                monthly_data[key].append(point)

        # Calculate completeness for each month
        result = {}
        for (year, month), month_data in monthly_data.items():
            # Calculate expected days in month
            if month == 2:  # February
                expected_days = 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28
            elif month in [4, 6, 9, 11]:  # 30-day months
                expected_days = 30
            else:  # 31-day months
                expected_days = 31

            completeness = len(month_data) / expected_days

            result[(year, month)] = {
                'data_points': len(month_data),
                'expected_days': expected_days,
                'completeness': completeness
            }

        return result

    def get_quality_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate quality monitoring report."""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Recent alerts
            alerts = conn.execute('''
                SELECT * FROM quality_alerts
                WHERE detected_at > ?
                ORDER BY detected_at DESC
            ''', (cutoff_date,)).fetchall()

            # Quality summary by station
            quality_summary = conn.execute('''
                SELECT station_triplet, element,
                       AVG(data_completeness) as avg_completeness,
                       AVG(quality_score) as avg_quality,
                       COUNT(*) as snapshots
                FROM quality_snapshots
                WHERE timestamp > ?
                GROUP BY station_triplet, element
                ORDER BY station_triplet, element
            ''', (cutoff_date,)).fetchall()

            # Availability index summary
            availability_summary = conn.execute('''
                SELECT station_triplet, element,
                        AVG(completeness) as avg_monthly_completeness,
                        COUNT(*) as months_tracked
                FROM availability_index
                GROUP BY station_triplet, element
                ORDER BY station_triplet, element
            ''').fetchall()

            # SMS depth completeness summary
            sms_depth_summary = conn.execute('''
                SELECT station_triplet, depth_inches, ordinal,
                        AVG(completeness) as avg_monthly_completeness,
                        COUNT(*) as months_tracked,
                        SUM(data_points) as total_data_points
                FROM sms_depth_completeness
                GROUP BY station_triplet, depth_inches, ordinal
                ORDER BY station_triplet, depth_inches, ordinal
            ''').fetchall()

        return {
            'report_period_days': days,
            'generated_at': datetime.now().isoformat(),
            'quality_alerts': [
                {
                    'detected_at': row[1],
                    'station': row[2],
                    'element': row[3],
                    'alert_type': row[4],
                    'description': row[5],
                    'severity': row[8]
                }
                for row in alerts
            ],
            'quality_summary': [
                {
                    'station': row[0],
                    'element': row[1],
                    'avg_completeness': row[2],
                    'avg_quality_score': row[3],
                    'snapshots': row[4]
                }
                for row in quality_summary
            ],
            'availability_summary': [
                {
                    'station': row[0],
                    'element': row[1],
                    'avg_monthly_completeness': row[2],
                    'months_tracked': row[3]
                }
                for row in availability_summary
            ],
            'sms_depth_summary': [
                {
                    'station': row[0],
                    'depth_inches': row[1],
                    'ordinal': row[2],
                    'avg_monthly_completeness': row[3],
                    'months_tracked': row[4],
                    'total_data_points': row[5]
                }
                for row in sms_depth_summary
            ]
        }

    async def scan_multiple_stations(self, station_list: List[str], elements: List[str] = None,
                                max_stations: int = 10) -> Dict[str, Any]:
        """Scan quality for multiple stations (with limit to avoid API throttling)."""
        print(f" Scanning quality for {min(len(station_list), max_stations)} stations...")

        results = {
            'stations_scanned': 0,
            'total_alerts': 0,
            'total_index_updates': 0,
            'scan_timestamp': datetime.now().isoformat(),
            'station_results': []
        }

        for station in station_list[:max_stations]:
            try:
                station_result = await self.scan_station_quality(station, elements)
                results['station_results'].append(station_result)
                results['stations_scanned'] += 1
                results['total_alerts'] += len(station_result['alerts_generated'])
                results['total_index_updates'] += station_result['availability_index_updates']

            except Exception as e:
                print(f" Failed to scan {station}: {e}")
                results['station_results'].append({
                    'station': station,
                    'error': str(e)
                })

        return results

    async def scan_state_stations(self, state_code: str, network_codes: List[str] = None,
                            elements: List[str] = None, max_stations: int = 50,
                            chunk_size: int = 10) -> Dict[str, Any]:
        """
        Scan all stations in a specific state, processing in chunks to avoid timeouts.

        Args:
            state_code: Two-letter state code (e.g., 'CA', 'OR')
            network_codes: List of network codes to filter by
            elements: Elements to scan for each station
            max_stations: Maximum total stations to scan
            chunk_size: Process stations in chunks of this size

        Returns:
            Comprehensive scan results for the state
        """
        print(f" Scanning all stations in {state_code}...")

        # Get all stations in the state using station triplet wildcards for better filtering
        print(f"   Fetching stations for state: {state_code}, networks: {network_codes}")

        # Build station triplet patterns for filtering
        station_triplets = []
        if network_codes:
            for network in network_codes:
                # Use wildcard pattern: *:STATE:NETWORK
                station_triplets.append(f"*:{state_code}:{network}")
        else:
            # If no specific networks, get all networks for the state
            station_triplets.append(f"*:{state_code}:*")

        print(f"   Using station triplet patterns: {station_triplets}")

        async with self.client:
            stations = await self.client.get_stations(
                station_triplets=station_triplets,
                active_only=True
            )
        print(f"   API returned {len(stations)} stations")

        if not stations:
            return {
                'state': state_code,
                'error': f'No active stations found in {state_code}',
                'stations_found': 0,
                'stations_scanned': 0
            }

        print(f" Found {len(stations)} active stations in {state_code}")

        # Limit total stations if specified
        if max_stations and len(stations) > max_stations:
            stations = stations[:max_stations]
            print(f" Limiting to first {max_stations} stations")

        # Process in chunks
        all_results = {
            'state': state_code,
            'stations_found': len(stations),
            'stations_scanned': 0,
            'total_alerts': 0,
            'total_index_updates': 0,
            'chunks_processed': 0,
            'scan_timestamp': datetime.now().isoformat(),
            'chunk_results': []
        }

        station_triplets = [s.station_triplet for s in stations]

        for i in range(0, len(station_triplets), chunk_size):
            chunk = station_triplets[i:i + chunk_size]
            chunk_num = (i // chunk_size) + 1
            total_chunks = (len(station_triplets) + chunk_size - 1) // chunk_size

            print(f"\n Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} stations)...")

            try:
                chunk_result = await self.scan_multiple_stations(
                    chunk, elements=elements, max_stations=len(chunk)
                )

                all_results['chunk_results'].append({
                    'chunk_number': chunk_num,
                    'stations_in_chunk': len(chunk),
                    'result': chunk_result
                })

                all_results['stations_scanned'] += chunk_result['stations_scanned']
                all_results['total_alerts'] += chunk_result['total_alerts']
                all_results['total_index_updates'] += chunk_result['total_index_updates']
                all_results['chunks_processed'] += 1

                print(f" Chunk {chunk_num} complete: {chunk_result['stations_scanned']} stations, "
                      f"{chunk_result['total_alerts']} alerts, {chunk_result['total_index_updates']} index updates")

            except Exception as e:
                print(f" Failed to process chunk {chunk_num}: {e}")
                all_results['chunk_results'].append({
                    'chunk_number': chunk_num,
                    'error': str(e)
                })

        print(f"\n {state_code} scan complete: {all_results['stations_scanned']}/{all_results['stations_found']} stations processed")
        return all_results


async def main():
    """Run data quality monitoring."""
    print(" AWDB Data Quality Monitor")
    print("=" * 50)

    monitor = DataQualityMonitor()

    # Check command line arguments for state scan
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--state":
        if len(sys.argv) < 3:
            print("Usage: python awdb_data_quality_monitor.py --state CA [network_codes]")
            sys.exit(1)

        state_code = sys.argv[2].upper()
        network_codes = sys.argv[3:] if len(sys.argv) > 3 else None

        print(f"\n Scanning all stations in {state_code}...")

        # Scan entire state in chunks
        state_results = await monitor.scan_state_stations(
            state_code=state_code,
            network_codes=network_codes,
            elements=['air_temp_avg', 'precipitation', 'snow_water_equivalent'],  # Use semantic names
            max_stations=100,  # Limit for testing
            chunk_size=5      # Small chunks for testing
        )

        print("\n State scan results:")
        print(f"   State: {state_results['state']}")
        print(f"   Stations found: {state_results['stations_found']}")
        print(f"   Stations scanned: {state_results['stations_scanned']}")
        print(f"   Total alerts: {state_results['total_alerts']}")
        print(f"   Index updates: {state_results['total_index_updates']}")
        print(f"   Chunks processed: {state_results['chunks_processed']}")

        # Generate report
        print("\n2. Generating quality report...")
        report = monitor.get_quality_report(days=1)  # Last day

        print(f"   Recent alerts: {len(report['quality_alerts'])}")
        print(f"   Stations tracked: {len(report['quality_summary'])}")
        print(f"   Availability records: {len(report['availability_summary'])}")

        if report['quality_alerts']:
            print("     Recent quality alerts:")
            for alert in report['quality_alerts'][:5]:  # Show first 5
                print(f"     {alert['station']} {alert['element']}: {alert['description']}")

        print(f"\n Quality monitoring complete. Database updated at: {monitor.db_path}")

    else:
        # Default: Scan a few key stations
        key_stations = [
            "301:CA:SNTL",    # SNTL station
            "2197:CO:SCAN",   # SCAN station with comprehensive sensors
            "AGP:CA:MSNT",    # MSNT station
        ]

        print("\n1. Scanning key stations for quality changes...")
        # Use semantic property names instead of element codes
        scan_results = await monitor.scan_multiple_stations(
            key_stations,
            elements=['air_temp_avg', 'precipitation', 'snow_water_equivalent', 'snow_depth', 'soil_moisture'],
            max_stations=3
        )

        print(f"   Scanned {scan_results['stations_scanned']} stations")
        print(f"   Generated {scan_results['total_alerts']} quality alerts")
        print(f"   Updated {scan_results['total_index_updates']} availability index entries")

        # Show sensor metadata summary for each station
        print("\n   Sensor inventory summary:")
        for station_result in scan_results.get('station_results', []):
            if 'sensor_metadata' in station_result and 'sensors' in station_result['sensor_metadata']:
                sensor_counts = {}
                for prop_name, sensors in station_result['sensor_metadata']['sensors'].items():
                    sensor_counts[prop_name] = len(sensors)
                total_sensors = sum(sensor_counts.values())
                print(f"     {station_result['station']}: {total_sensors} sensors across {len(sensor_counts)} properties")

        # Generate report
        print("\n2. Generating quality report...")
        report = monitor.get_quality_report(days=7)  # Last week

        print(f"   Recent alerts: {len(report['quality_alerts'])}")
        print(f"   Stations tracked: {len(report['quality_summary'])}")
        print(f"   Availability records: {len(report['availability_summary'])}")

        if report['quality_alerts']:
            print("     Recent quality alerts:")
            for alert in report['quality_alerts'][:3]:  # Show first 3
                print(f"     {alert['station']} {alert['element']}: {alert['description']}")

        print(f"\n Quality monitoring complete. Database updated at: {monitor.db_path}")

        # Show sample availability data
        if report['availability_summary']:
            print("\n Sample availability data:")
            for item in report['availability_summary'][:3]:
                print(".1%")

if __name__ == "__main__":
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print("\n  Quality monitoring interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        sys.exit(1)