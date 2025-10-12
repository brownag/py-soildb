"""
Response handling for SDA query results with proper data type conversion.
"""

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Iterator, List

from .exceptions import SDAResponseError

if TYPE_CHECKING:
    try:
        import pandas as pd
    except ImportError:
        pd = None  # type: ignore
    from soilprofilecollection import SoilProfileCollection

    try:
        import polars as pl
    except ImportError:
        pl = None  # type: ignore


class SDAResponse:
    """Represents a response from the Soil Data Access web service."""

    # SDA data type mapping to Python/pandas/polars types
    SDA_TYPE_MAPPING = {
        # Numeric types
        "int": "int64",
        "integer": "int64",
        "bigint": "int64",
        "smallint": "int32",
        "tinyint": "int16",
        "bit": "bool",
        # Floating point types
        "float": "float64",
        "real": "float32",
        "double": "float64",
        "decimal": "float64",
        "numeric": "float64",
        "money": "float64",
        "smallmoney": "float64",
        # String types
        "varchar": "string",
        "nvarchar": "string",
        "char": "string",
        "nchar": "string",
        "text": "string",
        "ntext": "string",
        # Date/time types
        "datetime": "datetime64[ns]",
        "datetime2": "datetime64[ns]",
        "smalldatetime": "datetime64[ns]",
        "date": "datetime64[ns]",
        "time": "string",  # Keep as string for time-only values
        "timestamp": "datetime64[ns]",
        # Spatial/binary types
        "geometry": "string",  # Keep WKT as string
        "geography": "string",
        "varbinary": "string",
        "binary": "string",
        "image": "string",
        # Other types
        "uniqueidentifier": "string",
        "xml": "string",
    }

    def __init__(self, raw_data: Dict[str, Any]):
        """Initialize from SDA JSON response."""
        self._raw_data = raw_data
        self._parse_response()

    def _parse_response(self) -> None:
        """Parse the SDA response format."""
        if "Table" not in self._raw_data:
            # Handle empty responses (no results) - SDA returns {} for empty result sets
            if self._raw_data == {}:
                # Empty result set
                self._columns: List[str] = []
                self._metadata: List[str] = []
                self._data = []
                return
            else:
                raise SDAResponseError(
                    "Invalid SDA response format: missing 'Table' key"
                )

        table_data = self._raw_data["Table"]

        if not isinstance(table_data, list) or len(table_data) < 2:
            raise SDAResponseError(
                "Invalid SDA response format: Table data is not a list or too short"
            )

        # First row contains column names
        self._columns = table_data[0] if table_data else []

        # Second row contains column metadata (data types, etc.)
        self._metadata = table_data[1] if len(table_data) > 1 else []

        # Remaining rows contain actual data
        self._data = table_data[2:] if len(table_data) > 2 else []

    @classmethod
    def from_json(cls, json_str: str) -> "SDAResponse":
        """Create response from JSON string."""
        try:
            data = json.loads(json_str)
            return cls(data)
        except json.JSONDecodeError as e:
            raise SDAResponseError(f"Failed to parse JSON response: {e}") from e

    @property
    def columns(self) -> List[str]:
        """Get column names."""
        return self._columns

    @property
    def data(self) -> List[List[Any]]:
        """Get raw data rows."""
        return self._data

    @property
    def metadata(self) -> List[str]:
        """Get column metadata."""
        return self._metadata

    def __len__(self) -> int:
        """Return number of data rows."""
        return len(self._data)

    def __iter__(self) -> "Iterator[List[Any]]":
        """Iterate over data rows."""
        return iter(self._data)

    def to_dict(self) -> List[Dict[str, Any]]:
        """Convert to list of dictionaries with basic type conversion.

        Returns:
            List of dictionaries where each dictionary represents a row with
            column names as keys and converted values as values.
        """
        if not self._columns:
            return []

        # Get column types for basic conversion
        column_types = self.get_column_types()

        result = []
        for row in self._data:
            # Pad row with None if it's shorter than columns
            padded_row = row + [None] * (len(self._columns) - len(row))

            # Convert values based on inferred types
            converted_row = {}
            for _, (col_name, value) in enumerate(
                zip(self._columns, padded_row[: len(self._columns)])
            ):
                sda_type = column_types.get(col_name, "varchar").lower()
                converted_row[col_name] = self._convert_value(value, sda_type)

            result.append(converted_row)

        return result

    def to_records(self) -> List[Dict[str, Any]]:
        """Alias for to_dict() for compatibility."""
        return self.to_dict()

    def _convert_value(self, value: Any, sda_type: str) -> Any:
        """Convert a single value based on SDA data type."""
        if value is None or value == "" or value == "NULL":
            return None

        sda_type = sda_type.lower()

        try:
            # Integer types
            if sda_type in ["int", "integer", "bigint", "smallint", "tinyint"]:
                return int(float(str(value))) if value is not None else None

            # Boolean type
            elif sda_type == "bit":
                if str(value).lower() in ["true", "1", "yes", "t"]:
                    return True
                elif str(value).lower() in ["false", "0", "no", "f"]:
                    return False
                else:
                    return None

            # Float types
            elif sda_type in [
                "float",
                "real",
                "double",
                "decimal",
                "numeric",
                "money",
                "smallmoney",
            ]:
                return float(value) if value is not None else None

            # Date/time types - basic parsing
            elif sda_type in [
                "datetime",
                "datetime2",
                "smalldatetime",
                "date",
                "timestamp",
            ]:
                if isinstance(value, str):
                    # Try common date formats
                    for fmt in [
                        "%Y-%m-%d %H:%M:%S",
                        "%Y-%m-%d",
                        "%m/%d/%Y",
                        "%Y-%m-%dT%H:%M:%S",
                    ]:
                        try:
                            return datetime.strptime(value, fmt)
                        except ValueError:
                            continue
                return str(value)  # Fall back to string if parsing fails

            # String types (default)
            else:
                return str(value) if value is not None else None

        except (ValueError, TypeError):
            # If conversion fails, return as string
            return str(value) if value is not None else None

    def _get_pandas_dtype_mapping(self) -> Dict[str, str]:
        """Get pandas-compatible dtype mapping."""
        column_types = self.get_column_types()
        dtype_mapping = {}

        for col_name, sda_type in column_types.items():
            sda_type_lower = sda_type.lower()

            # Map to pandas dtypes
            if sda_type_lower in ["int", "integer", "bigint"]:
                dtype_mapping[col_name] = "Int64"  # Nullable integer
            elif sda_type_lower in ["smallint", "tinyint"]:
                dtype_mapping[col_name] = "Int32"
            elif sda_type_lower == "bit":
                dtype_mapping[col_name] = "boolean"
            elif sda_type_lower in ["float", "real", "double", "decimal", "numeric"]:
                dtype_mapping[col_name] = "float64"
            elif sda_type_lower in ["datetime", "datetime2", "smalldatetime", "date"]:
                dtype_mapping[col_name] = "datetime64[ns]"
            else:
                dtype_mapping[col_name] = "string"

        return dtype_mapping

    def _get_polars_dtype_mapping(self) -> Dict[str, Any]:
        """Get polars-compatible dtype mapping."""
        try:
            import polars as pl
        except ImportError:
            return {}

        column_types = self.get_column_types()
        dtype_mapping = {}

        for col_name, sda_type in column_types.items():
            sda_type_lower = sda_type.lower()

            # Map to polars dtypes
            if sda_type_lower in ["int", "integer", "bigint"]:
                dtype_mapping[col_name] = pl.Int64  # type: ignore
            elif sda_type_lower in ["smallint", "tinyint"]:
                dtype_mapping[col_name] = pl.Int32  # type: ignore
            elif sda_type_lower == "bit":
                dtype_mapping[col_name] = pl.Boolean  # type: ignore
            elif sda_type_lower in ["float", "real", "double", "decimal", "numeric"]:
                dtype_mapping[col_name] = pl.Float64  # type: ignore
            elif sda_type_lower in ["datetime", "datetime2", "smalldatetime", "date"]:
                dtype_mapping[col_name] = pl.Datetime  # type: ignore
            else:
                dtype_mapping[col_name] = pl.Utf8  # type: ignore

        return dtype_mapping

    def to_dataframe(self, library: str = "pandas", convert_types: bool = True) -> Any:
        """Convert to pandas or polars DataFrame with proper type conversion."""
        data_dict = self.to_dict()

        if library.lower() == "pandas":
            try:
                import pandas as pd

                df = pd.DataFrame(data_dict)

                if convert_types and not df.empty:
                    # Apply dtype conversion
                    dtype_mapping = self._get_pandas_dtype_mapping()

                    for col_name, dtype in dtype_mapping.items():
                        if col_name in df.columns:
                            try:
                                if dtype == "datetime64[ns]":
                                    df[col_name] = pd.to_datetime(
                                        df[col_name], errors="coerce"
                                    )
                                elif dtype in ["Int64", "Int32"]:
                                    df[col_name] = pd.to_numeric(
                                        df[col_name], errors="coerce"
                                    ).astype(dtype)
                                elif dtype == "boolean":
                                    df[col_name] = df[col_name].astype("boolean")
                                elif dtype == "float64":
                                    df[col_name] = pd.to_numeric(
                                        df[col_name], errors="coerce"
                                    )
                                else:
                                    df[col_name] = df[col_name].astype(dtype)
                            except (ValueError, TypeError):
                                # If conversion fails, keep as object/string
                                continue

                return df
            except ImportError:
                raise ImportError(
                    "pandas is required for DataFrame conversion. Install with: pip install pandas"
                ) from None

        elif library.lower() == "polars":
            try:
                import polars as pl

                df = pl.DataFrame(data_dict)

                if convert_types and not df.is_empty():
                    # Apply dtype conversion
                    dtype_mapping = self._get_polars_dtype_mapping()

                    for col_name, dtype in dtype_mapping.items():
                        if col_name in df.columns:
                            try:
                                if dtype == pl.Datetime:
                                    df = df.with_columns(
                                        pl.col(col_name).str.strptime(
                                            pl.Datetime, format=None, strict=False
                                        )
                                    )
                                else:
                                    df = df.with_columns(
                                        pl.col(col_name).cast(dtype, strict=False)  # type: ignore
                                    )
                            except Exception:
                                # If conversion fails, keep original type
                                continue

                return df
            except ImportError:
                raise ImportError(
                    "polars is required for DataFrame conversion. Install with: pip install polars"
                ) from None

        else:
            raise ValueError(
                f"Unsupported library: {library}. Choose 'pandas' or 'polars'."
            )

    def to_pandas(self, convert_types: bool = True) -> Any:
        """Convert to pandas DataFrame with proper type conversion."""
        return self.to_dataframe("pandas", convert_types=convert_types)

    def to_polars(self, convert_types: bool = True) -> Any:
        """Convert to polars DataFrame with proper type conversion."""
        return self.to_dataframe("polars", convert_types=convert_types)

    def to_geodataframe(self, convert_types: bool = True) -> Any:
        """Convert to GeoPandas GeoDataFrame if geometry column exists."""
        try:
            import geopandas as gpd
            from shapely import wkt
        except ImportError:
            raise ImportError(
                "geopandas and shapely are required for GeoDataFrame conversion. Install with: pip install geopandas shapely"
            ) from None

        df = self.to_pandas(convert_types=convert_types)

        if df.empty:
            # Return empty GeoDataFrame with appropriate schema
            return gpd.GeoDataFrame([], geometry=[], crs="EPSG:4326")

        # Look for geometry column (case-insensitive)
        geometry_col = None
        for col in df.columns:
            if col.lower() in ["geometry", "geom", "shape", "wkt"]:
                geometry_col = col
                break

        if geometry_col is None:
            raise ValueError(
                "No geometry column found. Expected column name containing 'geometry', 'geom', 'shape', or 'wkt'"
            )

        # Convert WKT strings to shapely geometries
        def wkt_to_geom(wkt_str: Any) -> Any:
            if wkt_str is None or wkt_str == "" or str(wkt_str).lower() == "null":
                return None
            try:
                return wkt.loads(str(wkt_str))
            except Exception:
                return None

        df["geometry"] = df[geometry_col].apply(wkt_to_geom)

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

        # Remove invalid geometries
        gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]

        return gdf

    def to_soilprofilecollection(
        self,
        site_data: "pd.DataFrame | None" = None,
        site_id_col: str = "cokey",
        hz_id_col: str = "chkey",
        hz_top_col: str = "hzdept_r",
        hz_bot_col: str = "hzdepb_r",
    ) -> "SoilProfileCollection":
        """
        Converts the response data to a soilprofilecollection.SoilProfileCollection object.

        This method is intended for horizon-level data, which can be joined with
        site-level data (e.g., from the component table) to form a complete
        soil profile collection.

        Args:
            site_data: Optional pandas DataFrame containing site-level data.
                This will be joined with the horizon data.
            site_id_col: The name of the site ID column, used to link site and
                horizon data (default: "cokey").
            hz_id_col: The name of the unique horizon ID column (default: "chkey").
            hz_top_col: The name of the horizon top depth column (default: "hzdept_r").
            hz_bot_col: The name of the horizon bottom depth column (default: "hzdepb_r").

        Returns:
            A SoilProfileCollection object.

        Raises:
            ImportError: If the 'soilprofilecollection' package is not installed.
            ValueError: If required columns for creating the SoilProfileCollection
                are missing from the data.
        """
        try:
            from soilprofilecollection import SoilProfileCollection
        except ImportError:
            raise ImportError(
                "The 'soilprofilecollection' package is required to use "
                "to_soilprofilecollection(). Please install it with: "
                "pip install soildb[soil]"
            ) from None

        horizons_df = self.to_pandas()

        required_cols = [hz_id_col, site_id_col, hz_top_col, hz_bot_col]
        missing_cols = [col for col in required_cols if col not in horizons_df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in horizon data: {', '.join(missing_cols)}"
            )

        return SoilProfileCollection(
            horizons=horizons_df,
            site=site_data,
            idname=site_id_col,
            hzidname=hz_id_col,
            depthcols=(hz_top_col, hz_bot_col),
        )

    def get_column_types(self) -> Dict[str, str]:
        """Extract column data types from metadata."""
        if not self._metadata:
            return {}

        types = {}
        for i, col_name in enumerate(self._columns):
            if i < len(self._metadata):
                metadata_str = self._metadata[i]
                # Parse metadata like "ColumnOrdinal=0,DataTypeName=varchar"
                if "DataTypeName=" in metadata_str:
                    type_part = metadata_str.split("DataTypeName=")[1]
                    data_type = (
                        type_part.split(",")[0] if "," in type_part else type_part
                    )
                    types[col_name] = data_type

        return types

    def get_python_types(self) -> Dict[str, str]:
        """Get Python-compatible type mapping."""
        sda_types = self.get_column_types()
        python_types = {}

        for col_name, sda_type in sda_types.items():
            python_types[col_name] = self.SDA_TYPE_MAPPING.get(
                sda_type.lower(), "string"
            )

        return python_types

    def is_empty(self) -> bool:
        """Check if the response contains no data.

        Returns:
            True if the response contains no rows, False otherwise.
        """
        return len(self._data) == 0

    def __repr__(self) -> str:
        """String representation of the response."""
        return f"SDAResponse(columns={len(self._columns)}, rows={len(self._data)})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.is_empty():
            return "Empty SDA response"

        lines = [f"SDA Response: {len(self._data)} rows, {len(self._columns)} columns"]
        lines.append(
            f"Columns: {', '.join(self._columns[:5])}{'...' if len(self._columns) > 5 else ''}"
        )

        # Show column types
        column_types = self.get_column_types()
        if column_types:
            type_info = [
                f"{col}({column_types.get(col, 'unknown')})"
                for col in self._columns[:3]
            ]
            lines.append(
                f"Types: {', '.join(type_info)}{'...' if len(self._columns) > 3 else ''}"
            )

        # Show first few rows
        if self._data:
            lines.append("Sample data:")
            for i, row in enumerate(self._data[:3]):
                display_row = [str(x) if x is not None else "NULL" for x in row[:3]]
                lines.append(
                    f"  Row {i}: {', '.join(display_row)}{'...' if len(row) > 3 else ''}"
                )

            if len(self._data) > 3:
                lines.append(f"  ... and {len(self._data) - 3} more rows")

        return "\n".join(lines)
