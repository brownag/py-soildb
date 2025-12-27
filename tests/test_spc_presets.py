"""
Tests for SoilProfileCollection presets and column configurations.
"""

import pytest

from soildb.spc_presets import (
    PRESET_REGISTRY,
    CustomColumnConfig,
    LabPedonHorizonColumns,
    MapunitComponentHorizonColumns,
    PedonSiteHorizonColumns,
    StandardSDAHorizonColumns,
    get_preset,
    list_presets,
)


class TestColumnConfig:
    """Test base ColumnConfig class."""

    def test_column_config_attributes(self):
        """Test ColumnConfig base class has required attributes."""
        config = StandardSDAHorizonColumns()

        assert hasattr(config, "site_id_col")
        assert hasattr(config, "horizon_id_col")
        assert hasattr(config, "horizon_top_col")
        assert hasattr(config, "horizon_bottom_col")
        assert callable(config.get_config_dict)
        assert callable(config.get_required_columns)
        assert callable(config.get_all_columns)

    def test_get_config_dict(self):
        """Test get_config_dict returns proper structure."""
        config = StandardSDAHorizonColumns()
        config_dict = config.get_config_dict()

        assert isinstance(config_dict, dict)
        assert "site_id_col" in config_dict
        assert "hz_id_col" in config_dict
        assert "hz_top_col" in config_dict
        assert "hz_bot_col" in config_dict

    def test_get_required_columns(self):
        """Test get_required_columns returns list of column names."""
        config = StandardSDAHorizonColumns()
        required = config.get_required_columns()

        assert isinstance(required, list)
        assert len(required) == 4
        assert config.site_id_col in required
        assert config.horizon_id_col in required
        assert config.horizon_top_col in required
        assert config.horizon_bottom_col in required

    def test_get_all_columns(self):
        """Test get_all_columns includes all column names."""
        config = StandardSDAHorizonColumns()
        all_cols = config.get_all_columns()

        assert isinstance(all_cols, list)
        assert len(all_cols) >= 4
        # Should include at least the required columns
        required = config.get_required_columns()
        for col in required:
            assert col in all_cols


class TestStandardSDAHorizonColumns:
    """Test StandardSDAHorizonColumns preset."""

    def test_standard_sda_defaults(self):
        """Test StandardSDAHorizonColumns has correct default values."""
        config = StandardSDAHorizonColumns()

        assert config.site_id_col == "cokey"
        assert config.horizon_id_col == "chkey"
        assert config.horizon_top_col == "hzdept_r"
        assert config.horizon_bottom_col == "hzdepb_r"

    def test_standard_sda_config_dict(self):
        """Test StandardSDAHorizonColumns config dict."""
        config = StandardSDAHorizonColumns()
        config_dict = config.get_config_dict()

        assert config_dict["site_id_col"] == "cokey"
        assert config_dict["hz_id_col"] == "chkey"
        assert config_dict["hz_top_col"] == "hzdept_r"
        assert config_dict["hz_bot_col"] == "hzdepb_r"

    def test_standard_sda_required_columns(self):
        """Test StandardSDAHorizonColumns required columns."""
        config = StandardSDAHorizonColumns()
        required = config.get_required_columns()

        assert required == ["cokey", "chkey", "hzdept_r", "hzdepb_r"]


class TestLabPedonHorizonColumns:
    """Test LabPedonHorizonColumns preset."""

    def test_lab_pedon_defaults(self):
        """Test LabPedonHorizonColumns has correct default values."""
        config = LabPedonHorizonColumns()

        assert config.site_id_col == "pedon_id"
        assert config.horizon_id_col == "hzname"
        assert config.horizon_top_col == "hzdept_r"
        assert config.horizon_bottom_col == "hzdepb_r"

    def test_lab_pedon_config_dict(self):
        """Test LabPedonHorizonColumns config dict."""
        config = LabPedonHorizonColumns()
        config_dict = config.get_config_dict()

        assert config_dict["site_id_col"] == "pedon_id"
        assert config_dict["hz_id_col"] == "hzname"
        assert config_dict["hz_top_col"] == "hzdept_r"
        assert config_dict["hz_bot_col"] == "hzdepb_r"


class TestPedonSiteHorizonColumns:
    """Test PedonSiteHorizonColumns preset."""

    def test_pedon_site_defaults(self):
        """Test PedonSiteHorizonColumns has correct default values."""
        config = PedonSiteHorizonColumns()

        assert config.site_id_col == "pedon_id"
        assert config.horizon_id_col == "pedon_key_horizon"
        assert config.horizon_top_col == "hzdept_r"
        assert config.horizon_bottom_col == "hzdepb_r"

    def test_pedon_site_config_dict(self):
        """Test PedonSiteHorizonColumns config dict."""
        config = PedonSiteHorizonColumns()
        config_dict = config.get_config_dict()

        assert config_dict["site_id_col"] == "pedon_id"
        assert config_dict["hz_id_col"] == "pedon_key_horizon"


class TestMapunitComponentHorizonColumns:
    """Test MapunitComponentHorizonColumns preset."""

    def test_mapunit_component_defaults(self):
        """Test MapunitComponentHorizonColumns has correct default values."""
        config = MapunitComponentHorizonColumns()

        assert config.site_id_col == "mukey"
        assert config.horizon_id_col == "cokey"
        assert config.horizon_top_col == "hzdept_r"
        assert config.horizon_bottom_col == "hzdepb_r"

    def test_mapunit_component_has_mukey(self):
        """Test MapunitComponentHorizonColumns includes mukey."""
        config = MapunitComponentHorizonColumns()
        all_cols = config.get_all_columns()

        assert "mukey" in all_cols


class TestCustomColumnConfig:
    """Test CustomColumnConfig for user-defined columns."""

    def test_custom_config_basic(self):
        """Test CustomColumnConfig with basic column names."""
        config = CustomColumnConfig(
            site_id_col="site_id",
            horizon_id_col="horizon_id",
            horizon_top_col="top_depth",
            horizon_bottom_col="bottom_depth",
        )

        assert config.site_id_col == "site_id"
        assert config.horizon_id_col == "horizon_id"
        assert config.horizon_top_col == "top_depth"
        assert config.horizon_bottom_col == "bottom_depth"

    def test_custom_config_dict(self):
        """Test CustomColumnConfig config dict."""
        config = CustomColumnConfig(
            site_id_col="profile_id",
            horizon_id_col="layer_id",
            horizon_top_col="depth_top",
            horizon_bottom_col="depth_bottom",
        )
        config_dict = config.get_config_dict()

        assert config_dict["site_id_col"] == "profile_id"
        assert config_dict["hz_id_col"] == "layer_id"
        assert config_dict["hz_top_col"] == "depth_top"
        assert config_dict["hz_bot_col"] == "depth_bottom"

    def test_custom_config_required_columns(self):
        """Test CustomColumnConfig required columns."""
        config = CustomColumnConfig(
            site_id_col="sid",
            horizon_id_col="hid",
            horizon_top_col="top",
            horizon_bottom_col="bot",
        )
        required = config.get_required_columns()

        assert required == ["sid", "hid", "top", "bot"]


class TestPresetRegistry:
    """Test preset registry and lookup functions."""

    def test_preset_registry_exists(self):
        """Test PRESET_REGISTRY is defined."""
        assert isinstance(PRESET_REGISTRY, dict)
        assert len(PRESET_REGISTRY) > 0

    def test_preset_registry_has_standard(self):
        """Test PRESET_REGISTRY contains standard preset."""
        assert "standard_sda" in PRESET_REGISTRY

    def test_preset_registry_has_lab_pedon(self):
        """Test PRESET_REGISTRY contains lab_pedon preset."""
        assert "lab_pedon" in PRESET_REGISTRY

    def test_preset_registry_has_pedon_site(self):
        """Test PRESET_REGISTRY contains pedon_site preset."""
        assert "pedon_site" in PRESET_REGISTRY

    def test_preset_registry_has_mapunit_component(self):
        """Test PRESET_REGISTRY contains mapunit_component preset."""
        assert "mapunit_component" in PRESET_REGISTRY


class TestGetPreset:
    """Test get_preset function."""

    def test_get_preset_standard_sda(self):
        """Test get_preset returns StandardSDAHorizonColumns."""
        config = get_preset("standard_sda")

        assert isinstance(config, StandardSDAHorizonColumns)
        assert config.site_id_col == "cokey"
        assert config.horizon_id_col == "chkey"

    def test_get_preset_lab_pedon(self):
        """Test get_preset returns LabPedonHorizonColumns."""
        config = get_preset("lab_pedon")

        assert isinstance(config, LabPedonHorizonColumns)
        assert config.site_id_col == "pedon_id"

    def test_get_preset_pedon_site(self):
        """Test get_preset returns PedonSiteHorizonColumns."""
        config = get_preset("pedon_site")

        assert isinstance(config, PedonSiteHorizonColumns)
        assert config.site_id_col == "pedon_id"

    def test_get_preset_mapunit_component(self):
        """Test get_preset returns MapunitComponentHorizonColumns."""
        config = get_preset("mapunit_component")

        assert isinstance(config, MapunitComponentHorizonColumns)
        assert "mukey" in config.get_all_columns()

    def test_get_preset_invalid_name(self):
        """Test get_preset raises ValueError for invalid name."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("invalid_preset_name")

    def test_get_preset_case_sensitive(self):
        """Test get_preset is case-sensitive."""
        with pytest.raises(ValueError):
            get_preset("Standard_SDA")  # Wrong case


class TestListPresets:
    """Test list_presets function."""

    def test_list_presets_returns_dict(self):
        """Test list_presets returns a dictionary."""
        presets = list_presets()

        assert isinstance(presets, dict)
        assert len(presets) > 0

    def test_list_presets_includes_standard_sda(self):
        """Test list_presets includes standard_sda."""
        presets = list_presets()

        assert "standard_sda" in presets

    def test_list_presets_includes_lab_pedon(self):
        """Test list_presets includes lab_pedon."""
        presets = list_presets()

        assert "lab_pedon" in presets

    def test_list_presets_includes_pedon_site(self):
        """Test list_presets includes pedon_site."""
        presets = list_presets()

        assert "pedon_site" in presets

    def test_list_presets_includes_mapunit_component(self):
        """Test list_presets includes mapunit_component."""
        presets = list_presets()

        assert "mapunit_component" in presets

    def test_list_presets_has_descriptions(self):
        """Test list_presets values are descriptions."""
        presets = list_presets()

        for name, description in presets.items():
            assert isinstance(name, str)
            assert isinstance(description, str)
            assert len(description) > 0


class TestPresetIntegration:
    """Integration tests for presets with different data scenarios."""

    def test_preset_columns_are_consistent(self):
        """Test all presets have consistent required columns."""
        presets = [
            StandardSDAHorizonColumns(),
            LabPedonHorizonColumns(),
            PedonSiteHorizonColumns(),
            MapunitComponentHorizonColumns(),
        ]

        for preset in presets:
            required = preset.get_required_columns()
            assert len(required) == 4
            assert all(isinstance(col, str) for col in required)

    def test_custom_config_can_match_preset_values(self):
        """Test CustomColumnConfig can be configured to match preset values."""
        standard = StandardSDAHorizonColumns()
        custom = CustomColumnConfig(
            site_id_col=standard.site_id_col,
            horizon_id_col=standard.horizon_id_col,
            horizon_top_col=standard.horizon_top_col,
            horizon_bottom_col=standard.horizon_bottom_col,
        )

        assert custom.get_required_columns() == standard.get_required_columns()

    def test_get_preset_returns_independent_instances(self):
        """Test get_preset returns independent instances."""
        config1 = get_preset("standard_sda")
        config2 = get_preset("standard_sda")

        # Different instances
        assert config1 is not config2
        # But same values
        assert config1.get_config_dict() == config2.get_config_dict()
