"""Comprehensive unit tests for menagerie_loader.py"""

import os
import tempfile
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock

import pytest

from mujoco_mcp.menagerie_loader import MenagerieLoader


class TestMenagerieLoaderInit:
    """Test MenagerieLoader initialization."""

    def test_default_cache_dir(self):
        """Test that default cache directory is created in temp."""
        loader = MenagerieLoader()
        assert loader.cache_dir.exists()
        assert "mujoco_menagerie" in str(loader.cache_dir)

    def test_custom_cache_dir(self):
        """Test custom cache directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_cache = Path(tmpdir) / "custom_cache"
            loader = MenagerieLoader(cache_dir=str(custom_cache))
            assert loader.cache_dir == custom_cache
            assert loader.cache_dir.exists()

    def test_base_url(self):
        """Test that BASE_URL is correctly set."""
        loader = MenagerieLoader()
        assert loader.BASE_URL == "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main"


class TestDownloadFile:
    """Test file downloading functionality."""

    def test_download_file_success(self):
        """Test successful file download."""
        loader = MenagerieLoader()
        test_content = "<mujoco><worldbody/></mujoco>"

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.getcode.return_value = 200
            mock_response.read.return_value = test_content.encode("utf-8")
            mock_response.__enter__.return_value = mock_response
            mock_response.__exit__.return_value = False
            mock_urlopen.return_value = mock_response

            result = loader.download_file("test_model", "test.xml")

        assert result == test_content

    def test_download_file_from_cache(self):
        """Test that cached files are loaded from cache instead of downloading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = MenagerieLoader(cache_dir=tmpdir)

            # Create cached file
            cache_file = loader.cache_dir / "test_model" / "test.xml"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cached_content = "<cached>content</cached>"
            cache_file.write_text(cached_content)

            # Should return cached content without network request
            with patch("urllib.request.urlopen") as mock_urlopen:
                result = loader.download_file("test_model", "test.xml")
                mock_urlopen.assert_not_called()

            assert result == cached_content

    def test_download_file_http_error(self):
        """Test handling of HTTP errors."""
        loader = MenagerieLoader()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.getcode.return_value = 404
            mock_response.__enter__.return_value = mock_response
            mock_response.__exit__.return_value = False
            mock_urlopen.return_value = mock_response

            with pytest.raises(RuntimeError, match="HTTP error 404"):
                loader.download_file("test_model", "missing.xml")

    def test_download_file_url_error(self):
        """Test handling of URL errors (network failures)."""
        loader = MenagerieLoader()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("Network error")

            with pytest.raises(RuntimeError, match="Failed to download"):
                loader.download_file("test_model", "test.xml")

    def test_download_file_unicode_error(self):
        """Test handling of UTF-8 decode errors."""
        loader = MenagerieLoader()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.getcode.return_value = 200
            mock_response.read.return_value = b"\xff\xfe"  # Invalid UTF-8
            mock_response.__enter__.return_value = mock_response
            mock_response.__exit__.return_value = False
            mock_urlopen.return_value = mock_response

            with pytest.raises(UnicodeDecodeError):
                loader.download_file("test_model", "test.xml")


class TestResolveIncludes:
    """Test XML include resolution."""

    def test_no_includes(self):
        """Test XML without includes is returned unchanged."""
        loader = MenagerieLoader()
        xml = "<mujoco><worldbody/></mujoco>"
        result = loader.resolve_includes(xml, "test_model")

        # Parse both to compare structure
        root1 = ET.fromstring(xml)
        root2 = ET.fromstring(result)
        assert root1.tag == root2.tag

    def test_simple_include(self):
        """Test simple include resolution."""
        loader = MenagerieLoader()
        main_xml = '<mujoco><include file="included.xml"/></mujoco>'
        included_xml = '<mujoco><worldbody><body name="test"/></worldbody></mujoco>'

        with patch.object(loader, "download_file", return_value=included_xml):
            result = loader.resolve_includes(main_xml, "test_model")

        root = ET.fromstring(result)
        assert root.find(".//body[@name='test']") is not None

    def test_circular_include_detection(self):
        """Test that circular includes are detected and avoided."""
        loader = MenagerieLoader()
        # XML that includes itself
        xml = '<mujoco><include file="self.xml"/></mujoco>'

        visited = {"self.xml"}
        result = loader.resolve_includes(xml, "test_model", visited=visited)

        # Should not raise an error, just skip the circular include
        assert result is not None

    def test_invalid_xml(self):
        """Test handling of invalid XML."""
        loader = MenagerieLoader()
        invalid_xml = "<mujoco><unclosed"

        # Should return original content on parse error
        result = loader.resolve_includes(invalid_xml, "test_model")
        assert result == invalid_xml

    def test_include_without_file_attribute(self):
        """Test include elements without file attribute are skipped."""
        loader = MenagerieLoader()
        xml = '<mujoco><include/></mujoco>'

        result = loader.resolve_includes(xml, "test_model")
        # Should not raise error
        assert result is not None

    def test_failed_include_download(self):
        """Test that failed include downloads are gracefully handled."""
        loader = MenagerieLoader()
        xml = '<mujoco><include file="missing.xml"/></mujoco>'

        with patch.object(loader, "download_file", side_effect=RuntimeError("Not found")):
            result = loader.resolve_includes(xml, "test_model")

        # Should not raise, just keep the include as-is
        assert result is not None


class TestGetModelXml:
    """Test get_model_xml functionality."""

    def test_empty_model_name(self):
        """Test that empty model name raises ValueError."""
        loader = MenagerieLoader()

        with pytest.raises(ValueError, match="Model name cannot be empty"):
            loader.get_model_xml("")

    def test_whitespace_model_name(self):
        """Test that whitespace-only model name raises ValueError."""
        loader = MenagerieLoader()

        with pytest.raises(ValueError, match="Model name cannot be empty"):
            loader.get_model_xml("   ")

    def test_successful_load_first_attempt(self):
        """Test successful model load on first attempt."""
        loader = MenagerieLoader()
        test_xml = "<mujoco><worldbody/></mujoco>"

        with patch.object(loader, "download_file", return_value=test_xml):
            with patch.object(loader, "resolve_includes", return_value=test_xml):
                result = loader.get_model_xml("test_model")

        assert result == test_xml

    def test_fallback_to_second_file(self):
        """Test fallback to alternative XML file names."""
        loader = MenagerieLoader()
        test_xml = "<mujoco><worldbody/></mujoco>"

        # First file fails, second succeeds
        def download_side_effect(model_name, file_path):
            if file_path == "test_model.xml":
                raise RuntimeError("Not found")
            return test_xml

        with patch.object(loader, "download_file", side_effect=download_side_effect):
            with patch.object(loader, "resolve_includes", return_value=test_xml):
                result = loader.get_model_xml("test_model")

        assert result == test_xml

    def test_all_files_fail(self):
        """Test that RuntimeError is raised when all file attempts fail."""
        loader = MenagerieLoader()

        with patch.object(loader, "download_file", side_effect=RuntimeError("Not found")):
            with pytest.raises(RuntimeError, match="Could not load any XML files"):
                loader.get_model_xml("nonexistent_model")


class TestGetAvailableModels:
    """Test get_available_models functionality."""

    def test_returns_dict(self):
        """Test that get_available_models returns a dictionary."""
        loader = MenagerieLoader()
        models = loader.get_available_models()
        assert isinstance(models, dict)

    def test_contains_expected_categories(self):
        """Test that returned dict contains expected categories."""
        loader = MenagerieLoader()
        models = loader.get_available_models()

        expected_categories = ["arms", "quadrupeds", "humanoids", "grippers"]
        for category in expected_categories:
            assert category in models

    def test_categories_contain_models(self):
        """Test that each category contains model names."""
        loader = MenagerieLoader()
        models = loader.get_available_models()

        for _category, model_list in models.items():
            assert isinstance(model_list, list)
            assert len(model_list) > 0
            assert all(isinstance(m, str) for m in model_list)


class TestValidateModel:
    """Test model validation functionality."""

    def test_empty_xml_validation_error(self):
        """Test that empty XML content raises ValueError."""
        loader = MenagerieLoader()

        with patch.object(loader, "get_model_xml", return_value="   "):
            with pytest.raises(ValueError, match="empty XML content"):
                loader.validate_model("test_model")

    def test_invalid_xml_parse_error(self):
        """Test that invalid XML raises ParseError."""
        loader = MenagerieLoader()
        invalid_xml = "<mujoco><unclosed"

        with patch.object(loader, "get_model_xml", return_value=invalid_xml):
            with pytest.raises(ET.ParseError):
                loader.validate_model("test_model")

    def test_invalid_root_element(self):
        """Test that non-mujoco root element raises ValueError."""
        loader = MenagerieLoader()
        wrong_root = "<notmujoco><worldbody/></notmujoco>"

        with patch.object(loader, "get_model_xml", return_value=wrong_root):
            with pytest.raises(ValueError, match="root element is 'notmujoco'"):
                loader.validate_model("test_model")

    def test_validation_without_mujoco_installed(self):
        """Test validation when MuJoCo is not installed."""
        loader = MenagerieLoader()
        valid_xml = "<mujoco><worldbody/></mujoco>"

        with patch.object(loader, "get_model_xml", return_value=valid_xml):
            with patch.dict("sys.modules", {"mujoco": None}):
                result = loader.validate_model("test_model")

        assert result["valid"] is True
        assert "xml_size" in result
        assert "note" in result

    def test_validation_with_mujoco_installed(self):
        """Test validation when MuJoCo is installed."""
        loader = MenagerieLoader()
        valid_xml = "<mujoco><worldbody/></mujoco>"

        mock_mujoco = MagicMock()
        mock_model = MagicMock()
        mock_model.nbody = 5
        mock_model.njnt = 7
        mock_model.nu = 6
        mock_mujoco.MjModel.from_xml_path.return_value = mock_model

        with patch.object(loader, "get_model_xml", return_value=valid_xml):
            with patch.dict("sys.modules", {"mujoco": mock_mujoco}):
                with patch("os.unlink"):  # Don't actually delete temp file in test
                    result = loader.validate_model("test_model")

        assert result["valid"] is True
        assert result["n_bodies"] == 5
        assert result["n_joints"] == 7
        assert result["n_actuators"] == 6
        assert "xml_size" in result

    def test_mujoco_validation_failure(self):
        """Test handling of MuJoCo validation failure."""
        loader = MenagerieLoader()
        valid_xml = "<mujoco><worldbody/></mujoco>"

        mock_mujoco = MagicMock()
        mock_mujoco.MjModel.from_xml_path.side_effect = Exception("MuJoCo error")

        with patch.object(loader, "get_model_xml", return_value=valid_xml):
            with patch.dict("sys.modules", {"mujoco": mock_mujoco}):
                with patch("os.unlink"):
                    with pytest.raises(RuntimeError, match="Failed to load MuJoCo model"):
                        loader.validate_model("test_model")


class TestCreateSceneXml:
    """Test scene XML creation."""

    def test_complete_scene_returned_as_is(self):
        """Test that complete scene XML is returned unchanged."""
        loader = MenagerieLoader()
        complete_scene = "<mujoco><worldbody><geom/></worldbody></mujoco>"

        with patch.object(loader, "get_model_xml", return_value=complete_scene):
            result = loader.create_scene_xml("test_model")

        assert result == complete_scene

    def test_incomplete_model_wrapped_in_scene(self):
        """Test that incomplete model is wrapped in scene template."""
        loader = MenagerieLoader()
        model_xml = "<body name='robot'/>"

        with patch.object(loader, "get_model_xml", return_value=model_xml):
            result = loader.create_scene_xml("test_model")

        assert "<mujoco" in result
        assert "<worldbody>" in result
        assert model_xml in result
        assert "test_model_scene" in result

    def test_custom_scene_name(self):
        """Test custom scene name in generated XML."""
        loader = MenagerieLoader()
        model_xml = "<body name='robot'/>"

        with patch.object(loader, "get_model_xml", return_value=model_xml):
            result = loader.create_scene_xml("test_model", scene_name="custom_scene")

        assert "custom_scene" in result
