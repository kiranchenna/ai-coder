"""Tests for tools/tech_tools.py (mocked HTTP calls)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock
from tools.tech_tools import (
    get_pypi_version, get_npm_version, check_requirements_versions
)


# ─── get_pypi_version ────────────────────────────────────────────────────────

def test_pypi_version_success():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "info": {
            "name": "fastapi",
            "version": "0.115.0",
            "summary": "FastAPI framework",
            "home_page": "https://fastapi.tiangolo.com",
            "requires_python": ">=3.8",
            "license": "MIT",
        }
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=mock_resp):
        result = get_pypi_version("fastapi")

    assert result["name"] == "fastapi"
    assert result["version"] == "0.115.0"
    assert "error" not in result


def test_pypi_version_not_found():
    import httpx
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    error = httpx.HTTPStatusError("Not found", request=MagicMock(), response=mock_resp)

    with patch("httpx.get", side_effect=error):
        result = get_pypi_version("totally_fake_pkg_xyz")

    assert "error" in result
    assert "not found" in result["error"]


# ─── get_npm_version ─────────────────────────────────────────────────────────

def test_npm_version_success():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "name": "react",
        "version": "18.3.1",
        "description": "React is a JavaScript library",
        "homepage": "https://reactjs.org",
        "license": "MIT",
        "engines": {"node": ">=0.10.0"},
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=mock_resp):
        result = get_npm_version("react")

    assert result["name"] == "react"
    assert result["version"] == "18.3.1"
    assert "error" not in result


def test_npm_version_not_found():
    import httpx
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    error = httpx.HTTPStatusError("Not found", request=MagicMock(), response=mock_resp)

    with patch("httpx.get", side_effect=error):
        result = get_npm_version("totally_fake_npm_package_xyz")

    assert "error" in result


# ─── check_requirements_versions ─────────────────────────────────────────────

def test_check_requirements_parses_packages():
    req_text = "fastapi==0.100.0\nrequests>=2.28.0\n# a comment\n"
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "info": {"name": "pkg", "version": "99.0.0", "summary": "",
                 "home_page": "", "requires_python": "", "license": ""}
    }

    with patch("httpx.get", return_value=mock_resp):
        results = check_requirements_versions(req_text)

    assert len(results) == 2
    packages = [r["package"] for r in results]
    assert "fastapi" in packages
    assert "requests" in packages


def test_check_requirements_detects_outdated():
    req_text = "fastapi==0.50.0\n"
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "info": {"name": "fastapi", "version": "0.115.0", "summary": "",
                 "home_page": "", "requires_python": "", "license": ""}
    }

    with patch("httpx.get", return_value=mock_resp):
        results = check_requirements_versions(req_text)

    assert results[0]["outdated"] is True


def test_check_requirements_up_to_date():
    req_text = "fastapi==0.115.0\n"
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "info": {"name": "fastapi", "version": "0.115.0", "summary": "",
                 "home_page": "", "requires_python": "", "license": ""}
    }

    with patch("httpx.get", return_value=mock_resp):
        results = check_requirements_versions(req_text)

    assert results[0]["outdated"] is False


def test_check_requirements_ignores_comments_and_blanks():
    req_text = "# just a comment\n\n   \nfastapi==0.100.0\n"
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "info": {"name": "fastapi", "version": "0.110.0", "summary": "",
                 "home_page": "", "requires_python": "", "license": ""}
    }

    with patch("httpx.get", return_value=mock_resp):
        results = check_requirements_versions(req_text)

    assert len(results) == 1
