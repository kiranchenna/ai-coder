"""Tests for core/model_catalog.py — the curated /model recommendations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_catalog import RECOMMENDED_MODELS, TIER_LABELS, TIER_ORDER


def test_every_entry_has_a_valid_tier():
    for spec in RECOMMENDED_MODELS:
        assert spec.tier in TIER_ORDER, spec.tag


def test_every_entry_has_a_nonempty_tag_and_note():
    for spec in RECOMMENDED_MODELS:
        assert spec.tag.strip(), "empty tag in catalog"
        assert spec.note.strip(), f"{spec.tag} has no recommendation note"
        assert spec.size_gb > 0, f"{spec.tag} has a non-positive size"


def test_tags_are_unique():
    tags = [spec.tag for spec in RECOMMENDED_MODELS]
    assert len(tags) == len(set(tags)), "duplicate tag in the catalog"


def test_every_tier_has_at_least_one_recommendation():
    tiers_present = {spec.tier for spec in RECOMMENDED_MODELS}
    assert tiers_present == set(TIER_ORDER)


def test_tier_labels_cover_every_tier_order_entry():
    assert set(TIER_LABELS) == set(TIER_ORDER)


def test_current_default_model_is_in_the_catalog():
    # The shipped default should be a recognized recommendation, not an orphan
    # value nothing else in the catalog agrees is reasonable.
    tags = [spec.tag for spec in RECOMMENDED_MODELS]
    assert "qwen2.5-coder:7b" in tags
