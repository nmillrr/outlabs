"""
Tests for eGFR utility functions.
"""

import pytest
from eGFR.utils import (
    creatinine_mgdl_to_umoll,
    creatinine_umoll_to_mgdl,
    egfr_to_ckd_stage,
    lbs_to_kg,
    kg_to_lbs,
    inches_to_cm,
    cm_to_inches,
)


# ── Creatinine conversions ─────────────────────────────────────────────────

class TestCreatinineConversions:
    """Tests for creatinine mg/dL ↔ µmol/L conversions."""

    def test_mgdl_to_umoll_typical(self):
        # 1.0 mg/dL → 88.4 µmol/L
        assert creatinine_mgdl_to_umoll(1.0) == pytest.approx(88.4)

    def test_mgdl_to_umoll_high(self):
        # 2.5 mg/dL → 221.0 µmol/L
        assert creatinine_mgdl_to_umoll(2.5) == pytest.approx(221.0)

    def test_mgdl_to_umoll_zero(self):
        assert creatinine_mgdl_to_umoll(0.0) == pytest.approx(0.0)

    def test_umoll_to_mgdl_typical(self):
        # 88.4 µmol/L → 1.0 mg/dL
        assert creatinine_umoll_to_mgdl(88.4) == pytest.approx(1.0)

    def test_umoll_to_mgdl_high(self):
        # 221.0 µmol/L → 2.5 mg/dL
        assert creatinine_umoll_to_mgdl(221.0) == pytest.approx(2.5)

    def test_roundtrip_mgdl(self):
        """Converting mg/dL → µmol/L → mg/dL should return the original value."""
        original = 1.3
        assert creatinine_umoll_to_mgdl(creatinine_mgdl_to_umoll(original)) == pytest.approx(original)

    def test_roundtrip_umoll(self):
        """Converting µmol/L → mg/dL → µmol/L should return the original value."""
        original = 115.0
        assert creatinine_mgdl_to_umoll(creatinine_umoll_to_mgdl(original)) == pytest.approx(original)


# ── CKD stage classification ──────────────────────────────────────────────

class TestCKDStageClassification:
    """Tests for eGFR → CKD stage mapping (KDIGO 2012 thresholds)."""

    def test_g1(self):
        assert egfr_to_ckd_stage(120) == "G1"
        assert egfr_to_ckd_stage(90) == "G1"

    def test_g2(self):
        assert egfr_to_ckd_stage(89) == "G2"
        assert egfr_to_ckd_stage(60) == "G2"

    def test_g3a(self):
        assert egfr_to_ckd_stage(59) == "G3a"
        assert egfr_to_ckd_stage(45) == "G3a"

    def test_g3b(self):
        assert egfr_to_ckd_stage(44) == "G3b"
        assert egfr_to_ckd_stage(30) == "G3b"

    def test_g4(self):
        assert egfr_to_ckd_stage(29) == "G4"
        assert egfr_to_ckd_stage(15) == "G4"

    def test_g5(self):
        assert egfr_to_ckd_stage(14) == "G5"
        assert egfr_to_ckd_stage(5) == "G5"
        assert egfr_to_ckd_stage(0) == "G5"

    def test_boundary_90(self):
        """Exactly 90 should be G1."""
        assert egfr_to_ckd_stage(90.0) == "G1"

    def test_boundary_just_below_90(self):
        assert egfr_to_ckd_stage(89.9) == "G2"


# ── Weight conversions ────────────────────────────────────────────────────

class TestWeightConversions:
    """Tests for lbs ↔ kg conversions."""

    def test_lbs_to_kg(self):
        # 154 lbs ≈ 69.85 kg
        assert lbs_to_kg(154) == pytest.approx(69.853, rel=1e-3)

    def test_kg_to_lbs(self):
        # 70 kg ≈ 154.32 lbs
        assert kg_to_lbs(70) == pytest.approx(154.324, rel=1e-3)

    def test_lbs_to_kg_zero(self):
        assert lbs_to_kg(0) == pytest.approx(0.0)

    def test_roundtrip_weight(self):
        original = 180.0
        assert kg_to_lbs(lbs_to_kg(original)) == pytest.approx(original)


# ── Height conversions ────────────────────────────────────────────────────

class TestHeightConversions:
    """Tests for inches ↔ cm conversions."""

    def test_inches_to_cm(self):
        # 70 in = 177.8 cm
        assert inches_to_cm(70) == pytest.approx(177.8)

    def test_cm_to_inches(self):
        # 177.8 cm = 70 in
        assert cm_to_inches(177.8) == pytest.approx(70.0)

    def test_inches_to_cm_zero(self):
        assert inches_to_cm(0) == pytest.approx(0.0)

    def test_roundtrip_height(self):
        original = 65.5
        assert cm_to_inches(inches_to_cm(original)) == pytest.approx(original)
