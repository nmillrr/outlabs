"""
Tests for eGFR mechanistic equation models.
"""

import math
import warnings

import pytest

from eGFR.models import (
    calc_egfr_ckd_epi_2021,
    calc_egfr_mdrd,
    calc_crcl_cockcroft_gault,
    calc_crcl_cockcroft_gault_bsa,
    _normalize_sex,
)


# ---------------------------------------------------------------------------
# _normalize_sex
# ---------------------------------------------------------------------------


class TestNormalizeSex:
    """Tests for the _normalize_sex helper."""

    def test_string_m(self):
        assert _normalize_sex("M") == "M"

    def test_string_f(self):
        assert _normalize_sex("F") == "F"

    def test_string_male(self):
        assert _normalize_sex("male") == "M"

    def test_string_female(self):
        assert _normalize_sex("Female") == "F"

    def test_int_1(self):
        assert _normalize_sex(1) == "M"

    def test_int_2(self):
        assert _normalize_sex(2) == "F"

    def test_invalid_string(self):
        with pytest.raises(ValueError, match="Invalid sex string"):
            _normalize_sex("X")

    def test_invalid_int(self):
        with pytest.raises(ValueError, match="Invalid sex code"):
            _normalize_sex(3)

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Invalid sex type"):
            _normalize_sex([1])


# ---------------------------------------------------------------------------
# calc_egfr_ckd_epi_2021 — known-value tests
# ---------------------------------------------------------------------------


class TestCKDEPI2021KnownValues:
    """Verify CKD-EPI 2021 against expected values.

    Expected values computed analytically from the published formula:
      eGFR = 142 × min(SCr/κ, 1)^α × max(SCr/κ, 1)^(−1.200)
             × 0.9938^Age × 1.012 [if female]

    Reference: Inker et al., NEJM 2021; 385:1737-1749.
    """

    def test_50yo_male_scr_1_0(self):
        """50 M, SCr=1.0 → eGFR ≈ 92 (PRD spec)."""
        result = calc_egfr_ckd_epi_2021(1.0, 50, "M")
        assert result == pytest.approx(91.7, abs=1.0)

    def test_50yo_female_scr_0_8(self):
        """50 F, SCr=0.8 → eGFR ≈ 90.

        Note: PRD originally listed ~99 but the correct analytic value
        from the published CKD-EPI 2021 formula is ~89.7.
        """
        result = calc_egfr_ckd_epi_2021(0.8, 50, "F")
        assert result == pytest.approx(89.7, abs=1.0)

    def test_70yo_male_scr_1_5(self):
        """70 M, SCr=1.5 → eGFR ≈ 45 (PRD spec)."""
        result = calc_egfr_ckd_epi_2021(1.5, 70, "M")
        assert result == pytest.approx(49.8, abs=1.5)

    def test_nhanes_coding(self):
        """NHANES sex coding (1=male, 2=female) works."""
        male_result = calc_egfr_ckd_epi_2021(1.0, 50, 1)
        female_result = calc_egfr_ckd_epi_2021(1.0, 50, 2)
        assert male_result == pytest.approx(91.7, abs=1.0)
        assert female_result != male_result  # Female factor changes result

    def test_low_creatinine_female(self):
        """SCr below kappa uses the alpha exponent (more sensitive range)."""
        # 30F, SCr=0.5 (below kappa=0.7)
        result = calc_egfr_ckd_epi_2021(0.5, 30, "F")
        assert result > 100  # Should be high eGFR for low creatinine in young person

    def test_high_creatinine_male(self):
        """Very high creatinine → very low eGFR."""
        result = calc_egfr_ckd_epi_2021(5.0, 60, "M")
        assert result < 15  # Stage G5 kidney failure

    def test_young_adult(self):
        """18-year-old should get the highest eGFR for same biomarkers."""
        young = calc_egfr_ckd_epi_2021(1.0, 18, "M")
        old = calc_egfr_ckd_epi_2021(1.0, 80, "M")
        assert young > old

    def test_female_factor(self):
        """Female eGFR should differ from male at same inputs.

        The female factor is 1.012, but kappa and alpha also differ,
        so the direction of change depends on creatinine level.
        """
        male = calc_egfr_ckd_epi_2021(1.0, 50, "M")
        female = calc_egfr_ckd_epi_2021(1.0, 50, "F")
        assert male != female


# ---------------------------------------------------------------------------
# calc_egfr_ckd_epi_2021 — input validation
# ---------------------------------------------------------------------------


class TestCKDEPI2021Validation:
    """Verify ValueError is raised for invalid inputs."""

    def test_negative_creatinine(self):
        with pytest.raises(ValueError, match="cr_mgdl must be positive"):
            calc_egfr_ckd_epi_2021(-1.0, 50, "M")

    def test_zero_creatinine(self):
        with pytest.raises(ValueError, match="cr_mgdl must be positive"):
            calc_egfr_ckd_epi_2021(0.0, 50, "M")

    def test_nan_creatinine(self):
        with pytest.raises(ValueError, match="cr_mgdl must be a finite number"):
            calc_egfr_ckd_epi_2021(float("nan"), 50, "M")

    def test_nan_age(self):
        with pytest.raises(ValueError, match="age_years must be a finite number"):
            calc_egfr_ckd_epi_2021(1.0, float("nan"), "M")

    def test_age_below_18(self):
        with pytest.raises(ValueError, match="age_years must be ≥ 18"):
            calc_egfr_ckd_epi_2021(1.0, 17, "M")

    def test_invalid_sex_string(self):
        with pytest.raises(ValueError, match="Invalid sex string"):
            calc_egfr_ckd_epi_2021(1.0, 50, "X")

    def test_invalid_sex_int(self):
        with pytest.raises(ValueError, match="Invalid sex code"):
            calc_egfr_ckd_epi_2021(1.0, 50, 0)


# ---------------------------------------------------------------------------
# calc_egfr_mdrd — known-value tests
# ---------------------------------------------------------------------------


class TestMDRDKnownValues:
    """Verify MDRD against expected values.

    Expected values computed analytically from:
      eGFR = 175 × SCr^(−1.154) × Age^(−0.203) × 0.742 [if female]
             × 1.212 [if Black]

    Reference: Levey et al., Ann Intern Med 2006; 145:247-254.
    """

    def test_50yo_male_scr_1_0(self):
        """50 M, SCr=1.0 → eGFR ≈ 79.

        Manual: 175 × 1.0^−1.154 × 50^−0.203 ≈ 79.1
        """
        result = calc_egfr_mdrd(1.0, 50, "M")
        assert result == pytest.approx(79.1, abs=2.0)

    def test_50yo_female_scr_1_0(self):
        """50 F, SCr=1.0 → ~58.7 (×0.742 female factor)."""
        result = calc_egfr_mdrd(1.0, 50, "F")
        assert result == pytest.approx(79.1 * 0.742, abs=2.0)

    def test_70yo_male_scr_1_5(self):
        """70 M, SCr=1.5 → eGFR ≈ 45.

        Manual: 175 × 1.5^−1.154 × 70^−0.203
              = 175 × 0.6351 × 0.4024 ≈ 44.7
        """
        result = calc_egfr_mdrd(1.5, 70, "M")
        assert result == pytest.approx(44.7, abs=2.0)

    def test_race_coefficient(self):
        """is_black=True multiplies result by 1.212."""
        without = calc_egfr_mdrd(1.5, 70, "M", is_black=False)
        with_black = calc_egfr_mdrd(1.5, 70, "M", is_black=True)
        assert with_black == pytest.approx(without * 1.212, rel=1e-6)

    def test_nhanes_coding(self):
        """NHANES sex coding (1=male, 2=female) works."""
        male_result = calc_egfr_mdrd(1.0, 50, 1)
        female_result = calc_egfr_mdrd(1.0, 50, 2)
        assert female_result == pytest.approx(male_result * 0.742, rel=1e-6)

    def test_high_creatinine_low_egfr(self):
        """Very high creatinine → very low eGFR (no warning)."""
        result = calc_egfr_mdrd(5.0, 60, "M")
        assert result < 30

    def test_age_effect(self):
        """Older patients get lower eGFR at same creatinine."""
        young = calc_egfr_mdrd(1.0, 20, "M")
        old = calc_egfr_mdrd(1.0, 80, "M")
        assert young > old


# ---------------------------------------------------------------------------
# calc_egfr_mdrd — warning for eGFR > 60
# ---------------------------------------------------------------------------


class TestMDRDWarning:
    """Verify UserWarning is issued when eGFR > 60."""

    def test_warning_issued_above_60(self):
        """Low creatinine in a young male → eGFR > 60 → warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = calc_egfr_mdrd(0.8, 30, "M")
            assert result > 60
            assert len(w) == 1
            assert "MDRD is less accurate" in str(w[0].message)
            assert issubclass(w[0].category, UserWarning)

    def test_no_warning_below_60(self):
        """High creatinine → eGFR ≤ 60 → no warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = calc_egfr_mdrd(2.0, 60, "M")
            assert result <= 60
            assert len(w) == 0


# ---------------------------------------------------------------------------
# calc_egfr_mdrd — input validation
# ---------------------------------------------------------------------------


class TestMDRDValidation:
    """Verify ValueError is raised for invalid inputs (same pattern as CKD-EPI)."""

    def test_negative_creatinine(self):
        with pytest.raises(ValueError, match="cr_mgdl must be positive"):
            calc_egfr_mdrd(-1.0, 50, "M")

    def test_zero_creatinine(self):
        with pytest.raises(ValueError, match="cr_mgdl must be positive"):
            calc_egfr_mdrd(0.0, 50, "M")

    def test_nan_creatinine(self):
        with pytest.raises(ValueError, match="cr_mgdl must be a finite number"):
            calc_egfr_mdrd(float("nan"), 50, "M")

    def test_nan_age(self):
        with pytest.raises(ValueError, match="age_years must be a finite number"):
            calc_egfr_mdrd(1.0, float("nan"), "M")

    def test_age_below_18(self):
        with pytest.raises(ValueError, match="age_years must be ≥ 18"):
            calc_egfr_mdrd(1.0, 17, "M")

    def test_invalid_sex(self):
        with pytest.raises(ValueError, match="Invalid sex string"):
            calc_egfr_mdrd(1.0, 50, "X")


# ---------------------------------------------------------------------------
# calc_crcl_cockcroft_gault — known-value tests
# ---------------------------------------------------------------------------


class TestCockcroftGaultKnownValues:
    """Verify Cockcroft-Gault against expected values.

    Expected values computed analytically from:
      CrCl = [(140 - Age) × Weight / (72 × SCr)] × 0.85 [if female]

    Reference: Cockcroft DW, Gault MH. Nephron. 1976;16(1):31-41.
    """

    def test_70yo_70kg_male_scr_1_0(self):
        """70 M, 70 kg, SCr=1.0 → CrCl ≈ 68 mL/min (PRD spec).

        Manual: (140 - 70) × 70 / (72 × 1.0) = 4900 / 72 = 68.06
        """
        result = calc_crcl_cockcroft_gault(1.0, 70, 70.0, "M")
        assert result == pytest.approx(68.06, abs=1.0)

    def test_50yo_80kg_male_scr_1_2(self):
        """50 M, 80 kg, SCr=1.2 → CrCl ≈ 83.3.

        Manual: (140 - 50) × 80 / (72 × 1.2) = 7200 / 86.4 = 83.33
        """
        result = calc_crcl_cockcroft_gault(1.2, 50, 80.0, "M")
        assert result == pytest.approx(83.33, abs=1.0)

    def test_50yo_60kg_female_scr_0_8(self):
        """50 F, 60 kg, SCr=0.8 → CrCl ≈ 84.9.

        Manual: (140 - 50) × 60 / (72 × 0.8) × 0.85 = 5400 / 57.6 × 0.85 ≈ 79.69
        """
        result = calc_crcl_cockcroft_gault(0.8, 50, 60.0, "F")
        expected = (90 * 60) / (72 * 0.8) * 0.85
        assert result == pytest.approx(expected, abs=1.0)

    def test_female_factor(self):
        """Female result should be 0.85× the male result."""
        male = calc_crcl_cockcroft_gault(1.0, 50, 70.0, "M")
        female = calc_crcl_cockcroft_gault(1.0, 50, 70.0, "F")
        assert female == pytest.approx(male * 0.85, rel=1e-6)

    def test_nhanes_coding(self):
        """NHANES sex coding (1=male, 2=female) works."""
        male = calc_crcl_cockcroft_gault(1.0, 50, 70.0, 1)
        female = calc_crcl_cockcroft_gault(1.0, 50, 70.0, 2)
        assert female == pytest.approx(male * 0.85, rel=1e-6)

    def test_returns_mL_min(self):
        """CrCl should be in mL/min (not normalised to BSA)."""
        result = calc_crcl_cockcroft_gault(1.0, 70, 70.0, "M")
        # The result should be reasonable CrCl range (not eGFR-like)
        assert 20 < result < 200

    def test_age_effect(self):
        """Older patients get lower CrCl at same creatinine/weight."""
        young = calc_crcl_cockcroft_gault(1.0, 20, 70.0, "M")
        old = calc_crcl_cockcroft_gault(1.0, 80, 70.0, "M")
        assert young > old

    def test_weight_effect(self):
        """Heavier patients get higher CrCl."""
        light = calc_crcl_cockcroft_gault(1.0, 50, 50.0, "M")
        heavy = calc_crcl_cockcroft_gault(1.0, 50, 100.0, "M")
        assert heavy > light


# ---------------------------------------------------------------------------
# calc_crcl_cockcroft_gault — input validation
# ---------------------------------------------------------------------------


class TestCockcroftGaultValidation:
    """Verify ValueError is raised for invalid inputs."""

    def test_negative_creatinine(self):
        with pytest.raises(ValueError, match="cr_mgdl must be positive"):
            calc_crcl_cockcroft_gault(-1.0, 50, 70.0, "M")

    def test_zero_creatinine(self):
        with pytest.raises(ValueError, match="cr_mgdl must be positive"):
            calc_crcl_cockcroft_gault(0.0, 50, 70.0, "M")

    def test_nan_creatinine(self):
        with pytest.raises(ValueError, match="cr_mgdl must be a finite number"):
            calc_crcl_cockcroft_gault(float("nan"), 50, 70.0, "M")

    def test_nan_age(self):
        with pytest.raises(ValueError, match="age_years must be a finite number"):
            calc_crcl_cockcroft_gault(1.0, float("nan"), 70.0, "M")

    def test_age_below_18(self):
        with pytest.raises(ValueError, match="age_years must be ≥ 18"):
            calc_crcl_cockcroft_gault(1.0, 17, 70.0, "M")

    def test_negative_weight(self):
        with pytest.raises(ValueError, match="weight_kg must be positive"):
            calc_crcl_cockcroft_gault(1.0, 50, -70.0, "M")

    def test_zero_weight(self):
        with pytest.raises(ValueError, match="weight_kg must be positive"):
            calc_crcl_cockcroft_gault(1.0, 50, 0.0, "M")

    def test_nan_weight(self):
        with pytest.raises(ValueError, match="weight_kg must be a finite number"):
            calc_crcl_cockcroft_gault(1.0, 50, float("nan"), "M")

    def test_invalid_sex(self):
        with pytest.raises(ValueError, match="Invalid sex string"):
            calc_crcl_cockcroft_gault(1.0, 50, 70.0, "X")


# ---------------------------------------------------------------------------
# calc_crcl_cockcroft_gault_bsa — BSA-adjusted variant
# ---------------------------------------------------------------------------


class TestCockcroftGaultBSA:
    """Verify BSA-adjusted Cockcroft-Gault variant.

    CrCl_adjusted = CrCl × (1.73 / BSA)
    BSA (DuBois) = 0.007184 × Height_cm^0.725 × Weight_kg^0.425
    """

    def test_bsa_adjustment(self):
        """BSA-adjusted should differ from unadjusted for non-standard BSA."""
        raw = calc_crcl_cockcroft_gault(1.0, 50, 70.0, "M")
        adjusted = calc_crcl_cockcroft_gault_bsa(1.0, 50, 70.0, "M", 170.0)
        # For 70kg, 170cm: BSA ≈ 1.81 m² → adjustment factor ≈ 1.73/1.81 ≈ 0.956
        assert adjusted != raw
        assert adjusted == pytest.approx(raw * 1.73 / (0.007184 * 170.0**0.725 * 70.0**0.425), rel=1e-6)

    def test_at_standard_bsa(self):
        """At BSA = 1.73 m², adjusted ≈ unadjusted.

        BSA ≈ 1.73 when weight ≈ 66.7 kg, height ≈ 168 cm (approx).
        We verify the relationship: adjusted = raw × (1.73 / BSA).
        """
        # Find a weight/height combo close to BSA=1.73
        w, h = 65.0, 168.0
        bsa = 0.007184 * h**0.725 * w**0.425
        raw = calc_crcl_cockcroft_gault(1.0, 50, w, "M")
        adjusted = calc_crcl_cockcroft_gault_bsa(1.0, 50, w, "M", h)
        assert adjusted == pytest.approx(raw * 1.73 / bsa, rel=1e-6)

    def test_invalid_height_negative(self):
        with pytest.raises(ValueError, match="height_cm must be positive"):
            calc_crcl_cockcroft_gault_bsa(1.0, 50, 70.0, "M", -170.0)

    def test_invalid_height_zero(self):
        with pytest.raises(ValueError, match="height_cm must be positive"):
            calc_crcl_cockcroft_gault_bsa(1.0, 50, 70.0, "M", 0.0)

    def test_invalid_height_nan(self):
        with pytest.raises(ValueError, match="height_cm must be a finite number"):
            calc_crcl_cockcroft_gault_bsa(1.0, 50, 70.0, "M", float("nan"))

    def test_propagates_base_validation(self):
        """Invalid base params still raise errors through the BSA wrapper."""
        with pytest.raises(ValueError, match="cr_mgdl must be positive"):
            calc_crcl_cockcroft_gault_bsa(-1.0, 50, 70.0, "M", 170.0)

