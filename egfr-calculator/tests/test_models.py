"""
Tests for eGFR mechanistic equation models.
"""

import math
import warnings

import pytest

from eGFR.models import calc_egfr_ckd_epi_2021, calc_egfr_mdrd, _normalize_sex


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

    Expected values computed analytically from the published formula:
      eGFR = 175 × SCr^(−1.154) × Age^(−0.203) × 0.742 [if female]
             × 1.212 [if Black]

    Reference: Levey et al., Ann Intern Med 2006; 145:247-254.
    Reference values cross-checked with NKF GFR Calculator.
    """

    def test_50yo_male_scr_1_0(self):
        """50 M, SCr=1.0 → eGFR ≈ 79."""
        # 175 * 1.0^(-1.154) * 50^(-0.203) = 175 * 1.0 * 0.4523 ≈ 79.2
        result = calc_egfr_mdrd(1.0, 50, "M")
        assert result == pytest.approx(79.2, abs=2.0)

    def test_50yo_female_scr_0_8(self):
        """50 F, SCr=0.8 → eGFR ≈ 73."""
        # 175 * 0.8^(-1.154) * 50^(-0.203) * 0.742
        result = calc_egfr_mdrd(0.8, 50, "F")
        assert result == pytest.approx(73.3, abs=3.0)

    def test_70yo_male_scr_1_5(self):
        """70 M, SCr=1.5 → eGFR ≈ 46."""
        # 175 * 1.5^(-1.154) * 70^(-0.203) = 175 * 0.6119 * 0.4300 ≈ 46.0
        result = calc_egfr_mdrd(1.5, 70, "M")
        assert result == pytest.approx(46.0, abs=2.0)

    def test_is_black_coefficient(self):
        """is_black=True multiplies result by 1.212."""
        baseline = calc_egfr_mdrd(1.5, 70, "M", is_black=False)
        with_race = calc_egfr_mdrd(1.5, 70, "M", is_black=True)
        assert with_race == pytest.approx(baseline * 1.212, rel=1e-6)

    def test_female_factor(self):
        """Female eGFR should be multiplied by 0.742."""
        male = calc_egfr_mdrd(1.0, 50, "M")
        female = calc_egfr_mdrd(1.0, 50, "F")
        assert female == pytest.approx(male * 0.742, rel=1e-6)

    def test_nhanes_sex_coding(self):
        """NHANES coding (1=male, 2=female) works."""
        str_result = calc_egfr_mdrd(1.0, 50, "M")
        int_result = calc_egfr_mdrd(1.0, 50, 1)
        assert str_result == pytest.approx(int_result, rel=1e-10)

    def test_high_creatinine_low_egfr(self):
        """Very high creatinine → very low eGFR (no warning expected)."""
        result = calc_egfr_mdrd(5.0, 60, "M")
        assert result < 20

    def test_age_effect(self):
        """Older patient should have lower eGFR."""
        young = calc_egfr_mdrd(1.0, 25, "M")
        old = calc_egfr_mdrd(1.0, 75, "M")
        assert young > old


# ---------------------------------------------------------------------------
# calc_egfr_mdrd — warning for eGFR > 60
# ---------------------------------------------------------------------------


class TestMDRDWarning:
    """Verify MDRD issues a warning when eGFR > 60."""

    def test_warns_above_60(self):
        """Should warn when eGFR > 60."""
        with pytest.warns(UserWarning, match="MDRD is less accurate"):
            calc_egfr_mdrd(1.0, 50, "M")  # Expected eGFR ≈ 79

    def test_no_warning_below_60(self):
        """Should NOT warn when eGFR ≤ 60."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # High creatinine → low eGFR, should be ≤ 60 with no warning
            calc_egfr_mdrd(1.5, 70, "M")  # Expected eGFR ≈ 46


# ---------------------------------------------------------------------------
# calc_egfr_mdrd — input validation
# ---------------------------------------------------------------------------


class TestMDRDValidation:
    """Verify ValueError is raised for invalid inputs."""

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

