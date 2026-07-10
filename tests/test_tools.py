"""Unit tests for the pure underwriting tools.

These exercise the functional core with no model and no network, so the whole pipeline's
decision logic is asserted deterministically (and cheaply). The graph nodes are thin
wrappers over these, so getting these right is most of the correctness story.
"""

from __future__ import annotations

from agent.domain import LARGE_LOSS_INCURRED
from agent.tools import (
    compute_exposure,
    decide,
    price_policy,
    score_risk,
    validate_application,
)

# --- fixtures as plain dicts --------------------------------------------------

CLEAN_LOW_RISK = {
    "business_name": "Quiet Ledgers LLC",
    "industry": "office",
    "state": "PA",
    "years_in_business": 12,
    "annual_revenue": 1_200_000,
    "requested_limit": 500_000,
    "deductible": 25_000,
    "tiv": 500_000,
    "construction_type": "fire_resistive",
    "sprinklered": True,
    "prior_claims_count": 0,
    "prior_claims_incurred": 0,
}

HEAVY_LOSS_HIGH_RISK = {
    "business_name": "Blaze Chemicals Inc",
    "industry": "chemical",
    "state": "FL",
    "years_in_business": 4,
    "annual_revenue": 8_000_000,
    "requested_limit": 1_000_000,
    "deductible": 10_000,
    "tiv": 1_000_000,
    "construction_type": "frame",
    "sprinklered": False,
    "prior_claims_count": 6,
    "prior_claims_incurred": 400_000,
}


def _pipeline(app: dict):
    """Run the four pure stages in order and return their results."""
    normalized = validate_application(app)["normalized"]
    exposure = compute_exposure(normalized)
    risk = score_risk(normalized, exposure)
    pricing = price_policy(normalized, exposure, risk)
    decision = decide(normalized, risk, pricing)
    return normalized, exposure, risk, pricing, decision


# --- validation ---------------------------------------------------------------


def test_validate_coerces_numeric_strings_and_normalizes():
    result = validate_application(
        {
            "business_name": "  Café Corner  ",
            "industry": "Auto Service",
            "state": "ca",
            "tiv": "$1,250,000",
            "annual_revenue": "3000000",
            "requested_limit": "1000000",
        }
    )
    assert result["ok"] is True
    norm = result["normalized"]
    assert norm["business_name"] == "Café Corner"
    assert norm["industry"] == "auto_service"
    assert norm["state"] == "CA"
    assert norm["tiv"] == 1_250_000.0
    assert norm["annual_revenue"] == 3_000_000.0


def test_validate_flags_hard_errors():
    result = validate_application({"business_name": "", "tiv": 0, "annual_revenue": -5})
    assert result["ok"] is False
    assert any("business_name" in e for e in result["errors"])
    assert any("tiv" in e for e in result["errors"])
    assert any("annual_revenue" in e for e in result["errors"])


def test_validate_unknown_construction_warns_and_defaults():
    result = validate_application(
        {"business_name": "X", "tiv": 100, "annual_revenue": 100, "construction_type": "adobe"}
    )
    assert result["normalized"]["construction_type"] == "joisted_masonry"
    assert any("construction_type" in w for w in result["warnings"])


# --- exposure -----------------------------------------------------------------


def test_exposure_coverage_ratio_and_band():
    exposure = compute_exposure(
        {"tiv": 1_000_000, "requested_limit": 750_000, "annual_revenue": 1_000_000}
    )
    assert exposure["coverage_ratio"] == 0.75
    assert exposure["revenue_band"] == "small"


def test_exposure_divide_by_zero_guarded():
    exposure = compute_exposure({"tiv": 0, "requested_limit": 500_000, "annual_revenue": 0})
    assert exposure["coverage_ratio"] == 0.0  # no ZeroDivisionError


def test_exposure_revenue_bands():
    assert compute_exposure({"tiv": 1, "annual_revenue": 5_000_000})["revenue_band"] == "mid"
    assert compute_exposure({"tiv": 1, "annual_revenue": 50_000_000})["revenue_band"] == "large"


# --- scoring ------------------------------------------------------------------


def test_score_bounds_and_bands():
    _, exposure, risk, _, _ = _pipeline(CLEAN_LOW_RISK)
    assert 0 <= risk["score"] <= 100
    assert risk["band"] == "low"
    assert {f["name"] for f in risk["factors"]} == {
        "Industry hazard",
        "Loss history",
        "Construction",
        "CAT exposure",
        "Fire protection",
    }


def test_high_risk_scores_high():
    _, _, risk, _, _ = _pipeline(HEAVY_LOSS_HIGH_RISK)
    assert risk["score"] >= 70
    assert risk["band"] == "high"


def test_unsprinklered_adds_flat_load():
    sprinklered = dict(CLEAN_LOW_RISK, sprinklered=True)
    unsprinklered = dict(CLEAN_LOW_RISK, sprinklered=False)
    with_sprinkler = score_risk(validate_application(sprinklered)["normalized"], {})["score"]
    without_sprinkler = score_risk(validate_application(unsprinklered)["normalized"], {})["score"]
    assert round(without_sprinkler - with_sprinkler, 2) == 10.0


# --- pricing ------------------------------------------------------------------


def test_pricing_floor_and_rate_guard():
    pricing = price_policy({"tiv": 0, "deductible": 0}, {}, {"score": 10})
    assert pricing["premium"] >= 500.0  # MIN_PREMIUM floor
    assert pricing["rate_per_1k_tiv"] == 0.0  # tiv==0 guarded


def test_pricing_multiplier_rises_with_score():
    low = price_policy({"tiv": 1_000_000, "deductible": 0}, {}, {"score": 10})
    high = price_policy({"tiv": 1_000_000, "deductible": 0}, {}, {"score": 90})
    assert high["risk_multiplier"] > low["risk_multiplier"]
    assert high["premium"] > low["premium"]


# --- decision -----------------------------------------------------------------


def test_clean_low_risk_quotes():
    _, _, risk, pricing, decision = _pipeline(CLEAN_LOW_RISK)
    assert decision["decision"] == "quote"
    assert 0.5 <= decision["confidence"] <= 0.99


def test_heavy_loss_high_risk_declines():
    _, _, _, _, decision = _pipeline(HEAVY_LOSS_HIGH_RISK)
    assert decision["decision"] == "decline"


def test_large_single_loss_forces_referral_even_when_score_low():
    app = dict(CLEAN_LOW_RISK, prior_claims_incurred=LARGE_LOSS_INCURRED + 1)
    normalized = validate_application(app)["normalized"]
    risk = score_risk(normalized, {})
    pricing = price_policy(normalized, {}, risk)
    decision = decide(normalized, risk, pricing)
    assert decision["decision"] == "refer"
    assert any("referral trigger" in r for r in decision["reasons"])


def test_quote_requires_sprinkler_condition_when_unsprinklered():
    # Low-hazard but unsprinklered: still within appetite, quoted with a condition.
    app = dict(CLEAN_LOW_RISK, sprinklered=False)
    _, _, _, _, decision = _pipeline(app)
    assert decision["decision"] == "quote"
    assert any("sprinkler" in c.lower() for c in decision["conditions"])
