"""Deterministic underwriting tools — the functional core of the pipeline.

Every function here is pure: same input in, same output out, no I/O, no globals mutated,
no LLM calls. Inputs and outputs are plain JSON-serializable dicts so they thread cleanly
through LangGraph state and are trivial to unit-test without a model or network.

The graph nodes in ``graph.py`` call these to do the real work, then ask the model only
for narrative commentary on top of the numbers computed here.
"""

from __future__ import annotations

from agent.domain import (
    BASE_RATE,
    BENCHMARK_RATE_PER_1K_HIGH,
    BENCHMARK_RATE_PER_1K_LOW,
    CONSTRUCTION_FACTOR,
    DEFAULT_CAT_FACTOR,
    DEFAULT_CONSTRUCTION,
    DEFAULT_HAZARD_CLASS,
    INDUSTRY_HAZARD_CLASS,
    LARGE_LOSS_INCURRED,
    MIN_PREMIUM,
    POLICY_FEE,
    QUOTE_MAX_SCORE,
    REFER_MAX_SCORE,
    REVENUE_BAND_MID_MAX,
    REVENUE_BAND_SMALL_MAX,
    STATE_CAT_FACTOR,
    Application,
    DecisionResult,
    ExposureResult,
    PricingResult,
    RiskFactor,
    RiskResult,
    ValidationResult,
)

# --- small numeric helpers ----------------------------------------------------


def _num(value: object, default: float = 0.0) -> float:
    """Coerce a form value (number, numeric string, or junk) to a float.

    Never raises: unparseable input falls back to ``default``. This is the single
    choke point that turns untrusted form data into numbers the rest of the module
    can rely on.
    """
    if isinstance(value, bool):  # bool is an int subclass; treat as 0/1 intentionally
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("$", "")
        try:
            return float(cleaned)
        except ValueError:
            return default
    return default


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` into the inclusive ``[lo, hi]`` range."""
    return max(lo, min(hi, value))


def _normalize_industry(industry: object) -> str:
    """Lower-case + underscore an industry label so it can key the hazard table."""
    return str(industry or "").strip().lower().replace(" ", "_").replace("-", "_")


def _round2(value: float) -> float:
    """Round to cents; keeps JSON output tidy and comparisons stable."""
    return round(value, 2)


# --- 1. intake: validate + derive exposure ------------------------------------


def validate_application(app: Application) -> ValidationResult:
    """Coerce and range-check a raw application into a normalized one.

    Hard errors (empty required fields, non-positive TIV/revenue) set ``ok=False``;
    softer issues (unknown construction type, missing state) become warnings with a
    safe default applied. The returned ``normalized`` dict is what every downstream
    tool reads — callers should not re-read the raw ``app``.
    """
    errors: list[str] = []
    warnings: list[str] = []

    business_name = str(app.get("business_name") or "").strip()
    if not business_name:
        errors.append("business_name is required.")

    industry = _normalize_industry(app.get("industry"))
    if not industry:
        warnings.append("industry missing; defaulting hazard class to average.")

    state = str(app.get("state") or "").strip().upper()
    if len(state) != 2:
        warnings.append("state should be a 2-letter code; using average CAT exposure.")
        state = ""

    tiv = _num(app.get("tiv"))
    if tiv <= 0:
        errors.append("tiv (total insured value) must be greater than 0.")

    annual_revenue = _num(app.get("annual_revenue"))
    if annual_revenue <= 0:
        errors.append("annual_revenue must be greater than 0.")

    requested_limit = _num(app.get("requested_limit"))
    if requested_limit <= 0:
        warnings.append("requested_limit missing; defaulting to TIV.")
        requested_limit = tiv

    construction = str(app.get("construction_type") or "").strip().lower()
    if construction not in CONSTRUCTION_FACTOR:
        if construction:
            warnings.append(
                f"unknown construction_type '{construction}'; using {DEFAULT_CONSTRUCTION}."
            )
        construction = DEFAULT_CONSTRUCTION

    normalized: Application = {
        "business_name": business_name,
        "industry": industry,
        "state": state,
        "years_in_business": max(0.0, _num(app.get("years_in_business"))),
        "annual_revenue": max(0.0, annual_revenue),
        "requested_limit": max(0.0, requested_limit),
        "deductible": max(0.0, _num(app.get("deductible"))),
        "tiv": max(0.0, tiv),
        "construction_type": construction,  # type: ignore[typeddict-item]
        "sprinklered": bool(app.get("sprinklered", False)),
        "prior_claims_count": max(0.0, _num(app.get("prior_claims_count"))),
        "prior_claims_incurred": max(0.0, _num(app.get("prior_claims_incurred"))),
        "notes": str(app.get("notes") or "").strip(),
    }

    return {
        "ok": not errors,
        "normalized": normalized,
        "errors": errors,
        "warnings": warnings,
    }


def compute_exposure(app: Application) -> ExposureResult:
    """Derive exposure figures from a *normalized* application.

    Guards against divide-by-zero on TIV (returns a coverage ratio of 0 rather than
    raising). Assumes ``app`` came from :func:`validate_application`.
    """
    tiv = _num(app.get("tiv"))
    requested_limit = _num(app.get("requested_limit"))
    coverage_ratio = _round2(requested_limit / tiv) if tiv > 0 else 0.0

    revenue = _num(app.get("annual_revenue"))
    if revenue <= REVENUE_BAND_SMALL_MAX:
        band = "small"
    elif revenue <= REVENUE_BAND_MID_MAX:
        band = "mid"
    else:
        band = "large"

    return {"tiv": _round2(tiv), "coverage_ratio": coverage_ratio, "revenue_band": band}


# --- 2. risk: score the submission --------------------------------------------


def score_risk(app: Application, exposure: ExposureResult) -> RiskResult:
    """Score risk 0..100 (higher = riskier) with a transparent factor breakdown.

    Each factor contributes bounded points; the total is clamped to 0..100. Pure and
    deterministic so the score can be asserted in tests and audited in a trace.
    """
    factors: list[RiskFactor] = []

    # Industry hazard class -> 0..25 points.
    industry = _normalize_industry(app.get("industry"))
    hazard = INDUSTRY_HAZARD_CLASS.get(industry, DEFAULT_HAZARD_CLASS)
    hazard_points = _round2((hazard - 1) / 4 * 25)
    factors.append(
        {
            "name": "Industry hazard",
            "detail": f"{industry or 'unknown'} — class {hazard}/5",
            "points": hazard_points,
        }
    )

    # Loss history -> 0..35 points, from incurred-vs-TIV plus claim frequency.
    tiv = _num(app.get("tiv"))
    incurred = _num(app.get("prior_claims_incurred"))
    count = _num(app.get("prior_claims_count"))
    loss_ratio = incurred / tiv if tiv > 0 else 0.0
    loss_points = _clamp(loss_ratio * 200, 0, 25) + _clamp(count * 2, 0, 10)
    factors.append(
        {
            "name": "Loss history",
            "detail": f"{int(count)} claim(s), ${incurred:,.0f} incurred "
            f"({loss_ratio * 100:.1f}% of TIV)",
            "points": _round2(loss_points),
        }
    )

    # Construction quality -> 0..15 points.
    construction = str(app.get("construction_type") or DEFAULT_CONSTRUCTION).lower()
    cfactor = CONSTRUCTION_FACTOR.get(construction, CONSTRUCTION_FACTOR[DEFAULT_CONSTRUCTION])
    construction_points = _round2((cfactor - 0.80) / (1.35 - 0.80) * 15)
    factors.append(
        {
            "name": "Construction",
            "detail": f"{construction} (factor {cfactor})",
            "points": construction_points,
        }
    )

    # Catastrophe / state exposure -> 0..15 points.
    state = str(app.get("state") or "").upper()
    cat = STATE_CAT_FACTOR.get(state, DEFAULT_CAT_FACTOR)
    cat_points = _round2(_clamp((cat - 0.90) / (1.60 - 0.90) * 15, 0, 15))
    factors.append(
        {
            "name": "CAT exposure",
            "detail": f"{state or 'n/a'} (factor {cat})",
            "points": cat_points,
        }
    )

    # Protection: unsprinklered risks carry a flat load.
    sprinklered = bool(app.get("sprinklered", False))
    protection_points = 0.0 if sprinklered else 10.0
    factors.append(
        {
            "name": "Fire protection",
            "detail": "sprinklered" if sprinklered else "no sprinkler system",
            "points": protection_points,
        }
    )

    score = _round2(_clamp(sum(f["points"] for f in factors), 0, 100))
    if score < 30:
        band = "low"
    elif score < 50:
        band = "moderate"
    elif score < 70:
        band = "elevated"
    else:
        band = "high"

    return {"score": score, "band": band, "factors": factors}


# --- 3. pricing: build up the indicated premium -------------------------------


def price_policy(
    app: Application, exposure: ExposureResult, risk: RiskResult
) -> PricingResult:
    """Build up an indicated premium from TIV, the risk score, and a deductible credit.

    ``premium = max(base_premium * risk_multiplier * (1 - deductible_credit) + fee,
    floor)``. All divisions are TIV-guarded. Pure and deterministic.
    """
    tiv = max(0.0, _num(app.get("tiv")))
    base_premium = tiv * BASE_RATE

    score = _clamp(_num(risk.get("score")), 0, 100)
    risk_multiplier = _round2(0.80 + (score / 100) * 1.70)  # 0.80 .. 2.50

    # Higher deductibles earn a small credit, capped at 10%.
    deductible = max(0.0, _num(app.get("deductible")))
    deductible_credit = _round2(_clamp(deductible / tiv, 0, 0.10)) if tiv > 0 else 0.0

    premium = base_premium * risk_multiplier * (1 - deductible_credit) + POLICY_FEE
    premium = _round2(max(premium, MIN_PREMIUM))

    rate_per_1k = _round2(premium / (tiv / 1000)) if tiv > 0 else 0.0
    if rate_per_1k < BENCHMARK_RATE_PER_1K_LOW:
        adequacy = "below benchmark (possibly underpriced)"
    elif rate_per_1k <= BENCHMARK_RATE_PER_1K_HIGH:
        adequacy = "within benchmark range"
    else:
        adequacy = "above benchmark (rich)"

    return {
        "base_premium": _round2(base_premium),
        "risk_multiplier": risk_multiplier,
        "deductible_credit": deductible_credit,
        "fees": POLICY_FEE,
        "premium": premium,
        "rate_per_1k_tiv": rate_per_1k,
        "adequacy": adequacy,
    }


# --- 4. decision: quote / refer / decline -------------------------------------


def decide(app: Application, risk: RiskResult, pricing: PricingResult) -> DecisionResult:
    """Turn the risk score and appetite thresholds into a decision + conditions.

    A single large prior loss forces at least a referral regardless of score. Confidence
    is the normalized distance from the nearest appetite threshold, so borderline
    submissions report lower confidence.
    """
    score = _clamp(_num(risk.get("score")), 0, 100)
    incurred = max(0.0, _num(app.get("prior_claims_incurred")))
    large_loss = incurred >= LARGE_LOSS_INCURRED

    reasons: list[str] = []
    conditions: list[str] = []

    if score > REFER_MAX_SCORE:
        decision = "decline"
        reasons.append(f"Risk score {score:.0f} exceeds the referral ceiling {REFER_MAX_SCORE:.0f}.")
    elif score <= QUOTE_MAX_SCORE and not large_loss:
        decision = "quote"
        reasons.append(f"Risk score {score:.0f} is within appetite (<= {QUOTE_MAX_SCORE:.0f}).")
    else:
        decision = "refer"
        if large_loss:
            reasons.append(
                f"Prior incurred losses ${incurred:,.0f} at/above the "
                f"${LARGE_LOSS_INCURRED:,.0f} referral trigger."
            )
        else:
            reasons.append(
                f"Risk score {score:.0f} is in the referral band "
                f"({QUOTE_MAX_SCORE:.0f}–{REFER_MAX_SCORE:.0f})."
            )

    # Quote conditions.
    if decision == "quote":
        if not bool(app.get("sprinklered", False)):
            conditions.append("Install a monitored sprinkler system within 90 days of binding.")
        if _num(app.get("years_in_business")) < 3:
            conditions.append("Provide 3 years of prior-carrier loss runs before renewal.")
        if "above benchmark" not in pricing.get("adequacy", ""):
            reasons.append(f"Indicated rate is {pricing.get('adequacy', 'n/a')}.")

    # Confidence: distance from the nearest threshold, normalized to the band width.
    if decision == "quote":
        margin = (QUOTE_MAX_SCORE - score) / QUOTE_MAX_SCORE
    elif decision == "decline":
        margin = (score - REFER_MAX_SCORE) / (100 - REFER_MAX_SCORE)
    else:  # refer
        span = REFER_MAX_SCORE - QUOTE_MAX_SCORE
        midpoint = (QUOTE_MAX_SCORE + REFER_MAX_SCORE) / 2
        margin = 1 - abs(score - midpoint) / (span / 2) if span > 0 else 0.0
        if large_loss:
            margin = max(margin, 0.6)
    confidence = _round2(_clamp(0.5 + margin / 2, 0.5, 0.99))

    return {
        "decision": decision,  # type: ignore[typeddict-item]
        "confidence": confidence,
        "conditions": conditions,
        "reasons": reasons,
    }
