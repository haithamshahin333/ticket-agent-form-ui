"""Domain model + underwriting reference data for the pipeline.

Functional core: these are plain data shapes and constants. Every calculation that
consumes them lives in ``tools.py`` as pure functions; ``graph.py`` only wires nodes
around them. Decisions are modeled as a ``Literal`` so an illegal decision value is
unrepresentable.

All reference numbers here are synthetic and illustrative — this is a demo, not an
actuarial rating plan, and it deliberately uses no real customer data.
"""

from __future__ import annotations

from typing import Literal, TypedDict

# --- Input --------------------------------------------------------------------

ConstructionType = Literal[
    "frame",
    "joisted_masonry",
    "non_combustible",
    "masonry_non_combustible",
    "fire_resistive",
]


class Application(TypedDict, total=False):
    """A commercial-property underwriting submission (synthetic data only).

    Numeric fields arrive from a web form as numbers or numeric strings; they are
    coerced and range-checked by :func:`agent.tools.validate_application` before any
    other tool reads them. ``business_name`` and ``notes`` are free text (untrusted)
    and are only ever passed to the model inside data-boundary tags.
    """

    business_name: str
    industry: str
    state: str
    years_in_business: float
    annual_revenue: float
    requested_limit: float
    deductible: float
    tiv: float  # total insured value (building + contents), USD
    construction_type: ConstructionType
    sprinklered: bool
    prior_claims_count: float
    prior_claims_incurred: float  # total incurred losses over the lookback, USD
    notes: str


# --- Underwriting reference data (synthetic, illustrative) --------------------

# Hazard class 1 (low) .. 5 (high), keyed by a normalized industry token.
INDUSTRY_HAZARD_CLASS: dict[str, int] = {
    "office": 1,
    "professional_services": 1,
    "retail": 2,
    "warehouse": 2,
    "restaurant": 3,
    "hospitality": 3,
    "manufacturing": 4,
    "auto_service": 4,
    "chemical": 5,
    "woodworking": 5,
}
DEFAULT_HAZARD_CLASS = 3

# Catastrophe-exposure multiplier by US state (wind/quake/flood proxy).
STATE_CAT_FACTOR: dict[str, float] = {
    "FL": 1.60,
    "LA": 1.50,
    "CA": 1.40,
    "TX": 1.35,
    "OK": 1.30,
    "NY": 1.10,
    "NJ": 1.10,
    "IL": 1.00,
    "OH": 0.95,
    "PA": 0.95,
}
DEFAULT_CAT_FACTOR = 1.00
MIN_CAT_FACTOR = 0.90
MAX_CAT_FACTOR = 1.60

# Construction quality multiplier: lower is a better risk.
CONSTRUCTION_FACTOR: dict[str, float] = {
    "frame": 1.35,
    "joisted_masonry": 1.15,
    "non_combustible": 1.00,
    "masonry_non_combustible": 0.90,
    "fire_resistive": 0.80,
}
DEFAULT_CONSTRUCTION: ConstructionType = "joisted_masonry"
MIN_CONSTRUCTION_FACTOR = 0.80
MAX_CONSTRUCTION_FACTOR = 1.35

# Premium model.
BASE_RATE = 0.0035  # premium per $1 of TIV, before factors
POLICY_FEE = 250.0  # flat fee, USD
MIN_PREMIUM = 500.0  # floor, USD
BENCHMARK_RATE_PER_1K_LOW = 3.0  # $/‰ TIV — below this reads as underpriced
BENCHMARK_RATE_PER_1K_HIGH = 6.0  # above this reads as rich

# Appetite thresholds on the 0..100 risk score.
QUOTE_MAX_SCORE = 45.0  # score <= this -> quote
REFER_MAX_SCORE = 70.0  # score <= this -> refer; above -> decline
LARGE_LOSS_INCURRED = 250_000.0  # single prior loss at/above this forces a referral

# Revenue bands (USD annual revenue).
REVENUE_BAND_SMALL_MAX = 2_000_000.0
REVENUE_BAND_MID_MAX = 25_000_000.0


# --- Node result shapes -------------------------------------------------------

RevenueBand = Literal["small", "mid", "large"]
RiskBand = Literal["low", "moderate", "elevated", "high"]
Decision = Literal["decline", "refer", "quote"]


class ValidationResult(TypedDict):
    """Outcome of coercing + range-checking a raw application."""

    ok: bool
    normalized: Application
    errors: list[str]
    warnings: list[str]


class ExposureResult(TypedDict):
    """Derived exposure figures used by scoring and pricing."""

    tiv: float
    coverage_ratio: float  # requested_limit / tiv (0 if tiv <= 0)
    revenue_band: RevenueBand


class RiskFactor(TypedDict):
    """One contributor to the risk score, with its point value."""

    name: str
    detail: str
    points: float


class RiskResult(TypedDict):
    """Aggregate risk score (0..100, higher = riskier) and its factor breakdown."""

    score: float
    band: RiskBand
    factors: list[RiskFactor]


class PricingResult(TypedDict):
    """Indicated premium and how it was built up."""

    base_premium: float
    risk_multiplier: float
    deductible_credit: float
    fees: float
    premium: float
    rate_per_1k_tiv: float
    adequacy: str


class DecisionResult(TypedDict):
    """The underwriting decision, with conditions and plain-language reasons."""

    decision: Decision
    confidence: float  # 0..1
    conditions: list[str]
    reasons: list[str]
