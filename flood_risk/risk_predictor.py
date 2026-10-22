
"""
Advanced Flood Risk Predictor
Project: AI-Based Early Flood Risk Prediction and
         Risk-Aware Dynamic Evacuation Planning

Version: 2.0
Method: Weighted rule-based risk assessment
"""

import math

# =====================================================
# 1. LAND-COVER RISK WEIGHTS
# =====================================================

LAND_COVER_WEIGHTS = {
    "Agriculture": 0.65,
    "Barren": 0.35,
    "Roads": 0.75,
    "Urban": 0.85,
    "Vegetation": 0.45,
    "Water": 0.90,
}

# =====================================================
# 2. CONFIGURATION
# These are prototype assumptions, not calibrated values.
# =====================================================

RISK_WEIGHTS = {
    "land_cover": 0.25,
    "rainfall": 0.30,
    "water_proximity": 0.20,
    "elevation": 0.15,
    "drainage": 0.10,
}

MAX_RAINFALL_MM = 300.0
MAX_WATER_DISTANCE_M = 1000.0
ELEVATION_REFERENCE_M = 100.0

RISK_THRESHOLDS = {
    "LOW": 0.35,
    "MODERATE": 0.50,
    "HIGH": 0.70,
    "CRITICAL": 0.85,
}


# =====================================================
# 3. INPUT VALIDATION
# =====================================================

def validate_number(name, value):
    """Ensure a numeric input is finite and valid."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number.")

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    return float(value)


# =====================================================
# 4. RISK LEVEL CLASSIFICATION
# =====================================================

def classify_risk(score):
    """Convert a normalized score into a risk category."""

    if score >= RISK_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    elif score >= RISK_THRESHOLDS["HIGH"]:
        return "HIGH"
    elif score >= RISK_THRESHOLDS["MODERATE"]:
        return "MODERATE"
    else:
        return "LOW"


# =====================================================
# 5. DETAILED FLOOD RISK ASSESSMENT
# =====================================================

def calculate_flood_risk_detailed(
    land_cover,
    rainfall_mm,
    water_distance_m,
    elevation_m,
    drainage_score,
):
    """
    Calculate an experimental flood-susceptibility score.

    Assumptions:
      rainfall_mm: rainfall accumulation in mm over a specified period
      water_distance_m: distance to the nearest water body in metres
      elevation_m: relative elevation in metres; higher means higher ground
      drainage_score: drainage quality from 0 (poor) to 10 (excellent)

    Returns a dictionary containing score, level, and factor contributions.

    IMPORTANT:
    The scaling assumptions and weights must be calibrated against
    observed flood data before practical deployment.
    """

    # Validate class
    if land_cover not in LAND_COVER_WEIGHTS:
        raise ValueError(
            f"Unknown land cover '{land_cover}'. "
            f"Valid classes: {', '.join(LAND_COVER_WEIGHTS)}"
        )

    # Validate numeric inputs
    rainfall_mm = validate_number("Rainfall", rainfall_mm)
    water_distance_m = validate_number(
        "Water distance", water_distance_m
    )
    elevation_m = validate_number("Elevation", elevation_m)
    drainage_score = validate_number(
        "Drainage score", drainage_score
    )

    if rainfall_mm < 0:
        raise ValueError("Rainfall cannot be negative.")

    if water_distance_m < 0:
        raise ValueError("Water distance cannot be negative.")

    if not 0 <= drainage_score <= 10:
        raise ValueError("Drainage score must be between 0 and 10.")

    # -------------------------------------------------
    # Normalize the input factors to the range [0, 1]
    # -------------------------------------------------

    land_cover_score = LAND_COVER_WEIGHTS[land_cover]

    rainfall_score = min(
        rainfall_mm / MAX_RAINFALL_MM, 1.0
    )

    water_proximity_score = max(
        0.0,
        1.0 - water_distance_m / MAX_WATER_DISTANCE_M,
    )

    # Higher relative elevation reduces this prototype score.
    # Negative relative elevations are capped at maximum susceptibility.
    elevation_score = max(
        0.0,
        min(1.0, 1.0 - elevation_m / ELEVATION_REFERENCE_M),
    )

    drainage_risk_score = 1.0 - drainage_score / 10.0

    normalized_factors = {
        "land_cover": land_cover_score,
        "rainfall": rainfall_score,
        "water_proximity": water_proximity_score,
        "elevation": elevation_score,
        "drainage": drainage_risk_score,
    }

    # -------------------------------------------------
    # Weighted risk calculation
    # -------------------------------------------------

    contributions = {
        factor: normalized_factors[factor] * weight
        for factor, weight in RISK_WEIGHTS.items()
    }

    risk_score = sum(contributions.values())
    risk_score = max(0.0, min(1.0, risk_score))

    risk_level = classify_risk(risk_score)

    # -------------------------------------------------
    # Human-readable explanation
    # -------------------------------------------------

    explanations = {
        "land_cover": (
            f"{land_cover} land-cover class; "
            "the assigned class weight is a prototype assumption."
        ),
        "rainfall": (
            "Higher rainfall increases the score; rainfall duration "
            "and local intensity thresholds are not yet calibrated."
        ),
        "water_proximity": (
            "Closer proximity to mapped water increases the score; "
            "distance alone does not establish flood exposure."
        ),
        "elevation": (
            "Lower relative elevation increases the score; actual "
            "terrain, river stage, and drainage connectivity matter."
        ),
        "drainage": (
            "Lower drainage quality increases the score; use measured "
            "or defensibly estimated drainage information."
        ),
    }

    return {
        "risk_score": round(risk_score, 4),
        "risk_percentage": round(risk_score * 100, 2),
        "risk_level": risk_level,
        "land_cover": land_cover,
        "factor_scores": {
            key: round(value, 4)
            for key, value in normalized_factors.items()
        },
        "weighted_contributions": {
            key: round(value, 4)
            for key, value in contributions.items()
        },
        "explanations": explanations,
        "model_type": "Experimental weighted rule-based model",
        "validated_probability": False,
    }


# =====================================================
# 6. BACKWARD-COMPATIBLE FUNCTION
# Existing callers can continue unpacking two values.
# =====================================================

def calculate_flood_risk(
    land_cover,
    rainfall_mm,
    water_distance_m,
    elevation_m,
    drainage_score,
):
    result = calculate_flood_risk_detailed(
        land_cover,
        rainfall_mm,
        water_distance_m,
        elevation_m,
        drainage_score,
    )

    return result["risk_score"], result["risk_level"]


# =====================================================
# 7. INTERACTIVE TEST
# =====================================================

if __name__ == "__main__":

    print("\n" + "=" * 48)
    print("       ADVANCED FLOOD RISK PREDICTOR")
    print("=" * 48)

    print("\nAvailable land-cover classes:")
    for name in LAND_COVER_WEIGHTS:
        print(f"  - {name}")

    try:
        land_cover = input("\nLand cover: ").strip().title()

        rainfall = float(
            input("Rainfall accumulation (mm): ")
        )

        water_distance = float(
            input("Distance to nearest water body (m): ")
        )

        elevation = float(
            input("Relative elevation (m; higher = higher ground): ")
        )

        drainage = float(
            input("Drainage quality (0-10): ")
        )

        result = calculate_flood_risk_detailed(
            land_cover,
            rainfall,
            water_distance,
            elevation,
            drainage,
        )

        print("\n" + "=" * 48)
        print("             FLOOD RISK REPORT")
        print("=" * 48)

        print(f"Land cover     : {result['land_cover']}")
        print(f"Risk score     : {result['risk_score']:.4f} / 1.0000")
        print(f"Risk percentage: {result['risk_percentage']:.2f}%")
        print(f"Risk level     : {result['risk_level']}")

        print("\nWeighted factor contributions:")
        for factor, contribution in result[
            "weighted_contributions"
        ].items():
            print(
                f"  {factor.replace('_', ' ').title():20s}"
                f"{contribution:.4f}"
            )

        print("\nInterpretation:")
        for factor, explanation in result["explanations"].items():
            print(f"- {factor.replace('_', ' ').title()}: {explanation}")

        print("\nModel limitations:")
        print("- Score is not a calibrated flood probability.")
        print("- Thresholds and weights require validation.")
        print("- This prototype must not be used as an emergency warning.")

        print("=" * 48)

    except ValueError as error:
        print(f"\nInput error: {error}")
