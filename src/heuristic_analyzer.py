"""
Heuristic visual analyzer for plant disease detection.

Key design principles:
  - Each disease requires a SPECIFIC COMBINATION of signals, not just one.
  - A green-dominant leaf is healthy — we apply a green-dominance guard
    that suppresses all scores proportionally.
  - Co-occurrence of multiple signals is MANDATORY (not an optional bonus)
    to prevent single-color false positives.
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────
#  Color masks
# ─────────────────────────────────────────────

def _green_mask(hsv: np.ndarray) -> np.ndarray:
    """Healthy green leaf tissue."""
    lower = np.array([30, 40, 40])
    upper = np.array([90, 255, 255])
    return cv2.inRange(hsv, lower, upper)


def _yellow_mask(hsv: np.ndarray) -> np.ndarray:
    """Yellowing / chlorotic tissue."""
    lower = np.array([18, 50, 80])
    upper = np.array([36, 255, 255])
    return cv2.inRange(hsv, lower, upper)


def _brown_mask(hsv: np.ndarray) -> np.ndarray:
    """Dark brown / necrotic lesion tissue."""
    lower = np.array([5, 50, 25])
    upper = np.array([22, 210, 150])
    return cv2.inRange(hsv, lower, upper)


def _white_mask(bgr: np.ndarray) -> np.ndarray:
    """Whitish / powdery spots (high brightness, low saturation)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Low saturation + high value = white/grey
    lower = np.array([0, 0, 185])
    upper = np.array([180, 55, 255])
    return cv2.inRange(hsv, lower, upper)


def _pixel_ratio(mask: np.ndarray) -> float:
    total = mask.shape[0] * mask.shape[1]
    return float(np.count_nonzero(mask)) / total


def _edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 60, 160)
    return _pixel_ratio(edges)


def _ring_texture_score(gray: np.ndarray) -> float:
    """
    Checks for concentric ring texture (hallmark of Early Blight).
    Compares std-dev across 3 annular zones around image center.
    Returns normalized variance-between-zones / mean-within-zones.
    """
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    zone_stds = []
    for r_in, r_out in [(8, 35), (35, 65), (65, 95)]:
        ring = np.zeros_like(gray)
        cv2.circle(ring, (cx, cy), r_out, 255, -1)
        cv2.circle(ring, (cx, cy), r_in,  0,   -1)
        pixels = gray[ring > 0]
        if pixels.size > 50:
            zone_stds.append(float(np.std(pixels)))
    if len(zone_stds) < 2:
        return 0.0
    return float(np.std(zone_stds)) / (float(np.mean(zone_stds)) + 1e-6)


# ─────────────────────────────────────────────
#  Green-dominance guard
# ─────────────────────────────────────────────

def _green_dominance(hsv: np.ndarray) -> float:
    """
    Fraction of leaf pixels that are healthy green.
    If > 0.55, the leaf is predominantly healthy and all heuristic
    scores will be suppressed to avoid false positives.
    """
    return _pixel_ratio(_green_mask(hsv))


# ─────────────────────────────────────────────
#  Per-disease scorers
# ─────────────────────────────────────────────

def _score_early_blight(bgr: np.ndarray, hsv: np.ndarray, gray: np.ndarray) -> tuple[float, list[str]]:
    """
    Early Blight (Alternaria solani):
      REQUIRES both brown lesions AND yellow halos simultaneously.
      Single-colour presence alone is not enough.
    """
    signals: list[str] = []

    yellow_r   = _pixel_ratio(_yellow_mask(hsv))
    brown_r    = _pixel_ratio(_brown_mask(hsv))
    ring_score = _ring_texture_score(gray)

    # MANDATORY: both must be present above base thresholds
    if yellow_r < 0.04 or brown_r < 0.04:
        # Only one signal — not specific enough for Early Blight
        partial = 0.0
        if yellow_r > 0.07 and brown_r > 0.02:
            partial = 0.18   # weak partial signal
            signals.append("Slight yellow-brown discolouration")
        return partial, signals

    # Both present — start scoring
    score = 0.40  # base for co-occurrence of yellow + brown
    signals.append("Yellow halo + brown lesion co-occurrence")

    if yellow_r > 0.08:
        score += 0.15
        signals.append("Prominent yellow halo zones")
    if brown_r > 0.08:
        score += 0.15
        signals.append("Significant brown lesion patches")
    if ring_score > 0.10:
        score += 0.20
        signals.append("Concentric ring texture pattern")

    return min(score, 1.0), signals


def _score_late_blight(bgr: np.ndarray, hsv: np.ndarray, gray: np.ndarray) -> tuple[float, list[str]]:
    """
    Late Blight (Phytophthora infestans):
      Water-soaked greyish-green lesions + dark necrosis + irregular edges.
      Needs AT LEAST 2 of 3 signals.
    """
    signals: list[str] = []

    # Water-soaked tissue: dark, desaturated green/grey-green
    lower_ws = np.array([38,  8, 22])
    upper_ws = np.array([105, 85, 120])
    watersoaked_r   = _pixel_ratio(cv2.inRange(hsv, lower_ws, upper_ws))
    dark_necrosis_r = _pixel_ratio(_brown_mask(hsv))
    edge_irr        = _edge_density(gray)

    triggers = 0
    score    = 0.0

    if watersoaked_r > 0.06:
        score    += 0.30
        triggers += 1
        signals.append("Water-soaked grey-green tissue")
    if dark_necrosis_r > 0.06:
        score    += 0.30
        triggers += 1
        signals.append("Dark necrotic lesion areas")
    if edge_irr > 0.11:
        score    += 0.25
        triggers += 1
        signals.append("Irregular / ragged leaf border")

    # Require at least 2 of 3 signals
    if triggers < 2:
        return 0.0, []

    return min(score, 1.0), signals


def _score_powdery_mildew(bgr: np.ndarray, hsv: np.ndarray, gray: np.ndarray) -> tuple[float, list[str]]:
    """
    Powdery Mildew:
      White/grey powdery coating sitting ON green leaf tissue.
      Needs white patches AND green tissue co-existing.
    """
    signals: list[str] = []

    white_r = _pixel_ratio(_white_mask(bgr))
    green_r = _pixel_ratio(_green_mask(hsv))

    # Require white patches on a green leaf
    if white_r < 0.05 or green_r < 0.10:
        return 0.0, []

    # Spatial overlap: white pixels near green pixels
    white_m = _white_mask(bgr)
    green_m = _green_mask(hsv)
    # Dilate green to check proximity
    kernel  = np.ones((9, 9), np.uint8)
    green_d = cv2.dilate(green_m, kernel, iterations=1)
    overlap = cv2.bitwise_and(white_m, green_d)
    overlap_r = _pixel_ratio(overlap)

    if overlap_r < 0.03:
        return 0.0, []

    score = 0.45  # base: white on green confirmed
    signals.append("White coating over green tissue")

    if white_r > 0.10:
        score += 0.25
        signals.append("Extensive powdery white patches")
    if white_r > 0.18:
        score += 0.15
        signals.append("Severe powdery mildew coverage")

    return min(score, 1.0), signals


def _score_mosaic_virus(bgr: np.ndarray, hsv: np.ndarray, gray: np.ndarray) -> tuple[float, list[str]]:
    """
    Mosaic Virus (CMV / TMV):
      Alternating yellow-green mottling AND high local patchiness.
      Needs BOTH mottling colour pattern AND texture evidence.
    """
    signals: list[str] = []

    yellow_r = _pixel_ratio(_yellow_mask(hsv))
    green_r  = _pixel_ratio(_green_mask(hsv))
    edge_irr = _edge_density(gray)

    # Local patchiness: difference between original and heavily blurred
    blurred   = cv2.GaussianBlur(gray, (21, 21), 0)
    local_var = float(np.std(
        np.abs(gray.astype(np.float32) - blurred.astype(np.float32))
    ))

    # MANDATORY: must have both yellow-green mottling AND patchiness
    has_mottling  = (yellow_r > 0.05 and green_r > 0.10)
    has_patchiness = (local_var > 14.0)

    if not (has_mottling and has_patchiness):
        # Partial: only one signal
        if has_mottling and local_var > 9.0:
            signals.append("Mild yellow-green mottling")
            return 0.22, signals
        return 0.0, []

    score = 0.45
    signals.append("Yellow-green mottling with patchy variation")

    if edge_irr > 0.12:
        score += 0.25
        signals.append("Leaf curl / deformation pattern")
    if local_var > 20.0:
        score += 0.20
        signals.append("Strong colour variation (mottling)")

    return min(score, 1.0), signals


# ─────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────

DISEASE_SCORERS = {
    "Early Blight":   _score_early_blight,
    "Late Blight":    _score_late_blight,
    "Powdery Mildew": _score_powdery_mildew,
    "Mosaic Virus":   _score_mosaic_virus,
}


def analyze_heuristic(image: np.ndarray) -> dict:
    """
    Run all heuristic disease scorers on a preprocessed BGR image.

    Returns:
        {
          "label":      str,
          "confidence": float (0..1),
          "signals":    list[str],
          "all_scores": dict,
        }
    """
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ── Green dominance guard ──────────────────────────────────────
    # If the leaf is mostly healthy green, suppress all scores to
    # avoid false positives from tiny amounts of yellow/brown.
    green_dom = _green_dominance(hsv)
    # Suppression ramps from 0 (no suppression at green_dom=0.40)
    # to 0.80 (strong suppression at green_dom=0.80+)
    suppression = max(0.0, min(0.80, (green_dom - 0.40) * 2.0)) if green_dom > 0.40 else 0.0

    # ── Score each disease ─────────────────────────────────────────
    results: dict[str, tuple[float, list[str]]] = {}
    for disease, scorer in DISEASE_SCORERS.items():
        raw_score, sigs = scorer(image, hsv, gray)
        suppressed_score = raw_score * (1.0 - suppression)
        results[disease] = (suppressed_score, sigs)

    # ── Pick best ──────────────────────────────────────────────────
    best_disease          = max(results, key=lambda d: results[d][0])
    best_score, best_sigs = results[best_disease]

    all_scores = {d: round(s, 3) for d, (s, _) in results.items()}

    # Minimum confidence to be declared a disease (not just noise)
    MIN_CONFIDENCE = 0.42

    if best_score < MIN_CONFIDENCE:
        return {
            "label":      "Healthy",
            "confidence": round(max(0.50, 1.0 - best_score), 3),
            "signals":    [],
            "all_scores": all_scores,
        }

    return {
        "label":      best_disease,
        "confidence": round(best_score, 3),
        "signals":    best_sigs,
        "all_scores": all_scores,
    }
