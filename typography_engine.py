"""
Math-by-Design Typography Engine -- causal prototype
==========================================

A runnable, inspectable skeleton for the "Typography Engine" described in
the accompanying design note: a system that generates typographic
*relationships* (a job graph, a character vector, a pairing, a scale, a
hierarchy, a validation report) and only then goes looking for fonts that
can express them, instead of picking a font name first.

Core stages, each an inspectable numpy/sympy mechanism:

  1. Typographic Job Graph    -- infer_job_graph()
  2. Typography Genome        -- Genome, derive_genome()
  3. Genome <- project thesis -- ProjectThesis, derive_genome(), derive_grammar()
  4. Causal system selection    -- select_font_system(), PairingMode
  5. Realization + validation -- realization_error(), inter_test(), collision_similarity()

Plus the supporting pieces the note also asks for: a symbolic type-scale
ratio (sympy), a font catalog searched by *traits* rather than by name
(numpy nearest-neighbor), and a hierarchy/"convergence" generator that
walks the type scale from body text to a display size.

Honesty note: the font catalog's trait numbers and `supports` flags below
are illustrative placeholders estimated for this demo -- they are not
measured from the real font files or verified against their actual
metadata/feature tables. The *mechanism* (vectors, distances, scoring,
recurrences) is the real content here; swap in real metrics before using
this for actual typeface selection.

Run it directly for a demo or self-test:

    python3 typography_engine.py
    python3 typography_engine.py --self-test
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional

import numpy as np
import sympy as sp


# ----------------------------------------------------------------------------
# 1. Axes & Genome
# ----------------------------------------------------------------------------

# A typeface's *identity* -- what kind of face it is, independent of which
# weight/width instance gets used -- lives on these ten continuous axes.
AXES = [
    "width",         # condensed(0)    <-> expanded(1)
    "contrast",       # monoline(0)     <-> high-contrast(1)
    "construction",    # geometric(0)    <-> humanist(1)
    "texture",           # mechanical(0)   <-> literary(1)
    "x_height",            # low(0)          <-> high(1)
    "terminals",             # blunt(0)        <-> expressive(1)
    "density",                 # airy(0)         <-> compact(1)
    "voice",                     # neutral(0)      <-> eccentric(1)
    "historicity",                 # contemporary(0) <-> historical(1)
    "regularity",                    # systematic(0)   <-> irregular(1)
]
N_AXES = len(AXES)
MAX_DIST = float(np.sqrt(N_AXES))  # diagonal of the unit hypercube; normalizes distances to ~[0, 1]


def vec_from_axes(**kwargs: float) -> np.ndarray:
    """Build a genome vector from axis=value kwargs; axes left unspecified
    default to 0.5 (neutral)."""
    unknown = set(kwargs) - set(AXES)
    if unknown:
        raise ValueError(f"unknown axis name(s): {sorted(unknown)}")
    return np.array([kwargs.get(a, 0.5) for a in AXES], dtype=float)


class Job(Enum):
    CLAIM = "claim"
    READING = "reading"
    NAVIGATION = "navigation"
    INSTRUMENT = "instrument"
    ANNOTATION = "annotation"
    EMPHASIS = "emphasis"
    DISPLAY_DATA = "display-data"


@dataclass
class Genome:
    """A point in typographic-character space, plus the categorical traits
    that don't belong on a continuous axis."""
    vector: np.ndarray               # shape (N_AXES,), each entry in [0, 1]
    case: str = "sentence"           # sentence | title | upper | lower
    tracking: float = 0.0            # em
    line_height: float = 1.4         # multiplier
    label_mechanism: str = "none"    # eyebrow | rule-label | margin-note | none | ...

    def __post_init__(self) -> None:
        self.vector = np.asarray(self.vector, dtype=float)
        if self.vector.shape != (N_AXES,):
            raise ValueError(f"genome vector must have shape ({N_AXES},), got {self.vector.shape}")

    def as_dict(self) -> dict:
        return {
            **{axis: round(v, 3) for axis, v in zip(AXES, self.vector.tolist())},
            "case": self.case,
            "tracking": round(self.tracking, 4),
            "line_height": round(self.line_height, 3),
            "label_mechanism": self.label_mechanism,
        }


# ----------------------------------------------------------------------------
# 2. Project thesis -> genome derivation
# ----------------------------------------------------------------------------

THESIS_AXES = ["technicality", "formality", "warmth", "density", "historicity_pull", "playfulness"]


@dataclass
class ProjectThesis:
    """A short label plus the six numbers that drive everything downstream --
    stand-ins for 'material + geometry + information topology + color +
    content density + audience'. In a real build these would themselves be
    derived from the rest of the system (palette, layout, material
    metaphor); here they're the input."""
    label: str
    vector: np.ndarray  # shape (len(THESIS_AXES),), each roughly in [0, 1]

    def __post_init__(self) -> None:
        self.vector = np.asarray(self.vector, dtype=float)


def infer_job_graph(thesis: np.ndarray) -> dict:
    """Map each semantic job to a role ('display' | 'reading' | 'instrument'
    | 'reading-italic') or None when the project has no real use for that
    job. A simple threshold rule, not a learned model -- the point is that
    which jobs are even active is itself a decision, not a fixed list of
    seven boxes to fill in."""
    technicality, formality, warmth, density, historicity_pull, playfulness = thesis
    return {
        Job.CLAIM.value: "display",
        Job.READING.value: "reading",
        Job.NAVIGATION.value: "instrument" if technicality > 0.4 else "reading",
        Job.INSTRUMENT.value: "instrument",
        Job.ANNOTATION.value: "instrument" if technicality > 0.5 else "reading",
        Job.EMPHASIS.value: None if (technicality > 0.7 and playfulness < 0.2) else "reading-italic",
        Job.DISPLAY_DATA.value: "instrument" if density > 0.5 else "display",
    }


def derive_genome(thesis: np.ndarray, role: str) -> np.ndarray:
    """The project thesis pushes and pulls each axis differently depending
    on the role. Coefficients are hand-set and meant to be edited -- they
    are the 'ingredients', not a fitted model."""
    technicality, formality, warmth, density, historicity_pull, playfulness = thesis

    if role == "reading":
        width        = 0.56 - 0.12 * technicality
        contrast     = 0.30 + 0.30 * historicity_pull - 0.18 * technicality
        construction = 0.58 + 0.28 * warmth - 0.22 * technicality
        texture      = 0.30 + 0.40 * historicity_pull + 0.22 * warmth
        x_height     = 0.66 + 0.10 * technicality
        terminals    = 0.42 + 0.28 * warmth - 0.16 * technicality
        density_ax   = 0.42 + 0.28 * density
        voice        = 0.20 + 0.32 * playfulness
        historicity  = 0.25 + 0.50 * historicity_pull
        regularity   = 0.72 - 0.28 * playfulness

    elif role == "instrument":
        width        = 0.58 + 0.20 * technicality - 0.12 * density
        contrast     = 0.08 + 0.10 * formality
        construction = 0.22 + 0.10 * warmth - 0.12 * technicality
        texture      = 0.08 + 0.10 * warmth
        x_height     = 0.78 + 0.05 * technicality
        terminals    = 0.18 + 0.18 * playfulness
        density_ax   = 0.38 + 0.34 * density
        voice        = 0.16 + 0.28 * playfulness
        historicity  = 0.06 + 0.14 * historicity_pull
        regularity   = 0.88 + 0.08 * technicality

    elif role == "display":
        width        = 0.38 - 0.30 * technicality + 0.22 * formality
        contrast     = 0.16 + 0.45 * historicity_pull - 0.10 * technicality
        construction = 0.50 + 0.26 * warmth - 0.18 * technicality
        texture      = 0.22 + 0.42 * historicity_pull + 0.20 * warmth
        x_height     = 0.60 + 0.10 * technicality
        terminals    = 0.32 + 0.45 * playfulness + 0.15 * historicity_pull
        density_ax   = 0.28 + 0.24 * density
        voice        = 0.22 + 0.55 * playfulness
        historicity  = 0.18 + 0.55 * historicity_pull
        regularity   = 0.60 - 0.32 * playfulness

    else:
        raise ValueError(f"unknown role: {role!r}")

    vec = np.array([width, contrast, construction, texture, x_height,
                     terminals, density_ax, voice, historicity, regularity])
    return np.clip(vec, 0.0, 1.0)


def derive_grammar(thesis: np.ndarray) -> dict:
    """Casing, tracking and the label mechanism are themselves generated,
    not defaulted to 'uppercase + tracked + 10px' for anything technical."""
    technicality, formality, warmth, density, historicity_pull, playfulness = thesis

    if technicality > 0.6 and density > 0.5:
        label_mechanism = "rule-label"
    elif formality > 0.6 and historicity_pull > 0.5:
        label_mechanism = "margin-note"
    else:
        label_mechanism = "eyebrow"

    # tracked uppercase eyebrows are only earned when nothing else (a rule,
    # a margin, a mono prefix) is already doing the structural work of
    # separating a label from its content
    use_tracked_caps = label_mechanism == "eyebrow" and playfulness < 0.4
    claim_case = "title" if (formality > 0.55 and playfulness < 0.5) else "sentence"

    return {
        "label_mechanism": label_mechanism,
        "label_case": "upper" if use_tracked_caps else "sentence",
        "label_tracking": 0.12 if use_tracked_caps else 0.0,
        "claim_case": claim_case,
    }


# ----------------------------------------------------------------------------
# 3. Symbolic scale ratio (sympy) + hierarchy as a trajectory (numpy)
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 3. Symbolic scale ratio + thesis-responsive hierarchy trajectory
# ----------------------------------------------------------------------------

from itertools import product
import sys

phi = (1 + sp.sqrt(5)) / 2

SCALE_RATIOS: dict = {
    "quiet_dense_tool":     sp.sqrt(phi),
    "editorial_hierarchy":  phi ** sp.Rational(2, 3),
    "dramatic_sparse_hero": phi,
    "technical_compact":    sp.sqrt(2),
}

SCALE_LABELS: dict = {
    "quiet_dense_tool":     "sqrt(phi)",
    "editorial_hierarchy":  "phi^(2/3)",
    "dramatic_sparse_hero": "phi",
    "technical_compact":    "sqrt(2)",
}


def scale_ratio(name: str) -> tuple:
    """Return (exact SymPy expression, numeric float)."""
    expr = SCALE_RATIOS[name]
    return expr, float(expr.evalf())


def choose_scale_ratio(thesis: np.ndarray) -> str:
    """Choose a scale family from the thesis when the caller does not pin one.

    This is deliberately modest: ratio selection is a hierarchy decision, not a
    claim that phi or sqrt(2) is intrinsically 'better'.
    """
    technicality, formality, warmth, density, historicity_pull, playfulness = thesis
    if density >= 0.68:
        return "quiet_dense_tool"
    if playfulness >= 0.62 and density <= 0.45:
        return "dramatic_sparse_hero"
    if technicality >= 0.72 and historicity_pull <= 0.35:
        return "technical_compact"
    return "editorial_hierarchy"


HIER_FIELDS = ["size", "weight", "width", "tracking", "line_height", "measure"]


@dataclass
class HierarchyLevel:
    name: str
    size: float
    weight: float
    width: float
    tracking: float
    line_height: float
    measure: float


@dataclass
class HierarchyPolicy:
    body: dict
    per_level_delta: dict
    rationale: dict


def derive_hierarchy_policy(
    thesis: np.ndarray,
    reading_target: np.ndarray,
    display_target: np.ndarray,
) -> HierarchyPolicy:
    """Derive hierarchy motion from the project instead of applying one curve.

    Size still follows the selected symbolic ratio. The remaining channels are
    linked but thesis-responsive. Weight/width are explicitly design choices;
    tracking, line-height and measure generally tighten as display size rises.
    """
    technicality, formality, warmth, density, historicity_pull, playfulness = thesis
    ridx = {a: AXES.index(a) for a in AXES}

    body = {
        "size": 16.0,
        "weight": float(np.clip(390 + 65 * formality + 35 * density, 360, 520)),
        "width": float(np.clip(100 + 8 * (reading_target[ridx["width"]] - 0.5), 94, 106)),
        "tracking": 0.0,
        "line_height": float(np.clip(1.48 + 0.10 * warmth + 0.08 * historicity_pull - 0.08 * density, 1.36, 1.68)),
        "measure": float(np.clip(70 + 7 * historicity_pull + 4 * warmth - 12 * density, 54, 78)),
    }

    # Technical/dense systems tend toward emphatic compact display; warm or
    # historical systems are allowed to become lighter instead. This is a
    # policy, not a universal typographic law, and is exposed in the packet.
    weight_delta = float(np.clip(
        55 * (technicality + density - warmth - historicity_pull) / 2,
        -36, 48,
    ))

    target_width_delta = (
        display_target[ridx["width"]] - reading_target[ridx["width"]]
    ) * 18
    width_delta = float(np.clip(
        target_width_delta - 2.5 * technicality + 1.5 * playfulness,
        -8, 8,
    ))

    tracking_delta = float(-np.clip(0.008 + 0.010 * formality + 0.007 * density, 0.007, 0.024))
    line_height_delta = float(-np.clip(0.09 + 0.075 * density + 0.025 * formality, 0.08, 0.20))
    measure_delta = float(-np.clip(5.5 + 7.0 * density, 5.0, 13.5))

    per_level_delta = {
        "weight": weight_delta,
        "width": width_delta,
        "tracking": tracking_delta,
        "line_height": line_height_delta,
        "measure": measure_delta,
    }
    rationale = {
        "weight_direction": "heavier" if weight_delta > 2 else "lighter" if weight_delta < -2 else "stable",
        "width_direction": "wider" if width_delta > 0.5 else "narrower" if width_delta < -0.5 else "stable",
        "tracking": "tightens with hierarchy",
        "line_height": "tightens with hierarchy",
        "measure": "shortens with hierarchy",
    }
    return HierarchyPolicy(body=body, per_level_delta=per_level_delta, rationale=rationale)


def build_hierarchy(names: list, body: dict, ratio: float, per_level_delta: dict) -> list:
    base_vec = np.array([body[f] for f in HIER_FIELDS], dtype=float)
    delta_vec = np.array([0.0] + [per_level_delta.get(f, 0.0) for f in HIER_FIELDS[1:]])
    levels = []
    for n in range(len(names)):
        vec = base_vec + n * delta_vec
        vec[0] = body["size"] * (ratio ** n)
        # Safety clamps are about usability, not aesthetics.
        vec[1] = np.clip(vec[1], 250, 900)       # weight
        vec[2] = np.clip(vec[2], 75, 125)        # width percentage-ish
        vec[3] = np.clip(vec[3], -0.09, 0.12)    # em tracking
        vec[4] = np.clip(vec[4], 0.86, 1.8)      # line-height
        vec[5] = np.clip(vec[5], 10, 90)         # measure ch
        levels.append(HierarchyLevel(names[n], *vec.tolist()))
    return levels


# ----------------------------------------------------------------------------
# 4. Font catalog -- traits first, names second
# ----------------------------------------------------------------------------

@dataclass
class Font:
    family: str
    license: str
    classes: list
    vector: np.ndarray
    supports: dict
    provenance: str = "illustrative-placeholder"

    def __post_init__(self) -> None:
        self.vector = np.asarray(self.vector, dtype=float)
        if self.vector.shape != (N_AXES,):
            raise ValueError(f"font vector must have shape ({N_AXES},), got {self.vector.shape}")

    def as_genome(self) -> Genome:
        return Genome(vector=self.vector.copy())


class FontCatalog:
    def __init__(self, fonts: list):
        self.fonts = fonts
        self._matrix = np.stack([f.vector for f in fonts]) if fonts else np.zeros((0, N_AXES))

    def nearest(self, target: np.ndarray, k: int = 3, classes: Optional[list] = None) -> list:
        candidates = self.fonts
        matrix = self._matrix
        if classes:
            mask = np.array([any(c in f.classes for c in classes) for f in self.fonts])
            candidates = [f for f, m in zip(self.fonts, mask) if m]
            matrix = matrix[mask] if len(matrix) else matrix
        if not candidates:
            return []
        dists = np.linalg.norm(matrix - target[None, :], axis=1)
        order = np.argsort(dists)[:k]
        return [(candidates[i], float(dists[i])) for i in order]


# These values remain intentionally illustrative until a measurement/import
# pipeline extracts real metrics and OpenType capabilities from font files.
DEMO_CATALOG = FontCatalog([
    Font("Barlow Semi Condensed", "OFL", ["sans", "semi-condensed"],
         vec_from_axes(width=0.30, contrast=0.15, construction=0.55, texture=0.20,
                       x_height=0.72, terminals=0.35, density=0.60, voice=0.20,
                       historicity=0.15, regularity=0.85),
         {"italic": True, "variable": True, "tabular_nums": True}),
    Font("Martian Mono", "OFL", ["mono"],
         vec_from_axes(width=0.55, contrast=0.10, construction=0.30, texture=0.10,
                       x_height=0.78, terminals=0.25, density=0.45, voice=0.55,
                       historicity=0.10, regularity=0.95),
         {"italic": False, "variable": True, "tabular_nums": True}),
    Font("Familjen Grotesk", "OFL", ["sans", "grotesk"],
         vec_from_axes(width=0.50, contrast=0.18, construction=0.60, texture=0.30,
                       x_height=0.70, terminals=0.40, density=0.45, voice=0.35,
                       historicity=0.20, regularity=0.80),
         {"italic": False, "variable": True, "tabular_nums": True}),
    Font("Sometype Mono", "OFL", ["mono"],
         vec_from_axes(width=0.50, contrast=0.12, construction=0.35, texture=0.15,
                       x_height=0.75, terminals=0.30, density=0.45, voice=0.25,
                       historicity=0.15, regularity=0.95),
         {"italic": True, "variable": True, "tabular_nums": True}),
    Font("Aleo", "OFL", ["slab", "serif"],
         vec_from_axes(width=0.52, contrast=0.35, construction=0.65, texture=0.55,
                       x_height=0.65, terminals=0.55, density=0.48, voice=0.30,
                       historicity=0.45, regularity=0.70),
         {"italic": True, "variable": False, "tabular_nums": True}),
    Font("Azeret Mono", "OFL", ["mono"],
         vec_from_axes(width=0.55, contrast=0.15, construction=0.40, texture=0.20,
                       x_height=0.72, terminals=0.45, density=0.45, voice=0.45,
                       historicity=0.15, regularity=0.90),
         {"italic": True, "variable": True, "tabular_nums": True}),
    Font("Brygada 1918", "OFL", ["serif"],
         vec_from_axes(width=0.50, contrast=0.55, construction=0.75, texture=0.75,
                       x_height=0.62, terminals=0.60, density=0.42, voice=0.35,
                       historicity=0.65, regularity=0.60),
         {"italic": True, "variable": False, "tabular_nums": True}),
    Font("Spline Sans Mono", "OFL", ["mono"],
         vec_from_axes(width=0.50, contrast=0.12, construction=0.45, texture=0.18,
                       x_height=0.76, terminals=0.35, density=0.45, voice=0.25,
                       historicity=0.12, regularity=0.92),
         {"italic": True, "variable": True, "tabular_nums": True}),
])


# ----------------------------------------------------------------------------
# 5. Causal pairing + system selection
# ----------------------------------------------------------------------------

class PairingMode(Enum):
    SOLO = "solo"
    ECHO = "echo"
    KINSHIP = "kinship"
    UTILITY = "utility"
    COUNTERPOINT = "counterpoint"
    TENSION = "tension"


MODE_TARGET_DISTANCE = {
    PairingMode.SOLO: 0.03,
    PairingMode.ECHO: 0.08,
    PairingMode.KINSHIP: 0.15,
    PairingMode.UTILITY: 0.22,
    PairingMode.COUNTERPOINT: 0.30,
    PairingMode.TENSION: 0.40,
}

CLICHE_PAIRS = {
    frozenset({"Inter", "Playfair Display"}),
    frozenset({"Inter", "Georgia"}),
    frozenset({"Montserrat", "Merriweather"}),
    frozenset({"Poppins", "Playfair Display"}),
}

ROLE_ALLOWED_CLASSES = {
    "reading": ["sans", "serif", "slab", "grotesk"],
    "display": ["sans", "serif", "slab", "grotesk"],
    # Instrument text is not synonymous with monospace. A tabular sans can be
    # correct; technicality determines how strongly mono is preferred.
    "instrument": ["mono", "sans", "grotesk", "semi-condensed"],
}


def cliche_penalty_for(family_a: str, family_b: str) -> float:
    return 1.0 if frozenset({family_a, family_b}) in CLICHE_PAIRS else 0.0


def normalized_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / MAX_DIST)


def pairing_score(
    a: Genome,
    b: Genome,
    role_target_a: np.ndarray,
    role_target_b: np.ndarray,
    project_center: np.ndarray,
    mode: PairingMode,
    weights: tuple = (1.0, 1.0, 1.0, 0.8, 0.8),
    cliche_penalty: float = 0.0,
) -> dict:
    """Score a *realized* pair against a requested relationship.

    The score is peaked around the mode's target distance rather than blindly
    maximizing contrast. It therefore has causal meaning during selection.
    """
    w_c, w_h, w_r, w_m, w_d = weights
    dist_ab = normalized_distance(a.vector, b.vector)
    target_d = MODE_TARGET_DISTANCE[mode]
    sigma = 0.105 if mode in {PairingMode.SOLO, PairingMode.ECHO} else 0.13
    C = float(np.exp(-((dist_ab - target_d) ** 2) / (2 * sigma ** 2)))

    harmony_axes = [AXES.index(ax) for ax in ("construction", "x_height", "density", "regularity")]
    H = float(1.0 - np.mean(np.abs(a.vector[harmony_axes] - b.vector[harmony_axes])))
    R = float(1.0 - 0.5 * (
        normalized_distance(a.vector, role_target_a) + normalized_distance(b.vector, role_target_b)
    ))
    M = float(1.0 - 0.5 * (
        normalized_distance(a.vector, project_center) + normalized_distance(b.vector, project_center)
    ))
    D = float(cliche_penalty)

    raw = w_c * C + w_h * H + w_r * R + w_m * M - w_d * D
    denom = w_c + w_h + w_r + w_m
    normalized = float(np.clip(raw / denom, 0.0, 1.0))
    return {
        "score": round(raw, 4),
        "normalized_score": round(normalized, 4),
        "C": round(C, 4), "H": round(H, 4), "R": round(R, 4),
        "M": round(M, 4), "D": D, "distance": round(dist_ab, 4),
        "target_distance": target_d,
    }


def role_fit(font: Font, target: np.ndarray, role: str, thesis: np.ndarray) -> float:
    """Trait fit + role semantics + required feature fit, all in [0, 1]."""
    technicality, formality, warmth, density, historicity_pull, playfulness = thesis
    trait = float(np.clip(1.0 - normalized_distance(font.vector, target), 0.0, 1.0))

    classes = set(font.classes)
    if role == "instrument":
        if "mono" in classes:
            class_fit = 1.0 if technicality >= 0.45 else 0.82
        elif classes.intersection({"sans", "grotesk", "semi-condensed"}):
            class_fit = 0.90 if font.supports.get("tabular_nums") else 0.62
        else:
            class_fit = 0.45
        feature_fit = 1.0 if font.supports.get("tabular_nums") else 0.55
    elif role == "reading":
        class_fit = 1.0 if classes.intersection({"sans", "serif", "slab", "grotesk"}) else 0.4
        feature_fit = 1.0
    else:  # display
        class_fit = 1.0 if classes.intersection({"sans", "serif", "slab", "grotesk"}) else 0.45
        feature_fit = 1.0

    return float(np.clip(0.68 * trait + 0.22 * class_fit + 0.10 * feature_fit, 0.0, 1.0))


def family_structure_score(fonts: dict, mode: PairingMode) -> float:
    """Prefer family economy or separation according to the requested mode."""
    fams = [fonts[r].family for r in ("reading", "display", "instrument")]
    unique = len(set(fams))
    rd_same = fonts["reading"].family == fonts["display"].family

    if mode == PairingMode.SOLO:
        # In UI work a strict single-family system may still require a separate
        # instrument face; two families with shared reading/display is a valid
        # graceful fallback when the catalog lacks a true variable superfamily.
        return 1.0 if unique == 1 else 0.82 if (unique == 2 and rd_same) else 0.30
    if mode == PairingMode.ECHO:
        return 1.0 if unique <= 2 else 0.72
    if mode == PairingMode.KINSHIP:
        return 1.0 if unique == 2 else 0.82 if unique == 3 else 0.75
    if mode == PairingMode.UTILITY:
        return 1.0 if (rd_same and unique == 2) else 0.88 if unique == 3 else 0.72
    if mode == PairingMode.COUNTERPOINT:
        return 1.0 if unique >= 2 else 0.45
    if mode == PairingMode.TENSION:
        return 1.0 if unique == 3 else 0.72 if unique == 2 else 0.25
    return 0.5


def _candidate_pool(catalog: FontCatalog, target: np.ndarray, role: str, k: int) -> list:
    hits = catalog.nearest(target, k=max(k, 1), classes=ROLE_ALLOWED_CLASSES[role])
    return [font for font, _ in hits]


def system_identity_score(fonts: dict) -> dict:
    """Measure whether the *system* contributes identity beyond safe defaults.

    A neutral utility/instrument face is allowed. Display carries the largest
    identity burden, reading next, instrument least. This is closer to the
    intended "replace it all with Inter/Georgia/mono" thought experiment than
    requiring every individual role to be distinctive.
    """
    distances = {r: normalized_distance(fonts[r].vector, INTER_BASELINE.vector) for r in ROLES}
    weighted = (
        0.35 * distances["reading"]
        + 0.15 * distances["instrument"]
        + 0.50 * distances["display"]
    )
    expressive_peak = max(distances["reading"], distances["display"])
    score = float(np.clip(weighted / 0.20, 0.0, 1.0))
    passed = weighted >= 0.12 and expressive_peak >= 0.12
    return {
        "weighted_distance": round(weighted, 4),
        "expressive_peak": round(expressive_peak, 4),
        "per_role": {r: round(v, 4) for r, v in distances.items()},
        "score": round(score, 4),
        "pass": bool(passed),
    }


def score_font_system(
    fonts: dict,
    targets: dict,
    thesis: np.ndarray,
    project_center: np.ndarray,
    mode: PairingMode,
    emphasis_active: bool,
) -> dict:
    fits = {r: role_fit(fonts[r], targets[r], r, thesis) for r in ROLES}

    pair_specs = [
        ("display_instrument", "display", "instrument", 0.45),
        ("reading_display", "reading", "display", 0.35),
        ("reading_instrument", "reading", "instrument", 0.20),
    ]
    pairs = {}
    pair_total = 0.0
    for key, ra, rb, weight in pair_specs:
        pa = fonts[ra].as_genome()
        pb = fonts[rb].as_genome()
        p = pairing_score(
            pa, pb,
            role_target_a=targets[ra],
            role_target_b=targets[rb],
            project_center=project_center,
            mode=mode,
            cliche_penalty=cliche_penalty_for(fonts[ra].family, fonts[rb].family),
        )
        pairs[key] = p
        pair_total += weight * p["normalized_score"]

    structure = family_structure_score(fonts, mode)
    fit_mean = float(np.mean(list(fits.values())))
    identity = system_identity_score(fonts)

    feature_penalty = 0.0
    feature_notes = []
    if emphasis_active and not fonts["reading"].supports.get("italic", False):
        feature_penalty += 0.12
        feature_notes.append("reading face lacks italic while emphasis job is active")
    if not fonts["instrument"].supports.get("tabular_nums", False):
        feature_penalty += 0.14
        feature_notes.append("instrument face lacks tabular numerals")

    total = (
        0.48 * pair_total
        + 0.34 * fit_mean
        + 0.10 * structure
        + 0.08 * identity["score"]
        - feature_penalty
    )
    return {
        "score": round(float(total), 6),
        "pair_relationship": round(float(pair_total), 4),
        "role_fit": {k: round(v, 4) for k, v in fits.items()},
        "role_fit_mean": round(fit_mean, 4),
        "family_structure": round(structure, 4),
        "identity_contribution": identity,
        "feature_penalty": round(feature_penalty, 4),
        "feature_notes": feature_notes,
        "pairs": pairs,
    }


def select_font_system(
    catalog: FontCatalog,
    targets: dict,
    thesis: np.ndarray,
    mode: PairingMode,
    job_graph: dict,
    candidate_k: int = 6,
) -> dict:
    """Enumerate bounded candidate systems and let relationship scoring select.

    PairingMode is therefore causal: changing the requested relationship can
    change the selected families rather than merely changing a report score.
    """
    pools = {r: _candidate_pool(catalog, targets[r], r, candidate_k) for r in ROLES}
    if any(not pools[r] for r in ROLES):
        missing = [r for r in ROLES if not pools[r]]
        raise ValueError(f"no font candidates for roles: {missing}")

    project_center = np.mean([targets[r] for r in ROLES], axis=0)
    emphasis_active = job_graph.get(Job.EMPHASIS.value) is not None
    scored = []
    for reading, instrument, display in product(pools["reading"], pools["instrument"], pools["display"]):
        fonts = {"reading": reading, "instrument": instrument, "display": display}
        details = score_font_system(fonts, targets, thesis, project_center, mode, emphasis_active)
        scored.append((details["score"], fonts, details))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_fonts, best_details = scored[0]
    alternatives = []
    seen = set()
    for score, fonts, details in scored:
        key = tuple(fonts[r].family for r in ROLES)
        if key in seen:
            continue
        seen.add(key)
        alternatives.append({
            "families": {r: fonts[r].family for r in ROLES},
            "score": round(score, 6),
        })
        if len(alternatives) >= 5:
            break

    return {
        "fonts": best_fonts,
        "families": {r: best_fonts[r].family for r in ROLES},
        "score": best_score,
        "details": best_details,
        "alternatives": alternatives,
        "candidate_counts": {r: len(pools[r]) for r in ROLES},
    }


# ----------------------------------------------------------------------------
# 6. Validation: intent vs realized typography
# ----------------------------------------------------------------------------

INTER_BASELINE = Genome(vector=vec_from_axes(
    width=0.5, contrast=0.15, construction=0.55, texture=0.20,
    x_height=0.75, terminals=0.30, density=0.50, voice=0.10,
    historicity=0.10, regularity=0.90,
))


def inter_test(genome: Genome, threshold: float = 0.12) -> dict:
    dist = normalized_distance(genome.vector, INTER_BASELINE.vector)
    return {"distance_from_baseline": round(dist, 4), "pass": dist >= threshold}


def role_truth(roles: dict, threshold: float = 0.05) -> dict:
    names = list(roles.keys())
    collisions = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = roles[names[i]], roles[names[j]]
            dist = normalized_distance(a.vector, b.vector)
            if dist < threshold:
                collisions.append({"a": names[i], "b": names[j], "distance": round(dist, 4)})
    return {"collisions": collisions, "pass": len(collisions) == 0}


def realization_error(targets: dict, fonts: dict) -> dict:
    per_role = {r: normalized_distance(targets[r], fonts[r].vector) for r in ROLES}
    mean = float(np.mean(list(per_role.values())))
    return {
        "per_role": {r: round(v, 4) for r, v in per_role.items()},
        "mean": round(mean, 4),
        "pass": mean <= 0.22,
        "note": "threshold is provisional until real font metrics replace placeholder catalog traits",
    }


def feature_validation(job_graph: dict, fonts: dict) -> dict:
    failures = []
    if job_graph.get(Job.EMPHASIS.value) is not None and not fonts["reading"].supports.get("italic", False):
        failures.append("active emphasis role requires italic-capable reading face")
    if not fonts["instrument"].supports.get("tabular_nums", False):
        failures.append("instrument role requires tabular numerals")
    return {"failures": failures, "pass": not failures}


def _grammar_signature(grammar: dict) -> np.ndarray:
    mechanisms = ["none", "eyebrow", "rule-label", "margin-note"]
    cases = ["sentence", "title", "upper", "lower"]
    mech = [1.0 if grammar.get("label_mechanism") == x else 0.0 for x in mechanisms]
    label_case = [1.0 if grammar.get("label_case") == x else 0.0 for x in cases]
    claim_case = [1.0 if grammar.get("claim_case") == x else 0.0 for x in cases]
    tracking = [float(np.clip((grammar.get("label_tracking", 0.0) + 0.10) / 0.22, 0.0, 1.0))]
    return np.array(mech + label_case + claim_case + tracking, dtype=float)


def _hierarchy_signature(hierarchy: list, ratio: float) -> np.ndarray:
    values = [float(np.clip(ratio / float(phi.evalf()), 0.0, 1.25))]
    for lv in hierarchy:
        values.extend([
            np.clip(lv.size / 128.0, 0, 1.5),
            np.clip(lv.weight / 900.0, 0, 1),
            np.clip((lv.width - 75) / 50.0, 0, 1),
            np.clip((lv.tracking + 0.09) / 0.21, 0, 1),
            np.clip((lv.line_height - 0.86) / 0.94, 0, 1),
            np.clip(lv.measure / 90.0, 0, 1),
        ])
    return np.array(values, dtype=float)


def make_signature(role_vectors: dict, grammar: dict, hierarchy: list, ratio: float) -> np.ndarray:
    return np.concatenate([
        *[np.asarray(role_vectors[r], dtype=float) for r in ROLES],
        _grammar_signature(grammar),
        _hierarchy_signature(hierarchy, ratio),
    ])


def collision_similarity(signature: np.ndarray, history: list, threshold: float = 0.94) -> dict:
    if not history:
        return {"max_similarity": 0.0, "compared": 0, "pass": True}
    sims = []
    skipped = 0
    for h in history:
        h = np.asarray(h, dtype=float)
        if h.shape != signature.shape:
            skipped += 1
            continue
        sims.append(float(np.dot(signature, h) / (np.linalg.norm(signature) * np.linalg.norm(h) + 1e-9)))
    if not sims:
        return {"max_similarity": 0.0, "compared": 0, "skipped_shape_mismatch": skipped, "pass": True}
    max_sim = max(sims)
    return {
        "max_similarity": round(max_sim, 4),
        "compared": len(sims),
        "skipped_shape_mismatch": skipped,
        "pass": max_sim < threshold,
    }


# ----------------------------------------------------------------------------
# 7. End-to-end pipeline
# ----------------------------------------------------------------------------

ROLES = ("reading", "instrument", "display")


def run_engine(
    thesis: ProjectThesis,
    catalog: FontCatalog,
    pairing_mode: PairingMode,
    ratio_name: Optional[str] = None,
    level_names: tuple = ("body", "section", "display"),
    base_px: float = 16.0,
    history: Optional[list] = None,
    candidate_k: int = 6,
) -> dict:
    if thesis.vector.shape != (len(THESIS_AXES),):
        raise ValueError(f"thesis vector must have shape ({len(THESIS_AXES)},), got {thesis.vector.shape}")
    if not np.all(np.isfinite(thesis.vector)):
        raise ValueError("thesis vector must contain finite values")
    if np.any(thesis.vector < 0.0) or np.any(thesis.vector > 1.0):
        raise ValueError("thesis vector entries must be in [0, 1]")

    job_graph = infer_job_graph(thesis.vector)
    targets = {r: derive_genome(thesis.vector, r) for r in ROLES}
    genomes = {r: Genome(vector=targets[r].copy()) for r in ROLES}

    grammar = derive_grammar(thesis.vector)
    genomes["display"].case = grammar["claim_case"]
    genomes["instrument"].case = grammar["label_case"]
    genomes["instrument"].tracking = grammar["label_tracking"]
    for g in genomes.values():
        g.label_mechanism = grammar["label_mechanism"]

    selection = select_font_system(
        catalog=catalog,
        targets=targets,
        thesis=thesis.vector,
        mode=pairing_mode,
        job_graph=job_graph,
        candidate_k=candidate_k,
    )
    fonts = selection["fonts"]

    if ratio_name in (None, "auto"):
        ratio_name = choose_scale_ratio(thesis.vector)
    _, ratio_val = scale_ratio(ratio_name)

    policy = derive_hierarchy_policy(thesis.vector, targets["reading"], targets["display"])
    policy.body["size"] = base_px
    hierarchy = build_hierarchy(list(level_names), policy.body, ratio_val, policy.per_level_delta)

    intent_signature = make_signature(targets, grammar, hierarchy, ratio_val)
    realized_vectors = {r: fonts[r].vector for r in ROLES}
    realized_signature = make_signature(realized_vectors, grammar, hierarchy, ratio_val)

    realized_genomes = {r: fonts[r].as_genome() for r in ROLES}
    validation = {
        "intent_role_truth": role_truth(genomes),
        "realized_role_truth": role_truth(realized_genomes),
        "intent_inter_test": {r: inter_test(genomes[r]) for r in ROLES},
        "realized_inter_test": {r: inter_test(realized_genomes[r]) for r in ROLES},
        "system_identity_test": system_identity_score(fonts),
        "realization_error": realization_error(targets, fonts),
        "feature_validation": feature_validation(job_graph, fonts),
        "realized_signature_collision": collision_similarity(realized_signature, history or []),
    }

    return {
        "typography_engine": "mbd-type/0.2",
        "thesis": thesis.label,
        "thesis_axes": {k: round(float(v), 3) for k, v in zip(THESIS_AXES, thesis.vector)},
        "job_graph": job_graph,
        "pairing_mode": pairing_mode.value,
        "selection": {
            "families": selection["families"],
            "score": selection["score"],
            "details": selection["details"],
            "alternatives": selection["alternatives"],
            "candidate_counts": selection["candidate_counts"],
        },
        "intent": {r: genomes[r].as_dict() for r in ROLES},
        "realized": {
            r: {
                "family": fonts[r].family,
                "classes": fonts[r].classes,
                "license": fonts[r].license,
                "supports": fonts[r].supports,
                "provenance": fonts[r].provenance,
                "traits": {axis: round(float(v), 3) for axis, v in zip(AXES, fonts[r].vector)},
            }
            for r in ROLES
        },
        "scale": {
            "base_px": base_px,
            "ratio": round(ratio_val, 6),
            "source": SCALE_LABELS[ratio_name],
            "policy": ratio_name,
        },
        "hierarchy_policy": {
            "body": {k: round(float(v), 4) for k, v in policy.body.items()},
            "per_level_delta": {k: round(float(v), 4) for k, v in policy.per_level_delta.items()},
            "rationale": policy.rationale,
        },
        "hierarchy": [asdict(lv) for lv in hierarchy],
        "grammar": grammar,
        "validation": validation,
        "intent_signature": intent_signature.round(5).tolist(),
        "realized_signature": realized_signature.round(5).tolist(),
        "catalog_status": "ILLUSTRATIVE_TRAITS_NOT_MEASURED_FROM_FONT_FILES",
    }


# ----------------------------------------------------------------------------
# 8. Self-tests + demo
# ----------------------------------------------------------------------------

def _mode_causality_catalog(thesis: np.ndarray) -> FontCatalog:
    """Small synthetic catalog whose distances make relationship modes diverge.

    This is test data, not a recommendation catalog.
    """
    reading = derive_genome(thesis, "reading")
    instrument = derive_genome(thesis, "instrument")
    display = derive_genome(thesis, "display")

    def clipped(v):
        return np.clip(np.asarray(v, dtype=float), 0, 1)

    # Two reading choices, two instrument choices, and three display choices.
    # The display variants deliberately occupy different relationship bands.
    return FontCatalog([
        Font("Test Reading A", "TEST", ["sans"], clipped(reading),
             {"italic": True, "variable": False, "tabular_nums": True}, "synthetic-test"),
        Font("Test Reading B", "TEST", ["serif"], clipped(reading + 0.08),
             {"italic": True, "variable": False, "tabular_nums": True}, "synthetic-test"),
        Font("Test Instrument A", "TEST", ["mono"], clipped(instrument),
             {"italic": True, "variable": False, "tabular_nums": True}, "synthetic-test"),
        Font("Test Instrument B", "TEST", ["sans"], clipped(instrument + 0.12),
             {"italic": True, "variable": False, "tabular_nums": True}, "synthetic-test"),
        Font("Test Display Close", "TEST", ["sans"], clipped(instrument + 0.02),
             {"italic": True, "variable": False, "tabular_nums": True}, "synthetic-test"),
        Font("Test Display Mid", "TEST", ["sans"], clipped((display + instrument) / 2 + 0.10),
             {"italic": True, "variable": False, "tabular_nums": True}, "synthetic-test"),
        Font("Test Display Far", "TEST", ["serif"], clipped(1.0 - instrument * 0.72),
             {"italic": True, "variable": False, "tabular_nums": True}, "synthetic-test"),
    ])


def run_self_tests() -> dict:
    blueline = ProjectThesis(
        label="instrument lettering crossed with scientific editorial",
        vector=np.array([0.85, 0.55, 0.15, 0.75, 0.10, 0.05]),
    )

    # 1. Pairing mode must be causal, not advisory.
    test_catalog = _mode_causality_catalog(blueline.vector)
    mode_families = {}
    for mode in PairingMode:
        packet = run_engine(blueline, test_catalog, mode, ratio_name="technical_compact", candidate_k=7)
        mode_families[mode.value] = tuple(packet["selection"]["families"][r] for r in ROLES)
    unique_systems = set(mode_families.values())
    assert len(unique_systems) >= 2, f"pairing modes did not change selection: {mode_families}"

    # 2. The literal generic baseline must fail the Inter Test.
    assert inter_test(INTER_BASELINE)["pass"] is False

    # 3. Realized collision guard must catch a near-duplicate.
    packet = run_engine(blueline, DEMO_CATALOG, PairingMode.COUNTERPOINT, ratio_name="technical_compact")
    sig = np.array(packet["realized_signature"])
    rng = np.random.default_rng(0)
    near = sig + rng.normal(0, 0.002, size=sig.shape)
    collision = collision_similarity(sig, [near])
    assert collision["pass"] is False and collision["max_similarity"] > 0.99

    # 4. Hierarchy policy must move with project thesis, not remain fixed.
    gallery = ProjectThesis(
        label="gallery wall: titles, prose, accession records",
        vector=np.array([0.20, 0.70, 0.55, 0.35, 0.65, 0.15]),
    )
    g = run_engine(gallery, DEMO_CATALOG, PairingMode.UTILITY, ratio_name="editorial_hierarchy")
    assert packet["hierarchy_policy"]["per_level_delta"] != g["hierarchy_policy"]["per_level_delta"]

    # 5. Intent and realized validation are explicitly separate surfaces.
    assert "intent_inter_test" in packet["validation"]
    assert "realized_inter_test" in packet["validation"]
    assert "realization_error" in packet["validation"]

    return {
        "pass": True,
        "mode_causality": mode_families,
        "unique_mode_systems": len(unique_systems),
        "near_duplicate_similarity": collision["max_similarity"],
        "technical_hierarchy_delta": packet["hierarchy_policy"]["per_level_delta"],
        "gallery_hierarchy_delta": g["hierarchy_policy"]["per_level_delta"],
    }


def _demo() -> None:
    blueline = ProjectThesis(
        label="instrument lettering crossed with scientific editorial",
        vector=np.array([0.85, 0.55, 0.15, 0.75, 0.10, 0.05]),
    )
    packet = run_engine(
        thesis=blueline,
        catalog=DEMO_CATALOG,
        pairing_mode=PairingMode.COUNTERPOINT,
        ratio_name="technical_compact",
    )
    print(json.dumps(packet, indent=2))

    print("\n--- mode sensitivity on the same thesis/catalog ---")
    sensitivity = {}
    for mode in PairingMode:
        p = run_engine(blueline, DEMO_CATALOG, mode, ratio_name="technical_compact")
        sensitivity[mode.value] = {
            "families": p["selection"]["families"],
            "score": p["selection"]["score"],
        }
    print(json.dumps(sensitivity, indent=2))


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        print(json.dumps(run_self_tests(), indent=2))
    else:
        _demo()
