"""Controlled vocabulary constants and configuration for CVS action descriptions."""

CRITERION_DEFINITIONS = {
    "C1": "Two and only two tubular structures are visible entering the gallbladder.",
    "C2": "The hepatocystic triangle is cleared of fat and fibrous tissue.",
    "C3": "The lower third of the gallbladder is detached from the liver bed.",
}

# Derived: lowercase lookup for matching column prefixes (c1 -> C1)
_CRIT_DEF_LOWER = {k.lower(): (k, v) for k, v in CRITERION_DEFINITIONS.items()}

# =========================
# Controlled vocab
# =========================

VERBS = [
    "Grasp", "Retract", "Dissect", "Coagulate", "Clip", "Cut",
    "Aspirate", "Irrigate", "Pack", "Null"
]

TOOLS = [
    "grasper", "scissors", "bipolar", "hook", "clipper",
    "specimen bag", "irrigator", "Null"
]

# Items = what is manipulated/removed (OBJECT / MATERIAL).
# Priority order matters: model should prefer earlier entries if applicable.
ITEMS_PRIORITY = [
    # CVS-relevant structures
    "Gallbladder",
    "Cystic-duct",
    "Cystic-artery",
    "Blood-vessel",

    # Organs / major structures (only if truly being manipulated)
    "Liver",
    "Omentum",
    "Peritoneum",
    "Gastrointestinal tract",
    "Gut",

    # Materials / content commonly dissected/cleared
    "Fat",
    "Connective tissue",
    "Blood",
    "Fluid",

    # Devices / misc
    "Specimen-bag",

    # fallback
    "Null",
]

# Regions = WHERE the action happens (PLACE / FIELD).
# Priority order matters: model should prefer earlier entries if applicable.
REGIONS_PRIORITY = [
    "Hepatocystic triangle",
    "Gallbladder neck/infundibulum",
    "Gallbladder fundus/body",
    "Liver bed",
    "Abdominal-wall or cavity",
    "Unclear",
]

SCENE_FLAGS = [
    "clear scene",
    "occlusion",
    "bleeding",
    "crowded",
    "smoke",
    "blurring",
    "reflection",
    "trocar/net",
    "fouled lens",
]

DIRECTIONS = [
    "upward", "cephalad", "lateral", "medial", "anterior", "posterior", "unclear"
]

ALLOW_OTHER_ITEMS = True    # allows item="Other:<short name>" when not in list
ALLOW_OTHER_REGIONS = True  # allows region="Other:<short name>" when not in list

# =========================
# Default configuration
# =========================

CRITERIA = ["c1", "c2", "c3"]
RATERS = ["rater1", "rater2", "rater3"]
GAUSSIAN_SIGMA = 1.0
FRAME_STEP = 30  # extracted frames are every 30 frames (1s at 30fps)
COLS_PER_ROW = 6
