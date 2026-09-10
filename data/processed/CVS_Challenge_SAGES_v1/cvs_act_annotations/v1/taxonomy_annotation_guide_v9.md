# CVS-Act Action Taxonomy v9 — Annotation Guide

## Overview

v9 builds on v8 by **removing all “maintain” intentions**. The taxonomy now
covers only *active* surgical actions that change the state of the operative
field. Passive holding / stable traction is implicitly assumed whenever a
grasper is visible and retracted.

### Removed intentions

- ~~maintain exposure of the cystic plate~~
- ~~maintain exposure of the hepatocystic triangle~~
- ~~maintain exposure of the two tubular structures~~

If the *only* action in a frame interval is passive holding, the
`actions_ranked` list will be empty for that interval.

---

## Fields

### Actor Role

Who performs the action.

| Value | Count | Description |
|-------|------:|-------------|
| `right_instrument` | 171 |  |
| `left_instrument` | 99 |  |
| `camera` | 91 |  |
| `unknown` | 6 |  |
| `top_instrument` | 4 |  |
| `offscreen` | 1 |  |

### Tool Type

Which instrument is used.

| Value | Count | Description |
|-------|------:|-------------|
| `Hook` | 109 |  |
| `Grasper` | 103 |  |
| `camera` | 77 |  |
| `Maryland` | 40 |  |
| `(not set)` | 17 |  |
| `Irrigator` | 12 |  |
| `Unknown` | 2 |  |
| `Scissors` | 2 |  |

### Action Code

The dominant surgical action.

| Value | Count | Description |
|-------|------:|-------------|
| `DISSECT` | 131 |  |
| `RETRACT_LATERAL` | 63 |  |
| `CAMERA_ZOOM_OUT` | 42 |  |
| `CAMERA_REPOSITION` | 25 |  |
| `TOOL_WITHDRAW_UNBLOCKS_VIEW` | 25 |  |
| `CAMERA_ZOOM_IN` | 24 |  |
| `RETRACT_MEDIAL_TO_LATERAL` | 15 |  |
| `RETRACT_UPWARD` | 13 |  |
| `IRRIGATOR_ASPIRATE` | 9 |  |
| `RETRACT_MEDIAL` | 6 |  |
| `RETRACT_LATERAL_TO_MEDIAL` | 6 |  |
| `ICG_SWITCH` | 5 |  |
| `IRRIGATOR_COUNTERTRACTION_ASSIST` | 3 |  |
| `COAGULATE_HEMOSTASIS` | 2 |  |
| `RETRACT_DOWNWARD` | 1 |  |

### Target Structure

The anatomical structure physically manipulated.

| Value | Count | Description |
|-------|------:|-------------|
| `HepatocysticTriangle` | 180 |  |
| `GallbladderNeck_Infundibulum` | 101 |  |
| `CysticPlate` | 24 |  |
| `CysticArtery` | 19 |  |
| `CysticDuct` | 12 |  |
| `TwoTubularStructures` | 12 |  |
| `Gallbladder` | 9 |  |
| `BloodOrFluid` | 6 |  |
| `tissue between the cystic plate and the gallbladder` | 4 |  |
| `TissueNearCysticArtery` | 3 |  |

### Target Context

Spatial context describing where in the anatomy the action occurs.

| Value | Count | Description |
|-------|------:|-------------|
| `between presumed cystic duct and presumed cystic artery` | 58 |  |
| `between cystic artery and cystic plate` | 35 |  |
| `near the base of the hepatocystic triangle` | 18 |  |
| `between cystic artery and liver bed` | 7 |  |
| `moving from a posterior view to an anterior view` | 7 |  |
| `close to the gallbladder neck` | 7 |  |
| `moving from an anterior view to a posterior view` | 5 |  |
| `near the cystic duct` | 4 |  |
| `behind the two tubular structures` | 4 |  |
| `the position of the grasper moved more towards the the gallbladder body` | 1 |  |
| `on the side near the cystic plate` | 1 |  |
| `near cystic duct` | 1 |  |

### Intention

The surgical goal of this action.

| Value | Count | Description |
|-------|------:|-------------|
| `expose the hepatocystic triangle` | 81 | Create or significantly improve anatomical exposure of the HCT. |
| `clear the hepatocystic triangle` | 70 | Remove fat/fibrous tissue within HCT boundaries. |
| `expose the cystic plate` | 43 | Create or expand exposure of the gallbladder-liver interface (C3 plane). |
| `obtain a contextualized view` | 42 | Camera zooms out to show the surgical field in wider context. |
| `skeletonize the two tubular structures` | 31 | Simultaneous dissection isolating duct and artery together. |
| `adjust the viewing angle` | 25 | Camera repositions to change the viewing perspective on the surgical field. |
| `obtain a close-up focused view` | 24 | Camera zooms in to get a detailed view of the area of interest. |
| `expose the two tubular structures` | 11 | First clear visual isolation of cystic duct and artery together. |
| `clear blood/fluid to improve visualization` | 9 | Remove fluid obscuring the field (aspiration / irrigation-suction). |
| `skeletonize the presumed cystic artery` | 8 | Remove surrounding tissue to isolate the presumed cystic artery. |
| `improve visualization of the surgical field` | 6 | Tool withdraws from the field to unblock the camera's view of the surgical area. |
| `improve visualization and help confirm the identity of the biliary structures` | 5 | Use modality change (e.g. ICG) to confirm structure identity. |
| `assist retraction (allow regrasp / countertraction)` | 5 | Temporary assistance to enable traction setup or regrasp. |
| `achieve hemostasis` | 2 | Actively stop bleeding (coagulation applied to bleeding site). |
| `skeletonize the presumed cystic duct` | 1 | Remove surrounding tissue to isolate the presumed cystic duct. |

---

## Action Code → Natural Language

| Code | Phrase |
|------|--------|
| `CAMERA_REPOSITION` | repositions toward |
| `CAMERA_STABILIZE` | stabilizes on |
| `CAMERA_ZOOM_IN` | zooms in on |
| `CAMERA_ZOOM_OUT` | zooms out from |
| `COAGULATE_HEMOSTASIS` | coagulates |
| `DISSECT` | dissects the tissue around |
| `ICG_SWITCH` | switches to ICG view of |
| `IRRIGATOR_ASPIRATE` | aspirates |
| `IRRIGATOR_CLEAR_FIELD` | clears |
| `IRRIGATOR_COUNTERTRACTION_ASSIST` | provides countertraction on |
| `IRRIGATOR_GRASP` | grasps |
| `IRRIGATOR_IRRIGATE` | irrigates |
| `IRRIGATOR_PUSH_TISSUE` | pushes tissue near |
| `OTHER/UNKNOWN` | acts on |
| `REGRASP` | regrasps |
| `RETRACT_DOWNWARD` | retracts downward |
| `RETRACT_LATERAL` | retracts laterally |
| `RETRACT_LATERAL_TO_MEDIAL` | retracts from laterally to medially |
| `RETRACT_MEDIAL` | retracts medially |
| `RETRACT_MEDIAL_TO_LATERAL` | retracts from medially to laterally |
| `RETRACT_UPWARD` | retracts upward |
| `RETRACT_UPWARD_MEDIAL` | retracts upward and medially |
| `TOOL_WITHDRAW_UNBLOCKS_VIEW` | withdraws from view of |

---

## Common Presets

| Name | Actor | Tool | Action | Target | Intention |
|------|-------|------|--------|--------|-----------|
| left_grasper | left_instrument | Grasper | RETRACT_LATERAL | GallbladderNeck_Infundibulum | expose the hepatocystic triangle |
| camera_zoom_in | camera |  | CAMERA_ZOOM_IN | HepatocysticTriangle | obtain a close-up focused view |
| camera_zoom_out | camera |  | CAMERA_ZOOM_OUT | HepatocysticTriangle | obtain a contextualized view |
| right_hook_hct | right_instrument | Hook | DISSECT | HepatocysticTriangle | clear the hepatocystic triangle |
| right_hook_artery | right_instrument | Hook | DISSECT | CysticArtery | skeletonize the presumed cystic artery |
| right_hook_duct | right_instrument | Hook | DISSECT | CysticDuct | skeletonize the presumed cystic duct |

---

## Annotation Rules

1. Each frame interval gets an `actions_ranked` list ordered by importance.
2. If the only visible action is passive holding / stable traction, leave
   `actions_ranked` empty.
3. Pick **one** `action_code` per action entry (the dominant action).
4. `target_context_1` and `target_context_2` are optional spatial qualifiers.
   Use `(not set)` when not applicable.
5. Camera actions: `actor_role=camera`, `tool_type=camera`.
   Intention is determined by `action_code`:
   - `CAMERA_ZOOM_OUT` → "obtain a contextualized view"
   - `CAMERA_ZOOM_IN` → "obtain a close-up focused view"
   - `CAMERA_REPOSITION` → "adjust the viewing angle"

*Generated 2026-02-26 from taxonomy_v9.json*
