# CVS-Act Action Taxonomy Annotation Guide

This guide explains how to annotate sentences using the action taxonomy and the additional columns in the sorted file.

## Columns

- `action_code`: main action label
- `actor_role`: left_instrument / right_instrument / camera / offscreen / unknown
- `tool_type`: fixed enum
- `target_structure`: where the action is applied
- `target_relation`: on / near / between / unknown
- `direction`: cephalad/upward, lateral, medial, anterior, posterior, inferior/downward (may be combined)
- `view_change`: none / rotated/new_surface_exposed / posterior_view / new_surface_exposed
- `effect`: improve_exposure / clear_view / delineate_structures / reveal_detachment / remove_occlusion / enable_regrasp / hemostasis / confirm_identity / unknown
- `landmark_confidence`: high / medium / low / unknown
- `causal_for_mind_change`: (blank for humans) true/false

## Tool type rules

### Tool type assignment (fixed enum)
- **Grasper**: sentence mentions “grasper” (left/right/off-screen).
- **Bipolar**: sentence mentions “bipolar”.
- **Hook**: sentence mentions “hook”.
- **Scissors**: sentence mentions “scissors”.
- **Clipper**: sentence mentions “clipper” / “clip applier”.
- **Irrigator**: sentence mentions “irrigator” or “suction”; includes suction–irrigator.
- **Unknown**: none of the above (including all camera sentences).

## Anatomy identification rules

### Cystic duct vs cystic artery (do not guess)
Annotate **CysticDuct** / **CysticArtery** *only* when the sentence explicitly names them **and** the wording indicates confidence:
- `landmark_confidence=high`: “confirmed”, “clearly identified”, “helps confirm identity”.
- `landmark_confidence=low`: “presumed”, “suspected”, “apparent”.
- `landmark_confidence=medium`: explicitly named without hedging.

If identity is not explicit/clear:
- Use **TwoTubularStructures** when the sentence refers to “two tubular structures”.
- Otherwise use **HepatocysticTriangle**.

Never infer duct vs artery from context alone.

## Action codes

### DISSECT_TRIANGLE

**When to use:** Use when the instrument is removing fat/fibrous/connective tissue in the hepatocystic triangle (Calot’s triangle) to improve exposure/skeletonize structures.


**Typical phrasing:** Includes: “dissects connective tissue/fat in the hepatocystic triangle”, “clears the triangle”, “skeletonizing the structures.”


**Corner cases / tie-breakers:** If the sentence specifies cystic duct/artery *and it is visually certain*, set target_structure=CysticDuct/CysticArtery with landmark_confidence=high/medium. If it says presumed/suspected, keep landmark_confidence=low. If identity is unclear, prefer target_structure=HepatocysticTriangle or TwoTubularStructures.


### DISSECT_LIVER_BED

**When to use:** Use when the action is separating the gallbladder from the liver bed / gallbladder–liver interface (C3-related).


**Typical phrasing:** Includes: “dissects connective tissue from/at the liver bed”, “separating the lower third… from the liver bed”, “gallbladder is detached from liver bed.”


**Corner cases / tie-breakers:** If the sentence is only retraction that *reveals* detachment (no dissection), still annotate the primary action (often RETRACT_*) and set effect=reveal_detachment.


### DISSECT_ISOLATE_TUBULAR_STRUCTURES

**When to use:** Use when the dissection is explicitly described as separating/isolating the two tubular structures (duct vs artery) rather than generic triangle clearing.


**Typical phrasing:** Includes: “separating the two tubular structures”, “isolate tubular structures.”


**Corner cases / tie-breakers:** If the sentence uses “two tubular structures” but not “dissect/separate”, don’t use this code—use the retraction/camera code and set target_structure=TwoTubularStructures.


### COAGULATE_HEMOSTASIS

**When to use:** Use when the instrument (often Bipolar/Hook) is coagulating tissue or achieving hemostasis (energy application) rather than cutting/dissecting.


**Typical phrasing:** Includes: “coagulates…”, “achieves hemostasis”, “cautery coagulates…”


**Corner cases / tie-breakers:** If the sentence says “hook cautery dissects…”, prefer DISSECT_TRIANGLE unless it is clearly about hemostasis.


### RETRACT_UPWARD

**When to use:** Use when a grasper provides cephalad/upward traction to expose the field/triangle/liver bed.


**Typical phrasing:** Includes: “retracts upward/cephalad”, “maintains upward retraction.”


**Corner cases / tie-breakers:** If the sentence also mentions rotation as the main change (posterior exposure), use RETRACT_ROTATE_VIEW and keep direction=cephalad/upward as an attribute.


### RETRACT_LATERAL

**When to use:** Use when traction is lateral (left/right) to open the triangle or expose a dissection plane.


**Typical phrasing:** Includes: “retracts laterally”, “stable lateral retraction”, “toward patient’s left/right.”


**Corner cases / tie-breakers:** If the target is not the gallbladder (e.g., peritoneum/tissue), keep this code but set target_structure=PeritoneumOrSoftTissue.


### RETRACT_ROTATE_VIEW

**When to use:** Use when the key action is changing traction to rotate the gallbladder and/or shift exposure to a new surface (often posterior).


**Typical phrasing:** Includes: “rotates it… shifting exposure to posterior surface”, “shifts from upward→lateral traction”, “exposes posterior view/surface.”


**Corner cases / tie-breakers:** Set view_change=rotated/new_surface_exposed. If posterior is explicitly stated, keep it in view_change but avoid claiming anterior↔posterior unless the sentence clearly states it.


### REGRASP

**When to use:** Use when the sentence explicitly says the grasper regrasped (changed grasp point).


**Typical phrasing:** Includes: “allowing the left grasper to regrasp…”


**Corner cases / tie-breakers:** If regrasp is only implied, do not use this code; put enable_regrasp in effect (or leave unknown).


### CAMERA_REPOSITION

**When to use:** Use for camera pan/rotate/tilt/recenter movements not primarily described as zoom in/out.


**Typical phrasing:** Includes: “camera rotates”, “pans and rotates”, “repositions for clearer view.”


**Corner cases / tie-breakers:** If the sentence says “zooms in/out”, use CAMERA_ZOOM_IN/OUT instead (even if it also rotates).


### CAMERA_ZOOM_IN

**When to use:** Use when the camera zooms in/centers for closer view.


**Typical phrasing:** Includes: “zooms in”, “centers … for a clearer view.”


**Corner cases / tie-breakers:** If both zoom and pan/rotate are present, choose zoom as the action code and keep pan/rotate as implicit in effect/notes if needed.


### CAMERA_ZOOM_OUT

**When to use:** Use when the camera zooms out or pulls back for a wider contextual view.


**Typical phrasing:** Includes: “zooms out”, “pulls back”, “wider view.”


**Corner cases / tie-breakers:** If the sentence is just “pulls back for visibility” treat as zoom out.


### CAMERA_STABILIZE

**When to use:** Use when the camera stabilizes from shakiness.


**Typical phrasing:** Includes: “stabilizes from a shaky motion.”


**Corner cases / tie-breakers:** If you want fewer codes, you can map this to CAMERA_REPOSITION and set effect=stabilize.


### TOOL_WITHDRAW_UNBLOCKS_VIEW

**When to use:** Use when a tool moves away/withdraws and the main effect is removing occlusion (unblocking the view).


**Typical phrasing:** Includes: “withdraws… stop occluding”, “moves away from blocking the view.”


**Corner cases / tie-breakers:** If the withdrawal is primarily to pick up a clip, still use this code *if* the sentence emphasizes improved visibility/unblocked view.


### IRRIGATOR_CLEAR_FIELD

**When to use:** Use when the irrigator/suction clears blood/fluid/debris to improve visibility to the field/triangle.


**Typical phrasing:** Includes: “aspirates blood”, “evacuates fluid and debris”, “clears the field.”


**Corner cases / tie-breakers:** If the irrigator provides countertraction specifically to enable regrasp, use IRRIGATOR_COUNTERTRACTION_ASSIST.


### IRRIGATOR_COUNTERTRACTION_ASSIST

**When to use:** Use when the irrigator provides temporary countertraction/stabilization to enable another action (usually grasper regrasp).


**Typical phrasing:** Includes: “provides temporary countertraction… allowing the left grasper to regrasp.”


**Corner cases / tie-breakers:** If it only aspirates fluid without assisting regrasp, use IRRIGATOR_CLEAR_FIELD.


### ICG_SWITCH

**When to use:** Use when the view switches to ICG fluorescence (imaging mode change).


**Typical phrasing:** Includes: “switches to ICG fluorescence.”


**Corner cases / tie-breakers:** Set effect=confirm_identity when stated. Do not infer anatomy identity unless explicitly described.


## General corner cases

### General corner-case rules
1) **Multi-verb sentences**: pick the *dominant* action as `action_code`; encode the rest as attributes:
   - Example: “retracts upward and rotates… posterior surface” → RETRACT_ROTATE_VIEW (direction=cephalad/upward).
2) **Exposure vs action**: If a sentence’s main information is “a clearer view” caused by camera or tool withdrawal, code that (CAMERA_* or TOOL_WITHDRAW_UNBLOCKS_VIEW), not the implied dissection.
3) **Off-screen tools**: use `actor_role=offscreen`. Use tool_type only if explicitly stated; otherwise Unknown.
4) **Causal vs incidental**: use `causal_for_mind_change` as a human label:
   - `true` if the sentence plausibly explains the belief change (e.g., triangle clearing for C2; liver bed separation for C3).
   - `false` for purely visibility management or incidental repositioning, unless the belief change is about visibility.
