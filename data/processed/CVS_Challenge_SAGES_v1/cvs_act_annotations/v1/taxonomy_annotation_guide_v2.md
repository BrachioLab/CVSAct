# CVS-Act Action Taxonomy Annotation Guide (v2)

This guide explains how to annotate short action sentences in **CVS-Act** using a **single dominant Action Code**
plus structured attributes. The goal is **consistency**: different annotators should produce the same labels.

---

## 1. Core Principle (Most Important)

Each sentence should be mapped to:

1) **ONE** `action_code` (dominant action)  
2) supporting attributes (`tool_type`, `target_structure`, `direction`, etc.)

If a sentence contains multiple actions (e.g., retraction + rotation + exposure), **choose the dominant action** as the label
and encode the rest in attributes such as `direction`, `view_change`, and `effect`.

---

## 2. Columns (Fields)

### Required / main fields
- `action_code`: dominant action label (single choice)
- `actor_role`: who performed the action
- `tool_type`: which tool performed the action
- `target_structure`: what the tool physically manipulates / touches

### Supporting fields
- `target_relation`: on / near / between / unknown
- `direction`: cephalad/upward, lateral, medial, anterior, posterior, inferior/downward (may be combined)
- `view_change`: none / rotated/new_surface_exposed / posterior_view / new_surface_exposed
- `effect`: improve_exposure / clear_view / delineate_structures / reveal_detachment / remove_occlusion / enable_regrasp / hemostasis / confirm_identity / unknown
- `effect_structure`: the anatomical region whose visibility/clarity is improved
- `landmark_confidence`: high / medium / low / unknown
- `causal_for_mind_change`: human judgment (true/false)

---

## 3. UI Recommendation: Use Radio Dropdowns

All fields should be **single-choice dropdowns (radio)**.

This prevents annotators from selecting multiple incompatible labels.

The only field that may contain combinations is `direction`
(e.g., `cephalad/upward+lateral`), but it is still stored as **one value**.

---

## 4. Tool Type Rules (Fixed Enum)

Tool types must be one of:

- `Grasper`
- `Bipolar`
- `Hook`
- `Scissors`
- `Clipper`
- `Irrigator`
- `Unknown`

### Tool type assignment
- **Grasper**: sentence mentions “grasper”.
- **Bipolar**: sentence mentions “bipolar”.
- **Hook**: sentence mentions “hook”.
- **Scissors**: sentence mentions “scissors”.
- **Clipper**: sentence mentions “clipper” / “clip applier”.
- **Irrigator**: sentence mentions “irrigator” or “suction” (includes suction–irrigator).
- **Unknown**: none of the above (including all camera actions).

---

## 5. Actor Role Rules

Actor roles must be one of:

- `left_instrument`
- `right_instrument`
- `camera`
- `offscreen`
- `unknown`

### Assignment
- If sentence begins with “The camera …” → `camera`
- If sentence contains “off-screen”, “unseen” → `offscreen`
- If sentence says “left …” → `left_instrument`
- If sentence says “right …” → `right_instrument`
- Otherwise → `unknown`

---

## 6. Target vs Effect Structure (Important Distinction)

### 6.1 `target_structure` (WHAT is manipulated)
**Target structure = what the tool physically acts on.**

Examples:
- “grasper retracts the gallbladder upward” → `target_structure=Gallbladder`
- “hook dissects connective tissue in the hepatocystic triangle” → `target_structure=HepatocysticTriangle`
- “irrigator aspirates blood” → `target_structure=BloodOrFluid`

### 6.2 `effect_structure` (WHERE the benefit is observed)
**Effect structure = what region becomes clearer / more exposed.**

Example:
> “The left grasper retracts the gallbladder upward, exposing the hepatocystic triangle.”

Annotate:
- `target_structure = Gallbladder`
- `effect = improve_exposure`
- `effect_structure = HepatocysticTriangle`

### Recommended enum for `effect_structure`
- `HepatocysticTriangle`
- `TwoTubularStructures`
- `LiverBed`
- `Gallbladder`
- `GallbladderNeck_Infundibulum`
- `CysticDuct`
- `CysticArtery`
- `SurgicalFieldGeneral`
- `Unknown`

---

## 7. Anatomy Identification Rules (Do Not Guess)

### Cystic duct vs cystic artery
Annotate `CysticDuct` / `CysticArtery` ONLY if the sentence explicitly names them.

Set `landmark_confidence`:
- `high`: “confirmed”, “clearly identified”
- `medium`: explicitly named without hedging
- `low`: “presumed”, “suspected”, “apparent”
- `unknown`: not stated

If identity is unclear:
- Use `TwoTubularStructures` if sentence says “two tubular structures”.
- Otherwise use `HepatocysticTriangle`.

**Never infer duct vs artery from context alone.**

---

## 8. Action Codes

Below are the official action codes. Always select ONE.

---

### DISSECT_TRIANGLE

**Use when:** removing fat/fibrous/connective tissue in the hepatocystic triangle.

**Typical phrasing:**
- “dissects connective tissue in the hepatocystic triangle”
- “clears fat and connective tissue from the hepatocystic triangle”
- “skeletonizing the structures”

**Corner cases:**
- If the sentence says “near cystic artery/duct”, annotate target_structure as cystic artery/duct ONLY if explicit.
- If it says “suspected/presumed”, set landmark_confidence=low.

---

### DISSECT_LIVER_BED

**Use when:** separating gallbladder from liver bed (C3-related).

**Typical phrasing:**
- “dissects connective tissue from the liver bed”
- “separating the lower third of the gallbladder from the liver bed”
- “detached from the liver bed”

**Corner cases:**
- If detachment is only *revealed* by retraction, label RETRACT_* and set effect=reveal_detachment.

---

### DISSECT_ISOLATE_TUBULAR_STRUCTURES

**Use when:** dissection explicitly separates/isolate the two tubular structures.

**Typical phrasing:**
- “separating the two tubular structures”
- “isolating tubular structures”

**Corner cases:**
- If it only says “two tubular structures visible” without dissection, do NOT use this code.

---

### COAGULATE_HEMOSTASIS

**Use when:** coagulation/hemostasis is the main action.

**Typical phrasing:**
- “coagulates tissue”
- “achieves hemostasis”

**Corner cases:**
- “hook cautery dissects…” usually maps to DISSECT_TRIANGLE unless hemostasis is explicitly stated.

---

### RETRACT_UPWARD

**Use when:** grasper provides cephalad/upward traction.

**Typical phrasing:**
- “retracts upward”
- “maintains cephalad retraction”

**Corner cases:**
- If rotation/posterior exposure is emphasized, use RETRACT_ROTATE_VIEW.

---

### RETRACT_LATERAL

**Use when:** lateral traction to expose the triangle or dissection plane.

**Typical phrasing:**
- “retracts laterally”
- “toward patient’s left/right”

**Corner cases:**
- If retracting peritoneum/tissue, keep this code and set target_structure=PeritoneumOrSoftTissue.

---

### RETRACT_ROTATE_VIEW

**Use when:** traction change rotates gallbladder and shifts view (posterior/new surface).

**Typical phrasing:**
- “rotates it, shifting exposure to posterior surface”
- “changes traction, rotating the gallbladder”

**Corner cases:**
- Set `view_change=rotated/new_surface_exposed`.
- Do not claim anterior↔posterior unless explicitly stated.

---

### REGRASP

**Use when:** sentence explicitly says regrasp happened.

**Typical phrasing:**
- “allows the left grasper to regrasp”

**Corner cases:**
- If regrasp is implied but not explicit, do NOT use this code.

---

### CAMERA_REPOSITION

**Use when:** pan/rotate/tilt/recenter without explicit zoom in/out.

**Typical phrasing:**
- “camera rotates”
- “camera pans and rotates”
- “camera repositions”

**Corner cases:**
- If zoom is explicitly mentioned, use CAMERA_ZOOM_IN/OUT.

---

### CAMERA_ZOOM_IN

**Use when:** zooms in / centers for closer view.

**Typical phrasing:**
- “zooms in”
- “centers the hepatocystic triangle”

---

### CAMERA_ZOOM_OUT

**Use when:** zooms out / pulls back for wider view.

**Typical phrasing:**
- “zooms out”
- “pulls back”

---

### CAMERA_STABILIZE

**Use when:** stabilizes from shakiness.

**Typical phrasing:**
- “camera stabilizes from a shaky motion”

---

### TOOL_WITHDRAW_UNBLOCKS_VIEW

**Use when:** a tool withdraws/moves away to remove occlusion.

**Typical phrasing:**
- “withdraws from view to stop occluding”
- “moves away from blocking the view”

**Corner cases:**
- If tool withdraws to pick up a clip, still use this code if visibility improvement is emphasized.

---

### IRRIGATOR_CLEAR_FIELD

**Use when:** suction/irrigator clears blood/fluid/debris.

**Typical phrasing:**
- “aspirates blood”
- “evacuates fluid and debris”
- “clears the field”

---

### IRRIGATOR_COUNTERTRACTION_ASSIST

**Use when:** irrigator provides countertraction to assist regrasp or exposure.

**Typical phrasing:**
- “provides temporary countertraction… allowing the grasper to regrasp”

---

### ICG_SWITCH

**Use when:** view switches to ICG fluorescence mode.

**Typical phrasing:**
- “switches to ICG fluorescence”

**Corner cases:**
- Do not infer duct/artery identity unless explicitly stated.

---

## 9. General Corner-Case Rules

### Rule 1: Multi-verb sentences
Pick the dominant action.

Example:
> “retracts upward and rotates, exposing posterior surface”

→ `action_code=RETRACT_ROTATE_VIEW`  
→ `direction=cephalad/upward`  
→ `view_change=rotated/new_surface_exposed`

### Rule 2: Visibility vs anatomy manipulation
If the sentence is mostly about visibility:
- camera movement → CAMERA_*
- tool withdrawal → TOOL_WITHDRAW_UNBLOCKS_VIEW
- suction clearing → IRRIGATOR_CLEAR_FIELD

### Rule 3: Off-screen tools
If it says “unseen/off-screen grasper”, set:
- `actor_role=offscreen`
- `tool_type=Grasper` ONLY if explicitly stated; otherwise Unknown.

### Rule 4: Causal vs incidental
Annotate `causal_for_mind_change=true` if the action plausibly caused the belief change.

Examples:
- triangle clearing for C2 → likely causal
- liver bed separation for C3 → likely causal
- minor camera zoom-out → usually incidental (false)

---

## 10. Quick Reference: Target vs Effect

| Sentence snippet | action_code | target_structure | effect | effect_structure |
|---|---|---|---|---|
| retracts gallbladder upward, exposing triangle | RETRACT_UPWARD | Gallbladder | improve_exposure | HepatocysticTriangle |
| hook dissects tissue in triangle | DISSECT_TRIANGLE | HepatocysticTriangle | delineate_structures | HepatocysticTriangle |
| suction aspirates blood from triangle | IRRIGATOR_CLEAR_FIELD | BloodOrFluid | clear_view | HepatocysticTriangle |
| camera zooms out for broader view | CAMERA_ZOOM_OUT | Unknown | improve_exposure | SurgicalFieldGeneral |

---

**End of guide.**
