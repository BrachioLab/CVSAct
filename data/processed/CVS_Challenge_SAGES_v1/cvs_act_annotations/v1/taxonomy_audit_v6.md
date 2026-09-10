# Surgical Intention Annotation Guide (v6)

This guide defines each intention label with precise operational rules.
The ontology encodes **goal-directed anatomical state changes**, not just tool motion.

---

# GLOBAL PRINCIPLES

1. Label the **intended anatomical goal**, not just the motion.
2. Distinguish:
   - Fluid obstruction → clear blood/fluid
   - Camera movement → improve visualization
   - New exposure state → expose
   - Sustained exposure → maintain
   - Active tissue removal → clear / skeletonize / separate
3. C3 plane → cystic plate
4. HCT plane → hepatocystic triangle
5. Duct + artery together → two tubular structures
6. Single structure targeted → skeletonize specific structure

---

# 1️⃣ Achieve hemostasis

**Definition:** Actively stop bleeding.

**Use when:**
- Coagulation applied to bleeding site.
- Primary goal is bleeding control.

**Do NOT use when:**
- Energy is used primarily for dissection.

---

# 2️⃣ Assist retraction (allow regrasp / countertraction)

**Definition:** Temporary assistance to enable traction setup or regrasp.

**Use when:**
- Irrigator pushes to enable repositioning.
- Brief countertraction to allow tool adjustment.

**Do NOT use when:**
- Sustained traction for exposure.

---

# 3️⃣ Clear blood/fluid to improve visualization

**Definition:** Remove fluid obscuring the field.

**Use when:**
- Aspirating blood.
- Irrigation-suction clearing pooled fluid.

**Do NOT use when:**
- Camera movement improves view.
- Retraction improves view.

---

# 4️⃣ Improve visualization of the surgical field

**Definition:** Camera adjustment or tool withdrawal to improve view angle.

**Use when:**
- CAMERA_ZOOM_IN / OUT
- CAMERA_REPOSITION
- TOOL_WITHDRAW_UNBLOCKS_VIEW

**Do NOT use when:**
- Fluid is cleared (use clear blood/fluid).
- Tissue retraction creates anatomical exposure (use expose/maintain).

---

# 5️⃣ Clear the hepatocystic triangle

**Definition:** Remove fat/fibrous tissue within HCT boundaries.

**Use when:**
- Hook or Maryland dissecting triangle tissue.
- Goal is tissue clearance.

**Do NOT use when:**
- Only retracting to reveal the triangle.

---

# 6️⃣ Expose the hepatocystic triangle

**Definition:** Create or significantly improve anatomical exposure of the HCT.

**Use when:**
- Initial traction opens triangle.
- Gallbladder flipped (medial ↔ lateral) to newly reveal triangle.
- Retraction angle change improves visibility.
- Previously hidden boundaries become visible.

**Do NOT use when:**
- Simply holding stable traction (then maintain).

**Key Rule:**  
If the action changes what we can see → Expose.

---

# 7️⃣ Maintain exposure of the hepatocystic triangle

**Definition:** Sustain already established HCT exposure.

**Use when:**
- Stable lateral/upward traction.
- No meaningful change in visibility.

**Key Rule:**  
Holding exposure steady → Maintain.

---

# 8️⃣ Expose the cystic plate

**Definition:** Create or expand exposure of the gallbladder–liver interface (C3 plane).

**Use when:**
- Traction enables beginning liver bed dissection.
- Retraction reveals cystic plate plane.
- Exposure state of C3 improves.

**Do NOT use when:**
- Still working primarily in HCT.

---

# 9️⃣ Maintain exposure of the cystic plate

**Definition:** Sustain traction enabling ongoing C3 dissection.

**Use when:**
- Stable traction during detachment.
- No change in exposure state.

---

# 🔟 Expose the two tubular structures

**Definition:** First clear visual isolation of cystic duct and artery together.

**Use when:**
- Both structures become distinctly visible.
- Exposure state reveals them as separate entities.

**Do NOT use when:**
- Actively dissecting them (then skeletonize).

---

# 1️⃣1️⃣ Maintain exposure of the two tubular structures

**Definition:** Sustain visibility of already identified duct and artery.

**Use when:**
- Stable traction maintaining CVS view.

---

# 1️⃣2️⃣ Separate the gallbladder from the cystic plate

**Definition:** Actively dissect GB off liver bed.

**Use when:**
- Hook cutting between GB and cystic plate.
- Tissue plane actively divided in C3.

**Do NOT use when:**
- Only retracting (then expose cystic plate).

---

# 1️⃣3️⃣ Skeletonize the cystic artery

**Definition:** Remove surrounding tissue to isolate the cystic artery.

**Use when:**
- Dissection focused specifically on artery.

---

# 1️⃣4️⃣ Skeletonize the cystic duct

**Definition:** Remove surrounding tissue to isolate the cystic duct.

---

# 1️⃣5️⃣ Skeletonize the two tubular structures

**Definition:** Simultaneous dissection isolating duct and artery together.

**Use when:**
- Dissection within HCT clarifying both structures.
- CVS isolation phase.

---

# 1️⃣6️⃣ Improve visualization and help confirm identity of the biliary structures

**Definition:** Use modality change (e.g., ICG) to confirm structure identity.

**Use when:**
- ICG fluorescence activation.
- Imaging-based identity confirmation.

---

# DECISION SHORTCUT

Ask in order:

1. Is fluid obstructing view? → Clear blood/fluid
2. Is camera moving? → Improve visualization
3. Is tissue being removed?  
   - In HCT → Clear HCT / Skeletonize  
   - In C3 → Separate GB from cystic plate
4. Is traction changing exposure state? → Expose
5. Is traction holding steady? → Maintain
6. Is identity being confirmed via imaging? → Improve visualization & confirm identity

---

This guide encodes **anatomical state transitions**, not just instrument motion.