# 🔑 Keys to Reliable Cross-Validation (RSNA & Beyond)

A strong cross-validation (CV) setup is the **backbone of leaderboard success** in RSNA and any Kaggle competition.  
Below are the 10 golden rules for building a **trustworthy and LB-correlated CV**.

---

## 1. Stratification + Grouping
- ✅ Use **`StratifiedGroupKFold`**:
  - **Stratify** on `Aneurysm Present` → balances positives/negatives per fold.
  - **Group** by `patient_id` → prevents leakage (same patient in both train & val).
- ⚠️ Without grouping → CV will leak and look artificially inflated.

---

## 2. Multiple Seeds
- One split with `random_state=42` is **not enough**.
- Run CV with **3–5 different seeds** and average results.
- This reduces variance caused by “lucky” or “unlucky” splits.
- ✅ Final score = **mean OOF across seeds**.

---

## 3. OOF Predictions (Out-of-Fold)
- Always save **OOF predictions** (`shape = [N, num_labels]`).
- Benefits:
  - Recompute RSNA AUC **offline exactly**.
  - Essential for **ensembling/stacking** later.
- ✅ Strong OOF ↔ Public LB correlation = reliable CV.

---

## 4. Consistent Preprocessing in CV & Submission
- Do **not preprocess test differently** from train.
- If you use `.npz` conversion → apply same logic in CV.
- ⚠️ Avoid mismatches (e.g. train with `64` slices, predict with `32` slices).

---

## 5. Patient-Level Holdout Test
- After CV, hold out **5–10% patients** (never touched in CV).
- Train on the rest, evaluate once on holdout.
- ✅ This provides a **true generalization estimate** before submitting to Kaggle.

---

## 6. Monitor Per-Class AUC
- RSNA metric: `"Aneurysm Present"` weighted **13× more**.
- But track **all 14 AUCs** in validation.
- Improving small classes often stabilizes LB correlation.

---

## 7. Blend k-Folds Instead of Choosing Best Fold
- ❌ Never pick “Fold 3 looked best” → that’s **cherry-picking**.
- ✅ Train separate models for each fold.
- ✅ Average (or weighted average) predictions across folds.
- This reduces fold noise and matches Kaggle ensemble behavior.

---

## 8. Reproduce LB Distribution
- LB = evaluated on **hidden patients**.
- Replicate the **same patient scan distribution** in CV:
  - Example: Some patients have 20 scans, others only 3.
  - Keep ratios balanced across folds.

---

## 9. Error Analysis
- After CV:
  - Inspect **worst patients/scans**.
  - Identify **systematic failure modes** (e.g. consistent errors in Basilar Tip).
- ✅ Fixing systematic issues often yields a **bigger LB jump** than tuning layers.

---

## 10. Check Correlation with LB
- After first submissions:
  - Compare **OOF vs Public LB**.
  - Track correlation coefficient (**ρ**).
- ✅ High correlation (`ρ > 0.9`) → CV is reliable.
- ⚠️ Low correlation (`ρ < 0.7`) → redesign CV split strategy.

---

## 📌 Summary
Reliable CV =  
**Stratified + Grouped Splits** ➝ **Multiple Seeds** ➝ **OOF Predictions** ➝ **Consistent Preprocessing** ➝ **Holdout Test** ➝ **Error Analysis**.  

Master this cycle → your offline validation will **mirror Kaggle LB** and you’ll iterate with confidence.
