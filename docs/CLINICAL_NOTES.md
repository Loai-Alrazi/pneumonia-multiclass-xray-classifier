# Clinical Context & Academic Machine Learning Notes

---

## 1. Clinical Background: The Pediatric Pneumonia Triage Dilemma
Pneumonia remains a leading cause of pediatric morbidity and hospitalization worldwide. In anterior-posterior (AP) pediatric chest radiographs, differentiating between **bacterial** and **viral** pneumonia presents a central diagnostic challenge:
* **Bacterial Pneumonia:** Frequently presents with focal lobar or segmental consolidation, air bronchograms, and alveolar filling. Clinical practice requires prompt therapeutic antibiotic administration. Delayed therapy increases the risk of pleural effusion, empyema, and respiratory compromise.
* **Viral Pneumonia:** Characteristically presents with bilateral, diffuse peribronchial thickening, hyperinflation, and interstitial opacities. Management focuses on supportive oxygenation and hydration; antibiotics are clinically ineffective and their indiscriminate administration accelerates antimicrobial resistance (AMR).
* **Normal Controls:** Clear lung fields, sharp costophrenic angles, well-defined diaphragmatic domes, and normal cardiothoracic ratio.

---

## 2. Radiographic Findings & Distinguishing Features

| Radiographic Feature | Bacterial Pneumonia | Viral Pneumonia | Normal Pediatric Lung |
| :--- | :--- | :--- | :--- |
| **Parenchymal Infiltrate** | Focal, dense lobar/segmental consolidation | Diffuse, bilateral peribronchial/interstitial opacities | Clear lung parenchyma, no consolidations |
| **Air Bronchograms** | Frequently visualized within consolidations | Absent or rare | Absent |
| **Diaphragmatic Contours** | Often obscured adjacent to lower lobes (silhouette sign) | Typically preserved | Crisp, sharp bilateral diaphragmatic arches |
| **Typical Clinical Pathway** | Antibiotic administration & monitoring | Supportive care & antibiotic avoidance | Reassurance & discharge |

---

## 3. Machine Learning Observations & Generalization Considerations

In this academic study, two distinct modeling paradigms were evaluated on the pediatric cohort:

### A. The Controlled Experimental Baseline (Exp 2: Frozen MobileNetV2)
* **Architecture:** Pre-trained ImageNet features remain completely frozen; only the 3-class linear classification projection is trained with inverse-frequency class weights ($3,843$ trainable parameters).
* **Behavior:** By strictly constraining adaptation to a linear hyperplane over generic pre-trained representations, the model achieved balanced validation performance ($77.55\%$ Macro F1, $95.19\%$ Normal Recall, $61.75\%$ Viral Recall on Seed 42).
* **Historical Observation (exposed Test split; descriptive only):** On the historical Test cohort ($N=624$, $234$ Normal images), Exp 2 reached $67.52\%$ Normal recall ($158/234$) in the 3-seed ensemble and $64.53\%$ ($151/234$) on the single-seed checkpoint (whose weights are identical to the deployed Seed 42 baseline).

### B. The Validation-Optimized Variant (Exp 5: Top 4 Feature Blocks Fine-Tuned)
* **Architecture:** Unfreezes the top four feature blocks of MobileNetV2 (`features[15..18]`: three inverted residual blocks plus the final $1 \times 1$ convolution) with low-learning-rate fine-tuning ($5 \times 10^{-5}$ on the backbone, $5 \times 10^{-4}$ on the head, weight decay $10^{-4}$, cosine schedule). Fine-tuning also updates the BatchNorm statistics in these blocks.
* **Behavior:** Achieved the highest internal validation metrics ($79.56\%$ single seed, $79.79\%$ ensemble Macro F1; $97.41\%$ ensemble Normal Recall).
* **Historical Observation (exposed Test split; descriptive only):** On the historical Test benchmark, Exp 5's Normal recall declined to $54.27\%$ ($127/234$) in the 3-seed ensemble ($55.41\% \pm 6.25\%$ across individual seeds, and $53.42\%$ [$125/234$] on the early exploratory unsuffixed run), with a higher rate of healthy radiographs classified as viral pneumonia ($90$ misclassifications in ensemble).

### C. Scientific Interpretation & Epistemic Boundaries
* Validation metrics are optimistic: the same validation set was used for early stopping and for choosing among roughly ten configurations.
* The Test split was evaluated repeatedly during development, and those results were available when the default model was chosen. It therefore provides no unbiased performance estimate, and the Test differences above are descriptive observations.
* The causes of the validation–Test divergence between frozen and fine-tuned representations (e.g., distribution differences between the Kermany training pool and its test split, scanner or exposure differences, or BatchNorm adaptation during fine-tuning) remain **unverified hypotheses**.
* Neither model has been prospectively evaluated across independent hospital centers. Consequently, neither model is claimed to be clinically robust, safe, or ready for clinical deployment.

---

## 4. Visual Saliency Auditing via Grad-CAM
Deep learning models trained on medical imaging are susceptible to shortcut learning—relying on non-pathological confounders such as patient age markers, anatomical orientation tags, or radiographic borders.

By integrating **Grad-CAM (Gradient-weighted Class Activation Mapping)**:
1. **Parenchymal Inspection:** Saliency heatmaps let researchers qualitatively inspect whether high activations fall within the lung fields rather than on image borders or annotations. The maps are coarse ($7 \times 7$ before upsampling) and have not been validated against radiologist annotations.
2. **Interpretability:** Visualizing class-specific activation maps provides transparency into the features driving the differential prediction.

---

## 5. Non-Clinical Ethical Disclaimer

> [!CAUTION]
> **Strict Non-Clinical Use Notice:**  
> This project is developed strictly for **academic research, scientific demonstration, and educational purposes**. It is not certified, cleared, or approved by the U.S. FDA, European CE, or any health regulatory authority as a medical device or diagnostic decision-support system. It must **not** be used for clinical triage, primary diagnosis, patient management, or clinical decision-making. All radiographic evaluations must be performed by certified radiologists and licensed medical practitioners.
