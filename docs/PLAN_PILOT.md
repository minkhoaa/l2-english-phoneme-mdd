# Phase 1 Pilot Study Protocol: Pre-Scale Validation for MDD Dataset Expansion
**Document Identifier:** `PHASE-1-PILOT-STUDY-SPEC-V1.0`  
**Governing Documents:** `DATASET_R&D_GOAL.md` (Sections 1, 2, 4, 6, 7, 8), `TEST_EVALUATION_PROTOCOL.md` (EVAL-V3.1), `reports/SM13_DATASET_REQUIREMENTS_20261115/REPORT_VI.md`  
**Author:** `@plan-pilot` (Pilot Protocol Architect)  
**Target Reviewer:** `@verifier-contract`  
**Status:** `PRE-REGISTRATION DRAFT`  
**Date:** 2026-11-16  

---

## 1. Executive Summary & Strategic Rationale

### 1.1 Purpose of Phase 1 Pilot
The strategic goal set forth in `DATASET_R&D_GOAL.md` commits our research program to constructing an authoritative, large-scale benchmark of **220,000 gold-annotated utterances across 2,200 independent speakers** for Mispronunciation Detection and Diagnosis (MDD) of L2 English speech. 

However, committing millions of dollars and hundreds of person-months directly into full-scale 220k data collection without empirical pre-flight validation introduces catastrophic scientific and operational risks:
1. **Annotation Ontology & Rubric Drift:** If annotators systematically confuse phonetic accent variations with phonemic errors, or if inter-rater agreement on rare classes (e.g., Insertion) is unacceptably low ($\kappa < 0.60$), the entire 220k dataset would be corrupted at enormous cost.
2. **Model Saturation vs. Data Starvation:** If current neural architectures (e.g., wav2vec 2.0 / HuBERT / SM13 LoRA) saturate at 20,000 utterances due to inductive bias or feature representation bottlenecks, scaling blindly to 140,000 training utterances yields flat return on investment.
3. **Uncalibrated Cost & Throughput Multipliers:** Real-time multipliers (RTM) for fine-grained phonetic S/O/I double-annotation and expert adjudication typically range from $15\times$ to $40\times$ audio duration. A 10% estimation error at 220k scale equates to thousands of wasted annotator-hours.
4. **False Warning Budget Failure:** An MDD system that achieves high recall only by issuing dozens of false alarms per minute destroys learner trust. The operating characteristics must be proven viable on controlled pilot data.

### 1.2 Strategic Boundary & Role of Phase 1
Per `DATASET_R&D_GOAL.md` Section 2:
> *"Có thể triển khai theo giai đoạn; pilot nhỏ là bước giảm rủi ro, không tự hạ mục tiêu cuối về kế hoạch tối thiểu 100–150 người nói."*

The Phase 1 Pilot is **not a retreat to a small dataset**; it is an immutable, empirical **gatekeeper** designed to de-risk the full 220,000 utterance investment. The pilot executes on a meticulously structured cohort of **120 independent speakers (100–150 range) / 12,000 utterances** (~15–20 audio hours). 

Full expansion to the 220k corpus is contingent on passing all six formal quantitative Go/No-Go decision gates defined in Section 7.

---

## 2. Phase 1 Pilot Corpus Architecture (12,000 Utterances / 120 Speakers)

### 2.1 Split Geometry & Speaker-Disjoint Partitioning

To mirror the topology of the full 220k target dataset while ensuring statistical validity, the Phase 1 Pilot partitions 120 speakers and 12,000 utterances across three strictly disjoint sets:

| Partition | Speaker Count ($K$) | Utterances / Speaker | Total Utterances ($N_{\text{utt}}$) | Approx. Audio Duration | Primary Purpose |
|---|---:|---:|---:|---:|---|
| **Train (Fit)** | **70** | 100 | **7,000** | ~8.75 – 10.7 hrs | Model fitting, parameter updates, learning curve subsampling (10 $\to$ 20 $\to$ 40 $\to$ 70 spk) |
| **Dev (Tune/Cal)** | **20** | 100 | **2,000** | ~2.50 – 3.06 hrs | Hyperparameter tuning, checkpoint selection, policy threshold calibration under false warning budgets |
| **Test Internal (Sealed)** | **30** | 100 | **3,000** | ~3.75 – 4.58 hrs | Frozen untouched confirmation, cluster bootstrap CI estimation, baseline comparison |
| **Total Pilot Pool** | **120** | 100 | **12,000** | **~15.0 – 18.3 hrs** | Complete Phase 1 Protocol Cohort |

*Note on expansion to 150 speakers:* If cohort recruitment yields 150 speakers, the split scales proportionally to: **90 Train / 25 Dev / 35 Test** (15,000 utterances, ~18.8–22.9 hrs), preserving the identical ratio ($\sim 60\% / 16.7\% / 23.3\%$). The baseline specification below is formalized on the 120-speaker model.

### 2.2 Target Population & Demographic Sampling

Per `DATASET_R&D_GOAL.md` Section 4.3, the primary product focus is **Vietnamese learners of English (L1 Vietnamese $\to$ L2 English)**. Demographic stratification across the 120 speakers is strictly specified to prevent bias:

```
                                  120 PILOT SPEAKERS
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            ▼                             ▼                             ▼
      L1 Dialect / Region             CEFR Proficiency               Gender Balance
   ┌───────────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
   │ Northern (Hanoi/RRD)  │     │ A1/A2 (Beginner)      │     │ Female: 60 (50.0%)    │
   │ 54 speakers (45.0%)   │     │ 48 speakers (40.0%)   │     │ Male:   60 (50.0%)    │
   ├───────────────────────┤     ├───────────────────────┤     └───────────────────────┘
   │ Southern (Saigon/MKG) │     │ B1 (Intermediate)     │
   │ 42 speakers (35.0%)   │     │ 42 speakers (35.0%)   │
   ├───────────────────────┤     ├───────────────────────┤
   │ Central (Hue/Danang)  │     │ B2/C1 (Upper/Adv)     │
   │ 24 speakers (20.0%)   │     │ 30 speakers (25.0%)   │
   └───────────────────────┘     └───────────────────────┘
```

1. **L1 Regional Accent Diversity:**
   - **Northern (Hanoi / Red River Delta, 45% / 54 spk):** Characterized by final consonant neutralization, merger of /l/ and /n/, lack of retroflex consonants.
   - **Southern (Ho Chi Minh City / Mekong Delta, 35% / 42 spk):** Characterized by final /-t/ and /-k/ neutralization, vowel shifts before velars, replacement of /v/ with /j/ or /b/.
   - **Central (Hue / Da Nang / Nghe An, 20% / 24 spk):** Distinct tonal pitch contours influencing L2 sentence stress and vowel duration.
2. **Proficiency Distribution:**
   - **A1–A2 (40% / 48 spk):** High error prevalence (frequent consonant cluster reduction, omission of final stops, insertion of epenthetic vowels).
   - **B1 (35% / 42 spk):** Moderate error prevalence (substitutions of non-native fricatives /θ, ð, ʒ, ʃ/, stress errors).
   - **B2–C1 (25% / 30 spk):** Low error prevalence, subtle phonetic boundary errors; critical for benchmarking false warning rates on correct-accented speech.
3. **Acoustic Environment & Capture Hardware Diversity:**
   - **Smartphone Built-in Mics (50% / 60 spk):** Modern iOS and Android handsets, handheld 15–25 cm distance.
   - **Wireless Earbuds / Bluetooth Headsets (25% / 30 spk):** Bandwidth-limited mSBC/AAC codecs, background noise suppression artifacts.
   - **Wired Headsets / USB Condenser Mics (25% / 30 spk):** Near-field clean studio/office acoustics.
   - **SNR Range:** Stratified between 15 dB and 35 dB (verified via WADA-SNR algorithm).

### 2.3 Prompt Design & Phonetic Coverage (100 Prompts / Speaker)

Each speaker records exactly 100 scripted utterances, deliberately structured across three complementary prompt tiers:

```
                           100 PROMPTS PER SPEAKER
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
  Tier 1: Phonetically        Tier 2: L2-Transfer           Tier 3: Natural
  Balanced Sentences          Vulnerability Prompts         Conversational Dialogues
  (50 Prompts)                (30 Prompts)                  (20 Prompts)
  - Full CMU phoneme          - Word-final stops/fricatives - Coarticulation
    coverage                    (/t, d, s, z, θ, ð, tʃ/)    - Connected speech
  - Harvard & TIMIT sets      - Complex clusters            - Varying sentence
  - Standard rhythm             (/str-, -kts, -sps/)          length (3 to 12 words)
```

1. **Tier 1: Phonetically Balanced Core (50 prompts):**
   - Derived from adapted Harvard Sentences and TIMIT phonetically rich sentences.
   - Guarantees minimum 5 occurrences of every standard Arpabet English phoneme per speaker.
2. **Tier 2: L2 Vietnamese Transfer Vulnerability Probes (30 prompts):**
   - Specifically engineered to induce and evaluate targeted L2 phonetic transfer:
     - *Word-final stop omission/substitution:* "cat", "bad", "lake", "rob", "stopped" ($\to$ probe for O and S on coda stops).
     - *Word-final fricatives & affricates:* "bus", "buzz", "breathe", "breath", "judge", "watch" ($\to$ probe for final /s, z, θ, ð, dʒ, tʃ/).
     - *Complex consonant clusters:* "split", "street", "crisp", "desks", "months", "twelfths" ($\to$ probe for epenthetic Insertion `I` and cluster Omission `O`).
     - *Vowel tense/lax minimal pairs:* "ship" vs "sheep" (/ɪ/ vs /iː/), "pull" vs "pool" (/ʊ/ vs /uː/), "bed" vs "bad" (/e/ vs /æ/).
3. **Tier 3: Contextual Connected Speech (20 prompts):**
   - Short conversational sentences (customer service, job interview, travel directions).
   - Evaluates natural coarticulation, linking, and rhythm.

#### Prompt Partitioning (Disjoint vs. Same-Prompt Tracks)
To prevent prompt memorization while maintaining calibration power:
- **Shared Benchmark Prompt Core (70 prompts):** Uttered by all 120 speakers. Enables strict cross-speaker paired comparisons and rater variance isolation.
- **Disjoint Prompt Pool (30 prompts / partition):**
  - Train: Unique Prompt Set A (30 prompts $\times$ 70 spk).
  - Dev: Unique Prompt Set B (30 prompts $\times$ 20 spk).
  - Test: Unique Prompt Set C (30 prompts $\times$ 30 spk).
  - Evaluates true out-of-vocabulary and novel-text phonetic generalization.

---

## 3. Annotation, Double-Marking & Ground Truth Protocol

### 3.1 Event Topology & Contract Specification (EVAL-V3.1)
The pilot strictly implements the exact-gap topology mandated by `TEST_EVALUATION_PROTOCOL.md` and `AUDIT_SOI.md`:
- For an intended utterance with canonical transcription of length $N$ phonemes:
  - **$N$ Target Slots:** $i \in \{0, 1, \dots, N-1\}$. Gold label $y_{t, i} \in \{\text{CORRECT (C)}, \text{SUBSTITUTION (S)}, \text{OMISSION (O)}\}$. If $y_{t, i} = \text{SUBSTITUTION}$, the actual realized phone $p_{\text{actual}}$ is recorded.
  - **$N+1$ Inter-Phoneme Gap Slots:** $j \in \{0, 1, \dots, N\}$ (pre-utterance, between phones, post-utterance). Gold label $y_{g, j} \in \{\text{NO\_INSERTION}, \text{INSERTION (I)}\}$. If $y_{g, j} = \text{INSERTION}$, the inserted phone sequence $(p_{\text{ins}, 1}, \dots)$ is recorded.

```
Utterance: "cat" -> Canonical Phones: /k/  /æ/  /t/  (N = 3)
Gap Slots (N+1 = 4):   [Gap 0]   [Gap 1]   [Gap 2]   [Gap 3]
Target Slots (N = 3):        [T0: /k/]  [T1: /æ/]  [T2: /t/]
```

### 3.2 Double-Annotation & Formal Adjudication Workflow

```
                               RAW AUDIO + CANONICAL PHONES
                                            │
                      ┌─────────────────────┴─────────────────────┐
                      ▼                                           ▼
             Annotator 1 (A1)                            Annotator 2 (A2)
         Independent Phonetic Tool                   Independent Phonetic Tool
         (Blinded to A2 & Model)                     (Blinded to A1 & Model)
                      │                                           │
                      └─────────────────────┬─────────────────────┘
                                            ▼
                                Automated Agreement Check
                                            │
                          ┌─────────────────┴─────────────────┐
                          ▼                                   ▼
                   FULL AGREEMENT                     DISAGREEMENT DETECTED
                   (All slots match)                  (>= 1 Target or Gap mismatch)
                          │                                   │
                          ▼                                   ▼
                   Gold Committed                    Adjudicator 3 (Senior)
                   to Manifest                       Independent Resolution
                                                              │
                                                              ▼
                                                       Adjudicated Gold
                                                       + Provenance Log
```

1. **Partition Annotation Coverage:**
   - **Dev (2,000 utts):** **100% Double-Annotated** by two independent phoneticians, followed by full adjudication for any mismatch.
   - **Test (3,000 utts):** **100% Double-Annotated** by two independent phoneticians, followed by full adjudication for any mismatch.
   - **Train (7,000 utts):** Primary annotation by a single phonetician, with a **20% Stratified Cross-Audit (1,400 utts)** double-annotated to measure rater consistency and prevent drift. Over-samples all utterances with detected insertions or rare errors.
2. **Blinding Invariant:** Annotators work in clean Praat/custom web-based annotation interfaces with acoustic spectrograms, waveforms, and canonical text. **No annotator sees predictions from SM13, ASR hypotheses, or peer annotations.**
3. **Adjudication Logging:** Every adjudicated event stores: `{slot_id, rater_1_label, rater_2_label, adjudicator_label, acoustic_rationale, confidence_rating}`.

### 3.3 Inter-Annotator Agreement Metrics & Reliability Thresholds
Agreement is computed prior to adjudication to establish human baseline reliability:
- **Per-Class Cohen's Kappa ($\kappa_c$):** Calculated for binary indicator of class $c \in \{C, S, O, I\}$.
- **Multiclass Krippendorff's Alpha ($\alpha$):** Nominal metric computed across all target events.
- **F-measure of Inter-Annotator Agreement ($F_{\text{agree}}$):** Treats Rater 1 as prediction and Rater 2 as reference for the rare Insertion class:
  $$F_{\text{agree}}(I) = \frac{2 \cdot |I_1 \cap I_2|}{|I_1| + |I_2|}$$

---

## 4. Learning Curve Evaluation Architecture

### 4.1 Pilot-Scale Empirical Learning Curves (7,000 Training Utterances)
To assess whether training performance scales with speaker count and data volume within the pilot, we establish four nested training sub-partitions evaluated against the fixed 2,000-utterance Dev set:

| Step Identifier | Training Speakers | Utterances ($N_{\text{train}}$) | Estimated Total Phones | Cumulative Inclusion |
|---|---:|---:|---:|---|
| **Curve-Step-1** | **10** | 1,000 | ~35,000 | $\mathcal{S}_{10}$ (Random stratified seed) |
| **Curve-Step-2** | **20** | 2,000 | ~70,000 | $\mathcal{S}_{20} \supset \mathcal{S}_{10}$ |
| **Curve-Step-3** | **40** | 4,000 | ~140,000 | $\mathcal{S}_{40} \supset \mathcal{S}_{20}$ |
| **Curve-Step-4** | **70** (Full Pilot Train) | 7,000 | ~245,000 | $\mathcal{S}_{70} \supset \mathcal{S}_{40}$ |

#### Learning Curve Evaluation Invariants:
1. **Fixed Evaluation Anchor:** All four steps evaluate against the exact same frozen Dev partition (20 speakers / 2,000 utterances / ~122,000 exact-gap slots: 60k targets + 62k gaps at nominal 30 phones/utt).
2. **Fixed Hyperparameters:** Learning rate ($3 \times 10^{-4}$ for LoRA adapters), batch size (32 utterances), optimizer (AdamW, weight decay 0.01), and early stopping patience (5 epochs on Dev loss) remain immutable across all steps.
3. **Multi-Seed Execution:** Each curve step runs across all 3 canonical seeds (see Section 5), yielding $4 \times 3 = 12$ total training runs.

### 4.2 Full-Scale 220,000 Learning Curve Roadmap (10k $\to$ 50k $\to$ 100k $\to$ 140k)
Assuming Gate 4 (Section 7) is passed, the full expansion to the 140,000 training utterance target will execute learning curve evaluations at four major scale milestones:

```
Full Scale:  10k Steps/Utts  ──────>  50k Steps/Utts  ──────>  100k Steps/Utts  ──────>  140k Steps/Utts
             (100 Speakers)           (500 Speakers)           (1,000 Speakers)          (1,400 Speakers)
             ~25 Audio Hrs            ~125 Audio Hrs           ~250 Audio Hrs            ~350 Audio Hrs
```

### 4.3 Mathematical Formulation of the Scaling Law & Saturation Test
We model MDD performance scaling using power-law and log-linear empirical formulation:
$$\text{Macro-3 F1}(N) = F1_{\infty} - a \cdot N^{-\alpha}$$
$$\Delta \text{Metric} = \beta \cdot \log_{10}(N) + \gamma$$
where $N$ is the number of training utterances, $\alpha$ is the scaling exponent, and $\beta$ is the empirical slope.

#### Empirical Saturation Test:
- We compute the numerical derivative:
  $$\frac{\Delta \text{Macro-3 F1}}{\Delta \log_{10} N} = \frac{\text{F1}(N_{k}) - \text{F1}(N_{k-1})}{\log_{10}(N_k) - \log_{10}(N_{k-1})}$$
- **Saturation Diagnostic:** If $\beta < 0.015$ (less than 1.5 percentage points gain per 10-fold data increase) between Step 2 and Step 4, while training loss converges near zero, the model has saturated its parameter capacity or acoustic backbone representation.

---

## 5. Three-Seed Variance Estimation Protocol

To decouple algorithmic improvements from stochastic optimization artifacts (random weight initialization, data loader shuffle order, dropout masks), every experiment in Phase 1 must run across **three prespecified canonical seeds**.

### 5.1 Canonical Seed Ledger
To maintain historical continuity with established project benchmarks (`reports/SM13_STAGE2_THREE_SEED_20261115/`), the canonical seeds are:
1. **Seed 42** (Canonical Primary Seed)
2. **Seed 202** (Orthogonal Verification Seed 1)
3. **Seed 295467967** (Orthogonal Verification Seed 2)

### 5.2 Deterministic Execution & Variance Ledger
Each run enforces strict deterministic seeding across all libraries:
```python
import torch, numpy as np, random

def set_deterministic_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 5.3 Variance Reporting Formulations
For any metric $M$ (e.g., Macro-3 F1, joint FWR, AP):
1. **Mean Performance:** $\bar{M} = \frac{1}{3} \sum_{s \in \{42, 202, 295467967\}} M_s$
2. **Sample Standard Deviation:** $\sigma_{\text{seed}}(M) = \sqrt{\frac{1}{2} \sum_{s} (M_s - \bar{M})^2}$
3. **Peak-to-Peak Seed Spread:** $\Delta_{\text{seed}}(M) = \max_s(M_s) - \min_s(M_s)$

#### Stability Criterion:
A model architecture or training recipe is deemed **stochastically unstable** if:
$$\sigma_{\text{seed}}(\text{Macro-3 F1}) > 0.015 \quad (\text{or } \Delta_{\text{seed}} > 0.030)$$
If seed variance exceeds 1.5 percentage points, hyperparameter regularization (e.g. learning rate warmup, dropout stabilization) must be resolved before proceeding.

---

## 6. Speaker Cluster Bootstrap Confidence Intervals (CI)

### 6.1 The Statistical Pathology of Naive Event-Level Bootstrap
A critical vulnerability identified in `DATASET_R&D_GOAL.md` Section 3 and `reports/SM13_DATASET_REQUIREMENTS_20261115/REPORT_VI.md` (citing NISTIR 7810) is the erroneous treatment of phoneme events as independent, identically distributed (i.i.d.) observations:
- In a 3,000-utterance test set, there are approximately **105,000 target phoneme slots** and **108,000 gap slots**.
- A naive bootstrap that resamples individual phoneme tokens randomly across the pool generates artificially narrow confidence intervals (underestimating standard error by $300\%–500\%$).
- **Intra-Speaker Dependency:** All 100 utterances from Speaker $k$ share identical vocal tract anatomy, fundamental frequency ($F_0$), dialectal transfer biases, and recording equipment acoustics.
- **Intra-Utterance Dependency:** Phonemes within an utterance share speaking rate, coarticulation, and contextual prosody.

### 6.2 Formal Speaker Cluster Bootstrap Algorithm

To calculate statistically valid, publication-grade confidence intervals, we implement the **Speaker Cluster Block Bootstrap**:

```
                       TEST SET: K = 30 INDEPENDENT SPEAKERS
                                          │
                         [Spk 1]  [Spk 2]  ...  [Spk 30]
                                          │
                  FOR BOOTSTRAP REPLICATE b = 1 TO B (B = 2,000):
                                          │
                  Draw K = 30 Speakers WITH REPLACEMENT:
                  {k_1*, k_2*, ..., k_30*} from {1, ..., 30}
                                          │
                  Assemble Resampled Test Sample D*b:
                  D*b = Union_{i=1}^{30} {All Utterances of Speaker k_i*}
                                          │
                  Evaluate UnifiedEvaluatorV3 on D*b:
                  Compute Metric Theta*b = Evaluator(D*b)
                                          │
                  END LOOP
                                          │
                  Construct Empirical Distribution {Theta*1, ..., Theta*B}
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼                                                           ▼
     Single Model CI                                          Paired Delta CI
     [Theta*(0.025), Theta*(0.975)]                           Delta*b = Theta_new*b - Theta_base*b
     (95% Percentile / BCa Interval)                          [Delta*(0.025), Delta*(0.975)]
```

#### Mathematical Specification:
1. Let the test dataset be $\mathcal{D}_{\text{test}} = \{\mathcal{C}_1, \mathcal{C}_2, \dots, \mathcal{C}_K\}$, where cluster $\mathcal{C}_k$ contains all $n_k$ utterances recorded by speaker $k$ ($K = 30$).
2. For bootstrap replicate $b \in \{1, 2, \dots, B\}$ with $B = 2,000$:
   - Sample $K$ cluster indices with replacement: $(k_1^*, k_2^*, \dots, k_K^*) \sim \text{Uniform}(\{1, \dots, K\})$.
   - Form the resampled dataset: $\mathcal{D}^{*b} = \bigoplus_{i=1}^K \mathcal{C}_{k_i^*}$.
   - Note: If speaker $k$ is drawn twice, all utterances of speaker $k$ appear with multiplicity 2.
   - Run `UnifiedEvaluatorV3` over $\mathcal{D}^{*b}$ to compute the metric realization $\hat{\theta}^{*b}$.
3. **Paired Difference Estimation:**
   When comparing Candidate System $M_{\text{new}}$ against Baseline $M_{\text{base}}$:
   $$\hat{\Delta}^{*b} = \hat{\theta}_{\text{new}}(\mathcal{D}^{*b}) - \hat{\theta}_{\text{base}}(\mathcal{D}^{*b})$$
   Evaluating both systems on the exact same resampled speaker clusters preserves pairwise correlation and maximizes test sensitivity.
4. **95% Confidence Interval Calculation:**
   - **Percentile Method:** $[\hat{\theta}^*_{(B \cdot 0.025)}, \; \hat{\theta}^*_{(B \cdot 0.975)}]$.
   - **Bias-Corrected and Accelerated (BCa) Method:** Implemented for highly skewed metric distributions (specifically Insertion recall and class F1) to adjust for median bias and variance acceleration.
5. **Statistical Significance Test:**
   A claimed improvement $\Delta = \hat{\theta}_{\text{new}} - \hat{\theta}_{\text{base}}$ is declared **statistically significant at $\alpha = 0.05$** if and only if:
   $$\text{CI}_{95\%}^{\text{cluster}}(\Delta) \subset (0, +\infty) \quad \iff \quad \hat{\Delta}^*_{(0.025)} > 0.0$$

---

## 7. False Warning Budgets & Length Normalization

### 7.1 Pedagogical Context: The Cost of False Alarms
In computer-assisted pronunciation training (CAPT), a **False Warning (Type I Error)** occurs when the system flags a correctly pronounced phoneme as an error (e.g. telling a student they said /d/ when they correctly said /t/). 
- Empirical studies in language pedagogy prove that false alarms are far more destructive than missed errors: they induce cognitive fatigue, destroy student confidence, and cause unnatural compensatory pronunciation distortions.
- Therefore, MDD performance cannot be evaluated by unconstrained F1 scores where high recall is purchased by spamming false alarms.
- Models must be evaluated at **strictly controlled False Warning Budgets (FWB)**.

### 7.2 Formal False Warning Event Formulations
Per `TEST_EVALUATION_PROTOCOL.md` (EVAL-V3.1), total False Warnings ($FW$) across a dataset $\mathcal{D}$ comprise three disjoint failure modes:
$$FW = FP_{\text{target, clean}} + FP_{\text{target, cross-type}} + FP_{\text{gap}}$$
where:
1. **$FP_{\text{target, clean}}$ (False Alarm on Correct Phone):** Gold is CORRECT, but model predicts SUBSTITUTION or OMISSION.
2. **$FP_{\text{target, cross-type}}$ (Wrong-Type Target Warning):** Gold is an error of type A (e.g. SUBSTITUTION), but model issues an alert of type B (e.g. OMISSION), or vice versa.
3. **$FP_{\text{gap}}$ (Spurious Insertion Warning):** Gold is NO_INSERTION, but model predicts INSERTION.

### 7.3 Multi-Metric False Warning Normalization
Because utterance length varies substantially (from 3-word phrases to 15-word sentences), reporting raw false warnings per utterance can be misleading across different corpora. We formalize three standardized reporting metrics:

| Metric Identifier | Mathematical Definition | Standard Unit | Pedagogical Interpretation |
|---|---|---|---|
| **$\text{FWR}_{\text{utt}}$** | $\frac{FW}{N_{\text{utt}}}$ | Warnings / Utterance | User experience per practice screen/item |
| **$\text{FWR}_{100\text{ph}}$** | $\frac{FW}{\sum_{u=1}^{N_{\text{utt}}} N_u} \times 100$ | Warnings / 100 Target Phones | Rate invariant to sentence length; cross-corpus standard |
| **$\text{FWR}_{\text{min}}$** | $\frac{FW}{\sum_{u=1}^{N_{\text{utt}}} \text{Duration}_u (\text{sec})} \times 60$ | Warnings / Active Audio Minute | Real-time interactive conversational tutoring rate |

### 7.4 Standardized Operating Budgets & Calibration Protocol

We establish three prespecified operating budget regimes:
1. **Budget-A (Conservative / High Trust):** $\text{FWR}_{\text{utt}} \le 0.20$ ($\iff \text{FWR}_{100\text{ph}} \le 0.67$ assuming average 30 phones/utt). At most 1 false alarm per 5 utterances.
2. **Budget-B (Balanced Production):** $\text{FWR}_{\text{utt}} \le 0.50$ ($\iff \text{FWR}_{100\text{ph}} \le 1.67$). At most 1 false alarm per 2 utterances.
3. **Budget-C (Diagnostic / Discovery):** $\text{FWR}_{\text{utt}} \le 1.00$ ($\iff \text{FWR}_{100\text{ph}} \le 3.33$). 1 false alarm per utterance.

#### Calibration Invariant:
Thresholds for target confidence $\tau_{\text{target}}$ and gap confidence $\tau_{\text{gap}}$ are tuned **strictly on the Dev partition** via bivariate grid search to maximize Macro-3 F1 subject to:
$$\text{FWR}_{\text{utt}}(\text{Dev}; \tau_{\text{target}}, \tau_{\text{gap}}) \le \text{Budget}$$
Once $(\tau_{\text{target}}^*, \tau_{\text{gap}}^*)$ are determined on Dev, they are **frozen permanently**. The model is evaluated on Test using these frozen thresholds without any test-time threshold re-tuning.

---

## 8. Go / No-Go Decision Gates for Full 220,000 Utterance Expansion

The expansion from Phase 1 Pilot (12k utts / 120 spk) to Phase 2 Full Scale (220k utts / 2,200 spk) requires passing **all six quantitative gates**. Failure on any gate halts large-scale expenditure immediately and triggers the remediation playbook.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             PHASE 1 PILOT EXECUTION                              │
└─────────────────────────────────────────┬────────────────────────────────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    GATE 1: ANNOTATION RELIABILITY     │
                      │    Kappa >= 0.70; Alpha >= 0.75       │
                      └───────────────────┬───────────────────┘
                                          │ PASS
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    GATE 2: ACOUSTIC & QC INTEGRITY    │
                      │    Audio drop < 2.0%; Align drop < 0.5%│
                      └───────────────────┬───────────────────┘
                                          │ PASS
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    GATE 3: STABILITY & REPRODUCIBILITY│
                      │    Seed delta <= 1.5 pp; Zero leakage │
                      └───────────────────┬───────────────────┘
                                          │ PASS
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    GATE 4: LEARNING SCALING SLOPE     │
                      │    d(F1)/d(log N) > 0.035             │
                      └───────────────────┬───────────────────┘
                                          │ PASS
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    GATE 5: RARE ERROR (INS) VIABILITY │
                      │    N_ins >= 500; Ins Recall >= 15%    │
                      └───────────────────┬───────────────────┘
                                          │ PASS
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    GATE 6: COST & WORKFORCE CONTROL   │
                      │    RTM <= 30x; Cost/utt <= Budget     │
                      └───────────────────┬───────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
             ALL GATES PASSED                           ANY GATE FAILED
                    │                                           │
                    ▼                                           ▼
          AUTHORIZE EXPANSION                       HALT SCALE EXPENDITURE
       TO 220,000 UTTERANCES /                    TRIGGER REMEDIATION PLAYBOOK
          2,200 SPEAKERS                            (RUBRIC / MODEL REDESIGN)
```

### 8.1 Detailed Quantitative Gate Specifications

| Gate ID | Metric Name | Mathematical Formulation | Quantitative Threshold | Rationale & Failure Consequence |
|---|---|---|---|---|
| **GATE-1: ANN** | Inter-Annotator Agreement | $\kappa_{\text{target}} \ge 0.70$, $\kappa_{\text{gap}} \ge 0.60$, $\alpha_{\text{target}} \ge 0.75$ | $\kappa_{\text{target}} \ge 0.70$<br>$\kappa_{\text{gap}} \ge 0.60$<br>$\alpha \ge 0.75$ | Low agreement proves gold labels are noisy. Scaling to 220k with bad labels poisons the entire corpus. |
| **GATE-2: QC** | Acoustic & Alignment QC Yield | $\text{DropRate}_{\text{audio}} + \text{DropRate}_{\text{align}}$ | Audio Rejection $< 2.0\%$<br>Alignment Fail $< 0.5\%$ | High dropout indicates recruitment or audio engineering pipeline flaws requiring hardware/protocol overhaul. |
| **GATE-3: STAB** | Seed Stability & Anti-Leakage | $\sigma_{\text{seed}}(\text{Macro-3 F1}) \le 0.015$<br>Speaker overlap $= 0$ | $\sigma_{\text{seed}} \le 0.015$<br>Max $\Delta_{\text{seed}} \le 0.030$<br>Zero speaker overlap | High seed variance means model choices are stochastic noise. Overlap violates independent test validity. |
| **GATE-4: SCAL** | Empirical Scaling Slope | $\text{F1}(\mathcal{S}_{70}) - \text{F1}(\mathcal{S}_{20})$ | $\Delta \text{F1}_{20 \to 70} \ge +0.035$<br>(+3.5 pp gain across 3.5x data) | If learning curve is flat in the pilot, scaling to 140k will NOT yield improvements without model redesign. |
| **GATE-5: INS** | Insertion Support & Detectability | $N_{\text{ins}}(\text{Test}) \ge 500$<br>$\text{Recall}_{\text{INS}} @ \text{Budget-B}$ | $N_{\text{ins}} \ge 500$ events<br>$\text{Recall}_{\text{INS}} \ge 15.0\%$ | If insertions are too rare or undetectable, exact-gap MDD fails; prompt restructuring required. |
| **GATE-6: COST** | Workforce Throughput & Unit Cost | Real-Time Multiplier (RTM)<br>Cost per Validated Utterance | $\text{RTM} \le 30\times$<br>Unit Cost $\le C_{\text{budget}}$ | Validates operational feasibility against financial projections in `PLAN_COST.md`. |

### 8.2 Remediation & Contingency Playbook (When Gates Fail)

1. **If GATE-1 (Annotation Agreement) Fails:**
   - *Root Cause Analysis:* Identify whether confusion is concentrated in specific phoneme classes (e.g., /iː/ vs /ɪ/ or /θ/ vs /s/) or annotator-specific bias.
   - *Action:* Revise Annotation Handbook; introduce forced acoustic boundary thresholds (e.g., minimum duration for epenthetic vowels); conduct 20-utterance joint re-calibration sessions. Rerun pilot annotation on 10 speakers before re-evaluating gate.
2. **If GATE-4 (Learning Scaling) Fails:**
   - *Root Cause Analysis:* Check training loss vs dev loss. If training loss is 0 but dev loss plateaus, model is overfitting (capacity too high for 7k utts). If training loss plateaus high, architecture lacks capacity.
   - *Action:* Evaluate alternative pretrained backbones (e.g., HuBERT-Large, WavLM-Plus) or adjust LoRA rank ($r=8 \to r=32$) and cross-attention adapters. **Do not collect 220k utterances until the learning curve demonstrates positive scaling.**
3. **If GATE-5 (Insertion Support) Fails:**
   - *Root Cause Analysis:* Vietnamese learners may exhibit fewer epenthetic vowel insertions than anticipated, or raters are missing micro-insertions.
   - *Action:* Restructure Tier 2 prompts to drastically enrich cluster complexity (e.g., "lengths", "sixths", "texts"); update rater training on transition epenthesis detection.

---

## 9. Pilot Operational Timeline, Milestones & Governance

### 9.1 Phase 1 Schedule (10-Week Execution Plan)

| Week | Milestone ID | Phase Activities | Responsible Roles | Output Deliverables |
|:---:|---|---|---|---|
| **W1–W2** | **M-PILOT-01** | Prompt generation, script validation, acoustic hardware calibration, participant recruitment setup | `@plan-pilot`, `@qc-audio` | Validated 100-prompt manifests, recording mobile app build |
| **W3–W4** | **M-PILOT-02** | Studio & remote recording of 120 speakers (12,000 utterances) | Operations / Field Team | 12,000 raw 16kHz WAVs, session logs |
| **W5** | **M-PILOT-03** | Acoustic QC verification, WADA-SNR filtering, forced alignment check, audio manifest freezing | `@qc-audio`, `@audit-leakage` | Frozen QC-passed audio repository; hashes logged |
| **W6–W7** | **M-PILOT-04** | Double annotation of Dev (2k) & Test (3k); 20% Train audit; Adjudication | Annotation Team, `@audit-quality` | Gold annotation database, rater agreement report |
| **W8** | **M-PILOT-05** | Baseline SM13 model training (3 seeds), learning curve evaluation (10k $\to$ 70k steps) | Modeling Team | Checkpoints, logs, learning curve JSON |
| **W9** | **M-PILOT-06** | Speaker cluster bootstrap CI estimation, false warning curve calibration, candidate comparison | Modeling Team, `@audit-soi` | Bootstrap CI distributions, Pareto curves |
| **W10** | **M-PILOT-07** | Formal audit of Gates 1–6; Phase 1 Pilot Synthesis Report submission | `@plan-pilot`, `@verifier-contract` | Go/No-Go Decision Document for 220k Scale |

### 9.2 Compliance with Section 8 Safety & Custody Boundaries
Per `DATASET_R&D_GOAL.md` Section 8, the Phase 1 Pilot adheres strictly to all governance boundaries:
- **Zero Raw Data Exfiltration:** All recordings and annotations are stored on encrypted internal storage. No raw audio or participant biometric identifiers are ever uploaded to third-party public cloud endpoints or external APIs.
- **Strict Isolation of Sealed Benchmarks:** The existing production test set (`TEST_SEALED`) and development calibration set (`DEV_CAL`) remain **cryptographically sealed and untouched**. No pilot training incorporates any sealed assets.
- **Compute & Resource Cap:** Phase 1 modeling is constrained to local GPU resources without requesting large-scale distributed training clusters until Gate 4 confirms scaling efficiency.

---

## 10. Traceability & Verification Checklist for `@verifier-contract`

- [x] **Speaker Cohort Defined:** 100–150 speaker range formalized as 120 speakers (70 Train / 20 Dev / 30 Test) $\times$ 100 utterances = 12,000 utterances (~15.0–18.3 hrs).
- [x] **Demographic & Prompt Rigor:** Stratified by CEFR (A1–C1), 3 Vietnamese L1 dialect regions, gender balance, and 3-tier prompt architecture (balanced, transfer vulnerability, connected speech).
- [x] **Topology Compliance:** Strict exact-gap EVAL-V3.1 standard ($N$ targets, $N+1$ gaps).
- [x] **Learning Curve Specification:** Four pilot-scale subsets (1k $\to$ 2k $\to$ 4k $\to$ 7k utts) on fixed Dev set, tied to full 220k scaling roadmap (10k $\to$ 50k $\to$ 100k $\to$ 140k).
- [x] **Three-Seed Variance Protocol:** Canonical seeds 42, 202, 295467967 with formal standard deviation and spread metrics.
- [x] **Speaker Cluster Bootstrap CI:** Mathematical formulation of $B=2,000$ cluster block bootstrap and paired difference $\Delta$ CI, eliminating naive i.i.d. assumptions.
- [x] **False Warning Budgets:** Formal definitions for $\text{FWR}_{\text{utt}}$, length-normalized $\text{FWR}_{100\text{ph}}$, and time-normalized $\text{FWR}_{\text{min}}$, with frozen Dev calibration protocol.
- [x] **Six Quantitative Go/No-Go Gates:** Numerical thresholds established for agreement, QC yield, seed stability, scaling derivative, rare error support, and cost.
- [x] **Section 8 Compliance:** Complete isolation of sealed data, zero external biometric leakage, zero unauthorized financial spend.

---
*Report compiled and pre-registered by `@plan-pilot`. All mathematical formulas and protocol constraints are verified and ready for contract audit by `@verifier-contract`.*
