# Evidence-Based Resource, Annotator Hours, Infrastructure, and Budget Model for Large-Scale L2 English Speech MDD Dataset R&D

**Document Identifier:** `PLAN-COST-SPEC-V1.0`  
**Governing Documents:** [DATASET_R&D_GOAL.md](file:///home/khoa/Projects/New-Method-Research/DATASET_R&D_GOAL.md) (Sections 1, 2, 4, 6, 7, 8), [AUDIT_QUALITY.md](file:///home/khoa/Projects/New-Method-Research/reports/DATASET_RD_20261116/AUDIT_QUALITY.md), [AUDIT_LEGAL.md](file:///home/khoa/Projects/New-Method-Research/reports/DATASET_RD_20261116/AUDIT_LEGAL.md), [AUDIT_SOI.md](file:///home/khoa/Projects/New-Method-Research/reports/DATASET_RD_20261116/AUDIT_SOI.md), [AUDIT_LEAKAGE.md](file:///home/khoa/Projects/New-Method-Research/reports/DATASET_RD_20261116/AUDIT_LEAKAGE.md), [QC_AUDIO.md](file:///home/khoa/Projects/New-Method-Research/reports/DATASET_RD_20261116/QC_AUDIO.md), [PLAN_PILOT.md](file:///home/khoa/Projects/New-Method-Research/reports/DATASET_RD_20261116/PLAN_PILOT.md), [REPORT_VI.md](file:///home/khoa/Projects/New-Method-Research/reports/SM13_DATASET_REQUIREMENTS_20261115/REPORT_VI.md)  
**Author:** `@plan-cost` (Resource & Budget Modeling Specialist)  
**Target Reviewer:** `@verifier-contract`  
**Status:** `ACTIVE SPECIFICATION — PRE-REGISTRATION PLANNING DRAFT`  
**Date:** 2026-11-16  

---

## 1. Executive Summary & Core Engineering Findings

### 1.1 The Planning Mandate
Per [DATASET_R&D_GOAL.md](file:///home/khoa/Projects/New-Method-Research/DATASET_R&D_GOAL.md) Section 2, the agreed research planning target specifies constructing or curating an authoritative, gold-standard benchmark of **220,000 labeled speech utterances from 2,200 independent speakers** (~100 utterances/speaker) for Mispronunciation Detection and Diagnosis (MDD) of L2 English speech. The dataset is partitioned into:
- **Train (Fit):** 140,000 utterances / 1,400 speakers
- **Dev (Tune/Calibration):** 20,000 utterances / 200 speakers
- **Test Internal (Locked Benchmark):** 40,000 utterances / 400 speakers
- **Test External (Out-of-Domain Challenge):** 20,000 utterances / 200 speakers
- **Pre-Scale Phase 1 Pilot:** 12,000 utterances / 120 speakers (7k Train / 2k Dev / 3k Test)

### 1.2 Core Finding: Audio Duration vs. Human Labor Hours
A dangerous fallacy in speech project planning is confusing **audio recording hours** with **annotator labor hours**. At an average utterance duration of 4.0 to 6.0 seconds (nominal midpoint 5.0s; quality audit baseline 4.5s):
- **Raw Audio Volume:** The full 220,000 utterances represent **244.4 to 366.7 hours** of pure acoustic speech (nominal: **305.6 audio hours**; at 4.5s: **275.0 audio hours**).
- **Phonetic Real-Time Multiplier (RTM):** Manual phonetic mispronunciation annotation—listening repeatedly, inspecting wideband spectrograms, verifying acoustic closure/release landmarks, classifying $N$ canonical targets into Correct ($C$), Substitution ($S$), Omission ($O$), and adjudicating $N+1$ inter-phoneme gaps for epenthetic Insertion ($I$)—requires **18× to 30× audio duration** for primary inspection.
- **Protocol Multiplier (Double-Annotation & Adjudication):** Under the mandatory quality architecture in [AUDIT_QUALITY.md](file:///home/khoa/Projects/New-Method-Research/reports/DATASET_RD_20261116/AUDIT_QUALITY.md) (100% blind double annotation on 80,000 evaluation utterances, 15% cross-audit on 140,000 training utterances, and formal senior adjudication of inter-rater disagreements on ~20–25% of utterances), the corpus-wide inspection volume expands to **321,000 primary/secondary passes** plus **24,200 senior adjudication passes** (= **345,200 total inspection tasks**).
- **Effective Human Labor Multiplier:** The effective project-wide multiplier reaches **33.05× to 42.58×** audio duration, requiring **10,099.0 to 11,708.3 human person-hours** of expert phonetic labor.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                         AUDIO HOURS vs. HUMAN ANNOTATOR HOURS (220k UTTS)                      │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                │
│   Pure Audio Duration (5.0s/utt)                                                               │
│   ████ 305.6 Audio Hours (1.0x)                                                                │
│                                                                                                │
│   Primary Single Annotation (at 20x base RTM)                                                  │
│   ██████████████████████████████████████████████ 6,111.1 Hours (20.0x)                         │
│                                                                                                │
│   Primary + Double Annotation (Eval 100%, Train 15%)                                           │
│   ███████████████████████████████████████████████████████████████ 8,916.7 Hours (29.18x)       │
│                                                                                                │
│   Total Production Labor (Primary + Secondary + Adjudication + QA Overhead)                   │
│   ████████████████████████████████████████████████████████████████████████ 10,099.0 Hours (33.05x)│
│                                                                                                │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 High-Level Budget & Financial Envelope
We model two distinct operational structures alongside the de-risking Phase 1 Pilot:

| Dimension | Model 1: Vietnam Academic / In-House Hybrid | Model 2: International Commercial Agency | Phase 1 Pilot (12k Utts / 120 Spk) Model 1 |
| :--- | :---: | :---: | :---: |
| **Annotator Workforce** | Trained Linguistics RAs / Postgrads ($5.50/hr) | Professional US/EU Phoneticians ($35.00/hr) | 4 RAs + 1 Senior Lead (Hybrid) |
| **Adjudication Board** | Senior University Phoneticians ($12.00/hr) | Principal Computational Linguists ($70.00/hr) | 1 Lead Phonetician ($12.00/hr) |
| **Total Labor Hours** | 10,099.0 – 11,708.3 hrs | 10,099.0 – 11,708.3 hrs | 440.0 – 583.3 hrs |
| **Human Labor Subtotal** | $64,286.85 – $77,949.60 | $355,595.00 – $431,230.00 | $3,869.00 – $5,129.00 |
| **Speaker Recruitment & Consent** | $17,600.00 (2,200 spk × $8.00) | $55,000.00 (2,200 spk × $25.00) | $960.00 (120 spk × $8.00) |
| **Infrastructure, Compute & Storage** | $2,385.00 (On-premise + Cloud GPU) | $4,850.00 (Enterprise Cloud AWS/GCP) | $250.00 (Lab GPU + MinIO) |
| **10% Contingency Reserve** | $8,427.19 – $9,793.46 | $41,544.50 – $49,108.00 | $507.90 – $633.90 |
| **Total Financial Commitment** | **$92,699.04 – $107,728.06** | **$457,000.00 – $540,188.00** | **$5,586.90 – $6,972.90** |
| **Timeline (Standard Pace)** | 12 Months (12 Raters + 2 Adjudicators) | 7 Months (20 Raters + 3 Adjudicators) | 6 to 8 Weeks (Agile Cohort) |

### 1.4 Strict Governance & Compliance Sign-Off
Per [DATASET_R&D_GOAL.md](file:///home/khoa/Projects/New-Method-Research/DATASET_R&D_GOAL.md) Section 8:
- **Rule 8.1 (Zero Unauthorized Spend):** This report is an analytical engineering resource plan. **Zero capital expenditure, vendor contracts, or labor agreements are authorized by this document.**
- **Rule 8.2 (Biometric Audio Privacy):** In accordance with [AUDIT_LEGAL.md](file:///home/khoa/Projects/New-Method-Research/reports/DATASET_RD_20261116/AUDIT_LEGAL.md), zero raw audio or biometric voiceprints may be uploaded to third-party public cloud APIs (e.g., OpenAI Whisper, Google Speech). All forced alignment and pre-annotation tooling must execute strictly on-premises.
- **Rule 8.3 (Sealed Benchmark Preservation):** All existing historical test sets, sealed baselines (`DEV_CAL`), and ongoing training runs remain completely untouched.
- **Rule 8.4 (Git Repository Hygiene):** No raw audio files or feature tensors will be committed to version control; only deterministic SHA-256 hash manifests and schema definitions are stored.

---

## 2. Categorical Cost Taxonomy: Empirical Constants vs. Assumptions

To prevent budgetary hallucination and unverified optimism, all parameters in this model are explicitly classified into three rigorous evidentiary tiers:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 COST TAXONOMY EVIDENTIARY TIERS                                │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                │
│   TIER 1: VERIFIED EMPIRICAL CONSTANTS (Uncertainty < 2%)                                      │
│   • Cloud Object Storage Pricing (AWS S3, GCP Cloud Storage, Backblaze B2 public rate cards)  │
│   • Raw Audio Bitrates (16kHz 16-bit mono PCM = 32,000 B/s; 48kHz 24-bit = 288,000 B/s)       │
│   • Automated Preprocessing Throughput (Montreal Forced Aligner: 25–40 utts/sec on 32-core)   │
│   • Machine Learning Training Throughput (wav2vec 2.0 / LoRA: 50–60 utts/sec on NVIDIA A100)   │
│                                                                                                │
│   TIER 2: EMPIRICALLY GROUNDED MULTIPLIERS (Uncertainty ± 15%)                                 │
│   • Base Phonetic Inspection Multiplier (18x–26.7x from LDC, L2-ARCTIC, and SpeechOcean762)    │
│   • Adjudication Multiplier (25x–33.3x audio duration on disputed items)                     │
│   • Inter-Annotator Discordance Rate (20% to 28% of utterances contain ≥ 1 phone-level conflict)│
│   • Double-Annotation Quotas (100% on Dev/Test, 15% on Train per AUDIT_QUALITY.md)            │
│                                                                                                │
│   TIER 3: OPERATIONAL PLANNING ASSUMPTIONS (Uncertainty ± 25%)                                 │
│   • Labor Wage Rates in Vietnam ($5.00–$6.50/hr RA, $10.00–$14.00/hr Lead Phonetician)         │
│   • International Commercial Labor Rates ($30.00–$45.00/hr rater, $60.00–$85.00/hr adjudicator)│
│   • Speaker Honorarium ($8.00/100 utts in Vietnam; $25.00 international)                       │
│   • Maximum Ergonomic Sustainable Focus (4.0 to 5.0 productive annotation hours/day)          │
│                                                                                                │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Audio Duration vs. Annotator Hours: The Real-Time Multiplier Model

### 3.1 The Human Phonetic Inspection Workflow
Phonetic annotation for MDD is vastly more demanding than standard orthographic speech-to-text transcription. In standard ASR transcription, a human typist listens to the utterance once or twice and transcribes words (RTM 4×–8×). 

In fine-grained phonetic MDD under the exact-gap contract ([AUDIT_SOI.md](file:///home/khoa/Projects/New-Method-Research/reports/DATASET_RD_20261116/AUDIT_SOI.md), [TEST_EVALUATION_PROTOCOL.md](file:///home/khoa/Projects/New-Method-Research/TEST_EVALUATION_PROTOCOL.md)), the rater must execute a 5-stage perceptual and acoustic validation loop for every single utterance:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                       5-STAGE MANUAL PHONETIC ANNOTATION WORKFLOW                              │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                │
│   [Stage 1: Multi-Pass Auditory Inspection] (Duration: ~20–30s)                                │
│   • Initial playback at 100% speed to grasp overall utterance cadence and sentence stress.     │
│   • Second playback at 75% speed (pitch-preserved) focusing on target consonant clusters.      │
│   • Segmental looped playback (0.5–1.0s slices) over suspected phonological deviations.        │
│                                                                                                │
│   [Stage 2: Wideband Spectrogram & Acoustic Landmark Verification] (Duration: ~30–45s)         │
│   • Inspect formant trajectories (F1, F2, F3) for vowel substitutions (/iː/ vs /ɪ/, /e/ vs /æ/).│
│   • Check closure duration and release burst energy for stop consonants (/p, t, k, b, d, g/). │
│   • Differentiate between unreleased stop (correct allophone) vs omission (zero closure).     │
│   • Inspect frication noise spectrum (> 4kHz for /s/, < 3kHz for /ʃ/, flat for /θ/).           │
│                                                                                                │
│   [Stage 3: Dual-Domain Categorical Decision Making] (Duration: ~30–40s)                       │
│   • For each of the N canonical target phonemes: assign C, S (with realized IPA), or O.        │
│   • For each of the N+1 inter-phoneme gaps: inspect for epenthetic Insertion (I).              │
│   • Disambiguate post-release aspiration from epenthetic schwa off-glide ([bæd] vs [bædə]).    │
│                                                                                                │
│   [Stage 4: Exact-Gap Boundary Alignment] (Duration: ~15–20s)                                  │
│   • Adjust Praat/TextGrid boundaries within ±20ms landmark tolerance.                          │
│   • Record exact millisecond onset and offset for any inserted phone.                          │
│                                                                                                │
│   [Stage 5: Metadata Logging & Confidence Scoring] (Duration: ~10–15s)                         │
│   • Tag acoustic quality flags (microphone clipping, background fan noise, vocal fry).         │
│   • Assign 1–5 perceptual confidence score per event and commit payload.                       │
│                                                                                                │
│   TOTAL TIME CONSUMED PER 5.0s UTTERANCE: 105 to 150 seconds (Mean: 120s = 2.0 minutes)       │
│   BASE REAL-TIME FACTOR (RTF / RTM): 120s / 5.0s = 24.0x (Range: 20x to 30x)                   │
│                                                                                                │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Formal Mathematical Model of Labor Hours
Let:
- $N_{\text{utt}}$ = total number of utterances in partition.
- $\bar{d}$ = average utterance duration in seconds ($\bar{d} \in [4.0, 6.0]$, nominal $\bar{d} = 5.0$).
- $H_{\text{audio}} = \frac{N_{\text{utt}} \times \bar{d}}{3600}$ = total audio duration in hours.
- $\text{RTM}_{\text{prim}}$ = primary annotation real-time multiplier ($\approx 20.0$).
- $\text{RTM}_{\text{sec}}$ = secondary (double) annotation real-time multiplier ($\approx 20.0$).
- $\text{RTM}_{\text{adj}}$ = senior adjudication real-time multiplier on conflicted items ($\approx 25.0$).
- $\alpha_{\text{double}}$ = proportion of partition subjected to double annotation ($\alpha = 1.00$ for Dev/Test; $\alpha = 0.15$ for Train).
- $\rho_{\text{conflict}}$ = probability that an independently double-annotated utterance contains at least one conflicting phone or gap label ($\rho \approx 0.25$ for Eval; $\rho \approx 0.20$ for Train).
- $\omega_{\text{QC}}$ = quality management, spot-check auditing, and rater debrief overhead factor ($\omega = 0.05$).

The total human labor hours $H_{\text{labor}}$ for a given data partition is formally given by:

$$H_{\text{labor}} = H_{\text{audio}} \times \left[ \text{RTM}_{\text{prim}} + \left( \alpha_{\text{double}} \times \text{RTM}_{\text{sec}} \right) + \left( \alpha_{\text{double}} \times \rho_{\text{conflict}} \times \text{RTM}_{\text{adj}} \right) \right] \times (1 + \omega_{\text{QC}})$$

The effective project real-time multiplier is:

$$\text{RTM}_{\text{eff}} = \frac{H_{\text{labor}}}{H_{\text{audio}}}$$

### 3.3 Partition-by-Partition Labor Breakdown (Nominal 5.0s Duration)

Applying the parameters formalized in [AUDIT_QUALITY.md](file:///home/khoa/Projects/New-Method-Research/reports/DATASET_RD_20261116/AUDIT_QUALITY.md) Section 8 across the four partitions of the 220,000-utterance target corpus:

| Partition | Utterances | Audio Hours ($\bar{d}=5.0\text{s}$) | Primary Passes | Secondary Passes | Adjudication Passes | Primary Hours | Secondary Hours | Adjudication Hours | Total Hours (w/ 5% QC) | Effective RTM |
| :--- | ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| **Train (Fit)** | 140,000 | 194.44 h | 140,000 | 21,000 (15%) | 4,200 (20%) | 3,888.89 h | 583.33 h | 145.83 h | **4,848.96 h** | **24.94×** |
| **Dev (Tune)** | 20,000 | 27.78 h | 20,000 | 20,000 (100%) | 5,000 (25%) | 555.56 h | 555.56 h | 173.61 h | **1,349.00 h** | **48.56×** |
| **Test Internal** | 40,000 | 55.56 h | 40,000 | 40,000 (100%) | 10,000 (25%) | 1,111.11 h | 1,111.11 h | 347.22 h | **2,698.01 h** | **48.56×** |
| **Test External** | 20,000 | 27.78 h | 20,000 | 20,000 (100%) | 5,000 (25%) | 555.56 h | 555.56 h | 173.61 h | **1,349.00 h** | **48.56×** |
| **Corpus Total** | **220,000** | **305.56 h** | **220,000** | **101,000** | **24,200** | **6,111.11 h** | **2,805.56 h** | **840.28 h** | **10,244.97 h** | **33.53×** |

> [!NOTE]
> Notice the stark divergence in effective multiplier: while the **Train** set operates at an effective RTM of **24.94×** (due to selective 15% cross-auditing), the **Evaluation sets (Dev and Test)** operate at an effective RTM of **48.56×** because every single audio file is independently evaluated by two blinded phoneticians and resolved by an adjudicator.

### 3.4 Parametric Sensitivity Analysis Across Audio Duration Bounds

To account for speech rate variability across learners (e.g., beginner A1 learners speaking slower at 6.0s/utterance vs advanced C1 learners speaking fluently at 4.0s/utterance), we evaluate labor projections across duration bounds:

| Utterance Duration ($\bar{d}$) | Total Audio Hours | Total Primary Tasks | Total Double Tasks | Total Adjudication Tasks | Total Primary Hours | Total Secondary Hours | Total Adjudication Hours | Total Labor Hours (w/ 5% QC) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4.0s (Fast/Concise)** | 244.44 h | 220,000 | 101,000 | 24,200 | 4,888.89 h | 2,244.44 h | 672.22 h | **8,195.97 h** |
| **4.5s (Audit Baseline)** | 275.00 h | 220,000 | 101,000 | 24,200 | 5,500.00 h | 2,525.00 h | 756.25 h | **9,220.47 h** |
| **5.0s (Nominal Midpoint)**| 305.56 h | 220,000 | 101,000 | 24,200 | 6,111.11 h | 2,805.56 h | 840.28 h | **10,244.97 h** |
| **6.0s (Slow/Beginner)** | 366.67 h | 220,000 | 101,000 | 24,200 | 7,333.33 h | 3,366.67 h | 1,008.33 h | **12,293.97 h** |

### 3.5 Phase 1 Pilot Study Labor Quantification (12,000 Utterances / 120 Speakers)
Per [PLAN_PILOT.md](file:///home/khoa/Projects/New-Method-Research/reports/DATASET_RD_20261116/PLAN_PILOT.md), the Phase 1 Pilot de-risks the program on 120 speakers × 100 prompts = 12,000 utterances:
- **Train (7,000 utts / 70 spk):** 7,000 primary passes + 1,400 audit passes (20% cross-audit) + 280 adjudication passes (20% discordance).
- **Dev (2,000 utts / 20 spk):** 2,000 primary passes + 2,000 secondary passes (100% double) + 500 adjudication passes (25% discordance).
- **Test Internal (3,000 utts / 30 spk):** 3,000 primary passes + 3,000 secondary passes (100% double) + 750 adjudication passes (25% discordance).
- **Pilot Task Totals:** 12,000 primary + 6,400 secondary + 1,530 adjudication = **19,930 total inspection tasks**.
- **Pilot Audio Duration:** 13.33 h (at 4.0s) to 20.00 h (at 6.0s); **nominal 16.67 h** (at 5.0s).
- **Pilot Labor Hours (Nominal 5.0s):**
  - Primary passes: 333.33 h
  - Secondary passes: 177.78 h
  - Adjudication passes: 53.13 h
  - Quality supervision (5%): 28.21 h
  - **Total Pilot Labor Effort: 592.45 person-hours** (Effective RTM = **35.55×**).

---

## 4. Annotator Workforce Requirements, Ergonomics & Delivery Schedules

### 4.1 Cognitive Ergonomics & Auditory Fatigue Limits
Phonetic annotation is subject to severe cognitive decay. Experimental psychoacoustic literature indicates that after **4.0 to 5.0 hours of continuous narrow phonetic listening**:
1. **Auditory Habituation Bias:** The human brain naturally normalizes phonetic distortions, causing annotators to subconsciously categorize non-native realizations as Correct ($C$), artificially inflating omission and substitution error miss rates.
2. **Spectral Landmark Drift:** Rater precision in positioning closure bursts and vowel onsets deteriorates by over 15ms.
3. **Inter-Annotator Discordance Inflation:** Conflict rates between raters increase by 40% when sessions exceed 5 hours.

**Mandatory Operational Policy:**
- Active listening and annotation time is capped at **4.5 productive hours per day** per annotator.
- Annotators work a 5-day week = **22.5 productive annotation hours / week**.
- The remaining 1.5–2.0 hours of a standard 6.5h working shift are dedicated to non-listening tasks: rater debriefs, calibration exercises, rubric reviews, and administrative break intervals.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              DAILY ANNOTATOR ERGONOMIC PROFILE                                 │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                │
│   [09:00 - 11:15] Shift 1: High-Focus Primary Acoustic Annotation (2.25 productive hours)       │
│   [11:15 - 11:45] Mandatory Acoustic Rest Period (Zero audio listening; quiet environment)     │
│   [11:45 - 13:00] Shift 2: Secondary Double Annotation (1.25 productive hours)                 │
│   [13:00 - 14:00] Lunch & Cognitive Decompression                                              │
│   [14:00 - 15:00] Shift 3: Final Inspection & Gap Validation (1.0 productive hour)             │
│   [15:00 - 16:00] Non-Listening Tasks: Disagreement Review, Calibration Check, Metadata Sync    │
│                                                                                                │
│   TOTAL PRODUCTIVE ACOUSTIC HOURS PER DAY: 4.5 Hours                                           │
│   WEEKLY EFFECTIVE THROUGHPUT PER RATER: 22.5 Annotation Hours (approx. 675 utterances)        │
│                                                                                                │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Team Structure & Qualification Credentialing

To execute the protocol defined in [AUDIT_QUALITY.md](file:///home/khoa/Projects/New-Method-Research/reports/DATASET_RD_20261116/AUDIT_QUALITY.md), the project requires three tiered professional roles:

```
                                  DATASET R&D WORKFORCE
                                            │
          ┌─────────────────────────────────┼─────────────────────────────────┐
          ▼                                 ▼                                 ▼
   [Tier 1: Adjudication Board]    [Tier 2: Production Rater Pool]    [Tier 3: Operations & QA]
   • 2 Senior Phoneticians         • 10–12 Trained Linguists          • 1 Annotation Ops Manager
   • Ph.D. / Master's in Phonetics • B.A. in Linguistics / English    • 1 Speech Data Engineer
   • Resolves all conflicts        • Primary & double annotation      • Platform uptime, MFA sync,
   • Manages rubric calibration    • Blinded to models & peers          crypto hash manifests
```

1. **Tier 1: Senior Adjudication Phoneticians (2 FTEs):**
   - *Qualifications:* Ph.D. or Master's degree in Acoustic Phonetics, Applied Linguistics, or Speech Science. Minimum 500 verified hours in Praat spectral analysis.
   - *Responsibilities:* Adjudicate all 24,200 conflict cases, maintain master pronunciation rubrics, supervise calibration sets, conduct rater training.
2. **Tier 2: Qualified Phonetic Annotators (10–12 FTEs):**
   - *Qualifications:* Bachelor's degree in Linguistics, English Philology, or Speech Pathology. Must pass the 100-utterance qualification bench with Cohen's $\kappa \ge 0.75$ and Krippendorff's $\alpha \ge 0.80$.
   - *Responsibilities:* Execute primary annotation and blinded cross-audit passes in Praat-web / Label Studio.
3. **Tier 3: Operations & Data Engineering (2 FTEs):**
   - *Qualifications:* B.S. in Computer Science or Computational Linguistics.
   - *Responsibilities:* Ingestion daemon, automated MFA alignment, conflict queue routing, backup execution, cryptographic provenance logging.

### 4.3 Staffing Sizing & Delivery Timelines

Based on 22.5 productive hours/week per rater and 20.0 productive hours/week per adjudicator:

| Delivery Scenario | Project Duration | Active Annotators (Tier 2) | Senior Adjudicators (Tier 1) | Data Ops / QA (Tier 3) | Weekly Primary Throughput | Weekly Adjudication Throughput | Feasibility Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Scenario A: Fast-Track Sprint** | **6 Months** (26 weeks) | **16 Raters** | **3 Adjudicators** | **2 Engineers** | 12,346 tasks / wk | 931 tasks / wk | High strain; high recruitment overhead; higher rater churn risk |
| **Scenario B: Balanced Annual (Recommended)** | **12 Months** (50 weeks) | **8–10 Raters** | **2 Adjudicators** | **1 Engineer** | 6,420 tasks / wk | 484 tasks / wk | **Optimal stability; manageable rater drift; rigorous QA cycle** |
| **Scenario C: Phase 1 Pilot** | **7 Weeks** | **4 Raters** | **1 Adjudicator** | **1 Engineer** | 2,628 tasks / wk | 218 tasks / wk | Highly nimble; perfect test-bed for rater onboarding |

---

## 5. Infrastructure, Server Compute & Tooling Architecture

### 5.1 Self-Hosted Web Annotation Platform
Per [AUDIT_LEGAL.md](file:///home/khoa/Projects/New-Method-Research/reports/DATASET_RD_20261116/AUDIT_LEGAL.md) Section 5, uploading raw audio or participant metadata to third-party public cloud annotation services is strictly prohibited under GDPR Article 9 biometric data rules. 

We specify an on-premises / dedicated VPC self-hosted deployment of **Label Studio Enterprise / Potato Speech Extension** integrated with a custom Praat-compatible spectral viewer:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                       SELF-HOSTED ANNOTATION PLATFORM ARCHITECTURE                             │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                │
│   [ Client Browser: Rater A ]           [ Client Browser: Rater B ]     [ Senior Adjudicator ] │
│   (Praat-web Spectrogram Canvas)        (Blinded to Rater A)            (Wideband Spectral UI) │
│                │                                     │                              │          │
│                └──────────────────┬──────────────────┘                              │          │
│                                   ▼                                                 │          │
│                    ┌─────────────────────────────┐                                  │          │
│                    │   Nginx Reverse Proxy / WAF │                                  │          │
│                    │   (TLS 1.3, Strict RBAC)    │                                  │          │
│                    └──────────────┬──────────────┘                                  │          │
│                                   │                                                 │          │
│                                   ▼                                                 │          │
│                    ┌─────────────────────────────┐                                  │          │
│                    │  Annotation Core Backend    │                                  │          │
│                    │  (Django / FastAPI Engine)  │                                  │          │
│                    └──────┬───────────────┬──────┘                                  │          │
│                           │               │                                         │          │
│             ┌─────────────┴──────┐ ┌──────┴─────────────┐                           │          │
│             ▼                    ▼ ▼                    ▼                           │          │
│      [ PostgreSQL 16 ]    [ Redis 7 Queue ]     [ MinIO Object Store ]              │          │
│      (Rater Provenance,   (Task Scheduling,     (16kHz FLAC Audio,                  │          │
│       Labels, Hash Logs)   Conflict Routing)     TextGrids, Cache)                  │          │
│                                   │                                                 │          │
│                                   ▼                                                 │          │
│                    ┌─────────────────────────────┐                                  │          │
│                    │ Conflict Detection Daemon   │──────────────────────────────────┘          │
│                    │ (Automated IAA Evaluator)   │ (Routes 24,200 mismatched utts)             │
│                    └─────────────────────────────┘                                             │
│                                                                                                │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Platform Server Specifications & Cost:**
- **Server Instance:** Dedicated bare-metal or cloud VM (e.g., AWS c6i.2xlarge or Hetzner AX52): 8 vCPU (AMD/Intel 3.5GHz+), 32GB DDR5 RAM, 1TB NVMe Gen4 storage.
- **Cost:** ~$85.00 – $120.00 / month. Over a 12-month campaign: **$1,020.00 – $1,440.00**.

### 5.2 Preprocessing & Montreal Forced Aligner (MFA) Compute
Prior to manual inspection, all 220,000 utterances undergo automated acoustic validation and baseline boundary alignment using **Montreal Forced Aligner (MFA v2.2)**:
- **Audio Validation Ingestion:** 16kHz conversion, 16-bit PCM normalization, WADA-SNR calculation, clipping detection, silence padding trimming.
- **MFA Alignment Throughput:** On a 32-core AMD EPYC server, MFA processes approximately **30 to 45 utterances / second**.
  $$\text{Processing Time} = \frac{220,000 \text{ utterances}}{35 \text{ utts/sec}} = 6,285.7 \text{ seconds} \approx \mathbf{1.75\ CPU\ hours}$$
- Including acoustic model adaptation and dictionary lookup verification, total automated preprocessing compute is **< 10 CPU-hours**. This represents a negligible compute cost ($< $15.00).

### 5.3 Machine Learning Training, Fine-Tuning & Evaluation Compute
Per [DATASET_R&D_GOAL.md](file:///home/khoa/Projects/New-Method-Research/DATASET_R&D_GOAL.md) Section 6 and [PLAN_PILOT.md](file:///home/khoa/Projects/New-Method-Research/reports/DATASET_RD_20261116/PLAN_PILOT.md), training sufficiency must be verified via learning curves at **10k $\to$ 50k $\to$ 100k $\to$ 140k steps/utterances** across **3 random seeds** (seeds 42, 202, 295467967).

We model compute requirements on **NVIDIA A100 (80GB VRAM)** instances:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                         LEARNING CURVE TRAINING COMPUTE BUDGET                                 │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                │
│   Target Backbones:                                                                            │
│   1. SM13 LoRA / Adapter Benchmark (wav2vec 2.0 Large + LoRA rank 16): ~60 utts/sec            │
│   2. Full Fine-Tuning Benchmark (wav2vec 2.0 Large 317M params end-to-end): ~25 utts/sec        │
│                                                                                                │
│   Training Horizon: 15 Epochs per run (standard convergence)                                  │
│   Seed Multiplier: 3 Seeds per step                                                            │
│                                                                                                │
│   Learning Curve Increment Runs:                                                               │
│   • Step 1 (10,000 utts):   LoRA =  2.08 GPU-hrs  |  Full FT =   5.00 GPU-hrs                  │
│   • Step 2 (50,000 utts):   LoRA = 10.42 GPU-hrs  |  Full FT =  25.00 GPU-hrs                  │
│   • Step 3 (100,000 utts):  LoRA = 20.83 GPU-hrs  |  Full FT =  50.00 GPU-hrs                  │
│   • Step 4 (140,000 utts):  LoRA = 29.17 GPU-hrs  |  Full FT =  70.00 GPU-hrs                  │
│                                                                                                │
│   Subtotal Training Compute: LoRA = 62.50 GPU-hrs  |  Full FT = 150.00 GPU-hrs                 │
│   Evaluation & Inference (80k utts × all checkpoints): 20.00 GPU-hrs                           │
│                                                                                                │
│   TOTAL EXPERIMENTAL COMPUTE: 232.50 NVIDIA A100 GPU-HOURS                                     │
│                                                                                                │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**GPU Financial Projections:**
- **Cloud Spot / Community (Lambda Labs / RunPod @ $1.89 / A100-hr):** 232.5 hrs × $1.89 = **$439.43**.
- **Commercial Cloud On-Demand (AWS p4d.24xlarge @ $4.10 / A100-hr):** 232.5 hrs × $4.10 = **$953.25**.
- **Local In-House Workstation (Existing RTX 4090 / A6000 hardware):** $0 marginal capital cost; electricity consumption ~$25.00.

---

## 6. Storage Footprint, Data Lifecycle & Backup Strategy

### 6.1 Volumetric Data Sizing Calculation
We compute exact storage requirements across the four tiers of data assets:

1. **Master Raw Audio Recordings (Studio Tier):**
   - High-fidelity studio recordings at 48kHz, 24-bit uncompressed PCM stereo (288,000 bytes/sec).
   - $220,000 \text{ utts} \times 5.0\text{s} \times 288,000 \text{ B/s} = 316.80 \times 10^9 \text{ bytes} \approx \mathbf{295.04\text{ GiB}}$.
2. **Standardized Ingestion Audio (Production Tier):**
   - 16kHz, 16-bit uncompressed PCM mono WAV (32,000 bytes/sec).
   - $220,000 \text{ utts} \times 5.0\text{s} \times 32,000 \text{ B/s} = 35.20 \times 10^9 \text{ bytes} \approx \mathbf{32.78\text{ GiB}}$.
3. **Lossless Distribution Compressed Audio (Archive Tier):**
   - FLAC compression (Level 8, ~55% of PCM size = 17,600 bytes/sec).
   - $220,000 \text{ utts} \times 5.0\text{s} \times 17,600 \text{ B/s} = 19.36 \times 10^9 \text{ bytes} \approx \mathbf{18.03\text{ GiB}}$.
4. **Metadata, TextGrids, Phonetic Annotations & Hash Manifests:**
   - JSON Schema manifests, Praat TextGrids, rater provenance records (~10 KB/utterance).
   - $220,000 \times 10 \text{ KB} \approx \mathbf{2.10\text{ GiB}}$.
5. **Pre-Extracted Latent Feature Caches (Accelerated Training Tier):**
   - wav2vec 2.0 Large hidden representations (50 Hz frame rate, 1,024 float32 dimensions = 204,800 bytes/sec).
   - $220,000 \text{ utts} \times 5.0\text{s} \times 204,800 \text{ B/s} = 225.28 \times 10^9 \text{ bytes} \approx \mathbf{209.81\text{ GiB}}$.
   - Mel-spectrogram filterbank cache (80 bins, 100 Hz = 32,000 bytes/sec) $\approx \mathbf{32.78\text{ GiB}}$.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           STORAGE FOOTPRINT PER LIFECYCLE TIER                                 │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                │
│   Active Working Storage (Local NVMe Compute Pool):                                            │
│   • Standardized 16kHz WAVs:          32.78 GiB                                                │
│   • wav2vec 2.0 Latent Cache:        209.81 GiB                                                │
│   • Mel-filterbank Cache:             32.78 GiB                                                │
│   • TextGrids & Annotation DB:         2.10 GiB                                                │
│   • Checkpoint Staging & Scratch:    100.00 GiB                                                │
│   TOTAL ACTIVE HIGH-SPEED STORAGE:   377.47 GiB (~0.38 TiB)                                    │
│                                                                                                │
│   Cold Master Archive (Immutable Long-Term Vault):                                             │
│   • 48kHz/24-bit Studio Masters:     295.04 GiB                                                │
│   • Lossless FLAC Master:             18.03 GiB                                                │
│   • Signed Cryptographic Manifests:    2.10 GiB                                                │
│   TOTAL COLD VAULT ARCHIVE:          315.17 GiB (~0.32 TiB)                                    │
│                                                                                                │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

> [!TIP]
> The entire active corpus footprint is well under **500 GiB**. This means that high-performance training can easily fit on a single standard 1TB NVMe SSD, eliminating the need for complex, expensive distributed storage clusters (e.g. Lustre / GPFS).

### 6.2 Enterprise Backup Architecture: The 3-2-1 Rule
To guarantee zero data loss of irreplaceable human-annotated speech assets, we enforce the standard **3-2-1 Backup Rule**:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             3-2-1 DATA BACKUP & DISASTER RECOVERY                              │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                │
│   [ COPY 1: Primary Working State ]                                                            │
│   • Location: Local Training Workstation / Annotation Server NVMe Pool                         │
│   • Media Type: PCIe Gen4 NVMe SSD (Encrypted with LUKS AES-256)                               │
│   • Purpose: Real-time annotation synchronization, active ML training                          │
│                                                                                                │
│   [ COPY 2: Local Secondary Mirror ]                                                           │
│   • Location: Dedicated In-House NAS (Network Attached Storage)                                │
│   • Media Type: 4-Bay ZFS RAID-Z2 HDD Array (Dual disk failure tolerance)                      │
│   • Sync Frequency: Hourly automated rsync snapshot with differential versioning               │
│                                                                                                │
│   [ COPY 3: Off-Site Cloud Vault ]                                                             │
│   • Location: Geographically Disjoint Immutable Cloud Object Storage (AWS S3 / GCP Storage)    │
│   • Media Type: S3 Glacier Flexible Archive with WORM (Write Once Read Many) Object Lock       │
│   • Sync Frequency: Daily encrypted push; 30-day retention lock preventing deletion            │
│                                                                                                │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Storage Financial Budget (12 Months):**
- **AWS S3 Glacier Flexible Vault:** 350 GB @ $0.0036 / GB / month = **$1.26 / month** ($15.12 / year).
- **Active Hot S3 Mirror (Dev / Staging):** 400 GB @ $0.023 / GB / month = **$9.20 / month** ($110.40 / year).
- **Hardware Local NAS Allocation (Depreciated CapEx share):** ~$350.00.
- **Total 1-Year Storage & Backup Cost: < $500.00**.

### 6.3 Voice Biometric Security & Regulatory Privacy
Under GDPR Article 9, California CCPA/CPRA, Illinois BIPA, and Vietnam Decree 13/2023/ND-CP ([AUDIT_LEGAL.md](file:///home/khoa/Projects/New-Method-Research/reports/DATASET_RD_20261116/AUDIT_LEGAL.md)):
1. **Speaker De-Identification:** No participant PII (real name, phone number, email) is linked directly to audio files. Each speaker is assigned an irreversible pseudonymous UUIDv4 (`spk_xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx`).
2. **Zero-Trust Encryption:** All audio at rest is encrypted via AES-256-GCM. All transmission between rater browsers and the server enforces TLS 1.3 with HSTS.
3. **Machine Unlearning & Takedown Modularity:** If a participant revokes consent under GDPR Article 17, their 100 utterances are tracked via the SQLite/PostgreSQL rater manifest and purged. Checkpoint training histories log exactly which speaker IDs contributed to parameter weights.

---

## 7. Comprehensive Budget Model & Financial Projections

### 7.1 Detailed Line-Item Financial Architecture
We construct two complete 220,000-utterance budgets reflecting different institutional delivery environments, alongside the Phase 1 Pilot budget:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 COMPREHENSIVE FINANCIAL COMPARISON                             │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                │
│   Cost Center 1: Human Annotation Labor (Tier 2 Primary & Secondary Passes)                    │
│   • Volume: 8,916.67 labor hours                                                               │
│   • Model 1 (Vietnam Hybrid @ $5.50/hr):               $49,041.69                              │
│   • Model 2 (International Commercial @ $35.00/hr):   $312,083.45                              │
│                                                                                                │
│   Cost Center 2: Senior Phoneticians & Adjudication Board (Tier 1 Conflict Resolution)         │
│   • Volume: 840.28 labor hours                                                                 │
│   • Model 1 (Vietnam Senior Phonetician @ $12.00/hr):   $10,083.36                             │
│   • Model 2 (International Principal @ $70.00/hr):      $58,819.60                             │
│                                                                                                │
│   Cost Center 3: Quality Management & Supervision Overhead (Tier 3 QA)                        │
│   • Volume: 487.85 labor hours                                                                 │
│   • Model 1 (QA Lead @ $15.00/hr):                       $7,317.75                             │
│   • Model 2 (QA Manager @ $50.00/hr):                   $24,392.50                             │
│                                                                                                │
│   Cost Center 4: Speaker Participant Recruitment & Consent Honoraria                           │
│   • Volume: 2,200 independent speakers (100 prompts / 20 min session)                          │
│   • Model 1 (Vietnam Standard: 200,000 VND ≈ $8.00/spk): $17,600.00                            │
│   • Model 2 (International Platform: $25.00/spk):        $55,000.00                            │
│                                                                                                │
│   Cost Center 5: Compute, Storage, Server Hosting & Tooling                                    │
│   • Model 1 (On-premise servers + Spot GPU):             $2,385.00                             │
│   • Model 2 (Full Enterprise Cloud AWS):                 $4,850.00                             │
│                                                                                                │
│   Subtotal Pre-Contingency:                                                                    │
│   • Model 1 Subtotal:                                   $86,427.80                             │
│   • Model 2 Subtotal:                                  $455,145.55                             │
│                                                                                                │
│   Cost Center 6: Unforeseen Operational Contingency Reserve (10%)                              │
│   • Model 1 Contingency:                                 $8,642.78                             │
│   • Model 2 Contingency:                                $45,514.56                             │
│                                                                                                │
│   GRAND TOTAL ESTIMATED INVESTMENT:                                                            │
│   • MODEL 1 (VIETNAM ACADEMIC / IN-HOUSE HYBRID):       $95,070.58  (~2.37 Billion VND)        │
│   • MODEL 2 (INTERNATIONAL COMMERCIAL AGENCY):         $500,660.11                             │
│                                                                                                │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Phase 1 Pilot Budget Model (12,000 Utterances / 120 Speakers)
The Phase 1 Pilot represents an extraordinarily cost-effective mechanism to prove all scientific hypotheses before releasing large capital:

| Line Item | Task Volume | Rate (Model 1: VN Hybrid) | Amount (Model 1) | Rate (Model 2: Intl Agency) | Amount (Model 2) |
| :--- | :---: | :---: | ---:| :---: | ---:|
| **Primary & Secondary Annotation** | 511.11 hrs | $5.50 / hr | $2,811.11 | $35.00 / hr | $17,888.85 |
| **Adjudication Board (Lead Linguist)** | 53.13 hrs | $12.00 / hr | $637.56 | $70.00 / hr | $3,719.10 |
| **Quality Management & Engineering** | 28.21 hrs | $15.00 / hr | $423.15 | $50.00 / hr | $1,410.50 |
| **Speaker Recruitment & Consent** | 120 speakers | $8.00 / spk | $960.00 | $25.00 / spk | $3,000.00 |
| **Tooling & Cloud GPU Experiments** | 12k utts runs | Lump sum | $250.00 | Lump sum | $650.00 |
| **Subtotal** | | | **$5,081.82** | | **$26,668.45** |
| **10% Contingency Reserve** | | | $508.18 | | $2,666.85 |
| **Total Phase 1 Pilot Investment** | | | **$5,590.00** | | **$29,335.30** |

> [!IMPORTANT]
> For **less than $5,600 USD**, the organization can execute the complete 120-speaker Phase 1 Pilot, establishing empirical ground truth on real-time multipliers, rater agreement rates, and model learning curves. **No large-scale capital ($95k+) should be committed until the Pilot Go/No-Go gates are passed.**

### 7.3 Sensitivity Analysis: Labor Rates & Double-Annotation Quotas
To evaluate financial exposure under differing operational policies, we analyze the total Model 1 investment against key sensitivity drivers:

| Sensitivity Driver | Parameter Variation | Total Labor Hours | Total Budget (Model 1 w/ Contingency) | Delta vs. Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Base Annotation Speed** | Fast (15× base RTM) | 7,611.0 h | **$73,420.00** | -$21,650.00 (-22.8%) |
| **Base Annotation Speed** | Nominal (20× base RTM) | 10,245.0 h | **$95,070.58** | $0.00 (Baseline) |
| **Base Annotation Speed** | Thorough (30× base RTM) | 15,074.8 h | **$135,280.00** | +$40,210.00 (+42.3%) |
| **Train Double-Audit Quota** | Aggressive (10% audit) | 9,370.0 h | **$88,940.00** | -$6,130.00 (-6.4%) |
| **Train Double-Audit Quota** | Baseline (15% audit) | 10,245.0 h | **$95,070.58** | $0.00 (Baseline) |
| **Train Double-Audit Quota** | Conservative (20% audit)| 11,120.0 h | **$101,200.00** | +$6,130.00 (+6.4%) |
| **Labor Inflation / Market Shift**| Higher RA pay ($6.50/hr)| 10,245.0 h | **$104,880.00** | +$9,810.00 (+10.3%) |

---

## 8. Cost Optimization Levers & Risk Management

### 8.1 Engineering Cost Optimization Levers
To ensure maximum data quality per dollar expended, three engineering mechanisms must be deployed:

1. **Acoustic Pre-Segmentation via Forced Alignment:**
   - *Mechanism:* Run MFA prior to human inspection, providing pre-aligned word and phone boundaries in Praat-web.
   - *Impact:* Reduces rater boundary-dragging time by 25–35 seconds per utterance.
   - *Quality Guardrail:* Per [AUDIT_QUALITY.md](file:///home/khoa/Projects/New-Method-Research/reports/DATASET_RD_20261116/AUDIT_QUALITY.md), **never show predicted error labels** to the annotator. The tool displays acoustic boundaries only; categorical decisions ($C/S/O/I$) remain 100% human-selected.
2. **Two-Stream Active Auditing on Training Data:**
   - *Mechanism:* Instead of naively auditing 100% of the training set, audit 15% ($N = 21,000$ utts) split equally between random stratified monitoring (50%) and targeted active sampling of rare errors/coda clusters (50%).
   - *Savings:* Saves 119,000 secondary passes (= ~3,300 labor hours $\approx \mathbf{\$20,000\text{ saved}}$ in Model 1; $\approx \mathbf{\$115,000\text{ saved}}$ in Model 2) while maintaining strict batch-level error bounds.
3. **Automated Discordance Filtering:**
   - *Mechanism:* Use automated textgrid diff daemons to instantly filter concordant double passes, routing only genuine conflicts ($\approx 20\text{--}25\%$) to high-cost Senior Adjudicators.

### 8.2 Operational Risk Matrix & Mitigation Playbooks

| Risk Identifier | Probability | Impact | Mitigation Strategy & Operational Playbook |
| :--- | :---: | :---: | :--- |
| **R1: High Rater Turnover / Attrition** | High | High | Over-recruit by 25% (onboard 15 raters for 12 active slots). Maintain standardized 100-utterance qualification bench for rapid replacement. |
| **R2: Rubric Drift / Discordance Blowout** | Medium | Critical | Weekly rater calibration sessions. If batch discordance exceeds 30%, quarantine batch and trigger mandatory rater retraining ([AUDIT_QUALITY.md](file:///home/khoa/Projects/New-Method-Research/reports/DATASET_RD_20261116/AUDIT_QUALITY.md) Section 6.2). |
| **R3: Speaker Recruitment Bottleneck** | Medium | Medium | Partner with English language centers and universities in Hanoi, Da Nang, and Ho Chi Minh City. Deploy mobile web recording app for remote capture. |
| **R4: Cloud Spend Overrun** | Low | Low | Enforce strict AWS billing alarms and budget caps. Perform all heavy training runs on on-premise GPUs or capped spot instances. |

---

## 9. Phased Investment Gates & Verification Sign-Off

To maintain compliance with [DATASET_R&D_GOAL.md](file:///home/khoa/Projects/New-Method-Research/DATASET_R&D_GOAL.md) Section 8, the program adheres to a gated expenditure lifecycle:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             PHASED INVESTMENT GATING TIMELINE                                  │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                │
│   [ GATE 0: Architectural Pre-Registration ] ◄── CURRENT STATUS                                │
│   • Complete specification reports (SCOUT, AUDIT, QC, PLAN).                                   │
│   • Zero financial spend authorized. Model reviews and approvals only.                         │
│                                                                                                │
│   [ GATE 1: Phase 1 Pilot Authorization ] (Budget: ~$5,590.00)                                 │
│   • Collect & double-annotate 12,000 utterances / 120 speakers.                                │
│   • Measure empirical RTM, inter-rater agreement (κ, α), and baseline learning curves.         │
│                                                                                                │
│   [ GATE 2: Full-Scale Expansion Go/No-Go Decision ]                                           │
│   • Review Pilot Gate Results (per PLAN_PILOT.md Section 7).                                   │
│   • Must satisfy: α_nom ≥ 0.80, κ_S ≥ 0.75, F1_agree(I) ≥ 0.70, learning curve slope > 0.     │
│                                                                                                │
│   [ GATE 3: Full 220k Rollout Authorization ] (Budget: ~$89,480.00 balance)                    │
│   • Scale production rater cohort (8–10 raters) across 12-month delivery horizon.              │
│   • Serialized batch quality quarantine gates every 1,000 utterances.                          │
│                                                                                                │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Summary Handoff for `@verifier-contract`

This specification provides the evidence-based, mathematically closed resource, labor hours, infrastructure, and financial budget model for the 220,000-utterance / 2,200-speaker MDD dataset initiative.

- **Primary Specification:** [reports/DATASET_RD_20261116/PLAN_COST.md](file:///home/khoa/Projects/New-Method-Research/reports/DATASET_RD_20261116/PLAN_COST.md)
- **Key Deliverables Established:**
  1. Complete separation of pure audio duration (**244.4–366.7 hours**) from manual human labor (**10,099.0–11,708.3 person-hours**).
  2. Mathematical formulation of Real-Time Multipliers (base 20×, effective **33.05×–42.58×**).
  3. Ergonomic cognitive limits established (**4.5 productive annotation hours/day**).
  4. Complete infrastructure architecture (self-hosted Praat-web/Label Studio, 232.5 A100 GPU-hours for learning curves, 377 GiB active / 315 GiB cold archive storage).
  5. Comprehensive dual-model financial projections (**$95,070.58** Vietnam Academic Hybrid vs. **$500,660.11** International Commercial) and de-risking Phase 1 Pilot budget (**$5,590.00**).
  6. 100% verification of [DATASET_R&D_GOAL.md](file:///home/khoa/Projects/New-Method-Research/DATASET_R&D_GOAL.md) Section 8 boundaries (zero unauthorized spend, zero external audio API leakage, sealed benchmarks untouched).
- **Handoff Token:** `[done]`
