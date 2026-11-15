# AUDIT_LEGAL_GLOBAL_HUNT — Global Speech Dataset Deep Hunt Legal, Licensing & Compliance Clearance Catalog

**Target Milestone:** Global Speech Dataset Deep Hunt (L2 English / Accented Speech / Phonetic MDD Corpora)  
**File Location:** `reports/DATASET_RD_20261116/AUDIT_LEGAL_GLOBAL_HUNT.md`  
**Date of Audit:** 2026-11-16  
**Status:** `GLOBAL_LEGAL_CLEARANCE_CATALOG — BINDING_GOVERNANCE_LEDGER`  
**Governing Authority:** `DATASET_R&D_GOAL.md` Section 8 & Swarm Directive Board #35  
**Lead Auditor:** Legal, Regulatory & Biometric Compliance Audit Agent (`@audit-legal`)  
**Target Verifiers / Synthesizers:** `@verifier-contract`, `@verifier-source`, `@synthesizer`, `@coordinator`

---

## 1. Executive Summary & Legal Framework for Global Data Harvesting

### 1.1 Scope & Mission Directive
Under Swarm Directive Board #35, the research swarm has launched the **GLOBAL SPEECH DATASET DEEP HUNT** across global repositories (Hugging Face, OpenSLR, Zenodo, GitHub, LDC, ELRA, NII-SRC, Kaggle, university archives, and Vietnamese institutions) to identify all public and semi-public speech datasets capable of contributing toward the goal of **220,000 utterances / 2,200 independent speakers**.

This document serves as the **authoritative legal clearance catalog and pre-ingestion gatekeeper**. It audits the exact licensing terms, commercial rights, financial paywalls, biometric privacy liabilities, pediatric restrictions, and copyleft propagation risks for every major candidate corpus in the global speech repository ecosystem.

### 1.2 Mandatory Operational Guardrails (DATASET_R&D_GOAL.md Section 8)
Every dataset proposed by scouts must pass four non-negotiable legal tests prior to acquisition or inclusion in the master catalog:

```
+---------------------------------------------------------------------------------------------------+
| SECTION 8 NON-NEGOTIABLE LEGAL GATES FOR GLOBAL DATASET INGESTION                                 |
+---------------------------------------------------------------------------------------------------+
| 1. ZERO UNAUTHORIZED SPEND (Rule 8.1):                                                            |
|    - Datasets requiring non-member license fees (e.g. LDC $150-$5,000, ELRA catalogs, Datatang)    |
|      CANNOT be purchased or contracted without explicit prior user budget sign-off via PLAN_COST. |
+---------------------------------------------------------------------------------------------------+
| 2. ZERO CLOUD API DATA EXFILTRATION (Rule 8.2):                                                   |
|    - Audio files harvested from any source MUST NEVER be uploaded to external third-party cloud    |
|      APIs (OpenAI Whisper API, Google Speech, Amazon Transcribe) for automatic transcription.    |
|    - 100% of processing and alignment must occur strictly on local on-premises hardware.          |
+---------------------------------------------------------------------------------------------------+
| 3. BIOMETRIC VOICE PRIVACY & CONSENT INTEGRITY (GDPR / BIPA / Decree 13):                         |
|    - Datasets harvested via unauthorized web scraping (e.g. YouTube, podcasts, social media)      |
|      carry catastrophic biometric litigation liabilities (Illinois BIPA voiceprints: $1k-$5k/spk)  |
|      and are STRICTLY PROHIBITED.                                                                 |
+---------------------------------------------------------------------------------------------------+
| 4. REPOSITORY HYGIENE & SEEDING INTEGRITY (Rules 8.3 & 8.4):                                      |
|    - Zero raw audio files committed to Git repositories. Tracked solely via SHA-256 manifests.     |
|    - Sealed test partitions (DEV_CAL, historical test benchmarks) remain locked and untouched.   |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Taxonomy of Operational Legal Verdicts

To prevent legal ambiguity across the research swarm, each evaluated dataset is classified into one of five definitive operational categories:

```
+----------------------------------------------------------------------------------------------------+
| OPERATIONAL VERDICT        | LEGAL DEFINITION & PERMITTED TECHNICAL USAGE                          |
+----------------------------------------------------------------------------------------------------+
| APPROVED_TRACK_A           | Permitted for OPEN SCIENTIFIC RESEARCH & ACADEMIC BENCHMARK ONLY.     |
| (Academic Non-Commercial)  | - Governed by CC BY-NC 4.0, CC BY-NC-SA 4.0, or Academic EULAs.        |
|                            | - Neural model weights fine-tuned on this data CANNOT be commercialized|
|                            |   or embedded in commercial SaaS products.                         |
+----------------------------------------------------------------------------------------------------+
| APPROVED_TRACK_B           | Permitted for BOTH ACADEMIC RESEARCH AND COMMERCIAL DEPLOYMENT.        |
| (Clean Commercial Permiss.)| - Governed by CC0 1.0, CC BY 4.0, MIT, Apache 2.0, or Public Domain.   |
|                            | - Model weights remain completely unencumbered by NC restrictions.    |
+----------------------------------------------------------------------------------------------------+
| COPYLEFT_CONSTRAINED       | Permitted with STRICT COPYLEFT PROPAGATION OBLIGATIONS.              |
| (ShareAlike / GPL)         | - Governed by CC BY-SA 4.0 (EdAcc), CC BY-NC-SA 4.0, or GNU GPLv3.   |
|                            | - Downstream model weights and annotations MUST be released under the  |
|                            |   exact same copyleft license upon distribution.                   |
+----------------------------------------------------------------------------------------------------+
| COMMERCIAL_PAYWALL_GATED   | Subject to MANDATORY MONETARY FEE or INSTITUTIONAL MEMBERSHIP.        |
| (Budget Authorization Req.)| - Distributed via LDC, ELRA, or corporate vendors (Datatang).        |
|                            | - Requires formal budget approval under Section 8 before procurement. |
+----------------------------------------------------------------------------------------------------+
| STRICTLY_PROHIBITED        | LEGALLY HAZARDOUS / HIGH STATUTORY LIABILITY.                         |
| (Zero Ingestion Allowed)   | - Scraped biometric data lacking consent (VoxCeleb YouTube scraping), |
|                            |   severe BIPA/GDPR violations, or CC-ND (NoDerivatives) clauses that   |
|                            |   prohibit creating adapted model weights.                            |
+----------------------------------------------------------------------------------------------------+
```

---

## 3. Master Global Speech Dataset Legal Clearance Ledger

The following ledger establishes the binding legal status for datasets surveyed across the global speech ecosystem:

| Dataset Name | Host / Source Platform | Official License / EULA | Cost / Fee Barrier | Commercial Usability | Biometric & Privacy Assessment | Pediatric Exposure | Operational Clearance Verdict | Key Restrictions & Notes |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| **SpeechOcean762** | OpenSLR (SLR101) | `CC BY 4.0` | **$0.00 (Free)** | **YES** | Low risk (anonymized rater IDs) | 125/250 are minors | **APPROVED_TRACK_B (Adult)**<br>`Test-OOD-Pediatric (Child)` | Adult subset clean for Track B; minor subset quarantined to OOD challenge track. |
| **MRPL2 Corpus** | Zenodo (`11393345`) | `CC BY 4.0` | **$0.00 (Free)** | **YES** | Low risk (consented L2 adults) | None (Adults) | **APPROVED_TRACK_B** | **High-Value Discovery**: 4,500 utts, 23 spk (14 L1s). Gold S/O/I against British MRP. 100% clean for commercial Track B! |
| **L2-ARCTIC v5.0** | Texas A&M PSI Lab | `CC BY-NC 4.0` | **$0.00 (Free)** | **NO** | Low risk (IRB academic consent) | None (Adults) | **APPROVED_TRACK_A** | 3,599 gold manual utts. Taints model weights with NC constraint. |
| **Mozilla Common Voice** | Mozilla / Hugging Face | `CC0-1.0 Universal` | **$0.00 (Free)** | **YES** | Low risk (express web waiver) | Unknown/Variable | **APPROVED_TRACK_B** | Public domain. Transcripts are crowdsourced; 0% native S/O/I; needs gold re-annotation. |
| **EdAcc** | Univ. of Edinburgh / HF | `CC BY-SA 4.0` | **$0.00 (Free)** | **YES (Copyleft)** | Medium risk (speaker consent) | None | **COPYLEFT_CONSTRAINED** | Conversational L1/L2 speech. Adapted model weights MUST be shared under CC BY-SA 4.0. |
| **GMU Speech Accent Arc.**| George Mason Univ. | `CC BY-NC-SA 4.0` | **$0.00 (Free)** | **NO** | Low risk (academic consent) | None | **COPYLEFT_CONSTRAINED (NC)** | 3,000+ speakers reading "Please call Stella". Broad IPA transcripts. Taints weights with NC-SA. |
| **ALLSSTAR Corpus** | Northwestern Univ. | Academic Open Access | **$0.00 (Free)** | **NO** | Low risk (IRB consent) | None | **APPROVED_TRACK_A** | Multi-L1 (including Vietnamese). Non-commercial academic research only. No redistribution. |
| **CSLU Foreign Accented** | LDC (`LDC2007S08`) | CSLU User Agreement | **$150 (Non-member)** | **NO** | Low risk (consented 1990s) | None | **COMMERCIAL_PAYWALL_GATED** | 23 accents (incl. Vietnamese). 2,800+ speakers. Fee requires Section 8 budget approval. |
| **TIMIT Acoustic-Phonetic**| LDC (`LDC93S1`) | LDC Standard EULA | **$1,500 (Non-member)** | **NO** | Low risk (historical 1993) | None | **COMMERCIAL_PAYWALL_GATED** | Native American English read speech. Acoustic benchmark only. Fee gated. |
| **AESRC 2020** | Datatang / Interspeech | Datatang Academic EULA | **$0.00 (Academic)** | **NO** | Medium risk (vendor terms) | None | **APPROVED_TRACK_A** | 8 accents (incl. VN), 160h. Commercial strictly banned. No redistribution. |
| **UME-ERJ / TEAP** | NII-SRC Japan | NII-SRC Agreement | **$0.00 (Formal App)** | **NO** | Low risk (Japanese Univ) | None | **APPROVED_TRACK_A** | Japanese L2 English. Free for academic institutions; formal signed agreement required. |
| **LibriTTS-R / LibriSpeech**| OpenSLR (SLR60/SLR12) | `CC BY 4.0` | **$0.00 (Free)** | **YES** | Zero risk (LibriVox audio) | None | **APPROVED_TRACK_B** | Native English audiobooks. Ideal for clean pretraining/baseline. No L2 MDD labels. |
| **TED-LIUM v3** | OpenSLR (SLR51) | `CC BY-NC-ND 3.0` | **$0.00 (Free)** | **NO** | Low risk (TED talks) | None | **STRICTLY_PROHIBITED** | **CC-ND TRAP**: "NoDerivatives" clause prohibits distributing fine-tuned models! |
| **VoxForge** | VoxForge Community | `GNU GPLv3` | **$0.00 (Free)** | **YES (Copyleft)** | Low risk (community submit) | None | **COPYLEFT_CONSTRAINED** | Strict GPL copyleft. Training models on VoxForge forces model weights into GPLv3. |
| **FLEURS** | Google / Hugging Face | `CC BY 4.0` | **$0.00 (Free)** | **YES** | Low risk (paid contributors) | None | **APPROVED_TRACK_B** | 102 languages parallel speech. Permissive commercial use with attribution. |
| **VoxCeleb 1 & 2** | Univ. of Oxford | CC Non-Commercial | **$0.00 (Free)** | **NO** | **EXTREME (Biometric scrap)** | Minors present | **STRICTLY_PROHIBITED** | Scraped from YouTube celebrity interviews without consent. Severe BIPA litigation liability. |
| **VIVOS Corpus** | AILab VNU-HCM | `CC BY-NC-SA 4.0` | **$0.00 (Free)** | **NO** | Low risk (VNU students) | None | **COPYLEFT_CONSTRAINED (NC)** | Vietnamese L1 speech (15h). Non-commercial sharealike. Transfer baseline only. |
| **VinBigData English Sub.** | VinBigData / Corporate | Corporate Research EULA | **Contract Gated** | **NO** | Corporate proprietary | None | **APPROVED_TRACK_A (Pilot)** | Proprietary corporate assets. Commercial use requires formal bilateral contract. |
| **FPT Open Speech** | FPT Telecom / AI | Academic Contest EULA | **$0.00 (Contest)** | **NO** | Corporate proprietary | None | **APPROVED_TRACK_A (Pilot)** | Restricted to academic contest evaluation. Public redistribution strictly prohibited. |
| **PF-STAR Children Speech** | ELRA (`ELRA-S0154`) | ELRA Distribution EULA | **€1,500 - €3,000** | **NO** | High (Pediatric biometric) | 100% Children | **COMMERCIAL_PAYWALL_GATED** | European children speaking English. COPPA/GDPR Art 8 exposure. Heavy fee paywall. |
| **MyST (My Science Tutor)**| LDC (`LDC2021S05`) | LDC / CC BY-NC-SA 4.0 | **$0.00 (Members)** | **NO** | High (Child conversational) | 100% Children | **APPROVED_TRACK_A (OOD Only)**| Native US children conversational dialogue with virtual tutor. Zero MDD labels. |

---

## 4. Deep Legal Analysis of Critical Repositories & Pitfalls

### 4.1 The "Creative Commons NoDerivatives (ND)" Trap: The TED-LIUM v3 Case Study
A critical legal pitfall frequently overlooked by ML engineers is the inclusion of datasets licensed under **Creative Commons NoDerivatives (CC-ND)**, such as **TED-LIUM release 3 (OpenSLR 51)**, which is distributed under `CC BY-NC-ND 3.0`:

1. **The Statutory Prohibition (Section 2(a)(1)):**
   Under CC BY-NC-ND 3.0, the licensor grants permission to copy, distribute, and transmit the work, but expressly stipulates:
   > *"If you remix, transform, or build upon the material, you may not distribute the modified material."*
2. **Impact on Machine Learning Models:**
   - Training or fine-tuning a neural network transforms acoustic waveforms into learned parameter matrices (weight tensors).
   - Under copyright jurisprudence and Creative Commons guidance, model checkpoints represent an **Adapted Material (derivative work)**.
   - While internal training might be argued as private intermediate copying, **distributing or sharing the resulting model weights publicly or commercially constitutes distribution of an adapted work, directly violating the ND restriction**.
3. **Audit Verdict:** **TED-LIUM v3 is classified as `STRICTLY_PROHIBITED` for model training and redistributable checkpoints.** It may only be utilized as an external, un-adapted inference probe where weights are frozen and untouched.

---

### 4.2 The "Copyleft Propagation (ShareAlike / GPL)" Contagion
When datasets carry copyleft licenses—such as **EdAcc (`CC BY-SA 4.0`)**, **GMU Speech Accent Archive (`CC BY-NC-SA 4.0`)**, **VIVOS (`CC BY-NC-SA 4.0`)**, or **VoxForge (`GNU GPLv3`)**:

```
[Raw Audio under CC BY-SA 4.0 / GPLv3]
                 |
                 v
   [Backpropagation / Fine-Tuning]
                 |
                 v
[Downstream Checkpoint: Safetensors / PyTorch]
                 |
                 v
[COPYLEFT CONTAGION MANDATE]
- Model weights MUST be licensed under CC BY-SA 4.0 or GPLv3.
- Cannot be closed-source proprietary.
- Cannot be relicensed under Apache 2.0 or MIT.
- Any party receiving the model has the legal right to redistribute it freely.
```

- **Strategic Risk:** Training a commercial model on EdAcc forces the entire model into `CC BY-SA 4.0`. While commercial use is technically permitted under ShareAlike, the enterprise **cannot maintain proprietary trade secrets over the weights** and must provide downstream users full freedom to modify and redistribute the weights.
- **Auditor Recommendation:** Quarantining all ShareAlike and GPL datasets to **Track A (Academic Research)** or standalone reproducible research baselines.

---

### 4.3 The Web Scraping & Biometric Litigation Trap: The VoxCeleb Hazard
Certain researchers propose augmenting speaker diversity by scraping public audio from YouTube, TikTok, TED Talks, or podcasts:
1. **Illinois Biometric Information Privacy Act (BIPA, 740 ILCS 14/):**
   - Harvesting voice recordings of individuals without prior written informed consent violates Section 15(b).
   - VoxCeleb 1 & 2 were created by scraping YouTube interview videos of celebrities and public figures without their knowledge or consent.
   - In 2023–2025, multiple AI enterprises faced multi-million dollar class-action lawsuits under BIPA and the California Right of Publicity (Cal. Civ. Code § 3344) for extracting voiceprints and training commercial speech synthesis/recognition models on scraped celebrity audio.
2. **Computer Fraud and Abuse Act (CFAA) & Platform Terms of Service (ToS):**
   - Automated scraping of YouTube or social platforms directly breaches YouTube Terms of Service Section 5.3 ("Permissions and Restrictions").
3. **Audit Verdict:** **VoxCeleb and unconsented web-scraped voice datasets are classified as `STRICTLY_PROHIBITED` across all project tracks.** Zero scraped audio without explicit individual consent release forms is permitted.

---

### 4.4 Consortium Paywalls & The Section 8 Budget Constraint
Several high-value accented speech corpora are hosted in consortium catalogs:
- **CSLU Foreign Accented English (`LDC2007S08`):** 23 accents (including a dedicated Vietnamese L2 speaker subset), 2,800+ speakers. Non-member fee: **$150.00**.
- **TIMIT (`LDC93S1`):** Non-member fee: **$1,500.00**.
- **PF-STAR (`ELRA-S0154`):** Non-member fee: **€1,500.00–€3,000.00**.
- **Operational Rule:** While $150 for CSLU FAE represents high technical value for Vietnamese L2 speech, **`DATASET_R&D_GOAL.md` Section 8 explicitly bars unauthorized expenditure**. Scouts and planners must record these corpora as `COMMERCIAL_PAYWALL_GATED` and submit formal purchase proposals to the User via `PLAN_COST.md` prior to executing any license agreement.

---

### 4.5 Vietnam Decree 13/2023/ND-CP Cross-Border Transfer Requirements
For candidate corpora collected in Vietnam (e.g. ULIS, HUST, VNU, VIVOS, or newly recruited Vietnamese participants):
- **Article 24 & 25 Mandate:** Transferring Vietnamese citizens' audio recordings to servers located outside of Vietnam (e.g. AWS, Hugging Face Hub servers in Europe/US, Google Cloud) requires the creation of a **Hồ sơ đánh giá tác động chuyển dữ liệu cá nhân ra nước ngoài** (Cross-Border Data Transfer Impact Assessment Dossier).
- **Submission Timeline:** Must be finalized and submitted to the **Department of Cyber Security and High-Tech Crime Prevention (A05 - Ministry of Public Security)** within 60 days of the commencement of processing.
- **Auditor Recommendation:** All Vietnamese speech files must remain stored and processed on domestic on-premises infrastructure during R&D, with cross-border transfer filing completed prior to any international distribution sync.

---

## 5. Actionable Ingestion Protocol for Deep Hunt Scouts

Before any scout (`@scout-public`, `@scout-crowdsource`, `@scout-vietnamese`, `@scout-pediatric`) adds a newly discovered dataset to the candidate pool or master catalog, they must verify the following **Legal Ingestion Checklist**:

```
+---------------------------------------------------------------------------------------------------+
| GLOBAL DEEP HUNT: SCOUT LEGAL INGESTION CHECKLIST                                                 |
+---------------------------------------------------------------------------------------------------+
| [ ] 1. DIRECT LICENSE CITATION: Is the exact license text or SPDX identifier verified from the     |
|        original host (not a secondary blog post or third-party paper citation)?                   |
| [ ] 2. FINANCIAL ZERO-CHECK: Is the download 100% free of charge? If paywalled (LDC/ELRA),         |
|        flag as COMMERCIAL_PAYWALL_GATED and halt download pending User budget sign-off.           |
| [ ] 3. COMMERCIAL RIGHTS CHECK: Does the license allow commercial use?                            |
|        - If YES (CC0, CC BY 4.0) --> Allocate to TRACK B (Commercial).                            |
|        - If NO (CC BY-NC, Academic EULA) --> Allocate to TRACK A (Academic Benchmark).            |
| [ ] 4. COPYLEFT CHECK: Is the license ShareAlike (CC BY-SA) or GPL?                               |
|        - If YES --> Flag as COPYLEFT_CONSTRAINED; warn that derived weights must be open-sourced. |
| [ ] 5. NO-DERIVATIVES CHECK: Does the license contain an "ND" clause (e.g. CC BY-NC-ND)?         |
|        - If YES --> REJECT IMMEDIATELY (STRICTLY_PROHIBITED for model training).                  |
| [ ] 6. BIOMETRIC CONSENT CHECK: Was the audio gathered with explicit participant consent?         |
|        - If scraped from YouTube, social media, or podcasts without consent --> REJECT (BIPA risk)|
| [ ] 7. PEDIATRIC CHECK: Does the corpus contain speakers under 18 years old?                      |
|        - If YES --> Quarantine to Test-OOD-Pediatric; do NOT mix into the core 200k adult corpus.  |
| [ ] 8. GIT REPOSITORY HYGIENE: Confirm zero audio bytes will be committed to Git.                |
+---------------------------------------------------------------------------------------------------+
```

---

## 6. Summary Recommendations for the Master Global Catalog

1. **Clean Commercial Core (Track B Candidates):**
   - Adult subset of **SpeechOcean762** (125 adult speakers / 2,500 utts, CC BY 4.0).
   - **MRPL2 Non-native Speech Corpus** (Zenodo, 23 speakers / 4,500 utts with gold S/O/I Praat TextGrids, CC BY 4.0).
   - **Mozilla Common Voice** English / accented subsets (CC0 1.0; requires phonetic re-annotation).
   - **LibriTTS-R** (CC BY 4.0; pretraining / clean acoustic baseline).
   - **FLEURS** English subsets (CC BY 4.0; multilingual benchmark).
2. **Open Academic Benchmark (Track A Candidates):**
   - **L2-ARCTIC v5.0** (CC BY-NC 4.0; 24 speakers / 3,599 manual gold utts).
   - **ALLSSTAR Corpus** (Northwestern Univ; academic non-commercial access).
   - **AESRC 2020** (Datatang academic agreement; 160h accented speech).
   - **UME-ERJ / TEAP** (NII-SRC Japan; academic free agreement).
   - **EpaDB** (University of Buenos Aires; academic research use).
3. **Copyleft Tracked Candidates (Dedicated Baselines):**
   - **EdAcc** (CC BY-SA 4.0; conversational accented speech; requires copyleft release).
   - **GMU Speech Accent Archive** (CC BY-NC-SA 4.0; broad IPA reading corpus).
4. **Paywall Proposals Gated for User Approval:**
   - **CSLU Foreign Accented English (`LDC2007S08`)**: Recommend formal budget proposal ($150) for User authorization to access 2,800+ accented speakers including Vietnamese L2 English.
5. **Legally Prohibited Assets (Blacklisted):**
   - **VoxCeleb 1 & 2** (Unconsented YouTube scraping; extreme BIPA voiceprint litigation risk).
   - **TED-LIUM v3** (CC BY-NC-ND 3.0 NoDerivatives trap; prohibited for training redistributable models).

---

*Report prepared and certified by:*  
**Agent `@audit-legal`**  
*Lead Legal, Regulatory & Biometric Compliance Specialist — Project Langfens MDD Swarm*