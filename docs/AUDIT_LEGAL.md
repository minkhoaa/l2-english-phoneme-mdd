# AUDIT_LEGAL — Legal, Licensing, Biometric Voice Privacy & Regulatory Compliance Audit

**Target Milestone:** R&D Planning for 220,000 Utterances / 2,200 Independent Speakers (L2 English MDD Corpus)  
**File Location:** `reports/DATASET_RD_20261116/AUDIT_LEGAL.md`  
**Date of Audit:** 2026-11-16  
**Status:** `LEGAL_AUDIT_COMPLETE — BINDING_GOVERNANCE_FRAMEWORK`  
**Governing Authority:** `DATASET_R&D_GOAL.md` (Section 4.5 & Section 8)  
**Lead Auditor:** Legal & Regulatory Compliance Audit Agent (`@audit-legal`)  
**Target Verifier:** `@verifier-contract`

---

## 1. Executive Summary & Regulatory Perimeter

### 1.1 Objective & Context
This audit establishes the comprehensive legal, intellectual property, copyright, data protection, and biometric voice privacy governance framework for the planned expansion to **220,000 phonetically annotated utterances across 2,200 independent speakers** (Train: 140,000 / 1,400 spk; Dev: 20,000 / 200 spk; Test Internal: 40,000 / 400 spk; Test External: 20,000 / 200 spk) for L2 English Mispronunciation Detection and Diagnosis (MDD) targeting Substitution (S), Omission (O), and Insertion (I) error events.

This document serves as the mandatory legal gatekeeper under **`DATASET_R&D_GOAL.md` Section 8**. In accordance with established project governance:
> **Hard Legal Boundary:** The approval of planning goals (`SCALE_AGREED — DETAILED_R&D_PENDING`) does **not** constitute authorization to incur financial expenditure, procure commercial datasets, recruit human participants, upload biometric voice data to external cloud services, or unseal protected test sets (`DEV_CAL` / official benchmarks). All research activities must remain strictly compliant with local and international statutory requirements.

### 1.2 Core Compliance Findings & Risk Ledger

| Regulatory Domain | Core Statute / Instrument | Compliance Mandate | Risk Level | Actionable Constraint |
| :--- | :--- | :--- | :---: | :--- |
| **Open Licensing & IP** | Creative Commons (CC BY, CC BY-NC, CC0), Berne Convention, US Copyright Act | Prevent license contamination; maintain strict attribution and notice chains. | **CRITICAL** | Corpora with `CC BY-NC 4.0` (e.g. L2-ARCTIC) taint fine-tuned neural model weights with NonCommercial restrictions. Dual-track pipeline required. |
| **Voice as Biometric Data** | EU GDPR (Art. 4(14), Art. 9, Art. 17), EDPB Guidelines 02/2021 | Voice recordings processed for speaker embeddings/acoustic models trigger Special Category protections. | **HIGH** | Explicit opt-in consent (Art. 9(2)(a)) mandatory. Technical support for Right to Erasure (Art. 17) via speaker-isolated data architecture. |
| **US Biometric Privacy** | Illinois BIPA (740 ILCS 14/), Texas CUBI, California CCPA/CPRA | Voiceprints are statutory biometric identifiers; strict prohibition on unauthorized collection or commercial sale. | **CRITICAL** | Written informed release required prior to recording. Private right of action with statutory damages ($1k–$5k/violation) under BIPA. |
| **Pediatric Voice Protection** | US COPPA (15 U.S.C. §§ 6501–6506; 16 C.F.R. Part 312), GDPR Art. 8, VN Decree 13 Art. 20 | Child voice recordings are sensitive personal identifiers requiring verifiable parental consent. | **EXTREME** | Quarantining pediatric data into an isolated Out-of-Domain Challenge Track (`Test-OOD-Pediatric`). Exclude children from core 200k corpus. |
| **Vietnam Data Protection** | Decree 13/2023/ND-CP on Personal Data Protection | Voice audio constitutes personal data; biological characteristics constitute Sensitive Personal Data. | **HIGH** | Strict consent protocol; Mandatory Cross-Border Transfer Impact Assessment (DPIA) filed with MPS (A05) before international sharing. |
| **Operational Boundaries** | `DATASET_R&D_GOAL.md` Section 8 | Absolute ban on unauthorized spending, third-party cloud API uploads, test unsealing, or committing audio to Git. | **ZERO-TOLERANCE** | 100% on-premises execution for ASR/alignment; Git repo hygiene strictly enforced via SHA-256 hash manifests. |

---

## 2. Comparative Intellectual Property & Licensing Framework

### 2.1 Legal Anatomy of Speech Dataset Assets
A modern machine learning speech corpus is not a monolithic copyright entity. It is a composite multi-layered intellectual property artifact comprising four distinct legal tiers:

```
+-----------------------------------------------------------------------------------+
| 1. RAW AUDIO WAVEFORM (.wav / .flac)                                              |
|    - Performer's Right (Speaker's acoustic performance: WPPT Art. 5; US 17 USC 102)|
|    - Phonogram Producer's Right (Fixation and master recording ownership)         |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 2. TEXT PROMPTS & CANONICAL TRANSCRIPTS                                           |
|    - Literary Work Copyright (Sentences, paragraphs, prompts)                     |
|    - Status: Public domain (Project Gutenberg), CC-licensed, or proprietary        |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 3. PHONETIC & ERROR ANNOTATIONS (TextGrids, JSON, S/O/I alignments)               |
|    - Sui Generis Database Rights (EU Directive 96/9/EC)                           |
|    - Compilation Copyright / Original Derivative Authorship (Annotator analysis)  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 4. TRAINED NEURAL MODEL WEIGHTS (PyTorch / Safetensors / Checkpoints)              |
|    - Adapted Material / Derivative Work under CC 4.0 Section 1(a)                 |
|    - Parameter representations capturing acoustic & phonetic distribution         |
+-----------------------------------------------------------------------------------+
```

### 2.2 Granular License Comparison Matrix

| License Type | SPDX Identifier | Commercial Training Permitted? | Commercial Model Weights Distribution? | Attribution Mandatory? | Sublicensing Allowed? | Impact on Derived Neural Weights ("Contagion") |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Creative Commons Zero 1.0** | `CC0-1.0` | **YES** | **YES** | NO (Legally waived; required by academic ethics) | **YES** | **Zero restriction**. Model weights can be commercialized or dual-licensed freely. |
| **Creative Commons Attribution 4.0** | `CC-BY-4.0` | **YES** | **YES** | **YES** (Must credit author, URI, license text, & changes) | **YES** | **Permissive**. Downstream model weights can be distributed commercially provided original notices are retained. |
| **Creative Commons Attribution-NonCommercial 4.0** | `CC-BY-NC-4.0` | **NO** (Commercial exploitation strictly barred) | **NO** | **YES** | **NO** | **CONTAGIOUS NON-COMMERCIAL RESTRICTION**. Weights fine-tuned on this data become an "Adapted Material" restricted to NC use. |
| **Academic / Research Use Only** | Custom EULA / Non-standard | **NO** | **NO** | **YES** | **NO** | **SEVERE RESTRICTIONS**. Prohibits commercial SaaS, fee-charging tutoring, external APIs, and often requires annual renewal or data return. |
| **Commercial / Proprietary License** | Custom Commercial Contract | **YES** | **YES** | Dependent on SLA / contract | Dependent on contract | Fully compliant for commercial enterprise deployment; backed by indemnification and warranty. |

### 2.3 Deep Dive: The "Contagious NonCommercial Constraint" of CC BY-NC 4.0

#### 2.3.1 Definition of "NonCommercial" (Section 1(k))
Under Section 1(k) of the CC BY-NC 4.0 legal code:
> *"NonCommercial means not primarily intended for or directed towards commercial advantage or monetary compensation."*

Legal jurisprudence and Creative Commons official guidance establish that "commercial advantage" is interpreted broadly:
1. **Direct Commercial Exploitation:** Selling model weights, offering a paid API, bundling the model in a subscription SaaS tutoring platform, or charging consultation fees based on model output.
2. **Indirect Commercial Advantage:** An enterprise training an internal model to reduce operating costs, deploying pronunciation scoring within an internal corporate training portal for employees, or generating proprietary marketing leads.
3. **Academic Exemption:** Pure scholarly research conducted by recognized non-profit academic institutions leading to open peer-reviewed publications qualifies unambiguously as NonCommercial. However, venture-backed startups or corporate research labs collaborating with universities face severe legal exposure if they commercialize the derived artifacts.

#### 2.3.2 Model Weights as "Adapted Material" (Derivative Works)
Under Section 1(a) of CC BY-NC 4.0:
> *"Adapted Material means material that is subject to Copyright and Similar Rights that is derived from or based upon the Licensed Material and in which the Licensed Material is translated, altered, arranged, transformed, or otherwise modified in a manner requiring permission under the Copyright and Similar Rights held by the Licensor."*

When a neural acoustic model (e.g., wav2vec 2.0, HuBERT, or Conformer) is fine-tuned on L2-ARCTIC v5.0 (which is licensed under `CC BY-NC 4.0`):
- The backpropagation algorithm updates weight tensors based directly on the acoustic waveforms and phonetic error labels of L2-ARCTIC.
- The resulting checkpoint incorporates parameter patterns and phonetic representations extracted directly from the protected corpus.
- Under both Creative Commons guidance and prevailing legal consensus, **fine-tuned model checkpoints represent an Adapted Material (derivative work)**.
- **The Legal Impact:** Any model fine-tuned on `CC BY-NC 4.0` data automatically inherits the NonCommercial restriction. It **cannot** be relicensed under Apache 2.0 or MIT, nor deployed in commercial products, without securing a separate commercial license from Texas A&M University.

#### 2.3.3 The Boundary: Training vs. Fine-Tuning vs. Direct Evaluation
To prevent accidental license pollution across the 220k R&D project, we define three legally distinct operational regimes:

```
+-------------------------------------------------------------------------------------------------+
| OPERATIONAL REGIME                                     | LEGAL STATUS OF DOWNSTREAM WEIGHTS     |
+-------------------------------------------------------------------------------------------------+
| 1. PRETRAINING / FINE-TUNING ON CC BY-NC 4.0           | TAINTED: Inherits CC BY-NC 4.0          |
|    Model learns internal representations from NC data. | Commercial deployment prohibited.      |
+-------------------------------------------------------------------------------------------------+
| 2. DIRECT BENCHMARK EVALUATION ON CC BY-NC 4.0         | CLEAN: Remains under base model license|
|    Model weights are FROZEN. No gradients computed.   | Commercial deployment permitted, but   |
|    Test data only acts as an inference probe.         | benchmark results must cite attribution.|
+-------------------------------------------------------------------------------------------------+
| 3. PSEUDO-LABELING / KNOWLEDGE DISTILLATION            | HIGH RISK: Legally ambiguous derivative|
|    Teacher trained on NC data generates pseudo-labels | Often deemed an indirect adaptation     |
|    used to train a student model.                     | violating the spirit of Section 1(k).   |
+-------------------------------------------------------------------------------------------------+
```

### 2.4 Text and Data Mining (TDM) & Fair Use Limitations
Certain technical teams attempt to bypass CC-NC or academic restrictions by invoking **Fair Use (US 17 U.S.C. § 107)** or **Text and Data Mining (TDM) Exceptions (EU DSM Directive 2019/790 Articles 3 & 4)**:
- **US Fair Use (17 U.S.C. § 107):** While intermediate copying for training foundational AI models has some defense under transformative use arguments (*Authors Guild v. Google*), Fair Use **never authorizes public redistribution** of full audio datasets or copyrighted phonetic alignments. Furthermore, using a dataset specifically marketed for research in a commercial product usurps the licensor's commercial licensing market (the 4th Fair Use factor).
- **EU TDM Exceptions (DSM Articles 3 & 4):** Article 3 permits TDM for research organizations for scientific purposes; Article 4 permits general TDM unless rights holders have expressly reserved their rights (opt-out). Creators who license under `CC BY-NC 4.0` have expressly reserved commercial rights.
- **Auditor Verdict:** **Relying on Fair Use or TDM exceptions to commercialize models trained on CC BY-NC 4.0 or academic-only corpora is an unacceptable legal risk.** The project must enforce architectural separation.

---

## 3. Biometric Voice Data & Privacy Regulations

### 3.1 Technical & Legal Threshold: When Does Voice Become Biometric Data?
A fundamental question in speech processing is whether raw audio constitutes biometric data per se, or only when subjected to specific technical processing:

```
[Raw Acoustic Waveform]
         |
         +---> (1) Phonetic Transcription / MDD (Phoneme error: S/O/I)
         |         Legal Status: General Personal Data (GDPR Art. 6; VN Decree 13 Basic)
         |
         +---> (2) Acoustic Feature Extraction (F0, Formants, Energy)
         |         Legal Status: Borderline Sensitive Data
         |
         +---> (3) Speaker Embeddings (x-vectors, d-vectors, ECAPA-TDNN)
                   Legal Status: SPECIAL CATEGORY BIOMETRIC DATA (GDPR Art. 9; BIPA § 10)
                   Allows/confirms unique identification of an individual.
```

### 3.2 European Union — General Data Protection Regulation (GDPR - EU 2016/679)

#### 3.2.1 Statutory Definitions & Prohibitions
1. **Article 4(14) Definition:**
   > *"Biometric data means personal data resulting from specific technical processing relating to the physical, physiological or behavioural characteristics of a natural person, which allow or confirm the unique identification of that natural person, such as facial images or dactyloscopic data."*
2. **Article 9(1) General Prohibition:**
   Processing of biometric data for the purpose of uniquely identifying a natural person is **strictly prohibited**, unless one of the exceptions in Article 9(2) applies.
3. **Applicable Legal Bases under Article 9(2):**
   - **Article 9(2)(a) — Explicit Consent:** The data subject must give explicit, affirmative, freely given, specific, and informed consent for one or more specified purposes.
   - **Article 9(2)(j) — Scientific Research Exception:** Processing is necessary for scientific research purposes in accordance with Article 89(1), based on Union or Member State law, provided appropriate technical and organizational safeguards are implemented (pseudonymization, encryption, access control).
   - **Critical Warning:** Legitimate Interest (Article 6(1)(f)) is **insufficient** to override Article 9 prohibitions.

#### 3.2.2 EDPB Guidelines 02/2021 on Virtual Voice Assistants
The European Data Protection Board (EDPB) specifically addressed voice processing:
- **Purpose Specification:** Audio data processed to evaluate pronunciation cannot be repurposed for biometric authentication or marketing profiling without separate explicit consent.
- **Bystander Recordings:** Audio captured in shared acoustic environments containing unconsented third-party speech violates GDPR principles. Corpora must undergo strict acoustic filtering to eliminate non-target background voices.
- **Data Minimization:** Retaining raw uncompressed audio indefinitely when derived acoustic features or aligned transcripts suffice constitutes a potential breach of Article 5(1)(c).

#### 3.2.3 Data Subject Rights & The Engineering Impact of Article 17 (Right to Erasure)
Under GDPR Chapter III, participants possess enforceable rights:
- **Article 17 "Right to be Forgotten" (Erasure):** A speaker has the right to withdraw consent at any time and demand the erasure of their personal and biometric data.
- **Engineering Challenge — Machine Unlearning:** If a speaker demands deletion after a neural network has been trained on their voice:
  - Deleting the audio file from the server is necessary but legally insufficient if their acoustic biometric characteristics remain memorized in the model weights.
  - "Machine unlearning" in deep neural speech models remains an unsolved research problem with no mathematical certainty.
  - **Compliance Architecture:** The dataset architecture must maintain strict **Speaker Isolation Modules**. If a speaker requests deletion, their data must be purged from all distributed archives, manifests, and the model must be scheduled for retraining from the nearest checkpoint excluding that speaker ID.
- **Articles 15 (Access), 16 (Rectification), 18 (Restriction), and 21 (Right to Object):** The data controller must provide an automated or administrative mechanism to fulfill requests within 30 days.

#### 3.2.4 Data Protection Impact Assessment (DPIA) & Cross-Border Transfers
- **Article 35(3) DPIA Requirement:** Mandatory for "processing on a large scale of special categories of data". Curating 220,000 utterances across 2,200 human speakers represents large-scale processing. A formal DPIA must be maintained on record.
- **Chapter V International Transfers (Articles 44–49):** If EU participant audio is processed on compute infrastructure outside the EEA (e.g. Vietnam or US), valid transfer mechanisms—specifically **Standard Contractual Clauses (SCCs)** or an Adequacy Decision—are mandatory.

---

### 3.3 United States — Biometric Privacy Statutes & Jurisprudence

#### 3.3.1 Illinois Biometric Information Privacy Act (BIPA, 740 ILCS 14/)
BIPA is the most stringently enforced biometric statute globally and represents the highest commercial litigation risk for speech AI companies:
1. **Explicit Inclusion of Voiceprints (Section 10):**
   > *"Biometric identifier means a retina or iris scan, fingerprint, voiceprint, or scan of hand or face geometry."*
2. **Core Compliance Mandates (Section 15):**
   - **Section 15(a) Public Retention Schedule:** Must publish a written policy establishing a retention schedule and guidelines for permanent destruction when the initial purpose is satisfied or within 3 years of the individual's last interaction.
   - **Section 15(b) Informed Written Consent:** Prior to capturing or obtaining a voiceprint, the collector must inform the subject in writing of the specific purpose and duration, and receive an executed written release.
   - **Section 15(c) Absolute Prohibition on Sale / Profit:**
     > *"No private entity in possession of a biometric identifier or biometric information may sell, lease, trade, or otherwise profit from a person's or a customer's biometric identifier or biometric information."*
     *(Selling access to a proprietary dataset containing Illinois residents' raw voiceprints directly violates Section 15(c)).*
   - **Section 15(d) Disclosure Restrictions:** Prohibits dissemination without consent or statutory authorization.
3. **Private Right of Action & Liquidated Damages (Section 20):**
   - Plaintiffs do not need to prove actual injury or financial damages; a procedural statutory violation confers legal standing (*Rosenbach v. Six Flags Entertainment Corp.*).
   - Liquidated damages: **$1,000 per negligent violation**; **$5,000 per intentional or reckless violation**, plus attorney's fees.
   - In 2024, the Illinois legislature passed Senate Bill 2979, amending BIPA so that multiple collections using the same biometric identifier from the same person constitute a single violation, but class-action liability for thousands of speakers remains catastrophic (e.g., 2,200 speakers × $5,000 = $11,000,000 exposure).

#### 3.3.2 Other US State Statutes
- **Texas CUBI (Tex. Bus. & Com. Code § 503.001):** Prohibits capturing a biometric identifier (including voiceprints) without prior consent; civil penalties up to $25,000 per violation enforced by the Texas Attorney General.
- **Washington State (RCW 19.375):** Requires notice, consent, and security protections for biometric identifiers used in commercial databases.
- **California Consumer Privacy Act / CPRA (Cal. Civ. Code § 1798.140):** Voice recordings and biometric data are categorized as "Sensitive Personal Information" (SPI). Consumers possess statutory rights to limit processing, opt out, and demand deletion.

---

### 3.4 Vietnam — Personal Data Protection Decree (Decree 13/2023/ND-CP)

#### 3.4.1 Classification of Voice & Biometric Data
1. **Article 2(1) — Voice as Personal Data:**
   Personal data includes information in the form of symbols, letters, numbers, images, **sounds (âm thanh)**, or equivalent formats on electronic media that is associated with or helps identify a specific natural person.
2. **Article 2(4)(đ) — Voice Attributes as Sensitive Personal Data:**
   > *"Thông tin về thuộc tính vật lý, đặc điểm sinh học riêng của cá nhân"* (Information regarding physical attributes and biological characteristics unique to an individual) is classified as **Sensitive Personal Data (Dữ liệu cá nhân nhạy cảm)**.
   Extracting vocal tract characteristics, pitch contours, and speaker embeddings falls squarely under this definition.

#### 3.4.2 Statutory Processing Obligations
- **Article 9 & 11 (Valid Consent):** Consent must be voluntary, affirmative, informed, specific to the purpose, and expressed in a format that can be printed or reproduced in writing. Silence or non-response does not constitute consent.
- **Article 17 (Processing Sensitive Data):** Data subjects must be explicitly informed that their sensitive personal data (biometric/voice) is being processed prior to collection.
- **Articles 24 & 25 (Cross-Border Transfer Dossier):**
  If Vietnamese citizens' voice recordings or associated metadata are transferred to foreign cloud servers (e.g., AWS us-east-1, Google Cloud, Hugging Face Hub servers located abroad):
  - The Data Controller/Processor must formulate a **Cross-Border Transfer Impact Assessment Dossier (Hồ sơ đánh giá tác động chuyển dữ liệu cá nhân ra nước ngoài)**.
  - The dossier must be submitted to the **Department of Cyber Security and High-Tech Crime Prevention (A05 - Ministry of Public Security)** within 60 days of commencing processing.
  - Ministry inspections occur annually or upon suspected violations.

---

## 4. Pediatric & Minor Speech Data Governance

### 4.1 US Children's Online Privacy Protection Act (COPPA, 15 U.S.C. §§ 6501–6506; 16 C.F.R. Part 312)

#### 4.1.1 Scope & Biometric Classification
COPPA applies to online operators directed to children under 13 years of age, or general audience services knowingly collecting personal information from children under 13.
- In 2017, the Federal Trade Commission (FTC) issued an updated policy statement clarifying that **an audio file containing a child's voice constitutes "personal information"** because voice recordings are inherently personal and capable of identifying an individual.

#### 4.1.2 The Strict Limitations of the FTC "Audio File Exception"
In 2017, the FTC established an enforcement policy exception for voice commands:
- **The Exemption:** An operator is not required to obtain prior verifiable parental consent *only if* the audio file is collected solely as a replacement for written words (e.g., voice search or basic command execution).
- **The Catch:** The audio file **must be deleted immediately** after completing the transcription, and **cannot be used for any other purpose**, including machine learning training, behavioral profiling, or speaker identification.
- **CRITICAL AUDIT FINDING:** **The FTC Audio File Exception DOES NOT APPLY to AI dataset collection, model training, or corpus archiving.** Any retention of children's voice recordings for pronunciation research requires full Verifiable Parental Consent (VPC).

#### 4.1.3 Verifiable Parental Consent (VPC) Mechanics
Obtaining legal VPC under 16 C.F.R. § 312.5 requires rigorous, high-friction verification methods:
1. Providing a consent form to be signed and returned by mail, fax, or electronic scan.
2. Requiring the parent to use a credit card, debit card, or other online payment system that provides notification of each discrete transaction.
3. Connecting to trained personnel via video-conference.
4. Checking government-issued identification against specialized databases.
*Administrative cost:* Typically $3.00–$7.00 per verified consent transaction via third-party verification providers, introducing severe friction and drop-off.

---

### 4.2 European Union GDPR Article 8 (Children's Consent)
- Under GDPR Article 8, where information society services rely on consent, processing a child's data is lawful only if the child is at least **16 years old** (or a lower age established by national law, not below 13; e.g., 13 in the UK, 16 in Germany, France, and Ireland).
- For children below the statutory age, consent must be given or authorized by the holder of parental responsibility.
- Processing children's biometric data triggers heightened scrutiny under **Recital 38** (vulnerable natural persons requiring specific protection).

---

### 4.3 Vietnam Law on Children (2016) & Decree 13/2023/ND-CP Article 20
Vietnam enforces strict dual-consent rules regarding minor personal data:
- **Article 20(1) Classification:** A child is defined as a person **under 16 years of age**.
- **The Dual-Consent Rule (Article 20(2)):**
  - For children **under 7 years of age**: Consent must be provided by the child's parent or legal guardian.
  - For children **from 7 to under 16 years of age**: Consent **must be obtained from BOTH the child AND the parent/legal guardian**.
  - Collecting voice data from a Vietnamese child aged 7–15 with only parental consent (or only child consent) is an illegal statutory violation.

---

### 4.4 Architectural Recommendation for the 220k MDD Initiative
Given the severe statutory liabilities under COPPA, BIPA, GDPR Art. 8, and VN Decree 13 Art. 20:

```
+---------------------------------------------------------------------------------------------------+
| MANDATORY ARCHITECTURAL DECISION: TOTAL QUARANTINE OF PEDIATRIC SPEECH                           |
+---------------------------------------------------------------------------------------------------+
| 1. CORE 200,000-UTTERANCE CORPUS (Train 140k / Dev 20k / Test-Internal 40k):                     |
|    - STRICTLY ADULTS ONLY (Age >= 18; or >= 16 in compliant jurisdictions).                      |
|    - Zero minor participants. Eliminates COPPA VPC overhead and dual-consent liability.          |
+---------------------------------------------------------------------------------------------------+
| 2. ISOLATED PEDIATRIC OUT-OF-DOMAIN TRACK (Test-OOD-Pediatric):                                  |
|    - Allocated solely within the 20,000 External Test quota.                                     |
|    - Uses existing verified public academic corpora (e.g., SpeechOcean762 child subset).         |
|    - Fully quarantined: No pediatric data allowed in the training or validation splits.         |
|    - Requires separate IRB approval, localized consent verification, and restricted access ACLs.  |
+---------------------------------------------------------------------------------------------------+
```

---

## 5. Informed Consent Protocol & Legal Chain of Custody

### 5.1 The Four-Pillar Consent Standard
For any newly recruited speakers (e.g. Vietnamese L2 English speakers collected during pilot or rollout phases), consent must satisfy the highest common denominator among GDPR, BIPA, and Decree 13:

```
               +---------------------------------------------------------------+
               |                 THE FOUR-PILLAR CONSENT STANDARD              |
               +---------------------------------------------------------------+
                                               |
      +------------------------+---------------+-----------------------+
      |                        |                                       |
      v                        v                                       v
+------------------+  +------------------+  +-------------------+  +------------------+
| 1. SPECIFICITY   |  | 2. BIOMETRIC     |  | 3. TIERED         |  | 4. REVOCATION    |
| & PURPOSE        |  | DISCLOSURE       |  | PERMISSIONS       |  | & TAKEDOWN       |
| Explicit MDD     |  | Timbre, pitch,   |  | Tier 1: Internal  |  | Deletion policy; |
| phonetic model   |  | vocal tract,     |  | Tier 2: Academic  |  | clear terms on   |
| R&D and scoring. |  | voiceprints.     |  | Tier 3: Commercial|  | derivative model |
+------------------+  +------------------+  +-------------------+  +------------------+
```

### 5.2 Model Participant Agreement & Biometric Voice Release Template
The following legally binding template is approved for adaptation in participant onboarding:

```markdown
### PARTICIPANT INFORMED CONSENT & BIOMETRIC SPEECH DATA RELEASE FORM

1. RESEARCH PURPOSE & DATA USE:
I hereby authorize the Project Langfens Research Team ("Research Team") to record my voice speaking English sentences, and to process, analyze, and store the resulting audio recordings ("Recordings"). I understand that the Recordings will be used to develop, train, evaluate, and benchmark artificial intelligence algorithms for speech recognition, non-native pronunciation diagnosis, and phonetic error detection (Substitution, Omission, Insertion).

2. BIOMETRIC VOICE DATA DISCLOSURE:
I understand and acknowledge that my voice audio contains acoustic characteristics unique to my vocal anatomy (including pitch, formants, timbre, and acoustic embeddings) which may constitute "biometric data" or "biometric identifiers" under applicable law (including EU GDPR Article 4(14), Illinois BIPA 740 ILCS 14/, and Vietnam Decree 13/2023/ND-CP). I explicitly consent to the technical processing of my voice for speech and acoustic modeling purposes.

3. TIERED SHARING & DISTRIBUTION PERMISSIONS (Select all that apply):
[ ] TIER 1 (Internal R&D Only):
    My Recordings may be used solely by the Research Team for internal algorithm development. No audio or annotations will be publicly distributed.
[ ] TIER 2 (Open Scientific & Academic Benchmark):
    My Recordings and phonetic annotations may be published in anonymized research datasets under a Creative Commons Attribution-NonCommercial (CC BY-NC 4.0) license for non-commercial academic research.
[ ] TIER 3 (Permissive & Commercial AI Training):
    My Recordings may be used in commercial AI systems, licensed under Creative Commons Attribution (CC BY 4.0) or commercial enterprise licenses.

4. ANONYMIZATION & PRIVACY SAFEGUARDS:
My name, email, phone number, and official government identifiers will NEVER be linked to or released with the audio files. I will be assigned an anonymous cryptographic UUID (e.g., SPK_VN_XXXX). Raw audio metadata (EXIF/WAV headers) containing device serials or GPS coordinates will be stripped.

5. RETENTION SCHEDULE & RIGHT OF REVOCATION:
Recordings will be stored on secure, encrypted servers for up to five (5) years following the conclusion of the research project. I retain the right to withdraw my consent and demand the permanent erasure of my Recordings at any time by emailing privacy@project-langfens.org with my Participant ID.
*Limitation Notice:* I acknowledge that upon receiving a valid deletion request, my audio files will be permanently purged from active storage and future dataset releases within 30 days. However, pre-existing trained neural network model weights that have already been publicly released cannot be retroactively modified without complete model retraining.

6. COMPENSATION & VOLUNTARY PARTICIPATION:
I participate voluntarily. I have received the agreed remuneration/stipend as full and fair consideration. I waive any future claims for royalties, commercial licensing fees, or intellectual property rights arising from models trained on my voice.

Participant Full Name: ____________________________________________________
Date of Birth (Must be >= 18): __________________ Nationality: ____________
Participant Signature: __________________________ Date: ___________________
```

---

## 6. Operational Compliance with DATASET_R&D_GOAL.md Section 8 Boundaries

### 6.1 Section 8 Hard Guardrails & Enforcement Mechanisms

```
+---------------------------------------------------------------------------------------------------+
| SECTION 8 BOUNDARY ENFORCEMENT PROTOCOL                                                           |
+---------------------------------------------------------------------------------------------------+
| GUARDRAIL 8.1: NO UNAUTHORIZED EXPENDITURE                                                       |
| - Zero financial commitments permitted under current R&D planning status.                         |
| - Prohibited: Purchasing LDC corpora, paying crowd annotator invoices, signing vendor contracts.  |
| - Enforcement: All budget allocations must be submitted via PLAN_COST.md for explicit user sign-off|
+---------------------------------------------------------------------------------------------------+
| GUARDRAIL 8.2: STRICT PROHIBITION ON THIRD-PARTY CLOUD API DATA EXFILTRATION                     |
| - Zero uploading of raw voice audio or speaker metadata to external third-party cloud services:   |
|   * Prohibited APIs: OpenAI Whisper API, Google Cloud Speech, ElevenLabs, Amazon Transcribe,      |
|     AssemblyAI, or commercial cloud LLMs.                                                         |
| - Regulatory Basis: Exfiltration constitutes unauthorized data transfer under GDPR Chapter V      |
|   and VN Decree 13 Art. 25, violates participant consent, and breaches BIPA § 15(c).             |
| - Mandatory Architecture: 100% LOCAL ON-PREMISES EXECUTION on local GPUs for all ASR, MFA,         |
|   Kaldi, and feature extraction pipelines.                                                        |
+---------------------------------------------------------------------------------------------------+
| GUARDRAIL 8.3: PRESERVATION OF SEALED TEST SETS & HISTORICAL BASELINES                            |
| - The official test sets and DEV_CAL splits documented in TEST_EVALUATION_PROTOCOL.md remain      |
|   SEALED and IMMUTABLE.                                                                           |
| - Prohibited: Recycling historical test speakers into new training sets to inflate sample sizes.  |
| - Baseline Integrity: SM13 LoRA checkpoints, historical splits, and benchmark logs remain locked. |
+---------------------------------------------------------------------------------------------------+
| GUARDRAIL 8.4: ABSOLUTE GIT REPOSITORY HYGIENE                                                    |
| - Zero raw audio files (.wav, .flac, .mp3), participant PII, or unmasked biometric data in Git.   |
| - Storage Mechanism: External immutable object storage / GDrive, referenced solely by SHA-256     |
|   cryptographic checksum manifests and speaker UUIDs.                                             |
+---------------------------------------------------------------------------------------------------+
```

---

## 7. Corpus-by-Corpus Legal Risk Audit (Candidate Inventory)

We conduct a forensic legal audit of candidate corpora considered in current and previous project phases:

```
+----------------------------------------------------------------------------------------------------+
| CORPUS-BY-CORPUS LEGAL AUDIT MATRIX                                                                |
+----------------------------------------------------------------------------------------------------+
| Corpus Name       | Official License | Commercial Rights | Biometric / Privacy Risk | Audit Verdict |
+-------------------+------------------+-------------------+--------------------------+---------------+
| L2-ARCTIC v5.0    | CC BY-NC 4.0     | STRICTLY BANNED   | LOW (TAMU IRB Consent)   | APPROVED (NC) |
| SpeechOcean762    | CC BY 4.0        | PERMITTED         | MEDIUM (125 Children)    | SPLIT QUOTA   |
| Mozilla Common V. | CC0 1.0          | PERMITTED         | LOW (Explicit waiver)    | APPROVED      |
| EpaDB             | Academic / EULA  | STRICTLY BANNED   | LOW (Univ. Buenos Aires) | CONDITIONAL   |
| LibriTTS-R        | CC BY 4.0        | PERMITTED         | LOW (LibriVox public)    | APPROVED      |
| Tatoeba English   | CC BY 2.0 FR     | PERMITTED         | LOW (Crowdsourced)       | CONDITIONAL   |
| VoxCeleb 1 & 2    | Non-Comm Research| STRICTLY BANNED   | EXTREME (YouTube Scraped)| PROHIBITED    |
| VIVOS / VinBigData| Academic EULA    | STRICTLY BANNED   | MEDIUM (Corporate Terms) | PILOT ONLY    |
+----------------------------------------------------------------------------------------------------+
```

### 7.1 Detailed Evaluation of Target Corpora

#### 1. L2-ARCTIC Corpus v5.0 (Texas A&M University)
- **Authoritative License:** `CC BY-NC 4.0` (Confirmed by `manifests/RQ1-SPLIT-01/LICENSE_PROVENANCE_ERRATUM.md`).
- **Commercial Usability:** Prohibited. Any model fine-tuned on L2-ARCTIC cannot be commercialized.
- **Privacy & Consent:** Academic IRB approval at Texas A&M; anonymized speaker IDs (e.g. `BVT`, `ABA`).
- **Permitted Role:** Core non-commercial research baseline and academic evaluation benchmark.

#### 2. SpeechOcean762 (OpenSLR SLR101)
- **Authoritative License:** `CC BY 4.0` (Permissive commercial use with attribution).
- **Commercial Usability:** Permitted for adult subset.
- **Pediatric Compliance Risk:** 125 out of 250 speakers are children under 16. Deploying models trained on the child subset in commercial services triggers COPPA and Decree 13 liabilities unless independent parental consent documentation is audited.
- **Permitted Role:** Adult subset (125 speakers) approved for general training/dev; child subset (125 speakers) strictly relegated to the isolated `Test-OOD-Pediatric` track.

#### 3. Mozilla Common Voice (MCV)
- **Authoritative License:** `CC0-1.0 Universal` (Public Domain Dedication).
- **Commercial Usability:** Unrestricted. Model weights are completely free of licensing contamination.
- **Privacy & Consent:** Participants waive all rights via web interface prior to recording.
- **Data Quality Caveat:** Transcripts are crowdsourced; lacks gold S/O/I phoneme annotations. Audio requires expert phonetic re-annotation before serving as MDD ground truth.

#### 4. EpaDB (University of Buenos Aires)
- **Authoritative License:** Academic research use only under ISCA archive distribution terms.
- **Commercial Usability:** Prohibited.
- **Permitted Role:** Academic comparison benchmark only. Verification of raw distribution archive required before ingesting into active training splits.

#### 5. VoxCeleb 1 & 2 (University of Oxford)
- **Authoritative License:** Creative Commons Attribution-NonCommercial 4.0 International.
- **Biometric & Publicity Rights Hazard:** Audio was scraped from YouTube celebrity interview videos without direct individual consent.
- **Legal Risk:** High exposure under Illinois BIPA (740 ILCS 14/) and California Right of Publicity (Cal. Civ. Code § 3344) for commercial voice modeling.
- **Audit Verdict:** **STRICTLY PROHIBITED** from inclusion in both the training pipeline and redistributable releases.

#### 6. Vietnamese Corpora (VIVOS, VinBigData, FPT Open)
- **Authoritative License:** Custom non-commercial academic research agreements.
- **Commercial Usability:** Prohibited without formal bilateral enterprise licensing agreements.
- **Permitted Role:** Exploratory acoustic transfer analysis only; must not be bundled into redistributable public archives.

---

## 8. Actionable Recommendations, Decision Gates & Legal Roadmap

### 8.1 Strategic Dual-Track Licensing Architecture
To enable both cutting-edge open scientific publication and future commercial enterprise deployment, the project must implement a **Dual-Track Licensing Strategy**:

```
                                  [220,000 Utterance Initiative]
                                                 |
                   +-----------------------------+-----------------------------+
                   |                                                           |
                   v                                                           v
      +-------------------------+                                 +-------------------------+
      | TRACK A: OPEN SCIENTIFIC|                                 | TRACK B: CLEAN-ROOM     |
      | BENCHMARK (ACADEMIC)    |                                 | COMMERCIAL DEPLOYMENT   |
      +-------------------------+                                 +-------------------------+
      | - License: CC BY-NC 4.0 |                                 | - License: CC BY / CC0  |
      | - Incorporates:         |                                 | - Incorporates:         |
      |   * L2-ARCTIC v5.0      |                                 |   * SpeechOcean (Adult) |
      |   * Academic corpora    |                                 |   * Common Voice (Re-ann)|
      |   * Full S/O/I research |                                 |   * Tier-3 Consented New|
      | - Goal: State-of-the-art|                                 | - Goal: Commercial SaaS |
      |   scientific benchmark. |                                 |   tutoring products.    |
      +-------------------------+                                 +-------------------------+
```

### 8.2 Mandatory Compliance Gates Prior to Data Ingestion

```
[GATE L-1: CONSENT VERIFICATION]
   |--> Verify 100% of newly recruited speakers executed the 4-Pillar Consent Form.
   |--> Audit age verification (Zero speakers < 18 in the core 200k corpus).
   v
[GATE L-2: LICENSE TAINT CHECK]
   |--> Automated license linter checking manifest source URIs.
   |--> Block any CC BY-NC or Academic-Only corpus from entering Track B training manifests.
   v
[GATE L-3: BIOMETRIC & PII SCRUBBING]
   |--> Strip all acoustic file metadata (WAV headers, RIFF INFO chunks).
   |--> Verify speaker ID cryptographic anonymization (UUIDv4).
   v
[GATE L-4: INFRASTRUCTURE AIR-GAP CERTIFICATION]
   |--> Confirm all ASR, MFA, and feature extraction run on local on-premises GPUs.
   |--> Verify zero outbound network calls to cloud speech/LLM APIs.
   v
[GATE L-5: STATUTORY REGISTRATION (VIETNAM DECREE 13)]
   |--> Finalize Cross-Border Transfer Impact Assessment Dossier.
   |--> Archive dossier for inspection by MPS Department A05 prior to international sync.
```

---

## 9. Deliverable Verification Checklist

- [x] **File Path:** Saved to `reports/DATASET_RD_20261116/AUDIT_LEGAL.md`.
- [x] **License Comparison:** Comprehensive comparative analysis of CC BY-NC 4.0, CC BY 4.0, CC0 1.0, Academic Use, and Commercial Proprietary licenses.
- [x] **Contagious NC Analysis:** Detailed legal reasoning on how fine-tuned model weights inherit CC BY-NC 4.0 restrictions as "Adapted Materials".
- [x] **Biometric Privacy:** Exhaustive coverage of EU GDPR (Art. 4(14), Art. 9, Art. 17 right to erasure), US BIPA (740 ILCS 14/), Texas CUBI, California CCPA/CPRA, and Vietnam Decree 13/2023/ND-CP.
- [x] **Pediatric Governance:** Full audit of US COPPA, FTC 2017 Statement, the strict inapplicability of the "Audio File Exception" to AI training, and Decree 13 Art. 20 dual-consent mandates.
- [x] **Architectural Recommendation:** Strict quarantine of child speech into an isolated Out-of-Domain track (`Test-OOD-Pediatric`), excluding minors from the core 200k corpus.
- [x] **Informed Consent Framework:** Four-Pillar Consent Standard with complete model participant release agreement template.
- [x] **Section 8 Enforcement:** Strict operational enforcement of zero unauthorized spend, zero third-party cloud API uploads, sealed data preservation, and Git repository hygiene.
- [x] **Candidate Corpus Audit:** Forensic audit of L2-ARCTIC, SpeechOcean762, Common Voice, EpaDB, LibriTTS-R, Tatoeba, VoxCeleb, and Vietnamese corpora.
- [x] **Dual-Track Roadmap:** Actionable architecture separating Track A (Open Academic CC BY-NC 4.0) from Track B (Clean Commercial CC BY / CC0).

---

*Report prepared and certified by:*  
**Agent `@audit-legal`**  
*Legal, Regulatory & Biometric Compliance Specialist — Project Langfens MDD Swarm*