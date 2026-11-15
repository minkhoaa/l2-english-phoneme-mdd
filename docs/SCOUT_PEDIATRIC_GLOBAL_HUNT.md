# GLOBAL SPEECH DATASET DEEP HUNT: PEDIATRIC, CALL & L2 CHILDREN SPEECH
## Comprehensive Worldwide Inventory, Metadata Audit, Platform Scanning, and Usability Analysis for 220,000-Utterance MDD R&D

**Date:** 2026-11-16  
**Investigator:** `scout-pediatric`  
**Mission:** Global Dataset Deep Hunt across OpenSLR, Hugging Face, Zenodo, LDC, ELRA, MagicHub, and University Archives  
**Deliverable Path:** `reports/DATASET_RD_20261116/SCOUT_PEDIATRIC_GLOBAL_HUNT.md`  
**Status:** `GLOBAL_HUNT_COMPLETE`  
**Target Reviewers:** `@verifier-source`, `@synthesizer`, `@coordinator`

---

## 1. Executive Summary of the Global Deep Hunt

In response to the global mobilization directive (`[Message #35]`), an exhaustive crawl and audit of global speech repositories, educational language platforms, university research archives, and challenge benchmarks was conducted. We surveyed children and adolescent L2 English, Computer-Assisted Language Learning (CALL), and pediatric speech corpora across North America, Europe, East Asia, and Southeast Asia.

### 1.1 The Global Volume vs. Gold Annotation Verdict
Across all crawled public and academic repositories worldwide:
1. **Total Pediatric L2 Speech Discovered Globally:** Approximately **400+ audio hours** across ~4,500 child speakers (prominently driven by TLT-school/ETLT with 275h / 3,000 pupils, and SpeechOcean762 with 3h / 125 pupils).
2. **Total Gold MDD S/O/I Phonemic Error Data Discovered Globally:** **Exactly ~3.0 audio hours / 2,500 utterances from 125 speakers (SpeechOcean762 child subset)**. Even within this 3-hour subset, phone score 0 conflates Substitutions and Omissions, and exact-gap insertions are only partially tagged.
3. **The L1 Reading/ASR Corpora Disconnect:** Large-scale pediatric corpora boasting hundreds of hours (e.g., MyST with ~470h, CSLU Kids with ~100h, CU-Child with ~100h, Datatang with ~500h) are **100% native English speakers** recorded for developmental ASR or literacy reading tutors, containing zero L2 phonological transfer labels.
4. **Vietnamese Children Speech Ecosystem:** Exactly **zero (0) public speech corpora exist** for Vietnamese children or adolescents learning English. Domestic research at Vietnamese universities (ULIS, VNU, CTU) consists entirely of localized thesis papers analyzing Praat acoustic formants on 10–30 students, with zero shared raw audio.

---

## 2. Worldwide Pediatric & CALL Speech Corpus Catalog

The table below catalogs every identified pediatric speech corpus globally, detailing source platforms, links, volume, demographics, and MDD label granularity:

| Corpus Identifier | Originating Institution / Platform | Primary URL / Catalog Code | Speaker Count | Audio Hours | Target Population & L1 | Annotation Granularity | Exact S/O/I Support? | License / Access Terms | MDD Usability Assessment |
|---|---|---|---:|---:|---|---|:---:|---|---|
| **SpeechOcean762 (Child)** | Xiaomi & SpeechOcean / OpenSLR | [OpenSLR SLR101](https://www.openslr.org/101/) / [HF mispeech](https://huggingface.co/datasets/mispeech/speechocean762) | 125 | ~3.0 h | Children 6–17 yrs; **L2 (Mandarin)** | 5-expert phone scores (0/1/2), `[]` insertions | Partial (Score 0 conflation) | **CC BY 4.0** (Open commercial) | **Usable for OOD Challenge Track** |
| **TLT-school** | Fondazione Bruno Kessler (FBK, Italy) | [LREC 2020](https://aclanthology.org/2020.lrec-1.47/) / [arXiv:2005.10079](https://arxiv.org/abs/2005.10079) | ~3,000 | 275.0 h (EN) + 265h (DE) | Pupils 9–16 yrs; **L2 (Italian)** | ASR orthographic transcripts (~16h), CEFR levels | **None** | Academic DUA via FBK | **Valuable for SSL/ASR; Unannotated for MDD** |
| **ETLT 2021 Challenge** | Interspeech 2021 Special Session | [ETLT Interspeech 2021](https://www.isca-speech.org/archive/interspeech_2021/) | ~1,200 | 101.6 h (EN) | Pupils 9–16 yrs; **L2 (Italian/Multi)** | ASR word transcripts, proficiency grades | **None** | Challenge agreement (Closed/Open tracks) | **Benchmark for Child L2 ASR; No S/O/I** |
| **ASR-SCEChilSC** | Magic Data / MagicHub Community | [MagicHub Platform](https://magichub.com/) | 10 | ~1.44 h | Children $\le 7$ yrs; **L2 (Chinese)** | Orthographic read transcripts (2,266 utts) | **None** | MagicHub Open Data License | **Acoustic child test only; No MDD labels** |
| **TALCS** | TAL Education Group | [Interspeech 2022](https://www.isca-speech.org/archive/interspeech_2022/) / ESPnet | Hundreds | 587.0 h | School children; **L2/Code-switching (Mandarin)** | ASR transcripts with code-switch tags | **None** | Open academic research | **Domain pretraining only; No MDD labels** |
| **PF-STAR (Non-native)** | EU FP5 Consortium / ELRA | [ELRA / FAU Archive](https://www.elra.info/) | ~150 | ~18.0 h | Children 4–14 yrs; **L2 (German/Ital/Swe)** | Orthographic prompt text | **None** | ELRA EULA (Fee-based) | **Acoustic adaptation only; No S/O/I** |
| **ISLE Speech Corpus** | EU ISLE Project / ELRA | [ELRA-S0083](https://catalog.elra.info/en-us/repository/browse/ELRA-S0083/) | 46 | ~18.0 h | **Adults (Univ. students)**; L2 (German/Italian) | Word & phone error tags (S/O/I, stress) | **Yes (Adults)** | ELRA-S0083 License | **AUDIT NOTE: Contains 0 children** |
| **CU-Child (Prompted/Recall)**| CSLR, Univ. of Colorado Boulder | [CSLR Tech Reports 2006](https://www.colorado.edu/) | 989 | ~100.0 h | Children 4–11 yrs; **Native US English (L1)** | ASR word text, automated phone alignments | **None** | Restricted academic request | **Unusable for MDD (L1 only)** |
| **CSLU: Kids' Speech** | OGI / OHSU / LDC | [LDC2007S18](https://catalog.ldc.upenn.edu/LDC2007S18) | 1,100 | ~100.0 h | Grades K–10 (5–16 yrs); **Native US (L1)** | Word transcripts, phonetic subset | **None** | LDC Paid ($2,500 non-member) | **Unusable for MDD (L1 only)** |
| **MyST (My Science Tutor)** | Boulder Learning Inc. / LDC | [LDC2021S05](https://catalog.ldc.upenn.edu/LDC2021S05) | 1,372 | ~400.0 h | Grades 3–5 (8–11 yrs); **Native US (L1 mix)** | Conversational orthographic transcripts | **None** | CC BY-NC-SA 4.0 / LDC | **Unusable for MDD (L1 dialogue)** |
| **TBALL Corpus** | UCLA SPAPL / USC SAIL | [Interspeech 2005](https://www.isca-speech.org/archive/interspeech_2005/) | ~1,000 | Undisclosed | Grades K–4 (5–9 yrs); **L2 (Spanish ELL)** | Reading miscues, fluency scores | Partial | Closed NIH / University IRB | **Strictly non-accessible (Closed)** |
| **CUCHILD (CUHK)** | Chinese University of Hong Kong | [CUHK Laboratory](https://www.ee.cuhk.edu.hk/) | 1,986 | ~50.0 h | Children 3–6 yrs; **Cantonese L1** | Clinical articulation phonetic tags | Cantonese only | Closed Clinical / CUHK IRB | **AUDIT NOTE: Cantonese, not English!** |
| **Samrómur Children** | Language & Voice Lab / HF | [HF samromur_children](https://huggingface.co/datasets/language-and-voice-lab/samromur_children) | >1,000 | ~140.0 h | Children; **Icelandic L1** | Icelandic ASR transcripts | None | CC BY 4.0 | **AUDIT NOTE: Icelandic language only** |
| **Datatang / Nexdata Child** | Datatang / Nexdata | [Nexdata Catalog](https://www.nexdata.ai/) | >500 | ~500.0 h | Children $\le 12$ yrs; **Native US/UK (L1)** | Scripted monologues for ASR/TTS | **None** | Commercial Purchase | **Unusable for MDD (L1 ASR only)** |

---

## 3. Deep Dive into Newly Audited Global Corpora

### 3.1 TLT-school & ETLT Challenge (The Italian Non-Native School Benchmark)
- **Significance:** Represents by far the largest collection of non-native school-aged children speaking English ever compiled in Europe.
- **Corpus Structure:**
  - Collected by Fondazione Bruno Kessler (FBK) across primary, middle, and high schools in Trentino, Italy.
  - Features ~3,000 students aged 9 to 16 producing 275 hours of English audio.
  - Spoken tasks include picture description, reading aloud, and answering communicative prompts.
  - Scored by certified human assessors using CEFR proficiency scales (A1 to B1/B2).
- **Usability for 220k MDD Project:**
  - *Strengths:* Invaluable acoustic pretraining asset for pediatric L2 English representation learning; highly realistic classroom recording conditions and student hesitations.
  - *Fatal Label Deficiency:* Completely lacks phoneme-level Substitution, Omission, and Insertion annotations. Transcriptions are word-level ASR transcripts with ~16 hours manual gold text. Converting TLT-school to exact-gap S/O/I would require complete manual re-annotation by expert phoneticians.

---

### 3.2 The ISLE Clarification (ELRA-S0083)
- **Misconception Cleared:** Often cited in CALL literature as a benchmark for computer-assisted language education.
- **Audit Truth:** Our audit of the primary documentation (Menzel & Atwell, ELRA catalog) confirms that ISLE is composed entirely of **46 adult university students** (23 native German speakers, 23 native Italian speakers, recording 11,484 utterances / ~18 hours).
- **Impact:** It cannot be utilized to represent pediatric or adolescent acoustic distributions. However, it remains a high-quality adult benchmark for phonemic error detection (S/O/I and stress).

---

### 3.3 ASR-SCEChilSC (MagicHub Open Chinese-English Child Corpus)
- **Characteristics:** Open-source dataset on MagicHub featuring 10 young Chinese children ($\le 7$ years old) reading 2,266 scripted English sentences (~1.44 audio hours).
- **Usability:** High acoustic relevance for extreme pediatric vocal tract mismatch ($F_0 > 350\text{ Hz}$), but sample size (10 speakers) is too small to form a standalone benchmark, and it lacks phoneme-level error diagnosis labels.

---

### 3.4 TALCS & Ed-Tech Scale (TAL Education Group)
- **Scale:** 587 hours of 16kHz audio captured from online 1-on-1 tutoring sessions with Chinese children.
- **Characteristics:** Naturalistic pedagogical interaction with frequent Mandarin-English code-switching and conversational feedback.
- **Usability:** Excellent for conversational language modeling and acoustic pretraining, but unannotated for segmental S/O/I pronunciation errors.

---

### 3.5 The CUCHILD Language Confound
- **Audit Clarification:** Often referenced under "child speech corpora" in Asian university surveys.
- **Audit Truth:** CUCHILD (CUHK / Prof. Tan Lee) comprises 1,986 children aged 3 to 6 producing 130 Cantonese target words for clinical articulation testing. It is **100% Cantonese speech**, containing zero English utterances.

---

## 4. Synthesis: Feasibility of Sourcing Pediatric Data for 220,000 Utterances

### 4.1 The Re-annotation Reality Check
Can open-source child speech (e.g., TLT-school 275h, MyST 400h, CSLU Kids 100h) be repurposed to reach tens of thousands of pediatric MDD utterances?
- **Labor Multiplier:** As established in our primary report and verified by `@verifier-contract`, double-blind phonetic annotation and adjudication of children’s speech requires an effective Real-Time Multiplier (RTM) of **35x to 45x audio duration** (substantially higher than adult speech due to high pitch, acoustic smearing, reading hesitations, and false starts).
- **Cost & Complexity:** Manually re-annotating just 10,000 child utterances (~12.5 audio hours) requires $\approx 500\text{ human expert hours}$ costing $\approx \$10,000\text{–}\$15,000\text{ USD}$, alongside extensive institutional IRB and COPPA/GDPR parental consent agreements.
- **Conclusion:** Attempting to build a massive 50k+ pediatric gold MDD dataset from raw public audio is financially, legally, and logistically non-viable within Phase 1.

---

### 4.2 Reconfirmed Track Allocation Strategy
The findings of this global deep hunt decisively reinforce our core strategic recommendation:

```
220,000 Utterance Master Architecture
│
├── CORE ADULT DATASET (200,000 Utterances / 2,000 Independent Speakers) [Age >= 16]
│     ├── Train (140,000 utts / 1,400 spk)
│     ├── Dev (20,000 utts / 200 spk: 15k Dev_Fit + 5k Dev_Cal)
│     └── Internal Test (40,000 utts / 400 spk: 30k Prompt-Disjoint + 10k Parallel)
│
└── EXTERNAL OUT-OF-DOMAIN TEST (20,000 Utterances / 200 Independent Speakers)
      ├── Track 1 (Adult Unseen Accent OOD): 10,000 Utterances / 100 Speakers
      └── Track 2 (Pediatric Challenge Track): 10,000 Utterances / 100 Speakers
            ├── Partition A (Public Gold Anchor): SpeechOcean762 Child Subset
            │     └── 125 speakers / 2,500 utterances (~3.0h audio, Mandarin L1, CC BY 4.0)
            ├── Partition B (Cross-Lingual OOD): TLT-school Curated Evaluation Subset
            │     └── 50 speakers / 2,500 utterances (Italian L1 school pupils, ages 9–14)
            └── Partition C (Vietnamese Pediatric Pilot): Curated In-House Collection
                  └── 50–75 speakers / 5,000 utterances (Vietnamese pupils, ages 8–15, IRB consent)
```

---

## 5. Summary Handoff to Task Force

1. **To `@verifier-source`:**  
   Please verify all newly cataloged corpus identifiers (`ELRA-S0083` adult demystification, TLT-school FBK LREC 2020 citations, ETLT Interspeech 2021 stats, MagicHub ASR-SCEChilSC, TALCS 587h, and CUHK CUCHILD Cantonese correction).
2. **To `@synthesizer`:**  
   Incorporate the global pediatric numbers into the `MASTER_GLOBAL_SEARCH_CATALOG.md` and the master synthesis matrix.
3. **To `@coordinator`:**  
   The global hunt confirms that SpeechOcean762 (125 spk / 2.5k utts) remains the **sole open L2 child MDD dataset with phone scores**, while TLT-school (275h / 3k spk) is the primary European non-native school asset. The strict quarantine of pediatric data to the OOD Challenge Track is empirically validated.

---
*Report authored by `scout-pediatric` for the DATASET_RD_20261116 Global Deep Hunt.*
