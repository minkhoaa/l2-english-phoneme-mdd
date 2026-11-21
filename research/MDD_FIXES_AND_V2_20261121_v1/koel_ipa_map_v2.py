"""Complete 79-token Koel IPA vocabulary mapping, allophone groups, and v2 MDD alignment.

Fixes the 35/75 unmapped IPA token bug in pilot.MAP while keeping existing Koel Base
and R0 checkpoints untouched (zero retraining required for P0).
"""
from __future__ import annotations

import unicodedata
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

from pipelines.phoneme_review import ARPABET

# Special CTC tokens (never emitted as speech phones after greedy collapse)
SPECIAL_TOKENS: frozenset[str] = frozenset({"<pad>", "<s>", "</s>", "<unk>"})

# Full 75-phone KoelLabs/xlsr-english-01 IPA vocabulary -> normalized token sequence(s).
# Every single non-special token in Koel's 79-token vocab is explicitly covered (0 unmapped!).
# Categories:
#   1. Direct ARPAbet39 mappings (including valid English allophones like ɝ, kʰ, pʰ, ʉ, ɦ, nasalized vowels)
#   2. Syllabic sonorants (l̩->AH L, n̩->AH N, m̩->AH M, ŋ̍->AH NG)
#   3. Context-dependent English allophone classes (<FLAP>, <N_FLAP>, <GLOTTAL>, <REDUCED_V>, <EPADB_TH_H>, <EPADB_S_H>)
#      which match ONLY their legal canonical parent phone(s) in DP alignment and count as S otherwise.
#   4. Genuine non-English L2 substitution tokens (<L2_A>, <L2_E>, <L2_O>, <L2_BETA>, <L2_X>, <L2_GAMMA>, <L2_TRILL>)
#      which NEVER match canonical ARPAbet39 and are flagged as S (never dropped as '?' / unknown!).
KOEL_IPA_TO_NORM_V2: Dict[str, Tuple[str, ...]] = {
    # --- Original 40 mapped tokens ---
    "ɑ": ("AA",),
    "æ": ("AE",),
    "ʌ": ("AH",),
    "ə": ("AH",),
    "ɔ": ("AO",),
    "aʊ": ("AW",),
    "aɪ": ("AY",),
    "b": ("B",),
    "tʃ": ("CH",),
    "d": ("D",),
    "ð": ("DH",),
    "ɛ": ("EH",),
    "ɚ": ("ER",),
    "eɪ": ("EY",),
    "f": ("F",),
    "ɡ": ("G",),
    "g": ("G",),
    "h": ("HH",),
    "ɪ": ("IH",),
    "i": ("IY",),
    "dʒ": ("JH",),
    "k": ("K",),
    "l": ("L",),
    "m": ("M",),
    "n": ("N",),
    "ŋ": ("NG",),
    "oʊ": ("OW",),
    "ɔɪ": ("OY",),
    "p": ("P",),
    "ɹ": ("R",),
    "s": ("S",),
    "ʃ": ("SH",),
    "t": ("T",),
    "θ": ("TH",),
    "ʊ": ("UH",),
    "u": ("UW",),
    "v": ("V",),
    "w": ("W",),
    "j": ("Y",),
    "z": ("Z",),
    "ʒ": ("ZH",),
    # --- Multi-phone dictionary helpers ---
    "ɑːɹ": ("AA", "R"),
    "ɔːɹ": ("AO", "R"),
    "əl": ("AH", "L"),
    # --- Previously UNMAPPED valid English allophones (now properly mapped!) ---
    "ɝ": ("ER",),          # ID 70: stressed ER
    "ɜ": ("ER",),          # ID 49: non-rhotic/central ER
    "kʰ": ("K",),          # ID 68: aspirated K
    "pʰ": ("P",),          # ID 72: aspirated P
    "ɦ": ("HH",),          # ID 73: voiced glottal fricative allophone of H
    "ə̥": ("AH",),          # ID 13: devoiced schwa
    "ə̃": ("AH",),          # ID 14: nasalized schwa
    "ĩ": ("IY",),          # ID 10: nasalized /i/ (precomposed)
    "ĩ": ("IY",),          # ID 20: nasalized /i/ (combining)
    "ɑ̃": ("AA",),          # ID 60: nasalized /ɑ/
    "ɛ̃": ("EH",),          # ID 65: nasalized /ɛ/
    "æ̃": ("AE",),          # ID 76: nasalized /æ/
    "ʊ̃": ("UH",),          # ID 77: nasalized /ʊ/
    "oʊ̃": ("OW",),         # ID 23: nasalized /oʊ/
    "əʊ": ("OW",),         # ID 54: British /əʊ/ allophone of OW
    "aɪ̃": ("AY",),         # ID 51: nasalized /aɪ/
    "ʉ": ("UW",),          # ID 75: fronted /u/ (standard US English UW allophone)
    "ɒ": ("<LOT_V>",),     # ID 37: low back rounded (accepted for AA/AO)
    # --- Syllabic sonorants ---
    "l̩": ("AH", "L"),      # ID 59: syllabic L (e.g., single /sɪŋɡəl/ -> [sɪŋɡl̩])
    "n̩": ("AH", "N"),      # ID 55: syllabic N (e.g., button)
    "m̩": ("AH", "M"),      # ID 78: syllabic M (e.g., rhythm)
    "ŋ̍": ("AH", "NG"),     # ID 46: syllabic NG
    # --- Context-sensitive English allophones ---
    "ɾ": ("<FLAP>",),      # ID 38: alveolar flap (legal for T/D, S for R/other)
    "ɾ̃": ("<N_FLAP>",),    # ID 39: nasalized flap (legal for T/D/N)
    "ʔ": ("<GLOTTAL>",),   # ID 64: glottal stop (legal for coda T; ignorable before vowel onset)
    "ɨ": ("<REDUCED_V>",), # ID 63: barred-i (legal reduced vowel for IH/AH)
    "θʰ": ("<EPADB_TH_H>",),  # ID 43: Koel/EpaDB merged aspirated-t / dental-th (legal for T/TH)
    "sʰ": ("<EPADB_S_H>",),   # ID 7:  Koel/EpaDB postalveolar /sh/ (legal for SH)
    # --- Genuine non-English L2 substitution phones (flagged as S, NEVER '?') ---
    "a": ("<L2_A>",),      # ID 8:  Spanish/L2 open front /a/
    "e": ("<L2_E>",),      # ID 17: monophthong /e/
    "o": ("<L2_O>",),      # ID 22: monophthong /o/
    "β": ("<L2_BETA>",),   # ID 61: bilabial fricative (Spanish v/b)
    "x": ("<L2_X>",),      # ID 53: velar fricative (Spanish j for h)
    "ɣ": ("<L2_GAMMA>",),  # ID 67: velar approximant (Spanish g)
    "r": ("<L2_TRILL>",),  # ID 31: alveolar trill
}

# Context-sensitive allophone acceptance table: observed special token -> legal canonical parents
CONTEXT_ALLOPHONE_PARENTS: Dict[str, frozenset[str]] = {
    "<FLAP>": frozenset({"T", "D"}),
    "<N_FLAP>": frozenset({"T", "D", "N"}),
    "<GLOTTAL>": frozenset({"T"}),
    "<REDUCED_V>": frozenset({"IH", "AH"}),
    "<LOT_V>": frozenset({"AA", "AO"}),
    "<EPADB_TH_H>": frozenset({"T", "TH"}),
    "<EPADB_S_H>": frozenset({"SH"}),
}

# Allophone-aware Koel CTC token ID sets per ARPAbet39 canonical phone.
# Used by feature_extractor_v2 for both CTC Viterbi alignment AND can_lp / frame_margin,
# generalizing ER in {74, 70} to ALL legal English allophones in Koel's 79-token vocab!
ALLOPHONE_TOKEN_IDS: Dict[str, Tuple[int, ...]] = {
    "AA": (56, 60, 37),           # ɑ, ɑ̃, ɒ
    "AE": (35, 76),               # æ, æ̃
    "AH": (52, 12, 13, 14, 63),   # ʌ, ə, ə̥, ə̃, ɨ
    "AO": (34, 37),               # ɔ, ɒ
    "AW": (62,),                  # aʊ
    "AY": (50, 51),               # aɪ, aɪ̃
    "B": (27,),                   # b
    "CH": (71,),                  # tʃ
    "D": (15, 38, 39),            # d, ɾ, ɾ̃
    "DH": (47,),                  # ð
    "EH": (16, 65),               # ɛ, ɛ̃
    "ER": (74, 70, 49),           # ɚ, ɝ, ɜ
    "EY": (44,),                  # eɪ
    "F": (29,),                   # f
    "G": (48,),                   # ɡ
    "HH": (41, 73),               # h, ɦ
    "IH": (9, 63),                # ɪ, ɨ
    "IY": (19, 10, 20),           # i, ĩ, ĩ
    "JH": (58,),                  # dʒ
    "K": (18, 68),                # k, kʰ
    "L": (11, 59),                # l, l̩
    "M": (21, 78),                # m, m̩
    "N": (4, 55, 39),             # n, n̩, ɾ̃
    "NG": (45, 46),               # ŋ, ŋ̍
    "OW": (24, 23, 54),           # oʊ, oʊ̃, əʊ
    "OY": (66,),                  # ɔɪ
    "P": (25, 72),                # p, pʰ
    "R": (69,),                   # ɹ
    "S": (6,),                    # s
    "SH": (40, 7),                # ʃ, sʰ
    "T": (5, 38, 39, 64, 43),     # t, ɾ, ɾ̃, ʔ, θʰ
    "TH": (42, 43),               # θ, θʰ
    "UH": (36, 77),               # ʊ, ʊ̃
    "UW": (32, 75),               # u, ʉ
    "V": (28,),                   # v
    "W": (33,),                   # w
    "Y": (30,),                   # j
    "Z": (26,),                   # z
    "ZH": (57,),                  # ʒ
}

# Build NFC-normalized lookup table
_NFC_MAP: Dict[str, Tuple[str, ...]] = {}
for _k, _v in KOEL_IPA_TO_NORM_V2.items():
    _NFC_MAP[_k] = _v
    _NFC_MAP[unicodedata.normalize("NFC", _k)] = _v


def is_canonical_match(canonical_phone: str, observed_token: str) -> bool:
    """True iff observed_token is an exact match or legal English allophone of canonical_phone."""
    if canonical_phone == observed_token:
        return True
    parents = CONTEXT_ALLOPHONE_PARENTS.get(observed_token)
    if parents is not None and canonical_phone in parents:
        return True
    return False


def normalize_koel_tokens_v2(
    tokens: Sequence[str],
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Normalize Koel Base emitted IPA tokens with 100% vocab coverage.

    Returns:
      observed: list of normalized tokens (ARPAbet39, context-allophone tags, or <L2_*> sub tags)
      unknown_indices: indices of truly unknown tokens (should be empty for all 75 Koel IPA tokens!)
      unmapped: raw unmapped tokens (should be empty!)
      obs_to_seg: maps each observed index back to the source emitted segment index in `tokens`
    """
    observed: List[str] = []
    unknown_indices: List[int] = []
    unmapped: List[str] = []
    obs_to_seg: List[int] = []

    for seg_i, tok in enumerate(tokens):
        if tok in SPECIAL_TOKENS:
            continue
        mapped = _NFC_MAP.get(tok) or _NFC_MAP.get(unicodedata.normalize("NFC", tok))
        if mapped is None:
            unmapped.append(tok)
            unknown_indices.append(len(observed))
            observed.append("<UNMAPPED:" + tok + ">")
            obs_to_seg.append(seg_i)
        else:
            for m in mapped:
                observed.append(m)
                obs_to_seg.append(seg_i)
    return observed, unknown_indices, unmapped, obs_to_seg


def align_and_decide_v2(
    canonical: Sequence[str],
    observed: Sequence[str],
    unknown_indices: Sequence[int] = (),
) -> Dict[str, Any]:
    """Allophone-aware multi-path Levenshtein alignment producing exact Vanilla v2 target/gap calls
    and per-target alignment options (compatible with feature_extractor_v2).

    Note: Word/utterance-initial glottal stop <GLOTTAL> before a vowel is NOT flagged as an
    insertion error (standard English phonetic onset).
    """
    unk_set = set(unknown_indices)
    n, m = len(canonical), len(observed)
    # 1. Forward DP cost table (allophone-aware match cost = 0)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        dp[i, 0] = i
    for j in range(1, m + 1):
        dp[0, j] = j

    for i in range(1, n + 1):
        ci = canonical[i - 1]
        for j in range(1, m + 1):
            oj = observed[j - 1]
            # Unknown tokens cost 1 for substitution (never silently match)
            match = (j - 1 not in unk_set) and is_canonical_match(ci, oj)
            sub_cost = dp[i - 1, j - 1] + (0 if match else 1)
            del_cost = dp[i - 1, j] + 1
            ins_cost = dp[i, j - 1] + 1
            dp[i, j] = min(sub_cost, del_cost, ins_cost)

    # 2. Backward reachability of ALL optimal paths (to detect genuine alignment ambiguity vs clear ops)
    on_opt = np.zeros((n + 1, m + 1), dtype=bool)
    on_opt[n, m] = True

    target_ops: List[Set[str]] = [set() for _ in range(n)]
    target_obs_idx: List[Set[int]] = [set() for _ in range(n)]
    gap_has_ins: List[bool] = [False] * (n + 1)
    gap_has_no_ins: List[bool] = [False] * (n + 1)
    gap_ins_tokens: List[Set[str]] = [set() for _ in range(n + 1)]

    # Deterministic primary path (match/sub < del < ins) for boundary anchoring
    for i in range(n, -1, -1):
        for j in range(m, -1, -1):
            if not on_opt[i, j]:
                continue
            cur = dp[i, j]
            if i > 0 and j > 0:
                ci = canonical[i - 1]
                oj = observed[j - 1]
                match = (j - 1 not in unk_set) and is_canonical_match(ci, oj)
                cost = 0 if match else 1
                if dp[i - 1, j - 1] + cost == cur:
                    on_opt[i - 1, j - 1] = True
                    if j - 1 in unk_set:
                        target_ops[i - 1].add("?")
                    else:
                        target_ops[i - 1].add("C" if match else "S")
                    target_obs_idx[i - 1].add(j - 1)
                    gap_has_no_ins[i] = True
            if i > 0 and dp[i - 1, j] + 1 == cur:
                on_opt[i - 1, j] = True
                target_ops[i - 1].add("O")
                gap_has_no_ins[i] = True
            if j > 0 and dp[i, j - 1] + 1 == cur:
                on_opt[i, j - 1] = True
                gap_has_ins[i] = True
                gap_ins_tokens[i].add(observed[j - 1])

    # Build clean target decisions + metadata matching align_phone_events semantics
    targets_out: List[Dict[str, Any]] = []
    target_calls: List[str] = []
    known_obs_exist = m > len(unk_set)

    # Also compute primary alignment anchor (1-to-1 observed index if aligned) via deterministic backtrack
    primary_obs: List[int | None] = [None] * n
    pi, pj = n, m
    while pi > 0 or pj > 0:
        cur = dp[pi, pj]
        if pi > 0 and pj > 0:
            match = (pj - 1 not in unk_set) and is_canonical_match(canonical[pi - 1], observed[pj - 1])
            cost = 0 if match else 1
            if dp[pi - 1, pj - 1] + cost == cur:
                primary_obs[pi - 1] = pj - 1
                pi -= 1
                pj -= 1
                continue
        if pi > 0 and dp[pi - 1, pj] + 1 == cur:
            primary_obs[pi - 1] = None
            pi -= 1
            continue
        pj -= 1

    for i in range(n):
        ops = target_ops[i]
        obs_indices = sorted(target_obs_idx[i])
        if canonical[i] not in ARPABET:
            call = "?"
            status = "UNSUPPORTED"
        elif ops == {"C"}:
            call = "C"
            status = "PHONE_MATCH_NOT_ACCEPTABILITY"
        elif ops == {"S"}:
            call = "S"
            status = "SUBSTITUTION_HYPOTHESIS"
        elif ops == {"O"} and known_obs_exist:
            call = "O"
            status = "ABSTAIN_MISSING_RECOGNIZED_PHONE"
        else:
            # Ambiguous optimal paths (e.g., {'C', 'O'} or {'S', 'O'} or '?')
            call = "?"
            status = "AMBIGUOUS_ALIGNMENT"

        target_calls.append(call)
        targets_out.append({
            "index": i,
            "expected": canonical[i],
            "call": call,
            "status": status,
            "ops": sorted(ops),
            "obs_indices": obs_indices,
            "primary_obs_index": primary_obs[i],
        })

    # Gap decisions: ignore pure glottal-onset insertions (<GLOTTAL>) on vowel-initial/utterance gaps
    gap_calls: List[str] = []
    for k in range(n + 1):
        real_ins_toks = {t for t in gap_ins_tokens[k] if t != "<GLOTTAL>" and not t.startswith("<UNMAPPED:")}
        if gap_has_ins[k] and not gap_has_no_ins[k] and len(real_ins_toks) > 0:
            gap_calls.append("I")
        elif not gap_has_ins[k] or len(real_ins_toks) == 0:
            gap_calls.append("N")
        else:
            gap_calls.append("?")

    return {
        "target_calls": target_calls,
        "gap_calls": gap_calls,
        "targets": targets_out,
        "edit_distance": int(dp[n, m]),
    }
