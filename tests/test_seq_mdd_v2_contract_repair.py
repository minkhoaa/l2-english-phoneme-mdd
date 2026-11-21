from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.compare_repaired_controls import (
    compare_gap_slices,
    gap_bootstrap,
    replay_common_policy,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.data import (
    frame_path,
    save_json,
    save_jsonl,
    sha_file,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.features import WindowHead, pool_features
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.repair_model import LengthAwareV2
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.repair_prompts import (
    REPAIR_ID,
    ensure_separate,
    mdd_prompt,
    repair_core,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.run_repaired_control import (
    _load_control,
    fit_control,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.training import collate

CPU = torch.device("cpu")


def test_mdd_prompt_is_not_projected_to_per_inventory():
    row = dict(
        uid="one",
        speaker="AA1",
        audio_sha256="fake",
        canonical_phones=["AA", "SH", "K"],
    )
    source = dict(
        uid="one",
        speaker_id="aa1",
        audio_sha256="fake",
        canonical39_ref=["aa", "sh", "k"],
        words=["call", "beige", "oov"],
    )
    words = {"call": ["ao1"], "beige": ["zh"]}
    overrides = {"oov": ["k"]}
    result = repair_core(row, source, words, overrides)
    assert result["canonical_phones"] == ["AO", "ZH", "K"]
    assert row["canonical_phones"] == ["AA", "SH", "K"]
    assert source["canonical39_ref"] == ["aa", "sh", "k"]
    with pytest.raises(ValueError, match="UID/speaker/WAV"):
        repair_core(row, dict(source, audio_sha256="else"), words, overrides)
    with pytest.raises(ValueError, match="slot correspondence"):
        repair_core(row, source, {**words, "call": ["k"]}, overrides)


@pytest.mark.parametrize(
    "words,lex,err",
    [
        ([], {}, "Empty"),
        (["missing"], {}, "Missing"),
        (["bad"], {"bad": ["SIL"]}, "Unsupported"),
    ],
)
def test_mdd_prompt_rejects_unbound_phone_sequences(words, lex, err):
    with pytest.raises(ValueError, match=err):
        mdd_prompt(words, lex, {})


def test_archive_cannot_be_output(tmp_path):
    run = tmp_path / "runs" / "archive"
    for bad in (run, run.parent, run / "nested"):
        with pytest.raises(ValueError, match="archived run"):
            ensure_separate(run, bad)
    ensure_separate(run, run.with_name("repaired"))


def tiny_batches():
    rng = np.random.default_rng(9)
    h = rng.normal(size=(27, 6, 12)).astype(np.float16)
    lp = (
        torch.from_numpy(rng.normal(size=(27, 79)).astype(np.float32))
        .log_softmax(-1)
        .numpy()
    )
    rows = [
        {"canonical_phones": phones}
        for phones in (["AO"], ["T", "AA", "ZH"], ["D", "T", "L", "IY", "N", "R"])
    ]
    feats = [pool_features(h, lp, r["canonical_phones"]) for r in rows]
    together, _ = collate(rows, feats, {}, CPU, labels=False)
    each = [
        collate([r], [f], {}, CPU, labels=False)[0]
        for r, f in zip(rows, feats, strict=True)
    ]
    return rows, together, each


def test_mixed_length_valid_outputs_match_running_alone_and_legacy_unpadded():
    torch.manual_seed(7)
    rows, together, each = tiny_batches()
    repaired = LengthAwareV2(h_dim=12, d_model=24, dropout=0.0).eval()
    legacy = WindowHead(h_dim=12, d_model=24, dropout=0.0).eval()
    legacy.load_state_dict(repaired.state_dict(), strict=True)
    with torch.no_grad():
        merged = repaired(**together)
        legacy_mixed = legacy(**together)
        for j, (row, single) in enumerate(zip(rows, each, strict=True)):
            length = len(row["canonical_phones"])
            fixed = repaired(**single)
            old = legacy(**single)
            for key in fixed:
                count = length + (key == "gap_logits")
                torch.testing.assert_close(
                    merged[key][j, :count], fixed[key][0], rtol=2e-5, atol=3e-6
                )
                torch.testing.assert_close(old[key], fixed[key], rtol=2e-5, atol=3e-6)
            torch.testing.assert_close(
                legacy_mixed["gap_logits"][j, :length],
                merged["gap_logits"][j, :length],
                rtol=2e-5,
                atol=3e-6,
            )
        # The old code exhibits a meaningful deviation only at a padded EOS.
        assert (
            legacy_mixed["gap_logits"][1, 3] - merged["gap_logits"][1, 3]
        ).abs().max() > 1e-3


def test_gradient_reaches_eos_from_short_utterance_even_with_padding():
    torch.manual_seed(11)
    _, together, _ = tiny_batches()
    repaired = LengthAwareV2(h_dim=12, d_model=24, dropout=0.0).eval()
    legacy = WindowHead(h_dim=12, d_model=24, dropout=0.0).eval()
    legacy.load_state_dict(repaired.state_dict(), strict=True)
    for model in (legacy, repaired):
        # Shortest sample's true EOS, not final batch column.
        model(**together)["gap_logits"][0, 1, 1].backward()
        assert all(
            p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
        )
    assert torch.count_nonzero(legacy.eos_state.grad) == 0
    assert torch.count_nonzero(repaired.eos_state.grad) > 0


def training_fixture(tmp_path: Path):
    rng = np.random.default_rng(3)
    run, split = tmp_path / "legacy", tmp_path / "repaired_splits"
    (run / "splits").mkdir(parents=True)
    split.mkdir()
    frames = run / "frames"
    frames.mkdir()
    save_json(frames / "FRAME_CARD.json", {"backbone": {"model": "synthetic"}})
    rows = []
    for i, phones in enumerate(
        [
            ["AA", "AO", "SH", "T"],
            ["N", "ZH"],
            ["D", "N", "AO", "UW"],
            ["K", "P"],
            ["T", "K", "ZH"],
        ]
    ):
        r = dict(
            uid=f"utt{i}",
            speaker=f"SPK{i}",
            audio_sha256=f"{i:064x}",
            canonical_phones=phones,
        )
        r.update(
            gold_targets=(["C", "S", "C", "O"] if i % 2 == 0 else ["C", "O"])[
                : len(phones)
            ],
            gold_gaps=["N", "I", *(["N"] * (len(phones) - 1))],
            is_native_l1=False,
        )
        if i == 4:
            r.update(
                gold_targets=["C"] * len(phones),
                gold_gaps=["N"] * (len(phones) + 1),
                is_native_l1=True,
            )
        path = frame_path(frames, r)
        path.parent.mkdir(exist_ok=True)
        h = rng.normal(size=(25, 6, 1024)).astype(np.float16)
        lp = (
            torch.tensor(rng.normal(size=(25, 79)), dtype=torch.float32)
            .log_softmax(-1)
            .numpy()
        )
        np.savez(path, hidden=h, lp=lp)
        rows.append(r)
    save_jsonl(split / "fit.jsonl", rows[:2])
    save_jsonl(split / "cal.jsonl", rows[2:])
    hashes = {p.name: sha_file(p) for p in split.glob("*.jsonl")}
    hashes["eval_gold.jsonl"] = "unread-in-fit"
    parent = run / "splits/PREFLIGHT.json"
    save_json(parent, {"manifest_sha256": hashes})
    save_json(
        split / "PREFLIGHT.json",
        {
            "repair_id": REPAIR_ID,
            "parent_preflight_sha256": sha_file(parent),
            "manifest_sha256": hashes,
            "backbone": {"model": "synthetic"},
        },
    )
    return run, split


def test_matched_control_training_is_fit_cal_only_and_saves_repaired_model(tmp_path):
    run, split = training_fixture(tmp_path)
    cards = [
        fit_control(
            run, split, tmp_path / mode, CPU, mode=mode, seed=23, epochs=1, batch_size=2
        )
        for mode in ("fixed", "legacy-eos-control")
    ]
    assert cards[0]["init_state_sha256"] == cards[1]["init_state_sha256"]
    assert cards[0]["train_order_sha256"] == cards[1]["train_order_sha256"]
    assert cards[0]["best_epoch"] == cards[1]["best_epoch"] == 1
    m, *_ = _load_control(tmp_path / "fixed", sha_file(split / "PREFLIGHT.json"), CPU)
    assert isinstance(m, LengthAwareV2)
    assert not (split / "eval_gold.jsonl").exists()


def test_pair_comparison_uses_fixed_threshold_and_strict_label_tie_rule():
    x = pd.DataFrame(
        {
            "domain": ["target", "target", "gap", "gap"],
            "score": [0.6, 0.59, 0.825, 0.8],
            "p_s": [0.3, 0.1, None, None],
            "p_o": [0.3, 0.49, None, None],
            "call": ["C"] * 4,
        }
    )
    y = replay_common_policy(x, {"tau_t": 0.6, "tau_g": 0.825})
    assert y.call.tolist() == ["S", "C", "I", "N"]
    assert x.call.tolist() == ["C"] * 4


def test_gap_slice_uses_each_utterance_eos_before_masking_unknown():
    x = pd.DataFrame(
        {
            "cohort": ["held6"] * 5,
            "uid": ["a", "a", "b", "b", "b"],
            "speaker": ["x", "x", "y", "y", "y"],
            "domain": ["gap"] * 5,
            "index": [0, 1, 0, 1, 2],
            "gold_label": ["N", "I", "N", "I", "unknown"],
            "call": ["N", "N", "N", "I", "N"],
        }
    )
    y = x.copy()
    y.loc[1, "call"] = "I"
    r = compare_gap_slices(x, y)["held6"]
    assert r["eos"]["gaps"] == 1 and r["other"]["gaps"] == 3
    assert r["eos"]["changed_calls"] == 1 and r["other"]["changed_calls"] == 0
    assert r["eos"]["old"] == {"tp": 0, "fp": 0, "fn": 1}
    assert gap_bootstrap(x, y)["held6"]["delta_pp"] == pytest.approx(100 / 3)
