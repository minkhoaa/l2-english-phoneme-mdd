"""Bounded CPU probes of batching, loss scales, and LoRA autograd; no optimizer steps."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

from research.CONTROLLED_PILOT_AB_20261118_v1.run_pilot_ab import WEIGHTS_DIR
from research.KOEL_MDD_BRANCH_ABLATION_20261119_v1.models import load_base_koel
from research.MDD_FIXES_AND_V2_20261121_v1.seq_mdd_v2 import SeqMDDv2
from research.MDD_FIXES_AND_V2_20261121_v1.train_and_eval_seq_mdd_v2 import (
    calibrate_dual_policies,
)

from .audit_training_campaign import ARMS, q, tags
from .data import (
    HERE,
    SEED,
    V2_CKPT,
    V2_SHA,
    admitted,
    read_json,
    read_jsonl,
    save_json,
    sha_file,
)
from .features import Spec
from .prepare_l1_canonical_anchor import load_anchor
from .run_exp5_joint_lora import add_lora, cached_prefix, logits_and_loss, top_forward
from .training import (
    ARCHIVED_POLICY,
    cal_arrays,
    domain,
    features_from_cache,
    infer_logits,
    load_checkpoint,
    probabilities,
    sample_probs,
)

DEVICE = torch.device("cpu")


def pick(rows, count=4):
    ordered = sorted(rows, key=lambda r: (len(r["canonical_phones"]), r["uid"]))
    return [
        ordered[i] for i in np.linspace(0, len(ordered) - 1, count + 2, dtype=int)[1:-1]
    ]


def eos_exposure(rows, balanced=False, epochs=14):
    rng = np.random.default_rng(SEED)
    n = len(rows)
    term = Counter()
    l = np.array([len(r["canonical_phones"]) for r in rows])
    stats = []
    for _ in range(epochs):
        ids = (
            rng.choice(n, size=n, replace=True, p=sample_probs(rows))
            if balanced
            else rng.permutation(n)
        )
        affected = []
        for i in range(0, n, 16):
            sub = ids[i : i + 16]
            cut = l[sub] < l[sub].max()
            affected.extend(sub[cut])
            for j in sub[cut]:
                term[rows[j]["gold_gaps"][-1]] += 1
        stats.append(len(affected) / n)
    return {
        "pad_not_eos_fraction_by_epoch": stats,
        "terminal_gold_on_affected_exposures": term,
    }


def load_arm(run, name):
    if name == "v2":
        if sha_file(V2_CKPT) != V2_SHA:
            raise ValueError("V2 checkpoint SHA mismatch")
        d = torch.load(V2_CKPT, map_location=DEVICE, weights_only=False)
        head = SeqMDDv2(**d["model_kwargs"]).eval()
        head.load_state_dict(d["state_dict"])
        return (
            head,
            {
                "target_scalars": (d["t_mean"], d["t_std"]),
                "gap_scalars": (d["g_mean"], d["g_std"]),
            },
            Spec(),
            ARCHIVED_POLICY,
        )
    return load_checkpoint(run / ARMS[name], DEVICE)


def batched_vs_single(run, name, rows):
    h, n, s, p = load_arm(run, name)
    k = run / "ctc_delete.so" if s.ollr else None
    feat = features_from_cache(rows, s, run / "frames", k, "cal_pad_probe_" + name)
    batch = infer_logits(h, rows, feat, n, DEVICE, 16)
    solo = infer_logits(h, rows, feat, n, DEVICE, 1)
    tmove = []
    gmove = []
    swaps = {"batch_I_to_solo_N": 0, "batch_N_to_solo_I": 0}
    padded = 0
    thresholds = p["tau_g"]
    masked_end_changes = 0
    native = {"batch": Counter(), "solo": Counter()}
    ends = []
    term_gold = Counter()
    for offset, (r, b, a) in enumerate(zip(rows, batch, solo, strict=True)):
        size = len(r["canonical_phones"])
        start = offset // 16 * 16
        short = size < max(len(z["canonical_phones"]) for z in rows[start : start + 16])
        padded += int(short)
        at, ag = probabilities(a, r, p)
        bt, bg = probabilities(b, r, p)
        tmove.append(np.abs(at - bt).max())
        gmove.append(np.abs(ag[:-1, 1] - bg[:-1, 1]).max())
        delta = float(ag[-1, 1] - bg[-1, 1])
        if short:
            ends.append(delta)
        y, z = bg[-1, 1] >= thresholds, ag[-1, 1] >= thresholds
        term_gold[r["gold_gaps"][-1]] += int(y != z)
        swaps["batch_I_to_solo_N"] += int(y and not z)
        swaps["batch_N_to_solo_I"] += int(z and not y)
        masked_end_changes += int(short and y != z)
        if domain(r) == "l1":
            for name_, aa, gg in [("batch", bt, bg), ("solo", at, ag)]:
                native[name_].update(
                    n_utts=1,
                    n_targets=len(aa),
                    n_gaps=len(gg),
                    n_target_fa=int((aa[:, 1:].sum(1) >= p["tau_t"]).sum()),
                    n_gap_fa=int((gg[:, 1] >= thresholds).sum()),
                )
    l2, l1 = cal_arrays(rows, batch, p)
    # The historical Cal grid is bounded to Calibrate only. No new policy is applied to Eval.
    safe = calibrate_dual_policies(l2, l1)
    return {
        "rows": len(rows),
        "padded_utterances": padded,
        "valid_target_score_max_abs_diff": float(max(tmove)),
        "nonterminal_gap_score_max_abs_diff": float(max(gmove)),
        "padded_terminal_abs_delta_q": q(np.abs(ends)),
        "padded_terminal_signed_delta_q": q(ends),
        "padded_terminal_changes_call": masked_end_changes,
        "terminal_gold_by_changed_call": term_gold,
        "terminal_swaps": swaps,
        "native_cal_locked": native,
        "cal_only_two_policies_fixed_temp_and_fusion": safe,
    }


def seeded_head_rng():
    # Constructor alone is part of the V2 driver's random stream between manual_seed and the head.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        direct = SeqMDDv2()
        digest_direct = torch.sum(direct.target_head[-1].weight.detach()).item()
        torch.manual_seed(SEED)
        cfg = Wav2Vec2Config.from_dict(read_json(WEIGHTS_DIR / "config.json"))
        cfg._attn_implementation = "eager"
        model = Wav2Vec2ForCTC(cfg)
        del model
        after = SeqMDDv2()
    diffs = [
        (x - y).abs().max().item()
        for x, y in zip(direct.parameters(), after.parameters(), strict=True)
    ]
    return {
        "same_numeric_seed": SEED,
        "max_initial_parameter_abs_diff": max(diffs),
        "direct_sum_final_target_weight": digest_direct,
        "after_backbone_init_sum_final_target_weight": after.target_head[-1]
        .weight.sum()
        .item(),
        "scope": "constructor-only RNG probe, not re-created historical state after thousands of backbone forwards",
    }


def compare_loss_scales(rows):
    _, g = tags(rows)
    n, p = g["N"], g["I"]
    y = torch.tensor([0] * n + [1] * p)
    x = torch.zeros((len(y), 2), requires_grad=True)
    w = torch.tensor([1.0, 8.0])
    ce = F.cross_entropy(x, y, weight=w)
    log_pt = x.log_softmax(-1).gather(-1, y[:, None]).squeeze(-1)
    fo = (
        -(1 - log_pt.exp()).square() * log_pt * torch.where(y == 1, 0.75, 0.25)
    ).mean()
    (a,) = torch.autograd.grad(ce, x, retain_graph=True)
    (b,) = torch.autograd.grad(fo, x)
    return {
        "valid_gap_n": n,
        "valid_gap_i": p,
        "zero_logit_ce": ce.item(),
        "zero_logit_focal": fo.item(),
        "focal_over_weighted_ce_grad_l2": float(b.norm() / a.norm()),
        "focal_over_weighted_ce_positive_grad": float(b[-1, 1] / a[-1, 1]),
        "focal_over_weighted_ce_negative_grad": float(b[0, 1] / a[0, 1]),
        "weight_ratio_pos_to_neg_ce": float(a[-1, 1].abs() / a[0, 1].abs()),
        "weight_ratio_pos_to_neg_focal": float(b[-1, 1].abs() / b[0, 1].abs()),
    }


def compare_accumulation(fit4):
    d = {}
    i = np.random.default_rng(SEED).choice(
        len(fit4), size=len(fit4), p=sample_probs(fit4), replace=True
    )
    for name, rows in [
        ("first_16_draws", [fit4[x] for x in i[:16]]),
        ("first_4_draws", [fit4[x] for x in i[:4]]),
    ]:
        lengths = [sum(y != "unknown" for y in r["gold_targets"]) for r in rows]
        gap_weight = [
            sum(8 if y == "I" else 1 if y == "N" else 0 for y in r["gold_gaps"])
            for r in rows
        ]
        d[name] = {
            "targets": lengths,
            "weighted_gap_denominators": gap_weight,
            "target_per_slot_utt_weights": (
                np.asarray(lengths) / sum(lengths)
            ).tolist(),
            "gap_weighted_ce_utt_weights": (
                np.asarray(gap_weight) / sum(gap_weight)
            ).tolist(),
            "joint_per_utt_utt_weights": [1 / len(rows)] * len(rows),
        }
    d["frozen_updates_total"] = 14 * int(np.ceil(len(fit4) / 16))
    d["joint_updates_total"] = 6 * int(np.ceil(len(fit4) / 4))
    return d


def norm_vector(g):
    return float(
        torch.sqrt(
            sum(
                (x.detach().square().sum() for x in g if x is not None), torch.zeros(())
            )
        )
    )


def lora_grad_probe(run, fit4):
    from research.MDD_FIXES_AND_V2_20261121_v1.feature_extractor_v2 import (
        ctc_viterbi_align_allophone_aware,
    )
    from .data import frame_path

    root = run / ARMS["lora"]
    card = read_json(root / "CHECKPOINT_CARD.json")
    adapter_path = root / "lora_adapter.pt"
    if sha_file(adapter_path) != card["lora_adapter_sha256"]:
        raise ValueError("Adapter SHA mismatch")
    head, norm, spec, _ = load_arm(run, "lora")
    # No dropout noise and no optimizer: measure the actual backward paths on fixed examples.
    model = load_base_koel(DEVICE)
    pars = add_lora(model)
    with torch.no_grad():
        for k, v in torch.load(
            adapter_path, map_location=DEVICE, weights_only=True
        ).items():
            pars[k].copy_(v)
    vec = list(pars.values())
    all_rows = []
    for role in ("l2arctic", "epadb", "l1"):
        for r in pick([x for x in fit4 if domain(x) == role], count=3):
            x, low = cached_prefix(root / "prefix16", r, DEVICE)
            h, lp = top_forward(model, x, low, checkpointed=False)
            with np.load(frame_path(run / "frames", r), allow_pickle=False) as f:
                teacher = torch.tensor(f["lp"])
            mdd = logits_and_loss(model, head, norm, spec, r, h, lp, DEVICE, None)
            grads = torch.autograd.grad(
                mdd, [*vec, lp], retain_graph=True, allow_unused=True
            )
            kl = F.kl_div(lp, teacher, reduction="batchmean", log_target=True)
            kd_grads = torch.autograd.grad(0.5 * kl, vec)
            mdd_norm = norm_vector(grads[:-1])
            kd_norm = norm_vector(kd_grads)
            s0, _ = ctc_viterbi_align_allophone_aware(
                teacher.numpy(), r["canonical_phones"]
            )
            s1, _ = ctc_viterbi_align_allophone_aware(
                lp.detach().numpy(), r["canonical_phones"]
            )
            d = np.abs(np.asarray(s1)[:, -1] - np.asarray(s0)[:, -1])
            all_rows.append(
                {
                    "uid": r["uid"],
                    "domain": role,
                    "phones": len(r["canonical_phones"]),
                    "nonzero_mdd_grad_to_adapters": mdd_norm > 0,
                    "mdd_lp_has_gradient": grads[-1] is not None,
                    "mdd_lora_grad_l2": mdd_norm,
                    "raw_kl_teacher_to_student": float(kl.detach()),
                    "kd_grad_l2_scaled": kd_norm,
                    "kd_actually_on_this_example": role == "l1",
                    "argmax_frame_changes": int(
                        (teacher.argmax(-1) != lp.argmax(-1)).sum()
                    ),
                    "frames": len(lp),
                    "ctc_peak_changes": int(np.count_nonzero(d)),
                    "ctc_peak_abs_delta_frames": q(d),
                }
            )
    return all_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=HERE / "runs/campaign_d5f8e89")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--include-lora-grad", action="store_true")
    a = p.parse_args()
    run = a.run_dir.resolve()
    split = run / "splits"
    fit, cal = [admitted(split, k + ".jsonl") for k in ("fit", "cal")]
    fit4 = read_jsonl(run / "exp4_multiseed/final_fit.jsonl")
    cal4 = read_jsonl(run / "exp4_multiseed/final_cal.jsonl")
    for k in ("fit", "cal"):
        assert (
            sha_file(run / f"exp4_multiseed/final_{k}.jsonl")
            == read_json(run / "exp4_multiseed/DESIGN.json")[k + "_sha256"]
        )
    ca = load_anchor(run / "manifests/l1_canonical_anchor.jsonl")["cal"]
    both = {}
    for name, rows in [
        ("v2", cal),
        ("pool", cal),
        ("window", cal),
        ("canon", [r for r in cal if domain(r) != "l1"] + ca),
        ("gap", cal),
        ("joint", cal),
        ("seed21", cal4),
        ("seed22", cal4),
        ("seed23", cal4),
    ]:
        both[name] = batched_vs_single(run, name, rows)
        print(name, both[name]["terminal_swaps"], flush=True)
    ret = {
        "source_sha256": sha_file(__file__),
        "torch_version": torch.__version__,
        "run_status_sha256": sha_file(run / "CAMPAIGN_STATUS.json"),
        "eos_exposure": {
            "v2_recipe": eos_exposure(fit),
            "exp4": eos_exposure(fit4, True),
        },
        "cal_batched_vs_single": both,
        "seed_probe": seeded_head_rng(),
        "gap_loss_at_zero_logits": compare_loss_scales(fit),
        "loss_normalizations": compare_accumulation(fit4),
    }
    if a.include_lora_grad:
        ret["lora_backward_on_fit"] = lora_grad_probe(run, fit4)
    save_json(a.output, ret)
    print(a.output, flush=True)


if __name__ == "__main__":
    main()
