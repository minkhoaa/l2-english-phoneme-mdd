"""Trace MDD and realized heads on slot-bound reference labels (read-only CPU)."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..inputs import (
    RUN,
    BASE_KOEL_CACHE,
    PHONE_TO_ID,
    normalize_canonical_phone,
    roles_and_refs,
    sha_file,
)
from ..replay import Replay
from research.BACKBONE_REP_CAMPAIGN_20261122_v1.common import pool_koel_dataset


def probe(role, rows, ref, xf, arm):
    m = arm.model.eval()
    out_counts = defaultdict(Counter)
    angles = defaultdict(list)
    repair = []
    alt_rule = Counter()
    eg_by_speaker = defaultdict(Counter)
    for i in range(0, len(rows), 8):
        rs = rows[i : i + 8]
        inp, _ = arm.collate(rs, xf[i : i + 8])
        cap = []
        with (
            torch.no_grad(),
            m.target_head.register_forward_pre_hook(
                lambda mod, args: cap.append(args[0].detach())
            ),
        ):
            out = m(**inp)
        assert len(cap) == 1
        seq = cap[0].requires_grad_(True)
        t = m.target_head(seq)
        aux = m.realized_head(seq)
        assert torch.equal(t.detach(), out["target_logits"]) and torch.equal(
            aux.detach(), out["realized_logits"]
        )
        ps = t.detach().softmax(-1)
        tc = torch.where(
            ps[:, :, 1:].sum(-1) >= arm.policy["tau_t"], 1 + ps[:, :, 1:].argmax(-1), 0
        ).numpy()
        rc = aux.detach().argmax(-1).numpy()
        n = seq.shape[:2]
        yt = torch.full(n, -100, dtype=torch.long)
        ya = torch.full_like(yt, -100)
        yr = torch.full_like(yt, -100)
        subs = []
        for j, r in enumerate(rs):
            for k, z in enumerate(ref[r["uid"]]["target_phones"]):
                ph = r["canonical_phones"][k]
                ci = PHONE_TO_ID[ph]
                y = r["gold_targets"][k]
                observed = int(r["realized_ids"][k])
                real = (
                    39
                    if y == "O"
                    else PHONE_TO_ID[normalize_canonical_phone(z["realized_phone"])]
                )
                pred = int(rc[j, k])
                call = int(tc[j, k])
                alt_alert = pred not in (ci, 39)
                group = y if y != "S" else "S_same" if real == ci else "S_alt"
                counts = out_counts[group]
                counts["slots"] += 1
                counts["predicts_true_realized"] += int(pred == real)
                counts["predicts_canonical"] += int(pred == ci)
                counts["mdd_calls_C"] += int(call == 0)
                counts["both_mdd_C_and_true_realized"] += int(
                    call == 0 and pred == real
                )
                if call == 0 and alt_alert:
                    alt_rule[y] += 1
                if y == "S":
                    assert 0 <= observed <= 39 and 0 <= real < 39
                    yt[j, k] = 1
                    ya[j, k] = observed
                    yr[j, k] = real
                    category = "aux_mismatch" if observed != real else group
                    subs.append((j, k, category))
                    eg_by_speaker[r["speaker"]][category] += 1
                    if observed != real:
                        repair.append(
                            dict(
                                role=role,
                                uid=r["uid"],
                                slot_index=k,
                                canonical=ph,
                                training_realized_id=observed,
                                reference_realized_id=real,
                                reference_realized=normalize_canonical_phone(
                                    z["realized_phone"]
                                ),
                                unchanged_mdd="S",
                            )
                        )
        if subs:

            def grad(logits, y, retain=False):
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    y.flatten(),
                    ignore_index=-100,
                    reduction="sum",
                )
                return torch.autograd.grad(loss, seq, retain_graph=retain)[0]

            gt = grad(t, yt)
            ga = grad(aux, ya, True)
            gr = grad(aux, yr)
            orig = F.cosine_similarity(gt, ga, dim=-1).numpy()
            new = F.cosine_similarity(gt, gr, dim=-1).numpy()
            for j, k, cat in subs:
                angles[cat].append([orig[j, k], new[j, k]])
    stats = {}
    for key, matrix in angles.items():
        vals = np.array(matrix)
        assert np.isfinite(vals).all()
        stats[key] = dict(
            n=len(vals),
            median_original_corrected=np.median(vals, axis=0).tolist(),
            fraction_negative_orig_corrected=np.mean(vals < 0, axis=0).tolist(),
            median_paired_delta=float(np.median(vals[:, 1] - vals[:, 0])),
        )
    return dict(
        head_counts=out_counts,
        source_S_by_speaker=eg_by_speaker,
        hypothetical_add_S_on_C=alt_rule,
        gradient_cos_at_seq=stats,
    ), repair


def run():
    assert (
        os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available()
    )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    roles, refs, sha = roles_and_refs()
    result = {}
    patch = []
    for role, rr in roles.items():
        rs = [r for r in rr if r["source"] == "l2arctic_dev18"]
        xf = pool_koel_dataset(BASE_KOEL_CACHE, rs)
        result[role] = {}
        first = None
        for name in ["S0_20261121", "A1", "A3"]:
            print("CPU aux check", role, name, flush=True)
            a = Replay(name)
            d, res = probe(role, rs, refs, xf, a)
            d["checkpoint_sha"] = a.weights["checkpoint_sha"]
            result[role][name] = d
            if first is None:
                first = res
                patch.extend(res)
            else:
                assert res == first
        del xf
    return dict(
        scope="no optimization or new thresholds; only admitted Fit/Cal ARCTIC",
        code_sha=sha_file(__file__),
        run_sha=sha_file(RUN / "RUN_BINDINGS.json"),
        ref_sha=sha,
        analysis=result,
        source_slot_differences_not_applied=patch,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    path = p.parse_args().out.resolve()
    if path.exists() or path.is_relative_to(RUN.parents[1]):
        raise FileExistsError("Use a fresh path outside the campaign")
    payload = json.dumps(run(), indent=2, allow_nan=False) + "\n"
    with path.open("x") as f:
        f.write(payload)
    print(path, flush=True)
