"""Frozen-checkpoint CPU replay and conditional permutation of actual branch outputs."""

from contextlib import ExitStack
import numpy as np
import torch

from research.BACKBONE_REP_CAMPAIGN_20261122_v1.models import (
    DualRepConfig,
    DualRepSingleFactorV2,
)
from research.BACKBONE_REP_CAMPAIGN_20261122_v1.stages_0_3 import _collate_dualrep
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.run_single_factor_isolation import (
    collate_sf,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.single_factor_models import (
    SingleFactorConfig,
    SingleFactorV2,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.data import read_json, sha_file
from research.MDD_FIXES_AND_V2_20261121_v1.seq_mdd_v2 import compute_multitask_mdd_loss
from .inputs import RUN
from .metrics import view

CPU = torch.device("cpu")
SEEDS = (20261121, 20261122, 20261123)
ARMS = {
    **{f"S0_{s}": RUN / f"stage0_s0_clean/seed_{s}" for s in SEEDS},
    "A1": RUN / "stage2_dualrep/A1_top1",
    "A3": RUN / "stage2_dualrep/A3_top1_charsiu_proto",
}
PERM_SEEDS = (11, 22, 33)


def permutations(table):
    # No label/score used in the assignment; singletons stay as recorded.
    by = table.groupby(["src", "spk", "context"], sort=True).indices
    plans = {}
    for s in PERM_SEEDS:
        rng = np.random.default_rng(s)
        ix = np.arange(len(table))
        for xs in by.values():
            if len(xs) > 1:
                order = rng.permutation(xs)
                ix[order] = np.roll(order, 1)
        assert np.array_equal(np.sort(ix), np.arange(len(ix)))
        plans[s] = ix
    return plans


def perm_stats(table, plans):
    out = {}
    for seed, ix in plans.items():
        ok = table.l2.to_numpy() & table.y.ne("unknown").to_numpy()
        changed = ix != np.arange(len(ix))
        other = table.uid.to_numpy()[ix] != table.uid.to_numpy()
        pos = table.y.isin(["S", "O", "I"]).to_numpy()
        out[seed] = dict(
            valid_L2=int(ok.sum()),
            moved=int((ok & changed).sum()),
            different_uid=int((ok & other).sum()),
            positive=int((ok & pos).sum()),
            moved_positive=int((ok & pos & changed).sum()),
        )
    return out


def unpack(out, lengths):
    t = out["target_logits"].softmax(-1)
    g = out["gap_logits"].softmax(-1)
    return (
        np.concatenate([t[i, :l, 1:].detach().numpy() for i, l in enumerate(lengths)]),
        np.concatenate(
            [g[i, : l + 1, 1].detach().numpy() for i, l in enumerate(lengths)]
        ),
    )


def cat(parts):
    return tuple(np.concatenate(x) for x in zip(*parts, strict=True))


class Replay:
    def __init__(self, name):
        self.name = name
        d = ARMS[name]
        self.summary = sm = read_json(d / "SUMMARY.json")
        card = read_json(d / "CHECKPOINT_CARD.json")
        seal = read_json(d / "PREDICTIONS_SEAL.json")  # metadata only
        self.dual = name in ("A1", "A3")
        fn = "model.pt" if self.dual else "control.pt"
        assert (
            sha_file(d / "PREDICTIONS_SEAL.json") == sm["seal_sha256"]
            and sha_file(d / "CHECKPOINT_CARD.json") == seal["checkpoint_card_sha256"]
        )
        assert (
            sha_file(d / fn) == sm["checkpoint_sha256"] == card["checkpoint_sha256"]
            and sha_file(d / "cal_arrays.npz") == card["cal_arrays_sha256"]
        )
        ck = torch.load(d / fn, map_location="cpu", weights_only=True)
        self.policy = ck["policy"]
        assert self.policy == card["policy"] == sm["policy"]
        assert self.policy["alpha_realized"] == 0 and self.policy["temperature"] == {
            "default": [1.0, 1.0]
        }
        self.order = card["train_order_sha256"]
        torch.manual_seed(ck["seed"])
        self.model = m = (
            DualRepSingleFactorV2(DualRepConfig(**ck["config"]))
            if self.dual
            else SingleFactorV2(SingleFactorConfig(**ck["config"]))
        )
        self.parts = (
            [
                "target_scalar_proj",
                "seq_encoder",
                "realized_head",
                "target_head",
                "gap_head",
            ]
            + (["proj_ext1_target", "gap_ext1_proj", "af_head"] if self.dual else [])
            + (["aux_scalar_proj"] if name == "A3" else [])
        )
        init = {k: v.detach().clone() for k, v in m.named_parameters()}
        m.load_state_dict(ck["state_dict"], strict=True)
        m.eval()
        self.weights = dict(
            checkpoint_sha=sm["checkpoint_sha256"],
            seed=ck["seed"],
            best_epoch=ck["best_epoch"],
            params=sum(v.numel() for v in m.parameters()),
            relative_l2_from_seed_init={},
            phase2_adopted=ck.get("phase2_adopted"),
            phase2_target_drift=sm.get("phase2_target_max_abs_drift"),
            scalar_dropout_p=ck["config"].get("scalar_dropout_p"),
        )
        for key in self.parts:
            subset = [
                (k, v.detach())
                for k, v in m.named_parameters()
                if k.startswith(key + ".")
            ]
            assert subset
            self.weights["relative_l2_from_seed_init"][key] = float(
                torch.sqrt(
                    sum(((v - init[k]) ** 2).sum() for k, v in subset)
                    / sum(init[k].square().sum() for k, _ in subset)
                )
            )
        assert all(torch.isfinite(p).all() for p in m.parameters())
        if self.dual:
            self.weights["ext_mixing_target"] = (
                m.ext1_target_logits.detach().softmax(0).tolist()
            )
            self.weights["ext_mixing_gap"] = (
                m.gap_ext1_logits.detach().softmax(0).tolist()
            )
        self.norm = {
            k: tuple(np.array(x, dtype=np.float32) for x in pair)
            for k, pair in ck["norm"].items()
        }
        self.kw = {}
        if self.dual:
            self.kw["ext1_dir"] = RUN / "caches/slots_wavlm_base_plus"
        if name == "A3":
            with np.load(
                RUN / "stage1_probe/prototypes_wavlm_base_plus.npz", allow_pickle=False
            ) as z:
                proto = {k: z[k] for k in z.files}
            self.kw.update(
                charsiu_dir=RUN / "caches/charsiu_10ms",
                proto=proto,
                aux_norm=tuple(np.array(x, np.float32) for x in ck["aux_norm"]),
            )

    def collate(self, rows, feats, labels=False):
        fn = _collate_dualrep if self.dual else collate_sf
        return fn(rows, feats, self.norm, CPU, labels=labels, **self.kw)

    def gradient(self, rows, feats):
        # Controlled eval-mode backward on a loaded in-memory copy, dropout off. Not an optimizer trace.
        self.model.zero_grad(set_to_none=True)
        args, ys = self.collate(rows, feats, True)
        out = self.model(**args)
        loss = compute_multitask_mdd_loss(
            out,
            *ys,
            gap_pos_weight=8,
            lambda_gap=0.6,
            lambda_realized=0.25,
            native_fa_weight=1.5,
        )
        loss["loss"].backward()
        norms = {}
        for key in self.parts:
            grads = [
                v.grad
                for k, v in self.model.named_parameters()
                if k.startswith(key + ".") and v.grad is not None
            ]
            assert all(torch.isfinite(g).all() for g in grads)
            norms[key] = (
                float(torch.sqrt(sum(x.square().sum() for x in grads)))
                if grads
                else None
            )
        self.model.zero_grad(set_to_none=True)
        return dict(
            uids=[r["uid"] for r in rows],
            loss={k: float(v.detach()) for k, v in loss.items()},
            grad_l2=norms,
            dropout="off",
        )

    @torch.inference_mode()
    def original(self, rows, feats, *, capture=False):
        branches = {"target_scalar_proj": "t"}
        if self.dual:
            branches.update(proj_ext1_target="t", gap_ext1_proj="g")
        if self.name == "A3":
            branches["aux_scalar_proj"] = "t"
        outputs = []
        stash = {key: [] for key in branches} if capture else {}
        for i in range(0, len(rows), 8):
            rs = rows[i : i + 8]
            inp, _ = self.collate(rs, feats[i : i + 8])
            lens = [len(r["canonical_phones"]) for r in rs]

            def hook(key):
                def collect(module, args, z):
                    for j, n in enumerate(lens):
                        stash[key].append(
                            z[j, : n + int(branches[key] == "g")].numpy().copy()
                        )

                return collect

            with ExitStack() as ctx:
                for k in stash:
                    ctx.enter_context(
                        self.model.get_submodule(k).register_forward_hook(hook(k))
                    )
                outputs.append(unpack(self.model(**inp), lens))
        return cat(outputs), {
            key: (branches[key], np.concatenate(xs)) for key, xs in stash.items()
        }

    def parity(self, t, g, pred):
        with np.load(ARMS[self.name] / "cal_arrays.npz", allow_pickle=False) as z:
            diffs = {}
            for k, tab, sc in [("t", t, pred[0].sum(1)), ("g", g, pred[1])]:
                for domain, msk in [("l2", tab.l2), ("l1", ~tab.l2)]:
                    field = f"{domain}_{k}_sc"
                    ref = z[field]
                    assert ref.shape == sc[msk].shape
                    if domain == "l2":
                        assert np.array_equal(tab.y[msk], z[f"l2_{k}_g"])
                    d = float(np.max(np.abs(sc[msk] - ref)))
                    assert d < 5e-5, (self.name, field, d)
                    diffs[field] = d
        return diffs

    @torch.inference_mode()
    def intervene(self, rows, feats, t, g, stash, plans):
        # Replace within-group projected features only; other branches, valid lengths and padding unchanged.
        specs = {"koel_target_scalars": ["target_scalar_proj"]}
        if self.dual:
            specs["wavlm_joint"] = ["proj_ext1_target", "gap_ext1_proj"]
        if self.name == "A3":
            specs["aux_joint"] = ["aux_scalar_proj"]
        pred = {(name, s): [] for name in specs for s in PERM_SEEDS}
        off = {"t": 0, "g": 0}
        for i in range(0, len(rows), 8):
            rs = rows[i : i + 8]
            inp, _ = self.collate(rs, feats[i : i + 8])
            lengths = [len(r["canonical_phones"]) for r in rs]

            def replace(key, seed):
                kind, bank = stash[key]
                ix = plans[kind][seed]

                def hook(mod, args, z):
                    dest = z.clone()
                    p = off[kind]
                    for j, n in enumerate(lengths):
                        n += int(kind == "g")
                        dest[j, :n] = torch.from_numpy(bank[ix[p : p + n]])
                        p += n
                    return dest

                return hook

            for (name, s), store in pred.items():
                with ExitStack() as ctx:
                    for key in specs[name]:
                        ctx.enter_context(
                            self.model.get_submodule(key).register_forward_hook(
                                replace(key, s)
                            )
                        )
                    store.append(unpack(self.model(**inp), lengths))
            off["t"] += sum(lengths)
            off["g"] += sum(lengths) + len(lengths)
        assert off == {"t": len(t), "g": len(g)}
        return {
            name: [
                view(t, g, cat(pred[name, s]), self.policy, detailed=False)
                for s in PERM_SEEDS
            ]
            for name in specs
        }
