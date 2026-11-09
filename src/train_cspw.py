"""CS-PW v2 training + 5-arm ablation (Bigram Engram, Chunk-GQA, 4-slot Chrono-CfC, Qwen KD, spoiler-safe
story graph).

    python src/train_cspw.py --out-dir runs/cspw-v2 [--steps 2000] [--device cpu]
    python src/train_cspw.py --out-dir runs/cspw-smoke --smoke --steps 200        # local smoke

Arms (encoder | CS-PW core | pooling | decoder | spoiler gate):

    arm0_v1_base           v1 PLHN (unigram Engram)       | none (r = s)     | flat mean | phrases  | none
    arm1_arch_bigram_gqa   E1: + Bigram-Span Engram       | Chunk-GQA d=128  | flat mean | phrases  | none
    arm2_scale_d192        E1                             | Chunk-GQA d=192  | flat mean | phrases  | none
    arm3_qwen_kd           E3: E1 + Qwen relational KD    | Chunk-GQA d=192  | flat mean | phrases  | none
                                                            (+ teacher dossier / frame / phrase KD)
    arm4_full_cspw         E3                             | Chunk-GQA d=192  | Chrono-CfC| typed frame/phrase
                                                                                          pointers | SGRF gates

All encoders warm-start from the v1 ``arm3_platonic_engram_full`` checkpoint and train with int8 QAT
(bit-exact with the device).  CS-PW inputs are the int8 PLHN scene vectors of each arm's frozen encoder;
arm4 is evaluated through the exact device path (``story-graph.idx`` lookup + ``int8_cspw_weave`` +
frame fill) built from its own export tensors.  Metrics -> ``summary.json``:

* WSD Top-1 (FP32 / INT8); sentence cross-lingual R@1/R@5/MRR; term cross-lingual R@1 (all anchors,
  4-char idioms, anchors the router splits into >= 2 chunks)
* Temporal-Swap Accuracy: swap the first and last scene (pages stay with the positions); correct when
  Slot1 tracks the content at position 0 and Slot2 the content at position M-1 in both runs
* Spoiler Leakage: share of eval lookups whose refresher exists although the entity is not yet introduced /
  the alias not yet revealed, or uses an event past the reading position (must be 0.000 for arm4); also on
  the cspw_eval_benchmarks.json gate / pre_introduction / alias_reveal sets (``spoiler_leakage_bench``)
* Dossier Fact Precision: supported units / emitted units (frames and phrases in the teacher top-k for
  their slot, pointers into visible events)
* Cross-Book Concept-F1: token F1 of the echo segment vs the teacher cross_book dossier text

Inputs (``--data-dir``): the v1 data (train_ablation.Data), ``cspw_timelines.jsonl``, ``cspw_aliases.jsonl``,
``cspw_bigram_spans.jsonl``, ``cspw_frames.json``, ``teacher_embeddings.npz``, ``teacher_dossiers.jsonl``.
Missing story / bigram files fall back to a synthetic story set / bigrams mined from the lexicon with the
v1 router (reported in ``summary.json["data"]``); missing teacher files disable KD (also reported).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cspw_model as CM  # noqa: E402
import model as M  # noqa: E402
import train as TA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V1_CKPT = ROOT / "runs" / "v1-final" / "arm3_platonic_engram_full.pt"
ARMS = ["arm0_v1_base", "arm1_arch_bigram_gqa", "arm2_scale_d192", "arm3_qwen_kd", "arm4_full_cspw"]
ARM_CFG = {
    "arm0_v1_base": dict(encoder="E0", d_model=None, pool="flat", typed=False, gate=False, kd=False),
    "arm1_arch_bigram_gqa": dict(encoder="E1", d_model=128, pool="flat", typed=False, gate=False, kd=False),
    "arm2_scale_d192": dict(encoder="E1", d_model=192, pool="flat", typed=False, gate=False, kd=False),
    "arm3_qwen_kd": dict(encoder="E3", d_model=192, pool="flat", typed=False, gate=False, kd=True),
    "arm4_full_cspw": dict(encoder="E3", d_model=192, pool="cfc", typed=True, gate=True, kd=True),
}
ENCODERS = {"E0": dict(bigram=False, kd=False, train=False), "E1": dict(bigram=True, kd=False, train=True),
            "E3": dict(bigram=True, kd=True, train=True)}
SLOT_FIELDS = ("role", "origin", "arc", "cross_book")
HAN4 = re.compile(r"[\u3400-\u9fff]{4}")
TOKEN = re.compile(r"[a-z0-9']+|[\u3400-\u9fff]")


def read_jsonl(path: Path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str))
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------
# Story data (+ synthetic fallback in the cspw_timelines.jsonl schema)
# ----------------------------------------------------------------------------------------------

_EN_NAMES = ["Elizabeth Bennet", "Fitzwilliam Darcy", "Jane Bennet", "George Wickham", "Charlotte Lucas",
             "Edward Rochester", "Jane Eyre", "Pip", "Estella", "Miss Havisham", "Captain Ahab", "Ishmael"]
_ZH_NAMES = ["萧炎", "唐三", "林动", "药老", "美杜莎", "小舞", "叶凡", "石昊", "韩立", "王林", "白小纯", "孟浩"]
_PLACES = ["Pemberley", "Netherfield", "Thornfield", "the marshes", "the Pequod", "Satis House",
           "乌坦城", "史莱克学院", "天墉城", "迦南学院", "黄枫谷", "落云宗"]
_ACTS = [("meets", "遇见"), ("quarrels with", "与{o}争执"), ("rescues", "救下"), ("deceives", "欺骗"),
         ("trains with", "随{o}修炼"), ("loses to", "败给"), ("confesses to", "向{o}坦白"), ("hunts", "追杀"),
         ("betrays", "背叛"), ("is reunited with", "与{o}重逢")]
_ROLES = ["a proud heir", "a sharp-tongued sister", "a wandering cultivator", "a disgraced genius",
          "an obsessed captain", "a quiet orphan", "a scheming officer", "a spirit mentor"]


def synthetic_story(n: int, seed: int):
    """Schema-exact stand-in for cspw_timelines.jsonl / cspw_aliases.jsonl (no teacher dossiers)."""
    rng = random.Random(seed)
    timelines, aliases = [], []
    for i in range(n):
        zh = i % 2 == 1
        names = _ZH_NAMES if zh else _EN_NAMES
        name = names[(i // 2) % len(names)]
        book = f"/books/Novels/syn{i % 6}.epub" if zh else f"/books/Books/syn{i % 6}.epub"
        term = name if zh else name.split()[-1]
        nev = rng.randint(3, 8)
        spine = sorted(rng.sample(range(1, 60), nev))
        events = []
        for m in range(nev):
            other = rng.choice([x for x in names if x != name])
            place = rng.choice(_PLACES[6:] if zh else _PLACES[:6])
            en, zh_act = rng.choice(_ACTS)
            if zh:
                act = zh_act.format(o=other) if "{o}" in zh_act else zh_act + other
                scene = f"{name}在{place}{act}。"
                gloss = f"{act}于{place}"
            else:
                scene = f"{name} {en} {other} at {place}."
                gloss = f"{en} {other} at {place}"
            off = rng.randint(0, 4000)
            events.append({"m": m, "cutoff": ((spine[m] + 1) << 16) | (off >> 6), "spine_idx": spine[m],
                           "page": spine[m] * 12 + off // 400, "chapter_or_page": spine[m],
                           "slot_hint": 1 if m == 0 else 2, "salience_hint": rng.randint(0, 255),
                           "event_gloss": CM.utf8_fit(gloss, 63).decode(),
                           "target_or_loc": CM.utf8_fit(place, 55).decode(), "scene_text": scene,
                           "sentence_text": scene, "co_entities": [other]})
        tl_aliases = []
        if rng.random() < 0.35:
            alias = ("那个少年" if zh else "the stranger") + str(i)
            rev = events[min(nev - 1, rng.randint(1, nev - 1))]["cutoff"]
            ah = M.fnv1a64(alias.encode())
            tl_aliases.append({"alias": alias, "alias_hash": ah, "reveal_cutoff": rev})
            aliases.append({"alias": alias, "alias_hash": ah, "canonical": term, "canonical_hash": M.fnv1a64(term.encode()),
                            "reveal_cutoff": rev, "book_path": book, "kind": "epithet"})
        echoes = []
        if rng.random() < 0.4:
            other_book = f"syn{(i + 3) % 6}"
            g = f"echoes the {rng.choice(['rival', 'mentor', 'exile'])} in {other_book}"
            echoes.append({"echo_id": f"ec-{i}", "source": "books", "book_path": f"/books/Books/{other_book}.epub",
                           "title": other_book, "cutoff": 1 << 16, "concept": "parallel",
                           "text": f"Like the {g.split()[2]} of {other_book}, {name} carries an old debt.", "gloss": g})
        timelines.append({"timeline_id": f"tl-{i:06d}", "split": "eval" if i % 5 == 4 else "train",
                          "source": "synthetic", "book_id": f"syn{i % 6}", "book_path": book,
                          "book_hash": M.fnv1a64(book.encode()), "term": term, "term_hash": M.fnv1a64(term.encode()),
                          "lang": "zh" if zh else "en", "entity_kind": 1, "kind": "character",
                          "en_name_or_role": name if not zh else rng.choice(_ROLES), "aliases": tl_aliases,
                          "first_cutoff": events[0]["cutoff"], "mention_cutoffs": [e["cutoff"] for e in events],
                          "events": events, "cross_book_echoes": echoes})
    return timelines, aliases


def fallback_frames():
    """Typed frames / delexicalised phrases satisfying the agreed constraints (used only when
    cspw_frames.json is absent)."""
    heads = ["", "In short, ", "Recall: ", "So far, ", "Note: ", "Here, ", "Briefly, ", "Remember: "]
    tails = ["", " (%L)", " near %L", ""]
    shapes = [["%E is %R{t}.", "%E, %R{t}.", "%E: %R{t}."], ["%E %O{t}.", "First, %E %O{t}.", "%E began: %O{t}."],
              ["%E %A{t}.", "Lately %E %A{t}.", "Then %E %A{t}."], ["%E echoes %B.", "%E recalls %B.", "Compare %B."]]
    frames = []
    for k in range(4):
        out = []
        i = 0
        while len(out) < 64:
            s = heads[i % len(heads)] + shapes[k][(i // len(heads)) % 3].format(t=tails[(i // 24) % 4] if k < 3 else "")
            if s not in out and len(s.encode()) <= 63:
                out.append(s)
            i += 1
            if i > 2000:
                out.append(f"{shapes[k][0].format(t='')[:-1]} #{len(out)}.")
        frames += out
    adj = ["proud", "restless", "loyal", "wounded", "ambitious", "quiet", "reckless", "watchful"]
    noun = ["heir", "outsider", "rival", "guardian", "student", "wanderer", "schemer", "survivor",
            "mentor", "exile", "witness", "seeker", "keeper", "pretender", "avenger", "dreamer"]
    phrases = [f"a {a} {n}" for a in adj for n in noun][:128]
    phrases += [f"first seen as a {a} {n}" for a in adj for n in noun][:128]
    phrases += [f"now {a} and {n}-like" for a in adj for n in noun][:128]
    phrases += [f"the {a} {n} of another tale" for a in adj for n in noun][:128]
    return frames, phrases


def load_story(data_dir: Path, out_dir: Path, args, log):
    info = {}
    tp = data_dir / "cspw_timelines.jsonl"
    timelines = [t for t in read_jsonl(tp) if t.get("teacher_valid", True) is not False]
    aliases = read_jsonl(data_dir / "cspw_aliases.jsonl")
    if timelines:
        info["timelines"] = str(tp)
        info["synthetic"] = False
    else:
        timelines, aliases = synthetic_story(args.synthetic_timelines, args.seed)
        syn = out_dir / "synthetic"
        syn.mkdir(parents=True, exist_ok=True)
        with open(syn / "cspw_timelines.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(t, ensure_ascii=False) + "\n" for t in timelines)
        with open(syn / "cspw_aliases.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(a, ensure_ascii=False) + "\n" for a in aliases)
        info["timelines"] = str(syn / "cspw_timelines.jsonl")
        info["synthetic"] = True
        log({"event": "story_fallback", "reason": f"{tp} missing", "timelines": len(timelines)})
    info["aliases"] = str(data_dir / "cspw_aliases.jsonl") if not info["synthetic"] else \
        str(out_dir / "synthetic" / "cspw_aliases.jsonl")
    fp = data_dir / "cspw_frames.json"
    if fp.exists():
        fj = json.loads(fp.read_text())
        frames, phrases = list(fj["frames"]), list(fj["phrases"])
        info["frames"] = str(fp)
    else:
        frames, phrases = fallback_frames()
        info["frames"] = "fallback"
    assert len(frames) == CM.N_FRAMES and len(phrases) == CM.N_PHRASES, (len(frames), len(phrases))
    dossiers = {}
    for d in read_jsonl(data_dir / "teacher_dossiers.jsonl"):
        dossiers[(d["timeline_id"], int(d["m"]))] = d
    teacher = {}
    tnp = data_dir / "teacher_embeddings.npz"
    if tnp.exists():
        z = np.load(tnp, allow_pickle=False)
        for g in ("passages", "definitions", "scenes", "glosses", "entities", "frames", "phrases"):
            if f"{g}_ids" in z:
                ids = [str(x) for x in z[f"{g}_ids"]]
                teacher[g] = (dict(zip(ids, range(len(ids)))), z[f"{g}_emb"].astype(np.float32))
    info.update({"n_timelines": len(timelines), "n_train": sum(t.get("split") == "train" for t in timelines),
                 "n_eval": sum(t.get("split") == "eval" for t in timelines), "n_aliases": len(aliases),
                 "teacher_dossiers": len(dossiers), "teacher_embedding_groups": sorted(teacher),
                 "teacher_kd_available": bool(dossiers) or bool(teacher)})
    return timelines, aliases, frames, phrases, dossiers, teacher, info


# ----------------------------------------------------------------------------------------------
# Encoders
# ----------------------------------------------------------------------------------------------


def load_v1(device, use_bigram: bool):
    ck = torch.load(V1_CKPT, map_location="cpu", weights_only=False)
    net = M.PlatonicEngramHNet(use_engram=True, use_bigram=use_bigram)
    missing, unexpected = net.load_state_dict(ck["state_dict"], strict=False)
    assert missing == ["engram_hash"] and not unexpected, (missing, unexpected)
    net.quant = True
    return net.to(device)


@torch.no_grad()
def mine_bigram_spans(net, data: TA.Data, limit: int, log):
    """Lexicon headwords (<= 18 B) the router cuts into exactly two chunks (in isolation) and corpus
    adjacent chunk pairs equal to a headword -> {hash: count}."""
    heads = sorted({h.strip() for h, _ in data.lex if 2 <= len(h.strip().encode()) <= M.BIGRAM_MAX_BYTES})
    head_set = {h.encode() for h in heads}
    counts = {}
    net.eval()
    for i in range(0, len(heads), 512):
        chunk = heads[i:i + 512]
        enc = net.encode(chunk)
        for h, n in zip(chunk, enc["n_chunks"].tolist()):
            if n == 2:
                counts[M.fnv1a64(h.encode())] = counts.get(M.fnv1a64(h.encode()), 0) + 1
    for _ in range(max(1, limit // 256)):
        X, n, _ = data.lm_batch_t(256)
        enc = net.encode_tensor(X, n)
        st, nc, Xc = enc["starts"].cpu().numpy(), enc["n_chunks"].cpu().numpy(), X.cpu().numpy()
        for b in range(len(Xc)):
            row = bytes(Xc[b, :int(n[b])].astype(np.uint8))
            ends = list(st[b, 1:nc[b]]) + [len(row)]
            for c in range(1, nc[b]):
                span = row[st[b, c - 1]:ends[c]]
                if len(span) <= M.BIGRAM_MAX_BYTES and span.strip() in head_set and span == span.strip():
                    h = M.fnv1a64(span)
                    counts[h] = counts.get(h, 0) + 2
    log({"event": "bigram_mined", "headwords": len(heads), "spans": len(counts)})
    return sorted(counts.items(), key=lambda kv: -kv[1])


def bigram_table(spans) -> np.ndarray:
    """Training-time slot -> span_hash column (highest count wins a slot)."""
    tab = np.zeros(M.ENGRAM_SLOTS, np.uint64)
    for h, _ in spans:
        s = int(h) % M.ENGRAM_SLOTS
        if tab[s] == 0 and int(h) != 0:
            tab[s] = int(h)
    return tab


def teacher_text_kd(teacher: dict, data_dir: Path, limit: int):
    """(texts, normalised teacher embeddings) for encoder relational KD: definitions (good_views b
    text by id) + passages (train_corpus text by id)."""
    texts, embs = [], []
    if "definitions" in teacher:
        ix, E = teacher["definitions"]
        for r in TA.read_jsonl(data_dir / "good_views.jsonl"):
            j = ix.get(str(r.get("id")))
            if j is not None:
                texts.append(TA.view_text(r["b"]))
                embs.append(E[j])
            if len(texts) >= limit // 2:
                break
    if "passages" in teacher:
        ix, E = teacher["passages"]
        with open(data_dir / "train_corpus.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                j = ix.get(str(r.get("id")))
                if j is not None:
                    texts.append(M.text_bytes(r["text"]).decode("utf-8", "ignore"))
                    embs.append(E[j])
                if len(texts) >= limit:
                    break
    if not texts:
        return None
    e = torch.tensor(np.stack(embs), dtype=torch.float32)
    return texts, F.normalize(e, dim=-1)


def relational_kd(s, t, tau: float = 0.05):
    """KL(softmax(T T^T / tau) || softmax(S S^T / tau)) over the batch, self-similarity excluded."""
    n = len(s)
    eye = torch.eye(n, dtype=torch.bool, device=s.device)
    ls = (s @ s.T / tau).masked_fill(eye, -1e4)
    lt = (t @ t.T / tau).masked_fill(eye, -1e4)
    return F.kl_div(F.log_softmax(ls, -1), F.log_softmax(lt, -1), log_target=True, reduction="batchmean")


def finetune_encoder(name: str, net, data: TA.Data, args, kd_set, log):
    """Warm-started QAT fine-tune: next-byte + recon + boundary + view InfoNCE/LeJEPA + sense ranking +
    Engram lexicon (headword alone + in context, where the bigram slots fire) [+ relational KD]."""
    dev = next(net.parameters()).device
    eparams = [net.engram]
    eid = {id(p) for p in eparams}
    groups = [{"params": [p for p in net.parameters() if id(p) not in eid], "weight_decay": 0.01},
              {"params": eparams, "weight_decay": 0.0, "lr": args.enc_lr * args.engram_lr_mult}]
    opt = torch.optim.AdamW(groups, lr=args.enc_lr, betas=(0.9, 0.98))
    base = [g.get("lr", args.enc_lr) for g in opt.param_groups]
    net.quant = True
    net.engram_drop = args.engram_drop
    t0 = time.time()
    steps = args.enc_steps
    for step in range(steps):
        frac = step / max(1, steps)
        warm = min(1.0, (step + 1) / max(1, args.warmup))
        for g, lr in zip(opt.param_groups, base):
            g["lr"] = lr * warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        net.train()
        L = {}
        X, n, lab = data.lm_batch_t(args.batch_lm)
        enc = net.encode_tensor(X, n)
        bl = net.byte_losses(enc, recon=True)
        L["next_byte"] = bl["next_byte"]
        L["recon"] = args.w_recon * bl["recon"]
        bstats = M.boundary_stats(enc, lab, args.boundary_margin)
        pi = data.pair_batch_t(args.batch_views)
        si, spos = data.sense_batch_t(args.batch_sense)
        na, ns = len(pi), (0 if si is None else len(si))
        idx = torch.cat([pi[:, 0], pi[:, 1]] + ([si[:, 0], si[:, 1]] if ns else []))
        c, z, _ = TA.encode_views(net, data, idx, args.view_buckets, False)
        L["contrastive"] = M.info_nce(c[:na], c[na:2 * na], net.nce_log_tau)
        L["lejepa"] = M.lejepa_loss(z[:na], z[na:2 * na], args.sigreg_lambda)[0]
        if ns:
            L["sense_rank"] = args.w_sense * M.sense_rank_loss(c[2 * na:2 * na + ns], c[2 * na + ns:],
                                                               net.sense_log_tau, 0.1, pos_mask=spos, margin=0.2)
        if len(data.lex):
            li = torch.randint(0, len(data.lex), (args.batch_lex,), device=dev, generator=data.gen)
            hl = data.lex_len[li]
            Lh = max(1, int(hl.max()))
            lc = data.lex_ctx_t[torch.randint(0, max(1, len(data.lex_ctx_t)), (args.batch_lex_ctx,), device=dev,
                                              generator=data.gen)] if len(data.lex_ctx_t) else None
            drop, net.engram_drop = net.engram_drop, 0.0
            eh = net.encode_tensor(data.lex_X[li, :Lh].long(), hl)
            didx = data.lex_def[li]
            if lc is not None:
                Xc, nc, labc = data.lex_in_context(lc)
                ec = net.encode_tensor(Xc, nc)
                didx = torch.cat([didx, data.lex_def[lc]])
            net.engram_drop = drop
            cd, zd, _ = TA.encode_views(net, data, didx, args.view_buckets, False)
            L["lex_nce"] = M.info_nce(eh["c"], cd[:len(li)], net.nce_log_tau)
            L["lex_lejepa"] = M.lejepa_loss(eh["z"], zd[:len(li)], args.sigreg_lambda)[0]
            hs = M.boundary_stats(eh, data.lex_lab[li, :Lh], args.boundary_margin)
            if lc is not None:
                L["lex_ctx_nce"] = M.info_nce(ec["c"], cd[len(li):], net.nce_log_tau)
                hs = hs + M.boundary_stats(ec, labc, args.boundary_margin)
            bstats = bstats + hs
        L["boundary"] = M.boundary_loss(bstats)
        if kd_set is not None:
            texts, temb = kd_set
            j = torch.randint(0, len(texts), (args.batch_kd,), generator=torch.Generator().manual_seed(step))
            ck = net.encode([texts[i] for i in j.tolist()], device=dev)["c"]
            L["qwen_rkd"] = args.w_enc_kd * relational_kd(ck, temb[j].to(dev))
        loss = sum(L.values())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == steps - 1:
            log({"stage": "encoder", "encoder": name, "step": step, "loss": round(float(loss.detach()), 5),
                 "sec": round(time.time() - t0, 1), **{k: round(float(v.detach()), 5) for k, v in L.items()}})
    net.engram_drop = 0.0
    net.eval()
    return net


@torch.no_grad()
def enc_int(net, texts, cache: dict, bs: int = 128):
    todo = [t for t in dict.fromkeys(texts) if t not in cache]
    for i in range(0, len(todo), bs):
        for t, v in zip(todo[i:i + bs], net.encode_int(todo[i:i + bs])):
            cache[t] = list(v)
    return [cache[t] for t in texts]


@torch.no_grad()
def enc_float(net, texts, bs: int = 128):
    """Normalised int8 concept vectors (float) -- what the device sees."""
    if not texts:
        return torch.zeros(0, M.D_CONCEPT)
    return F.normalize(torch.tensor(enc_int(net, texts, {}), dtype=torch.float32), dim=-1)


@torch.no_grad()
def eval_terms(net, xl, int8: bool):
    cands = xl["candidates"]
    ids = {c["id"]: i for i, c in enumerate(cands)}
    C = TA.embed(net, [c["text"] for c in cands], int8=int8)
    gold = torch.tensor([ids[q["gold"]] for q in xl["queries"]])
    anchors = [q["anchor"] for q in xl["queries"]]
    Q = TA.embed(net, anchors, int8=int8)
    sims = Q @ C.T
    gs = sims[torch.arange(len(gold)), gold]
    rank = (sims > gs[:, None]).sum(1) + 1
    prev = net.quant
    net.quant = True
    nch = net.encode(anchors)["n_chunks"].cpu()
    net.quant = prev
    idiom = torch.tensor([bool(HAN4.fullmatch(a)) for a in anchors])
    multi = nch >= 2
    r1 = (rank <= 1).float()

    def sub(m):
        return float(r1[m].mean()) if m.any() else None

    return {"term_R@1": float(r1.mean()), "term_R@1_idiom4": sub(idiom), "term_R@1_multichunk": sub(multi),
            "term_MRR": float((1.0 / rank.float()).mean()), "term_n_idiom4": int(idiom.sum()),
            "term_n_multichunk": int(multi.sum())}


def eval_encoder(net, data: TA.Data, args):
    net.eval()
    wsd = data.wsd[:args.eval_wsd] if args.eval_wsd else data.wsd
    xl = data.xling
    if args.eval_xling:
        qs = xl["queries"][:args.eval_xling]
        golds = {q["gold"] for q in qs}
        keep = [c for c in xl["candidates"] if c["id"] in golds]
        keep += [c for c in xl["candidates"] if c["id"] not in golds][:max(0, 3 * len(qs) - len(keep))]
        xl = {"queries": qs, "candidates": keep}
    prev = net.quant
    net.quant = False
    res = {}
    res.update({f"fp32_{k}": v for k, v in TA.eval_wsd(net, wsd).items()})
    res.update({f"fp32_{k}": v for k, v in TA.eval_xling(net, xl).items()})
    res.update({f"fp32_{k}": v for k, v in eval_terms(net, xl, False).items()})
    res.update({f"int8_{k}": v for k, v in TA.eval_wsd(net, wsd, int8=True).items()})
    res.update({f"int8_{k}": v for k, v in TA.eval_xling(net, xl, int8=True).items()})
    res.update({f"int8_{k}": v for k, v in eval_terms(net, xl, True).items()})
    net.quant = prev
    return res


# ----------------------------------------------------------------------------------------------
# CS-PW queries / batches
# ----------------------------------------------------------------------------------------------


def story_events(tl: dict):
    """Timeline -> event dicts in SGRF form (vec looked up later by key)."""
    evs = []
    for ev in tl["events"]:
        evs.append({"cutoff": int(ev["cutoff"]), "chapter_or_page": min(0xFFFF, int(ev.get("chapter_or_page", 0) or 0)),
                    "slot_hint": int(ev.get("slot_hint", 1 if ev["m"] == 0 else 2)),
                    "salience_hint": int(ev.get("salience_hint", 0)), "event_gloss": ev.get("event_gloss", ""),
                    "target_or_loc": ev.get("target_or_loc", ""), "text": ev["scene_text"], "m": int(ev["m"]),
                    "sentence": ev.get("sentence_text", "")})
    for x in tl.get("cross_book_echoes", []) or []:
        evs.append({"cutoff": int(tl["first_cutoff"]), "chapter_or_page": 0, "slot_hint": CM.SLOT_HINT_CROSS,
                    "salience_hint": 0, "event_gloss": x.get("gloss", ""), "target_or_loc": x.get("title", ""),
                    "text": x.get("text") or x.get("gloss", ""), "m": -1, "echo_text": x.get("text", "")})
    evs.sort(key=lambda e: e["cutoff"])
    return evs


def make_queries(timelines, split: str, with_probes: bool):
    """One query per visible story event position (+ spoiler probes: before first appearance, alias before
    and after its reveal)."""
    qs = []
    for ti, tl in enumerate(timelines):
        if tl.get("split", "train") != split:
            continue
        evs = story_events(tl)
        story_cuts = sorted({e["cutoff"] for e in evs if e["slot_hint"] != CM.SLOT_HINT_CROSS})
        th = int(tl["term_hash"])
        for cur in story_cuts:
            qs.append({"tl": ti, "cur": cur, "qhash": th, "canonical": True, "reveal": 0})
        if with_probes:
            first = int(tl["first_cutoff"])
            if first > 1:
                qs.append({"tl": ti, "cur": first - 1, "qhash": th, "canonical": True, "reveal": 0})
            for a in tl.get("aliases", []) or []:
                rev = int(a["reveal_cutoff"])
                for cur in (rev - 1, rev):
                    if cur >= 1:
                        qs.append({"tl": ti, "cur": cur, "qhash": int(a["alias_hash"]), "canonical": False,
                                   "reveal": rev})
    return qs


def query_view(tl: dict, q: dict, gated: bool):
    """(scene events, echo event, used events, allowed) for a query; ungated = no spoiler / alias gate."""
    first = int(tl["first_cutoff"])
    allowed = q["cur"] >= first and (q["canonical"] or q["cur"] >= q["reveal"])
    evs = story_events(tl)
    if gated:
        if not allowed:
            return [], None, allowed
        evs = [e for e in evs if e["cutoff"] <= q["cur"]]
    scenes, echo = CM.select_scenes(evs)
    return scenes, echo, allowed


def last_m(scenes):
    return max([e["m"] for e in scenes if e["m"] >= 0] or [0])


def _gist(e: dict) -> str:
    """Event gloss, else (extract-stage data: glosses not yet distilled) the event's own sentence."""
    return e["event_gloss"] or e.get("sentence") or e["text"]


def slot_target_texts(tl, scenes, echo, dossier):
    if dossier is not None:
        return [dossier.get(f) or "" for f in SLOT_FIELDS]
    name = tl.get("en_name_or_role") or tl.get("term", "")
    return [f"{name}: {_gist(scenes[-1])}", _gist(scenes[0]), _gist(scenes[-1]) if len(scenes) > 1 else "",
            (echo.get("echo_text") or _gist(echo)) if echo is not None else ""]


class StoryTensors:
    """Per-encoder cached int8 vectors of every scene / echo / target / frame / phrase text."""

    def __init__(self, net, timelines, frames, phrases, dossiers, teacher, cache: dict):
        self.net, self.tl, self.dossiers, self.teacher, self.cache = net, timelines, dossiers, teacher, cache
        self.frame_v = enc_float(net, [f or "-" for f in frames])
        self.phrase_v = enc_float(net, [p or "-" for p in phrases])

    def vec(self, text):
        return enc_int(self.net, [text], self.cache)[0]

    def batch(self, qs, gated: bool, swap: bool = False):
        """Tensors for queries ``qs`` (only those with >= 1 scene)."""
        rows = []
        for q in qs:
            tl = self.tl[q["tl"]]
            scenes, echo, allowed = query_view(tl, q, gated)
            if not scenes:
                continue
            rows.append((q, tl, scenes, echo))
        texts = [e["text"] for _, _, sc, ec in rows for e in sc + ([ec] if ec is not None else [])]
        enc_int(self.net, texts, self.cache)
        tgt_texts = []
        for q, tl, sc, ec in rows:
            d = self.dossiers.get((tl["timeline_id"], last_m(sc)))
            tgt_texts += slot_target_texts(tl, sc, ec, d)
        tv = enc_float(self.net, [t or "-" for t in tgt_texts]).view(-1, 4, M.D_CONCEPT) if rows else None
        B = len(rows)
        S = torch.zeros(B, 4, M.D_CONCEPT)
        P = torch.zeros(B, 4, dtype=torch.long)
        E = torch.zeros(B, M.D_CONCEPT)
        Mv = torch.ones(B, dtype=torch.long)
        he = torch.zeros(B, dtype=torch.bool)
        tmask = torch.zeros(B, 4)
        T_t = torch.zeros(B, 4, 256)
        T_m = torch.zeros(B, 4)
        fsoft = torch.zeros(B, 4, CM.FRAMES_PER_SLOT)
        psoft = torch.zeros(B, 4, CM.PHRASES_PER_SLOT)
        has_teacher = torch.zeros(B, dtype=torch.bool)
        gl = self.teacher.get("glosses")
        for b, (q, tl, sc, ec) in enumerate(rows):
            seq = list(sc)
            if swap and len(seq) >= 2:
                seq[0], seq[-1] = seq[-1], seq[0]
            for m, e in enumerate(seq):
                S[b, m] = torch.tensor(self.cache[e["text"]], dtype=torch.float32)
            for m, e in enumerate(sc):
                P[b, m] = e["chapter_or_page"]
            Mv[b] = len(sc)
            if ec is not None:
                E[b] = torch.tensor(self.cache[ec["text"]], dtype=torch.float32)
                he[b] = True
            tts = tgt_texts[4 * b:4 * b + 4]
            for k in range(4):
                tmask[b, k] = float(bool(tts[k]) and (k != 2 or len(sc) >= 2) and (k != 3 or ec is not None))
            mm = last_m(sc)
            d = self.dossiers.get((tl["timeline_id"], mm))
            if d is not None:
                has_teacher[b] = True
                for k in range(4):
                    for i, p in d.get("frame_topk", {}).get(str(k), []):
                        if 64 * k <= i < 64 * k + 64:
                            fsoft[b, k, i - 64 * k] += p
                    for i, p in d.get("phrase_topk", {}).get(str(k), []):
                        if 128 * k <= i < 128 * k + 128:
                            psoft[b, k, i - 128 * k] += p
            if gl is not None:
                for k in range(4):
                    j = gl[0].get(f"{tl['timeline_id']}:{mm}:{k}")
                    if j is not None:
                        T_t[b, k] = torch.from_numpy(gl[1][j])
                        T_m[b, k] = 1.0
        if B:
            # no teacher distribution -> PLHN-similarity argmax of the slot target (one-hot)
            fv = self.frame_v.view(4, CM.FRAMES_PER_SLOT, -1)
            pv = self.phrase_v.view(4, CM.PHRASES_PER_SLOT, -1)
            fs = torch.einsum("bkd,kfd->bkf", tv, fv)
            ps = torch.einsum("bkd,kpd->bkp", tv, pv)
            f1 = F.one_hot(fs.argmax(-1), CM.FRAMES_PER_SLOT).float()
            p1 = F.one_hot(ps.argmax(-1), CM.PHRASES_PER_SLOT).float()
            ht = has_teacher[:, None, None]
            fsoft = torch.where(ht & (fsoft.sum(-1, keepdim=True) > 0), fsoft, f1)
            psoft = torch.where(ht & (psoft.sum(-1, keepdim=True) > 0), psoft, p1)
            fsoft = fsoft / fsoft.sum(-1, keepdim=True).clamp(min=1e-9)
            psoft = psoft / psoft.sum(-1, keepdim=True).clamp(min=1e-9)
            fsim, psim = fs, ps
        else:
            fsim = psim = None
        return {"rows": rows, "S": S, "P": P, "E": E, "M": Mv, "he": he, "tv": tv, "tmask": tmask,
                "T": F.normalize(T_t, dim=-1), "Tm": T_m, "fsoft": fsoft, "psoft": psoft, "fsim": fsim, "psim": psim}


def cspw_loss(cs, bt, cfg, args):
    out = cs(bt["S"], bt["P"], bt["M"], bt["E"], bt["he"])
    slots = out["slots"]
    sn = F.normalize(slots, dim=-1, eps=1e-6)
    tm = bt["tmask"]
    L = {"slot_cos": ((1 - (sn * bt["tv"]).sum(-1)) * tm).sum() / tm.sum().clamp(min=1)}
    # pointer logits from fixed-norm slots: the device argmax is invariant to a per-row positive scale, and
    # this stops the CE from inflating slot norms into int8 saturation (which erased scene order)
    sl = sn * 127.0
    if cfg["typed"]:
        fl = cs.frame_logits(sl)
        L["frame_ce"] = -((bt["fsoft"] * F.log_softmax(fl, -1)).sum(-1) * tm).sum() / tm.sum().clamp(min=1)
        pl = cs.phrase_logits(sl, typed=True)
        pt = torch.zeros(len(slots), CM.N_PHRASES)
        pt[:, :CM.PHRASES_PER_SLOT] = bt["psoft"][:, 0]
        L["phrase_ce"] = -(pt * F.log_softmax(pl, -1)).sum(-1).mean()
    else:
        pl = cs.phrase_logits(sl, typed=False)
        L["phrase_ce"] = -((bt["psoft"] * F.log_softmax(pl, -1)).sum(-1) * tm).sum() / tm.sum().clamp(min=1)
    if cfg["kd"] and bt["Tm"].any():
        kd = []
        for k in range(4):
            m = bt["Tm"][:, k] > 0
            if m.sum() >= 4:
                kd.append(relational_kd(sn[m, k], bt["T"][m, k], tau=0.1))
        if kd:
            L["qwen_rkd"] = args.w_kd * sum(kd) / len(kd)
    if args.w_anchor > 0:
        # chronological anchoring: Slot1 (origin) stays on the first gated scene and Slot2 (arc) on the latest,
        # so teacher-dossier slot targets cannot wash out scene order
        S, Mv = bt["S"], bt["M"]
        ar = torch.arange(len(Mv))
        a0 = F.normalize(S[:, 0], dim=-1, eps=1e-6)
        a1 = F.normalize(S[ar, (Mv - 1).clamp(min=0)], dim=-1, eps=1e-6)
        multi = (Mv >= 2).to(sn.dtype)
        pull = (1 - (sn[:, 1] * a0).sum(-1)) + (1 - (sn[:, 2] * a1).sum(-1))
        L["anchor"] = args.w_anchor * (pull * multi).sum() / multi.sum().clamp(min=1)
    return sum(L.values()), L


def train_cspw_arm(arm: str, st: StoryTensors, train_q, args, log):
    cfg = ARM_CFG[arm]
    torch.manual_seed(args.seed)
    cs = CM.ChronoSlotPointerWeaver(d_model=cfg["d_model"], pool=cfg["pool"])
    opt = torch.optim.AdamW(cs.parameters(), lr=args.cspw_lr, betas=(0.9, 0.98), weight_decay=0.0)
    rng = random.Random(args.seed)
    qat_start = int(args.steps * args.qat_start)
    t0 = time.time()
    batches = {}
    for step in range(args.steps):
        frac = step / max(1, args.steps)
        for g in opt.param_groups:
            g["lr"] = args.cspw_lr * min(1.0, (step + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        cs.quant = step >= qat_start
        key = step % args.batch_cache
        if key not in batches:
            batches[key] = st.batch(rng.sample(train_q, min(args.batch_cspw, len(train_q))), gated=True)
        bt = batches[key]
        if not bt["rows"]:
            continue
        loss, L = cspw_loss(cs, bt, cfg, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cs.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            log({"stage": "cspw", "arm": arm, "step": step, "quant": cs.quant, "loss": round(float(loss.detach()), 5),
                 "sec": round(time.time() - t0, 1), **{k: round(float(v.detach()), 5) for k, v in L.items()}})
    cs.quant = True
    cs.eval()
    return cs


# ----------------------------------------------------------------------------------------------
# CS-PW evaluation
# ----------------------------------------------------------------------------------------------


def token_f1(a: str, b: str) -> float:
    ta, tb = TOKEN.findall(a.lower()), TOKEN.findall(b.lower())
    if not ta or not tb:
        return 0.0
    from collections import Counter

    common = sum((Counter(ta) & Counter(tb)).values())
    if common == 0:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def topk_sets(bt, b: int, dossier, k_frames: int = 16, k_phrases: int = 32):
    """Supported frame / phrase indices per slot: the teacher top-k, else the PLHN-similarity top-k to the
    slot target text."""
    fsets, psets = [], []
    for k in range(4):
        if dossier is not None and dossier.get("frame_topk", {}).get(str(k)):
            fsets.append({int(i) for i, _ in dossier["frame_topk"][str(k)][:k_frames]})
        else:
            fsets.append({64 * k + int(i) for i in bt["fsim"][b, k].topk(k_frames).indices})
        if dossier is not None and dossier.get("phrase_topk", {}).get(str(k)):
            psets.append({int(i) for i, _ in dossier["phrase_topk"][str(k)][:k_phrases]})
        else:
            psets.append({128 * k + int(i) for i in bt["psim"][b, k].topk(k_phrases).indices})
    return fsets, psets


def bench_queries(path: Path, timelines):
    """cspw_eval_benchmarks.json spoiler sets (gate / pre_introduction / alias_reveal) as queries."""
    if not path.exists():
        return []
    d = json.loads(path.read_text(encoding="utf-8"))
    tix = {t["timeline_id"]: i for i, t in enumerate(timelines)}
    sg = d.get("spoiler_gate", {})
    qs = []
    for it in sg.get("gate", []) + sg.get("pre_introduction", []):
        i = tix.get(it["timeline_id"])
        if i is not None:
            qs.append({"tl": i, "cur": int(it["cutoff"]), "qhash": int(timelines[i]["term_hash"]),
                       "canonical": True, "reveal": 0})
    for it in sg.get("alias_reveal", []):
        i = tix.get(it["timeline_id"])
        if i is not None:
            qs.append({"tl": i, "cur": int(it["cutoff"]), "qhash": int(it["alias_hash"]), "canonical": False,
                       "reveal": int(it["reveal_cutoff"])})
    return qs


@torch.no_grad()
def eval_cspw_arm(arm, cs, st: StoryTensors, eval_q, frames, phrases, aliases_all, args, swap: bool = True):
    cfg = ARM_CFG[arm]
    res = {}
    # temporal swap (gated inputs, M >= 2; cspw_eval_benchmarks.json rule): the first and last scene
    # swap places (pages stay with the positions); Slot1 (origin) must track the content at position 0
    # and Slot2 (arc) the content at position M-1, in the original and in the swapped run
    multi = [q for q in eval_q if q["canonical"] and q["cur"] >= int(st.tl[q["tl"]]["first_cutoff"])] if swap else []
    bt = st.batch(multi, gated=True)
    bs = st.batch(multi, gated=True, swap=True)
    keep = bt["M"] >= 2
    if swap and keep.any():
        ar = torch.arange(len(bt["M"]))
        first = F.normalize(bt["S"][:, 0], dim=-1)
        last = F.normalize(bt["S"][ar, (bt["M"] - 1).clamp(min=0)], dim=-1)

        def tracks(run, a, b):
            sl = cs(run["S"], run["P"], run["M"], run["E"], run["he"])["slots"]
            o, r = F.normalize(sl[:, 1], dim=-1), F.normalize(sl[:, 2], dim=-1)
            return ((o * a).sum(-1) > (o * b).sum(-1)) & ((r * b).sum(-1) > (r * a).sum(-1))

        ok = tracks(bt, first, last) & tracks(bs, last, first)
        res["temporal_swap_acc"] = float(ok[keep].float().mean())
        res["temporal_swap_n"] = int(keep.sum())
    # refresher generation
    leaks = produced = 0
    units = supported = 0
    f1s = []
    examples = []
    if cfg["gate"]:
        w = CM.CspwWeights(CM.pack_cspw(*cs.export_tensors(), frames, phrases))
        tls = [st.tl[i] for i in sorted({q["tl"] for q in eval_q})]
        ents = CM.timeline_entities(tls, st.vec)
        graph = CM.StoryGraph(CM.pack_sgrf(ents, CM.alias_rows(tls, aliases_all)))
        for q in eval_q:
            tl = st.tl[q["tl"]]
            allowed = q["cur"] >= int(tl["first_cutoff"]) and (q["canonical"] or q["cur"] >= q["reveal"])
            hit = graph.lookup(q["qhash"], int(tl.get("book_hash") or 0), q["cur"])
            if hit is None:
                continue
            trace = []
            text, r = CM.weave_refresher(w, hit[0], hit[1], tl.get("term", ""), trace)
            if not text:
                continue
            produced += 1
            scenes, echo = CM.select_scenes(hit[1])
            used = scenes + ([echo] if echo is not None else [])
            leak = (not allowed) or any(e["cutoff"] > q["cur"] for e in used)
            leaks += leak
            sc_q, ec_q, _ = query_view(tl, q, True)
            d = st.dossiers.get((tl["timeline_id"], last_m(sc_q))) if sc_q else None
            one = st.batch([q], gated=True)
            if not one["rows"]:
                continue
            fsets, psets = topk_sets(one, 0, d)
            for k, fidx, tpl, _ in trace:
                if fidx >= 0:
                    units += 1
                    supported += fidx in fsets[k]
                for ph in re.findall(r"%([EROALB])", tpl):
                    units += 1
                    if ph == "R":
                        supported += r is not None and r["phrase_idx"] in psets[0]
                    else:
                        supported += not leak
            if " | Echo: " in text and ec_q is not None:
                ref = (d or {}).get("cross_book") or ec_q.get("echo_text") or ec_q["event_gloss"]
                f1s.append(token_f1(text.split(" | Echo: ", 1)[1], ref))
            if len(examples) < 6:
                examples.append({"term": tl.get("term"), "cur": q["cur"], "text": text})
    else:
        for i in range(0, len(eval_q), 256):
            qs = eval_q[i:i + 256]
            bu = st.batch(qs, gated=False)
            if not bu["rows"]:
                continue
            out = cs(bu["S"], bu["P"], bu["M"], bu["E"], bu["he"])["slots"]
            pidx = cs.phrase_logits(out, typed=False).argmax(-1)  # (B,4) within slot range
            for b, (q, tl, sc, ec) in enumerate(bu["rows"]):
                allowed = q["cur"] >= int(tl["first_cutoff"]) and (q["canonical"] or q["cur"] >= q["reveal"])
                produced += 1
                used = sc + ([ec] if ec is not None else [])
                leak = (not allowed) or any(e["cutoff"] > q["cur"] for e in used)
                leaks += leak
                d = st.dossiers.get((tl["timeline_id"], last_m(sc)))
                fsets, psets = topk_sets(bu, b, d)
                ph = [128 * k + int(pidx[b, k]) for k in range(4)]
                name = tl.get("en_name_or_role") or tl.get("term", "")
                text = f"Story: {name}: {phrases[ph[0]]} \u2014 {phrases[ph[1]]}"
                ks = [0, 1]
                if len(sc) >= 2:
                    text += f"; {phrases[ph[2]]}"
                    ks.append(2)
                text += "."
                units += 1  # %E pointer
                supported += 1
                if ec is not None:
                    text += f" | Echo: {phrases[ph[3]]}"
                    ks.append(3)
                    ref = (d or {}).get("cross_book") or ec.get("echo_text") or ec["event_gloss"]
                    f1s.append(token_f1(phrases[ph[3]], ref))
                for k in ks:
                    units += 1
                    supported += ph[k] in psets[k]
                if len(examples) < 6:
                    examples.append({"term": tl.get("term"), "cur": q["cur"], "text": text})
    res.update({"spoiler_leakage": leaks / max(1, len(eval_q)), "spoiler_queries": len(eval_q),
                "refreshers": produced, "dossier_fact_precision": supported / max(1, units), "fp_units": units,
                "crossbook_concept_f1": float(np.mean(f1s)) if f1s else None, "crossbook_n": len(f1s),
                "examples": examples})
    return res


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="all")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "cspw-v2")
    ap.add_argument("--device", default="cpu", help="encoder fine-tune device (CS-PW trains on CPU)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--steps", type=int, default=2000, help="CS-PW steps per arm")
    ap.add_argument("--enc-steps", type=int, default=None, help="encoder fine-tune steps (default = --steps)")
    ap.add_argument("--qat-start", type=float, default=0.5)
    ap.add_argument("--enc-lr", type=float, default=3e-4)
    ap.add_argument("--engram-lr-mult", type=float, default=5.0)
    ap.add_argument("--engram-drop", type=float, default=0.1)
    ap.add_argument("--cspw-lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch-lm", type=int, default=64)
    ap.add_argument("--batch-views", type=int, default=256)
    ap.add_argument("--batch-sense", type=int, default=128)
    ap.add_argument("--batch-lex", type=int, default=256)
    ap.add_argument("--batch-lex-ctx", type=int, default=128)
    ap.add_argument("--batch-kd", type=int, default=128)
    ap.add_argument("--batch-cspw", type=int, default=256)
    ap.add_argument("--batch-cache", type=int, default=64, help="distinct CS-PW batches cycled through")
    ap.add_argument("--view-buckets", type=int, default=4)
    ap.add_argument("--w-recon", type=float, default=0.5)
    ap.add_argument("--w-sense", type=float, default=1.5)
    ap.add_argument("--w-kd", type=float, default=1.0, help="CS-PW slot relational KD weight")
    ap.add_argument("--w-enc-kd", type=float, default=None, help="encoder relational KD weight (default = --w-kd)")
    ap.add_argument("--w-anchor", type=float, default=1.0, help="CS-PW origin/arc chronological anchor weight")
    ap.add_argument("--boundary-margin", type=float, default=0.2)
    ap.add_argument("--sigreg-lambda", type=float, default=0.02)
    ap.add_argument("--corpus-limit", type=int, default=None)
    ap.add_argument("--bigram-mine-windows", type=int, default=4096)
    ap.add_argument("--kd-texts", type=int, default=20000)
    ap.add_argument("--synthetic-timelines", type=int, default=240)
    ap.add_argument("--eval-wsd", type=int, default=0, help="0 = all WSD items")
    ap.add_argument("--eval-xling", type=int, default=0, help="0 = all cross-lingual queries")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="small batches / corpus / eval subsets")
    args = ap.parse_args(argv)
    if args.smoke:
        args.corpus_limit = args.corpus_limit or 3000
        args.batch_lm, args.batch_views, args.batch_sense = 8, 48, 24
        args.batch_lex, args.batch_lex_ctx, args.batch_kd, args.batch_cspw = 48, 16, 32, 64
        args.bigram_mine_windows, args.kd_texts, args.batch_cache = 512, 2000, 16
        args.eval_wsd, args.eval_xling = args.eval_wsd or 200, args.eval_xling or 150
        args.view_buckets = 2
    if args.enc_steps is None:
        args.enc_steps = args.steps
    if args.w_enc_kd is None:
        args.w_enc_kd = args.w_kd
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train_log.jsonl", "a", encoding="utf-8")

    def log(rec):
        rec = {"t": round(time.time(), 1), **rec}
        logf.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        logf.flush()
        print(json.dumps(rec, ensure_ascii=False, default=str), flush=True)

    arms = ARMS if args.arms == "all" else args.arms.split(",")
    dev = torch.device(args.device)
    log({"event": "start", "arms": arms, "args": vars(args)})
    data = TA.Data(args.data_dir, args.seed, args.corpus_limit, dev)
    timelines, aliases, frames, phrases, dossiers, teacher, story_info = load_story(args.data_dir, out, args, log)
    bp = args.data_dir / "cspw_bigram_spans.jsonl"
    rows = read_jsonl(bp)
    if rows:
        spans = sorted(((int(r["span_hash"]), int(r.get("count", 1))) for r in rows), key=lambda kv: -kv[1])
        story_info["bigram_spans"] = {"source": str(bp), "n": len(spans)}
    else:
        spans = mine_bigram_spans(load_v1(dev, False), data, args.bigram_mine_windows, log)
        story_info["bigram_spans"] = {"source": "mined (lexicon headwords split in two by the v1 router)",
                                      "n": len(spans)}
    btab = bigram_table(spans)
    kd_set = teacher_text_kd(teacher, args.data_dir, args.kd_texts) if teacher else None
    story_info["encoder_kd_texts"] = 0 if kd_set is None else len(kd_set[0])
    summary = {"arms": {}, "encoders": {}, "data": story_info, "steps": args.steps, "enc_steps": args.enc_steps}

    needed = sorted({ARM_CFG[a]["encoder"] for a in arms})
    encoders = {}
    for name in needed:
        ecfg = ENCODERS[name]
        torch.manual_seed(args.seed)
        net = load_v1(dev, ecfg["bigram"])
        if ecfg["bigram"]:
            net.set_engram_hashes(btab)
        if ecfg["train"]:
            if name == "E3" and "E1" in encoders:
                # E3 = E1 + Qwen relational KD: continue from the trained E1 weights
                net.load_state_dict(encoders["E1"].state_dict())
                if kd_set is None:
                    log({"event": "encoder_reuse", "encoder": "E3", "from": "E1", "reason": "no teacher embeddings"})
                else:
                    log({"event": "encoder_warm_start", "encoder": "E3", "from": "E1"})
                    net = finetune_encoder(name, net, data, args, kd_set, log)
            else:
                net = finetune_encoder(name, net, data, args, kd_set if ecfg["kd"] else None, log)
        net.eval()
        encoders[name] = net
        r = eval_encoder(net, data, args)
        r["bigram_slots"] = int((btab != 0).sum()) if ecfg["bigram"] else 0
        r["teacher_kd"] = bool(ecfg["kd"] and kd_set is not None)
        summary["encoders"][name] = r
        log({"event": "encoder_eval", "encoder": name, **r})
        atomic_json(out / "summary.json", summary)

    story_cpu = {}
    train_q = make_queries(timelines, "train", with_probes=False)
    eval_q = make_queries(timelines, "eval", with_probes=True)
    bench_q = bench_queries(args.data_dir / "cspw_eval_benchmarks.json", timelines)
    summary["data"].update({"train_queries": len(train_q), "eval_queries": len(eval_q),
                            "bench_spoiler_queries": len(bench_q)})
    for arm in arms:
        cfg = ARM_CFG[arm]
        en = cfg["encoder"]
        if en not in story_cpu:
            enet = copy.deepcopy(encoders[en]).cpu().eval()
            story_cpu[en] = StoryTensors(enet, timelines, frames, phrases, dossiers, teacher, {})
        st = story_cpu[en]
        cs = train_cspw_arm(arm, st, train_q, args, log)
        r = eval_cspw_arm(arm, cs, st, eval_q, frames, phrases, aliases, args)
        if bench_q:
            rb = eval_cspw_arm(arm, cs, st, bench_q, frames, phrases, aliases, args, swap=False)
            r["spoiler_leakage_bench"] = rb["spoiler_leakage"]
            r["spoiler_bench_queries"] = len(bench_q)
        r.update({"encoder": en, **{k: v for k, v in cfg.items() if k != "encoder"},
                  "teacher_kd": bool(cfg["kd"] and story_info["teacher_kd_available"]),
                  "cspw_params": sum(p.numel() for p in cs.parameters())})
        enc_r = summary["encoders"][en]
        for k in ("fp32_wsd_top1", "int8_wsd_top1", "int8_xling_R@1", "int8_xling_R@5", "int8_xling_MRR",
                  "int8_term_R@1", "int8_term_R@1_idiom4", "int8_term_R@1_multichunk"):
            r[k] = enc_r.get(k)
        summary["arms"][arm] = r
        log({"event": "arm_eval", "arm": arm, **{k: v for k, v in r.items() if k != "examples"}})
        ck = {"arm": arm, "encoder_name": en, "encoder": encoders[en].state_dict(),
              "encoder_cfg": {"use_bigram": ENCODERS[en]["bigram"]}, "cspw": cs.state_dict(),
              "cspw_cfg": {"d_model": cfg["d_model"], "pool": cfg["pool"]}, "frames": frames, "phrases": phrases,
              "bigram_spans": [[int(h), int(c)] for h, c in spans], "story": story_info, "args": vars(args)}
        tmp = out / f".{arm}.pt.tmp"
        torch.save(ck, tmp)
        os.replace(tmp, out / f"{arm}.pt")
        atomic_json(out / "summary.json", summary)
    a4 = summary["arms"].get("arm4_full_cspw")
    if a4 is not None:
        summary["arm4_spoiler_leakage_zero"] = a4["spoiler_leakage"] == 0.0 and a4.get("spoiler_leakage_bench", 0.0) == 0.0
    atomic_json(out / "summary.json", summary)
    log({"event": "done", "summary": str(out / "summary.json")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
