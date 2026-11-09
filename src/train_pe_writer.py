"""Train + evaluate the PE-Writer ablation arms.

  Arm 0  Flat AR micro-LM baseline     (c_page only; no planner, no WCLS bank, no DPO; float + PTQ)
  Arm 1  + Ouro 3-step latent planner   (LeJEPA SIGReg + cosine trajectory alignment to teacher clauses)
  Arm 2  + SD Clause-Engram WCLS bank   (64 x 64 slots, copy_gate_proj clause-guided logit boost)
  Arm 3  Full PE-Writer                 (Arm 2 init + on-policy loop-rejecting DPO + teacher top-K KD + int8 QAT)
  Arm 4  Frame-Locked PE-Writer v2      (proper-noun-free role-partitioned WCLS v2 bank, InfoNCE + cosine + routing
                                         Ouro planner, decoder locked to <EOS> + slot specials + 32 connector
                                         pieces: ``<ENT_SUBJECT>, <ENT_ROLE>: <CLAUSE_0><conn> <BOOK_ECHO>,
                                         <CLAUSE_1>.``; int8 QAT). Its own run dir: ``--arms 4``.

Every arm is evaluated on held-out ``val`` through its *exported int8* forward (``PeWriterIntTorch``, bit-exact
with the device kernel), across: thought-trajectory cosine, WCLS clause Recall@1/@5, content-F1 / ROUGE-L vs
the Qwen3-32B teacher riffs, Rep-3 doom-loop rate, entity/echo grounding precision.

    python src/train_pe_writer.py --smoke --steps 100            # CPU synthetic smoke
    python src/train_pe_writer.py --all-arms                     # GPU, data/pe_writer/writer_dataset.npz
    python src/train_pe_writer.py --arms 4 --run runs/pe_writer/arm4_frame_locked   # Mac CPU, minutes
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pe_writer_model as PW  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "pe_writer" / "writer_dataset.npz"
RUNS = ROOT / "runs" / "pe_writer"

ARMS = {
    0: dict(name="flat_ar_baseline", use_planner=False, use_bank=False, dpo=False, qat=False),
    1: dict(name="ouro_planner", use_planner=True, use_bank=False, dpo=False, qat=False),
    2: dict(name="wcls_clause_engram", use_planner=True, use_bank=True, dpo=False, qat=False),
    3: dict(name="full_pe_writer", use_planner=True, use_bank=True, dpo=True, qat=True),
    4: dict(name="frame_locked_pe_writer_v2", use_planner=True, use_bank=True, dpo=False, qat=True),
}
BANK_V2 = ROOT / "data" / "pe_writer" / "clause_bank_v2.npz"

STOPWORDS = set("""a an the and or but of to in on at by for with from as is are was were be been being it its this that
these those his her their our your my he she they we you i not no so than then there here into over under about
after before while which who whom whose what when where why how all any each both more most other some such only own
same too very can will just also yet nor if because between through during without within upon""".split())
# Capitalized template / transition words the writer emits mid-sentence (after ";", ",", "—", ":"). They are
# connective tissue, not entities, so they never count as invented proper names.
TEMPLATE_CAPS = frozenset("""Here Thus Yet Where While Instead Both Though Even Rather Within Across Beneath Through
This That When What""".split()) | frozenset(w.capitalize() for w in STOPWORDS)


def invented_name(word: str, prev: str, allowed) -> bool:
    """Capitalized, not sentence-initial, not a template/function word and absent from the grounding context."""
    return (word[:1].isupper() and prev not in (".", "!", "?") and word not in TEMPLATE_CAPS
            and word not in allowed)


def rendered_words(seq, subject: str, role: str, echo: str, ctok_rows, vocab, owners: bool = False):
    """Lower-cased alnum words of a template row as the device renders it (specials expanded, clause slots from
    the retrieved WCLS clause tokens); the same word stream the Rep-3 metric scores. ``owners=True`` also returns
    the index of the template token that produced each word."""
    words, own = [], []
    for t, v in enumerate(seq):
        if v == PW.ENT_SUBJECT:
            ws = PW.word_tokens(subject)
        elif v == PW.ENT_ROLE:
            ws = PW.word_tokens(role or PW.ROLE_FALLBACK.decode())
        elif v == PW.BOOK_ECHO:
            ws = PW.word_tokens(echo)
        elif PW.CLAUSE_0 <= v <= PW.CLAUSE_0 + 2:
            ws = [vocab[int(c)] for c in ctok_rows[v - PW.CLAUSE_0] if PW.N_SPECIAL <= int(c) < PW.N_VOCAB]
        elif v >= PW.N_SPECIAL:
            ws = [vocab[v]]
        else:
            ws = []
        for w in ws:
            if any(ch.isalnum() for ch in w):
                words.append(w.lower())
                own.append(t)
    return (words, own) if owners else words


def loop_token(words, own):
    """Template index of the token that completes the first repeated word 3-gram (None if no repeat)."""
    seen = set()
    for j in range(len(words) - 2):
        g = tuple(words[j:j + 3])
        if g in seen:
            return own[j + 2]
        seen.add(g)
    return None


# ----------------------------------------------------------------------------- data
def _norm_rows(x):
    return PW.normalize_i8_rows(np.asarray(x, np.int64)).astype(np.int8)


def synthetic_dataset(seed: int = 0, n_train: int = 512, n_val: int = 64) -> dict:
    """Small structured world: topic vectors drive passages, echoes and the 3 teacher clauses."""
    rng = np.random.default_rng(seed)
    n_top = 48
    topics = rng.standard_normal((n_top, PW.D))
    adjs = ["hidden", "quiet", "brittle", "stubborn", "borrowed", "inherited", "fragile", "patient", "restless",
            "silent", "public", "private", "sacred", "ordinary", "ruthless", "tender"]
    nouns = ["orthodoxy", "loyalty", "memory", "ambition", "grief", "power", "faith", "silence", "debt", "exile",
             "honor", "desire", "survival", "doubt", "class", "time"]
    verbs = ["reveals", "masks", "inverts", "tests", "exposes", "bends", "mirrors", "undoes", "sharpens", "frames"]
    links = ["while", "because", "until", "so", "yet", "as"]
    names = ["Elinor", "Winston", "Raskol", "Lizzy", "Darcy", "Ahab", "Ishmael", "Julia", "Sonya", "Marlow",
             "孙悟空", "林黛玉"]
    roles = ["the reluctant narrator", "a disgraced clerk", "the rebel", "the outsider", "", "the heir"]
    echoes = ["Nineteen Eighty-Four", "Moby-Dick", "Middlemarch", "Crime and Punishment", "", "Heart of Darkness"]

    def clause(top, s):
        r = np.random.default_rng(top * 7 + s * 1009)
        return (f"{r.choice(verbs)} {r.choice(adjs)} {r.choice(nouns)} {r.choice(links)} "
                f"{r.choice(adjs)} {r.choice(nouns)} {'survives' if s == 2 else 'hardens'}")

    def split(n, off):
        cp, ce, cc, ct, riffs, tpls, subj, rol, ech, pas = ([] for _ in range(10))
        for i in range(n):
            a, b = rng.integers(n_top, size=2)
            tops = [a, (a + 1 + rng.integers(3)) % n_top, (a + b) % n_top]
            cp.append(topics[a] + 0.35 * rng.standard_normal(PW.D))
            ce.append(topics[b] + 0.35 * rng.standard_normal(PW.D))
            cc.append([topics[t] + 0.15 * rng.standard_normal(PW.D) for t in tops])
            cls = [clause(int(t), s) for s, t in enumerate(tops)]
            sj, rl, ec = rng.choice(names), rng.choice(roles), rng.choice(echoes)
            tpl = rng.choice([
                "<ENT_SUBJECT> , <ENT_ROLE> , <CLAUSE_0> ; the scene <CLAUSE_1> , echoing <BOOK_ECHO> : <CLAUSE_2> .",
                "Here <ENT_SUBJECT> <CLAUSE_0> , and <CLAUSE_1> — as in <BOOK_ECHO> , the page <CLAUSE_2> .",
                "<ENT_SUBJECT> as <ENT_ROLE> <CLAUSE_0> ; underneath , <CLAUSE_1> , so <CLAUSE_2> ."])
            if not ec:
                tpl = tpl.replace("<BOOK_ECHO>", "the older story")
            fill = {"<ENT_SUBJECT>": sj, "<ENT_ROLE>": rl or "this concept", "<BOOK_ECHO>": ec,
                    "<CLAUSE_0>": cls[0], "<CLAUSE_1>": cls[1], "<CLAUSE_2>": cls[2]}
            riff = tpl
            for k, v in fill.items():
                riff = riff.replace(k, v)
            ct.append(cls)
            riffs.append(riff)
            tpls.append(tpl)
            subj.append(sj)
            rol.append(rl)
            ech.append(ec)
            pas.append(f"{sj} {rl} {ec} " + " ".join(cls))
        return {"c_page": _norm_rows(cp), "c_echo": _norm_rows(ce),
                "c_clause": _norm_rows(np.asarray(cc).reshape(-1, PW.D)).reshape(n, 3, PW.D),
                "clause_text": np.array(ct), "riff": np.array(riffs), "riff_template": np.array(tpls),
                "subject": np.array(subj), "role": np.array(rol), "echo_title": np.array(ech),
                "passage": np.array(pas)}

    tr, va = split(n_train, 0), split(n_val, 1)
    bank_vec, bank_text, bank_role = [], [], []
    for t in range(n_top):
        for s in range(3):
            for k in range(6):  # several distinct clauses per (topic, step)
                bank_vec.append(topics[t] + 0.1 * rng.standard_normal(PW.D))
                bank_text.append(clause(t, s) if k == 0 else clause(t * 13 + k, s))
                bank_role.append(s)
    out = {f"train_{k}": v for k, v in tr.items()} | {f"val_{k}": v for k, v in va.items()}
    out["bank_vec"] = _norm_rows(bank_vec)
    out["bank_text"] = np.array(bank_text)
    out["bank_step_role"] = np.array(bank_role, np.uint8)
    return out


def load_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _s(x) -> str:
    return str(x) if x is not None else ""


class Split:
    """Tokenized split held as numpy + torch tensors."""

    def __init__(self, d: dict, prefix: str, tok: PW.Tokenizer, max_n: int | None = None):
        g = lambda k, default=None: d.get(f"{prefix}_{k}", default)  # noqa: E731
        n = len(g("c_page"))
        if max_n:
            n = min(n, max_n)
        self.n = n
        self.c_page = np.asarray(g("c_page")[:n], np.int64)
        self.c_echo = np.asarray(g("c_echo")[:n], np.int64)
        self.c_clause = np.asarray(g("c_clause")[:n], np.int64)
        self.clause_text = [[_s(c) for c in row] for row in g("clause_text")[:n]]
        self.riff = [_s(x) for x in g("riff")[:n]]
        self.subject = [_s(x) for x in g("subject")[:n]]
        self.role = [_s(x) for x in g("role")[:n]]
        self.echo = [_s(x) for x in g("echo_title")[:n]]
        pas = g("passage")
        ctx = g("page_ctx")
        self.passage = [(_s(pas[i]) if pas is not None else "") + " " + (_s(ctx[i]) if ctx is not None else "")
                        for i in range(n)]
        tpl = g("riff_template")
        self.template = [_s(tpl[i]) if tpl is not None else
                         PW.riff_to_template(self.riff[i], self.subject[i], self.role[i], self.echo[i],
                                             self.clause_text[i]) for i in range(n)]
        toks = np.full((n, PW.T_MAX), -1, np.int64)
        for i in range(n):
            ids = tok.encode_template(self.template[i])
            ids = [v for v in ids if not ((v == PW.ENT_SUBJECT and not self.subject[i]) or
                                          (v == PW.BOOK_ECHO and not self.echo[i]))]
            toks[i, :len(ids)] = ids
        self.tokens = toks
        self.kd_ids = g("kd_topk_ids")
        self.kd_p = g("kd_topk_p")
        if self.kd_ids is not None:
            self.kd_ids, self.kd_p = np.asarray(self.kd_ids[:n], np.int64), np.asarray(self.kd_p[:n], np.float32)
        self.allowed = [set(PW.word_tokens(" ".join([self.subject[i], self.role[i], self.echo[i], self.passage[i]]
                                                     + self.clause_text[i]))) for i in range(n)]

    def batch(self, idx, dev) -> dict:
        T = lambda a, dt=torch.float32: torch.as_tensor(a[idx], dtype=dt, device=dev)  # noqa: E731
        b = {"c_page": T(self.c_page), "c_echo": T(self.c_echo), "c_clause": T(self.c_clause),
             "tokens": T(self.tokens, torch.long),
             "has_subject": torch.as_tensor([bool(self.subject[i]) for i in idx], device=dev),
             "has_echo": torch.as_tensor([bool(self.echo[i]) for i in idx], device=dev), "idx": idx}
        if self.kd_ids is not None:
            b["kd_ids"] = T(self.kd_ids, torch.long)
            b["kd_p"] = T(self.kd_p)
        return b


def bank_tensors(layout: dict, dev):
    vec = torch.as_tensor(layout["vec"].astype(np.float32), device=dev)
    tok = torch.as_tensor(layout["tok"].astype(np.int64), device=dev)
    tok = torch.where(tok >= PW.N_VOCAB, torch.zeros_like(tok), tok)
    valid = torch.as_tensor(np.array([[len(t) > 0 for t in row] for row in layout["text"]]), device=dev)
    return vec, tok, valid


# ----------------------------------------------------------------------------- losses
def ce_loss(logits, tokens, kd_ids=None, kd_p=None, kd_weight=0.0):
    tgt = tokens.clone()
    valid = tgt >= 0
    g = logits.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    valid &= g > -1e6  # never train toward an excluded token
    logp = F.log_softmax(logits, -1)
    nll = -(logp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1))
    loss = (nll * valid).sum() / valid.sum().clamp(min=1)
    if kd_ids is not None and kd_weight > 0:
        p = kd_p / kd_p.sum(-1, keepdim=True).clamp(min=1e-9)
        lq = logp.gather(-1, kd_ids.clamp(0, PW.N_VOCAB - 1))
        kd = -(p * lq).sum(-1)
        loss = loss + kd_weight * (kd * valid).sum() / valid.sum().clamp(min=1)
    return loss


def seq_logprob(model, batch, bank, tokens, qat):
    b = dict(batch)
    b["tokens"] = tokens
    out = model(b, bank, qat)
    logp = F.log_softmax(out["logits"], -1)
    valid = tokens >= 0
    lp = logp.gather(-1, tokens.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    lp = torch.where(valid & (lp > -1e6), lp, torch.zeros_like(lp))
    return lp.sum(-1), out


def plan_losses(c_plan, c_clause, sigreg_w, gen):
    cos = F.cosine_similarity(c_plan, c_clause, dim=-1)  # [B,3]
    l = (1 - cos).mean()
    if sigreg_w > 0:
        l = l + sigreg_w * PW.sigreg_loss(c_plan.reshape(-1, PW.D), generator=gen)
    return l, cos.detach().mean(0)


def reject_reasons(toks, subject_ok, cap_ids, allowed_words, vocab, words, echo_ok=False):
    """On-policy loop/grounding audit of a token row -> set of reasons (empty = acceptable).
    ``words``: rendered word stream of the row (``rendered_words``); Rep-3 is judged on it exactly as evaluated."""
    seq = []
    for v in toks:
        v = int(v)
        if v == PW.PAD:
            break
        seq.append(v)
    reasons = set()
    grams = [tuple(words[j:j + 3]) for j in range(len(words) - 2)]
    if len(grams) != len(set(grams)):
        reasons.add("rep3")
    if subject_ok and PW.ENT_SUBJECT not in seq:
        reasons.add("missing_entity")
    if echo_ok and PW.BOOK_ECHO not in seq:
        reasons.add("missing_echo")
    for i, v in enumerate(seq):
        if v in cap_ids and i > 0 and invented_name(vocab[v], vocab[seq[i - 1]], allowed_words):
            reasons.add("ungrounded_name")
    if not seq:
        reasons.add("empty")
    return reasons


# ----------------------------------------------------------------------------- training
def make_model(arm: int, layout: dict, dev, init_state=None) -> PW.PeWriterQAT:
    cfg = ARMS[arm]
    m = PW.PeWriterQAT(layout["centroids"], use_planner=cfg["use_planner"], use_bank=cfg["use_bank"]).to(dev)
    if init_state is not None:
        m.load_state_dict(init_state)
    return m


def optimizer(model, lr, lr_raw):
    raw, bias, w = [], [], []
    for n, p in model.named_parameters():
        if n in model.raw_param_names():
            raw.append(p)
        elif n.endswith("bias"):
            bias.append(p)
        else:
            w.append(p)
    return torch.optim.Adam([{"params": w, "lr": lr}, {"params": raw, "lr": lr_raw},
                             {"params": bias, "lr": lr_raw * 0.5}])


def train_arm(arm, model, tr: Split, bank, args, dev, ref_model=None, tok=None, log=print, select=None):
    """``select(model) -> (eligible, key, metrics)``: optional dev-split checkpoint selector (arm 3). The step-0
    state is always scored, so the returned weights never fall below the init on the selector's criteria."""
    cfg = ARMS[arm]
    s_lr = args.dpo_lr_scale if cfg["dpo"] else 1.0
    opt = optimizer(model, args.lr * s_lr, args.lr_raw * s_lr)
    steps = args.steps if arm < 3 else args.dpo_steps
    gen = torch.Generator(device="cpu").manual_seed(args.seed + arm)
    tgen = torch.Generator(device=dev).manual_seed(args.seed + 17 * arm) if dev.type == "cuda" else None
    rng = np.random.default_rng(args.seed + arm)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / max(1, steps // 20)) * 0.5 * (1 + math.cos(math.pi * min(i, steps) / steps)))
    vocab = tok.vocab
    cap_ids = {v for v in range(PW.N_SPECIAL, PW.N_VOCAB) if vocab[v][:1].isupper()}
    hist = []
    best = None

    def consider(step):
        nonlocal best
        model.eval()
        ok, key, m = select(model)
        model.train()
        log(f"[arm{arm} select step={step}] eligible={ok} key={key} " + json.dumps(
            {k: m[k] for k in ("content_f1", "rouge_l", "rep3_pct", "grounding_precision_pct")}))
        if ok and (best is None or key < best[0]):
            best = (key, step, {k: v.detach().clone() for k, v in model.state_dict().items()}, m)

    if select is not None:
        consider(0)
    t0 = time.time()
    for step in range(steps):
        idx = rng.choice(tr.n, size=min(args.batch, tr.n), replace=False)
        batch = tr.batch(idx, dev)
        out = model(batch, bank if cfg["use_bank"] else None, cfg["qat"])
        loss = ce_loss(out["logits"], batch["tokens"], batch.get("kd_ids"), batch.get("kd_p"),
                       args.kd_weight if cfg["dpo"] else 0.0)
        logs = {"ce": loss.item()}
        if cfg["dpo"] and ref_model is not None and args.dpo_kl > 0:
            # KL(ref || policy) on teacher-forced positions: anchors the arm-2 distribution DPO starts from
            with torch.no_grad():
                ref_lp = F.log_softmax(ref_model(batch, bank, True)["logits"], -1)
            kl = F.kl_div(F.log_softmax(out["logits"], -1), ref_lp, log_target=True, reduction="none").sum(-1)
            valid = batch["tokens"] >= 0
            kl = (kl * valid).sum() / valid.sum().clamp(min=1)
            loss = loss + args.dpo_kl * kl
            logs["kl"] = round(kl.item(), 5)
        if cfg["use_planner"]:
            pl, cos = plan_losses(out["c_plan"], batch["c_clause"], args.sigreg_weight, gen)
            loss = loss + args.plan_weight * pl
            logs["plan_cos"] = [round(float(c), 3) for c in cos]
        if cfg["dpo"] and ref_model is not None and step % args.dpo_every == 0:
            model.eval()
            # sample under the device decoder's repetition window: DPO targets the loops that survive it
            samp = model.generate(batch, bank, qat=True, rep_mask=True,
                                  temperature=args.dpo_temperature, generator=tgen)
            _, ctoks, _, _, _ = model.retrieve(model.plan(batch["c_page"], batch["c_echo"], True), *bank)
            model.train()
            samp_l, ctoks_l = samp.tolist(), ctoks.tolist()
            rej_rows, reasons_n = [], {}
            loops = {}
            for i in range(len(idx)):
                j = idx[i]
                seq = samp_l[i][:samp_l[i].index(PW.PAD)] if PW.PAD in samp_l[i] else samp_l[i]
                words, own = rendered_words(seq, tr.subject[j], tr.role[j], tr.echo[j], ctoks_l[i], vocab, True)
                rs = reject_reasons(seq, bool(tr.subject[j]), cap_ids, tr.allowed[j], vocab, words, bool(tr.echo[j]))
                if rs:
                    if "rep3" in rs:
                        loops[len(rej_rows)] = loop_token(words, own)
                    rej_rows.append(i)
                    for r in rs:
                        reasons_n[r] = reasons_n.get(r, 0) + 1
            logs["dpo_rejects"] = reasons_n
            if rej_rows:
                sel = np.asarray(rej_rows)
                sub = tr.batch(idx[sel], dev)
                rej = samp[sel].clone()
                # pad after EOS with -1 so only the emitted prefix (+EOS) is scored
                after = torch.cumsum((rej == PW.PAD).long(), 1) - (rej == PW.PAD).long() > 0
                rej[after] = -1
                lp_c, _ = seq_logprob(model, sub, bank, sub["tokens"], True)
                lp_r, out_r = seq_logprob(model, sub, bank, rej, True)
                with torch.no_grad():
                    rf_c, _ = seq_logprob(ref_model, sub, bank, sub["tokens"], True)
                    rf_r, _ = seq_logprob(ref_model, sub, bank, rej, True)
                dpo = -F.logsigmoid(args.dpo_beta * ((lp_c - rf_c) - (lp_r - rf_r))).mean()
                loss = loss + args.dpo_weight * dpo
                logs["dpo"] = dpo.item()
                loops = {k: t for k, t in loops.items() if t is not None}
                if loops and args.loop_ul > 0:
                    # unlikelihood on the exact token that closes each surviving loop (DPO saturates once CE/KD
                    # move the chosen side far from the reference; this term keeps a direct gradient on loops)
                    ks = torch.as_tensor(list(loops), device=dev)
                    ts = torch.as_tensor(list(loops.values()), device=dev)
                    p = torch.softmax(out_r["logits"][ks, ts], -1).gather(1, rej[ks, ts].clamp(min=0)[:, None])[:, 0]
                    ul = -torch.log((1 - p).clamp(min=1e-6)).mean()
                    loss = loss + args.loop_ul * ul
                    logs["loop_ul"] = round(ul.item(), 4)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == steps - 1:
            logs.update(step=step, loss=loss.item(), sec=round(time.time() - t0, 1))
            hist.append(logs)
            log(f"[arm{arm}] {json.dumps(logs)}")
        if select is not None and ((step + 1) % args.select_every == 0 or step == steps - 1):
            consider(step + 1)
    if best is not None:
        model.load_state_dict(best[2])
        hist.append({"selected_step": best[1], "selected_key": list(best[0]),
                     "selected_dev": {k: best[3][k] for k in ("content_f1", "rouge_l", "rep3_pct",
                                                              "grounding_precision_pct", "ungrounded_name_pct")}})
        log(f"[arm{arm}] selected step {best[1]} key={best[0]}")
    return hist


# ----------------------------------------------------------------------------- evaluation (exported int8 path)
def lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def _words(s: str):
    return [w.lower() for w in PW.word_tokens(s) if any(ch.isalnum() for ch in w)]


def prf(h, r):
    if not h or not r:
        return 0.0
    from collections import Counter
    c = sum((Counter(h) & Counter(r)).values())
    if c == 0:
        return 0.0
    p, q = c / len(h), c / len(r)
    return 2 * p * q / (p + q)


def evaluate(arm, blob, wblob, va: Split, layout, dev, vocab) -> dict:
    cfg = ARMS[arm]
    ref = PW.PeWriterIntTorch(blob, wblob, device=dev)
    b = lambda xs: [x.encode("utf-8") for x in xs]  # noqa: E731
    res = []
    for i in range(0, va.n, 256):
        sl = slice(i, min(va.n, i + 256))
        res += ref.generate(va.c_page[sl], va.c_echo[sl], b(va.subject[sl]), b(va.role[sl]), b(va.echo[sl]),
                            use_planner=cfg["use_planner"], use_bank=cfg["use_bank"])
    bank = PW.Wcls(wblob)
    flat_vec = bank.vec.reshape(-1, PW.D)
    flat_valid = bank.valid.reshape(-1)
    cos = np.zeros(3)
    r1 = np.zeros(3)
    r5 = np.zeros(3)
    routed_hit = np.zeros(3)
    f1s, rls, rep3, weave, ungrounded, grounded = [], [], [], [], [], []
    cap_ids = {v for v in range(PW.N_SPECIAL, PW.N_VOCAB) if vocab[v][:1].isupper()}
    for i, r in enumerate(res):
        cp = r["c_plan"].astype(np.float64)
        for s in range(3):
            t = va.c_clause[i, s].astype(np.float64)
            cos[s] += float(cp[s] @ t / max(1e-9, np.linalg.norm(cp[s]) * np.linalg.norm(t)))
            tgt_scores = np.where(flat_valid, (flat_vec @ va.c_clause[i, s]).astype(np.float64), -np.inf)
            target = int(np.argmax(tgt_scores))
            q = np.where(flat_valid, (flat_vec @ r["c_plan"][s]).astype(np.float64), -np.inf)
            top5 = np.argsort(-q, kind="stable")[:5]
            routed = r["buckets"][s] * PW.N_SLOTS + bank_best(bank, r["buckets"][s], r["c_plan"][s])
            r1[s] += top5[0] == target
            r5[s] += target in top5
            routed_hit[s] += routed == target
        hyp = r["text"].decode("utf-8", "replace")
        hw, rw = _words(hyp), _words(va.riff[i])
        f1s.append(prf([w for w in hw if w not in STOPWORDS], [w for w in rw if w not in STOPWORDS]))
        L = lcs(hw, rw)
        rls.append(0.0 if not L else 2 * (L / len(hw)) * (L / len(rw)) / (L / len(hw) + L / len(rw)))
        grams = [tuple(hw[j:j + 3]) for j in range(len(hw) - 2)]
        rep3.append(len(grams) != len(set(grams)))
        toks = [v for v in r["tokens"] if v]
        w_ok = (not va.subject[i] or PW.ENT_SUBJECT in toks) and (not va.echo[i] or PW.BOOK_ECHO in toks)
        ug = False
        for j, v in enumerate(toks):
            if v in cap_ids and j > 0 and invented_name(vocab[v], vocab[toks[j - 1]], va.allowed[i]):
                ug = True
        weave.append(w_ok)
        ungrounded.append(ug)
        grounded.append(w_ok and not ug)
    n = max(1, len(res))
    samples = [{"subject": va.subject[i], "echo": va.echo[i], "riff": res[i]["text"].decode("utf-8", "replace"),
                "teacher": va.riff[i]} for i in range(min(6, len(res)))]
    return {
        "arm": arm, "name": cfg["name"], "n_val": len(res),
        "thought_traj_cos": [round(c / n, 4) for c in cos], "thought_traj_cos_mean": round(cos.sum() / (3 * n), 4),
        "clause_recall_at1": [round(c / n, 4) for c in r1], "clause_recall_at5": [round(c / n, 4) for c in r5],
        "clause_recall_at1_mean": round(r1.sum() / (3 * n), 4), "clause_recall_at5_mean": round(r5.sum() / (3 * n), 4),
        "clause_routed_hit_at1_mean": round(routed_hit.sum() / (3 * n), 4),
        "content_f1": round(float(np.mean(f1s)), 4), "rouge_l": round(float(np.mean(rls)), 4),
        "rep3_pct": round(100 * float(np.mean(rep3)), 2),
        "entity_echo_weave_pct": round(100 * float(np.mean(weave)), 2),
        "ungrounded_name_pct": round(100 * float(np.mean(ungrounded)), 2),
        "grounding_precision_pct": round(100 * float(np.mean(grounded)), 2),
        "mean_tokens": round(float(np.mean([sum(1 for v in r["tokens"] if v) for r in res])), 2),
        "samples": samples,
    }


def bank_best(bank: PW.Wcls, b: int, q) -> int:
    best, bj = None, 0
    for j in range(PW.N_SLOTS):
        if bank.valid[b, j]:
            d = int(bank.vec[b, j] @ q)
            if best is None or d > best:
                best, bj = d, j
    return bj


def export_blobs(model: PW.PeWriterQAT, tok: PW.Tokenizer, layout: dict):
    return PW.pack_pewr(model.export_tensors(), tok.vocab_bytes()), PW.pack_wcls(layout)


# ----------------------------------------------------------------------------- Arm 4: Frame-Locked PE-Writer v2
PLANNER_PARAMS = ("step_embed", "plan_fc1.weight", "plan_fc1.bias", "plan_fc2.weight", "plan_fc2.bias")
FRAME_T = 12  # longest v2 frame (10 tokens) + EOS, padded


def frame_mask_(model: PW.PeWriterQAT) -> None:
    """Hard frame lock: lm_head rows >= N_FRAME_VOCAB are W = 0, bias = FRAME_MASK_BIAS; copy gate closed."""
    with torch.no_grad():
        model.lm_head.weight[PW.N_FRAME_VOCAB:] = 0
        model.lm_head.bias[PW.N_FRAME_VOCAB:] = PW.FRAME_MASK_BIAS
        model.copy_gate_proj.weight.zero_()
        model.copy_gate_proj.bias.fill_(-64.0)


@torch.no_grad()
def bank_slots(model: PW.PeWriterQAT, c_plan, bank):
    """Device routing + in-bucket retrieval -> (buckets [B,3], slots [B,3])."""
    bv, _, bval = bank
    q = torch.round(c_plan)
    buckets = (q @ model.centroids.T).argmax(-1)
    slots = torch.zeros_like(buckets)
    for s in range(PW.N_STEPS):
        sc = (bv[buckets[:, s]] @ q[:, s].unsqueeze(-1)).squeeze(-1)
        sc = torch.where(bval[buckets[:, s]], sc, torch.full_like(sc, -1e18))
        slots[:, s] = sc.argmax(-1)
    return buckets, slots


def random_inputs(n: int, rng) -> np.ndarray:
    """Off-distribution concept vectors (unit / saturated / raw int8 / zero), as in the golden + fuzz cases."""
    out = np.zeros((n, PW.D), np.int64)
    for i, k in enumerate(rng.choice(4, n, p=[0.55, 0.15, 0.2, 0.1])):
        if k == 0:
            out[i] = PW.normalize_i8(rng.integers(-60, 61, PW.D))
        elif k == 1:
            out[i] = rng.choice([-127, 127], PW.D)
        elif k == 2:
            out[i] = rng.integers(-128, 128, PW.D)
    return out


@torch.no_grad()
def plan_retrieval_cos(model, sp: Split, bank, dev) -> float:
    """Mean cos(retrieved WCLS slot vector, teacher clause vector) over the 3 steps (device routing)."""
    cp = torch.as_tensor(sp.c_page.astype(np.float32), device=dev)
    ce = torch.as_tensor(sp.c_echo.astype(np.float32), device=dev)
    b, s = bank_slots(model, model.plan(cp, ce, True), bank)
    v = F.normalize(bank[0][b, s], dim=-1)
    c = F.normalize(torch.as_tensor(sp.c_clause.astype(np.float32), device=dev), dim=-1)
    return float((v * c).sum(-1).mean())


def train_arm4_planner(model, tr: Split, va: Split, layout: dict, src: np.ndarray, bank, args, dev, log) -> list[dict]:
    """InfoNCE (temp ``--nce-temp``) over the step's WCLS partition + in-batch InfoNCE against the teacher clause
    vectors + cosine alignment + bucket-routing CE; only step_embed / plan_fc1 / plan_fc2 train. The bank positive
    of a train row is its nearest slot *excluding the slot mined from that row* (no memorisation target). plan_fc2
    starts at 0 (plan = l2(c_prev + c_echo + step tag)) and the checkpoint with the best internal-val retrieval
    cosine (``plan_retrieval_cos``) is kept, step 0 included."""
    V = torch.as_tensor(layout["vec"].reshape(-1, PW.D).astype(np.float32), device=dev)
    Vn = F.normalize(V, dim=-1)
    cent = model.centroids
    part = [torch.arange(b0 * PW.N_SLOTS, (b0 + k) * PW.N_SLOTS, device=dev)
            for k, b0 in zip(PW.ROLE_BUCKETS, PW.ROLE_BUCKET_START)]
    C = torch.as_tensor(tr.c_clause.astype(np.float32), device=dev)
    srcT = torch.as_tensor(src, device=dev)
    rows = torch.arange(tr.n, device=dev)[:, None]
    pos = [(C[:, r] @ V[part[r]].T).masked_fill(srcT[part[r]][None, :] == rows, -1e18).argmax(-1)
           for r in range(PW.N_STEPS)]  # positive slot (partition-local), own-row slot excluded
    tgt_b = [PW.ROLE_BUCKET_START[r] + pos[r] // PW.N_SLOTS for r in range(PW.N_STEPS)]
    with torch.no_grad():
        model.step_embed.copy_(torch.as_tensor(args.tag_scale * layout["tags"], dtype=torch.float32, device=dev))
        model.plan_fc2.weight.zero_()
        model.plan_fc2.bias.zero_()
    best = (plan_retrieval_cos(model, va, bank, dev), -1, copy.deepcopy(model.state_dict()))
    log(f"[arm4 planner] init val_retrieval_cos={best[0]:.4f}")
    params = [p for n, p in model.named_parameters() if n in PLANNER_PARAMS]
    for n, p in model.named_parameters():
        p.requires_grad_(n in PLANNER_PARAMS)
    opt = torch.optim.Adam([{"params": [model.step_embed], "lr": args.lr_raw},
                            {"params": [p for p in params if p is not model.step_embed], "lr": args.lr}])
    steps = args.plan_steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 * (1 + math.cos(math.pi * min(i, steps) / steps)))
    rng = np.random.default_rng(args.seed + 4)
    cp_all = torch.as_tensor(tr.c_page.astype(np.float32), device=dev)
    ce_all = torch.as_tensor(tr.c_echo.astype(np.float32), device=dev)
    hist, t0 = [], time.time()
    for step in range(steps):
        idx = torch.as_tensor(rng.choice(tr.n, size=min(args.batch, tr.n), replace=False), device=dev)
        P = model.plan(cp_all[idx], ce_all[idx], True)
        Pn = F.normalize(P, dim=-1)
        ar = torch.arange(len(idx), device=dev)
        l_bank = l_batch = l_cos = l_route = 0.0
        for r in range(PW.N_STEPS):
            l_bank = l_bank + F.cross_entropy(Pn[:, r] @ Vn[part[r]].T / args.nce_temp, pos[r][idx])
            Cn = F.normalize(C[idx, r], dim=-1)
            l_batch = l_batch + F.cross_entropy(Pn[:, r] @ Cn.T / args.nce_temp, ar)
            l_cos = l_cos + (1 - (Pn[:, r] * Cn).sum(-1)).mean()
            l_route = l_route + F.cross_entropy(P[:, r] @ cent.T * (args.route_scale / 127.0 ** 2), tgt_b[r][idx])
        loss = (l_bank + l_batch + args.cos_weight * l_cos + args.route_weight * l_route) / PW.N_STEPS
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.clip)
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == steps - 1:
            with torch.no_grad():
                b = (torch.round(P) @ cent.T).argmax(-1)
                part_ok = torch.stack([(b[:, r] >= PW.ROLE_BUCKET_START[r]) &
                                       (b[:, r] < PW.ROLE_BUCKET_START[r] + PW.ROLE_BUCKETS[r])
                                       for r in range(PW.N_STEPS)], 1).float().mean().item()
                bucket_ok = torch.stack([b[:, r] == tgt_b[r][idx] for r in range(3)], 1).float().mean().item()
            vcos = plan_retrieval_cos(model, va, bank, dev)
            if vcos > best[0]:
                best = (vcos, step, copy.deepcopy(model.state_dict()))
            logs = {"step": step, "loss": round(loss.item(), 4), "nce_bank": round(l_bank.item() / 3, 4),
                    "nce_batch": round(l_batch.item() / 3, 4), "cos": round(1 - l_cos.item() / 3, 4),
                    "route_ce": round(l_route.item() / 3, 4), "partition_hit": round(part_ok, 4),
                    "bucket_hit": round(bucket_ok, 4), "val_retrieval_cos": round(vcos, 4),
                    "sec": round(time.time() - t0, 1)}
            hist.append(logs)
            log(f"[arm4 planner] {json.dumps(logs)}")
    model.load_state_dict(best[2])
    hist.append({"selected_step": best[1], "val_retrieval_cos": round(best[0], 4)})
    log(f"[arm4 planner] selected step {best[1]} (val_retrieval_cos={best[0]:.4f}; -1 = untrained init)")
    return hist


def train_arm4_decoder(model, tr: Split, layout: dict, bank, args, dev, log) -> list[dict]:
    """Teacher-forced CE + integer-logit margin hinge on the v2 frames; planner frozen. Subject / echo presence is
    randomised (the masked special's runner-up must be the frame's fallback) and ``--rand-frac`` of each batch uses
    off-distribution concept vectors so the frame holds for any input. The connector follows the retrieved
    observation clause (``PW.frame_relation``)."""
    conn = torch.as_tensor([[PW.FRAME_CONNECTOR[PW.frame_relation(t.decode("utf-8", "replace"))] for t in row]
                            for row in layout["text"]], device=dev)
    for n, p in model.named_parameters():
        p.requires_grad_(n not in PLANNER_PARAMS and not n.startswith("copy_gate_proj"))
    frame_mask_(model)
    raw = [model.tok_embed, model.pos_embed]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam([{"params": raw, "lr": args.lr_raw},
                            {"params": [p for p in params if all(p is not q for q in raw)], "lr": args.lr}])
    steps = args.dec_steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 * (1 + math.cos(math.pi * min(i, steps) / steps)))
    rng = np.random.default_rng(args.seed + 44)
    hist, t0 = [], time.time()
    for step in range(steps):
        B = min(args.batch, tr.n)
        idx = rng.choice(tr.n, size=B, replace=False)
        cp, ce = tr.c_page[idx].copy(), tr.c_echo[idx].copy()
        rnd = rng.random(B) < args.rand_frac
        if rnd.any():
            cp[rnd] = random_inputs(int(rnd.sum()), rng)
            ce[rnd] = random_inputs(int(rnd.sum()), rng)
        hs = rng.random(B) >= args.p_no_subject
        he = rng.random(B) >= args.p_no_echo
        cp_t = torch.as_tensor(cp.astype(np.float32), device=dev)
        ce_t = torch.as_tensor(ce.astype(np.float32), device=dev)
        with torch.no_grad():
            buckets, slots = bank_slots(model, model.plan(cp_t, ce_t, True), bank)
        rel = conn[buckets[:, 0], slots[:, 0]].tolist()
        toks = np.full((B, FRAME_T), -1, np.int64)
        inv = {v: k for k, v in PW.FRAME_CONNECTOR.items()}
        for i in range(B):
            t = PW.frame_tokens(bool(hs[i]), bool(he[i]), inv[rel[i]])
            toks[i, :len(t)] = t
        batch = {"c_page": cp_t, "c_echo": ce_t, "tokens": torch.as_tensor(toks, device=dev),
                 "has_subject": torch.as_tensor(hs, device=dev), "has_echo": torch.as_tensor(he, device=dev)}
        out = model(batch, bank, True)
        logits = out["logits"]
        tgt = batch["tokens"]
        valid = tgt >= 0
        l_ce = ce_loss(logits, tgt)
        z = logits * model.tau
        zt = z.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        other = z.scatter(-1, tgt.clamp(min=0).unsqueeze(-1), -1e9).max(-1).values
        margin = zt - other
        l_hinge = (F.relu(args.margin - margin) * valid).sum() / valid.sum() / args.margin
        loss = l_ce + args.hinge_weight * l_hinge
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.clip)
        opt.step()
        sched.step()
        frame_mask_(model)
        if step % args.log_every == 0 or step == steps - 1:
            with torch.no_grad():
                ok = ((z.argmax(-1) == tgt) | ~valid).all(-1).float().mean().item()
                mmin = margin[valid].min().item()
            logs = {"step": step, "loss": round(loss.item(), 4), "ce": round(l_ce.item(), 5),
                    "hinge": round(l_hinge.item(), 5), "tf_seq_acc": round(ok, 4), "min_margin": round(mmin, 1),
                    "sec": round(time.time() - t0, 1)}
            hist.append(logs)
            log(f"[arm4 decoder] {json.dumps(logs)}")
    return hist


def frame_audit(blob: bytes, wblob: bytes, va: Split, n_fuzz: int, seed: int, dev) -> dict:
    """Exported int8 forward: share of outputs that are exactly a v2 frame (val rows under all 4 subject/echo
    combinations + off-distribution fuzz vectors), plus routing / distinct-clause / byte-cap checks."""
    ref = PW.PeWriterIntTorch(blob, wblob, device=dev)
    rng = np.random.default_rng(seed)
    cps = [va.c_page, random_inputs(n_fuzz, rng)]
    ces = [va.c_echo, random_inputs(n_fuzz, rng)]
    tot = bad = part_bad = dup = capped = 0
    worst = []
    for cp, ce in zip(cps, ces):
        n = len(cp)
        for hs in (True, False):
            for he in (True, False):
                subj = [b"Bathsheba" if hs else b""] * n
                role = [b"a central character"] * n
                echo = [b"Far from the Madding Crowd" if he else b""] * n
                for i in range(0, n, 512):
                    sl = slice(i, i + 512)
                    for r in ref.generate(cp[sl], ce[sl], subj[sl], role[sl], echo[sl]):
                        tot += 1
                        if not PW.frame_valid(r["tokens"], hs, he):
                            bad += 1
                            if len(worst) < 5:
                                worst.append(r["tokens"][:12])
                        part_bad += any(PW.bucket_partition(b) != s for s, b in enumerate(r["buckets"]))
                        dup += r["clause_text"][0] == r["clause_text"][1]
                        capped += len(r["text"]) >= PW.TEXT_CAP - 1
    return {"cases": tot, "frame_valid_pct": round(100 * (1 - bad / tot), 3), "invalid_examples": worst,
            "off_partition_route_pct": round(100 * part_bad / tot, 3), "clause0_eq_clause1_pct": round(100 * dup / tot, 3),
            "hit_239B_cap_pct": round(100 * capped / tot, 3)}


def run_arm4(args, d: dict, run: Path, log, dev) -> None:
    vocab = PW.frame_vocab()
    tok = PW.Tokenizer(vocab)
    z = np.load(args.bank_v2)
    texts = [str(x) for x in z["bank_text"]]
    layout = PW.build_partitioned_bank_layout(z["bank_vec"], texts, z["bank_step_role"], seed=args.seed,
                                              tag_gain=args.tag_gain)
    tr = Split(d, "train", tok)
    va = Split(d, "val", tok, args.max_val)
    log(f"[arm4] train={tr.n} val={va.n} bank={args.bank_v2} slots={len(texts)} frame_vocab={PW.N_FRAME_VOCAB}")
    json.dump(vocab, open(run / "vocab.json", "w"), ensure_ascii=False)
    wblob = PW.pack_wcls(layout)
    (run / "writer-clauses.bin").write_bytes(wblob)
    np.savez(run / "val_meta.npz", c_page=va.c_page.astype(np.int8), c_echo=va.c_echo.astype(np.int8),
             subject=np.array(va.subject), role=np.array(va.role), echo_title=np.array(va.echo))
    bank = bank_tensors(layout, dev)
    model = make_model(4, layout, dev)
    model.train()
    t2i = {t: i for i, t in enumerate(texts)}
    src = np.array([int(z["bank_src_idx"][t2i[t.decode("utf-8") if isinstance(t, bytes) else str(t)]]) if t else -1
                    for row in layout["text"] for t in row], np.int64)
    hist = train_arm4_planner(model, tr, va, layout, src, bank, args, dev, log)
    hist += train_arm4_decoder(model, tr, layout, bank, args, dev, log)
    model.eval()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save({"arm": 4, "cfg": ARMS[4], "state": state, "args": vars(args) | {
        "data": str(args.data), "run": str(run), "bank_v2": str(args.bank_v2)}}, run / "arm4.pt")
    blob, wblob2 = export_blobs(model, tok, layout)
    assert wblob2 == wblob
    m = evaluate(4, blob, wblob, va, layout, dev, vocab)
    m["frame_audit"] = frame_audit(blob, wblob, va, args.fuzz_audit, args.seed + 99, dev)
    m["train_hist_tail"] = hist[-3:]
    log(f"[eval arm4] " + json.dumps({k: v for k, v in m.items() if k not in ("samples", "train_hist_tail")}))
    for s in m["samples"][:6]:
        log(f"   riff: {s['riff']!r}\n   teacher: {s['teacher']!r}")
    json.dump({"source": str(args.data), "args": vars(args) | {"data": str(args.data), "run": str(run),
                                                                "bank_v2": str(args.bank_v2)},
               "metrics": {"4": m}}, open(run / "metrics.json", "w"), indent=1, ensure_ascii=False, default=str)
    log(f"wrote {run}")


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="synthetic data, CPU, small")
    ap.add_argument("--all-arms", action="store_true", help="train + evaluate arms 0..3 (default: same)")
    ap.add_argument("--arms", default="0,1,2,3")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--run", "--out-dir", dest="run", type=Path, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=4000, help="steps for arms 0-2")
    ap.add_argument("--dpo-steps", type=int, default=None, help="extra QAT+DPO steps for arm 3 (default steps/2)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr-raw", type=float, default=0.3)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--plan-weight", type=float, default=2.0)
    ap.add_argument("--sigreg-weight", type=float, default=0.002)
    ap.add_argument("--dpo-weight", type=float, default=0.5)
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--dpo-every", type=int, default=2)
    ap.add_argument("--dpo-temperature", type=float, default=1.0)
    ap.add_argument("--kd-weight", type=float, default=0.5)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpo-lr-scale", type=float, default=0.2, help="arm-3 lr multiplier (DPO step size)")
    ap.add_argument("--dpo-kl", type=float, default=0.1, help="arm-3 KL(ref||policy) anchor weight")
    ap.add_argument("--select-every", type=int, default=100, help="arm-3 dev checkpoint-selection interval")
    ap.add_argument("--dev-n", type=int, default=384, help="train rows scored by the arm-3 selector")
    ap.add_argument("--loop-ul", type=float, default=1.0, help="arm-3 unlikelihood weight on loop-closing tokens")
    ap.add_argument("--bank-v2", type=Path, default=BANK_V2, help="arm-4 proper-noun-free clause bank")
    ap.add_argument("--plan-steps", type=int, default=1500, help="arm-4 InfoNCE planner steps")
    ap.add_argument("--dec-steps", type=int, default=1500, help="arm-4 frame decoder steps")
    ap.add_argument("--nce-temp", type=float, default=0.07, help="arm-4 InfoNCE temperature")
    ap.add_argument("--cos-weight", type=float, default=1.0, help="arm-4 cosine alignment weight")
    ap.add_argument("--route-weight", type=float, default=1.0, help="arm-4 bucket routing CE weight")
    ap.add_argument("--route-scale", type=float, default=20.0, help="arm-4 routing logit scale (unit cosine)")
    ap.add_argument("--tag-gain", type=float, default=1.0, help="arm-4 step-tag share of each centroid")
    ap.add_argument("--tag-scale", type=float, default=200.0, help="arm-4 step_embed = scale * step tag axis")
    ap.add_argument("--margin", type=float, default=600.0, help="arm-4 integer logit margin target")
    ap.add_argument("--hinge-weight", type=float, default=1.0)
    ap.add_argument("--rand-frac", type=float, default=0.25, help="arm-4 off-distribution input share")
    ap.add_argument("--p-no-subject", type=float, default=0.2)
    ap.add_argument("--p-no-echo", type=float, default=0.35)
    ap.add_argument("--fuzz-audit", type=int, default=2000, help="arm-4 random inputs in the frame audit")
    args = ap.parse_args(argv)
    if args.all_arms:
        args.arms = "0,1,2,3"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.dpo_steps is None:
        args.dpo_steps = max(1, args.steps // 2)
    if args.run is not None and not args.run.is_absolute():
        args.run = ROOT / args.run
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() and not args.smoke else "cpu"))
    run = args.run or (RUNS / ("smoke" if args.smoke else time.strftime("arms-%Y%m%d-%H%M%S")))
    run.mkdir(parents=True, exist_ok=True)
    logf = open(run / "train.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    if args.smoke:
        d = synthetic_dataset(args.seed)
        source = "synthetic"
    else:
        if not args.data.exists():
            raise SystemExit(f"missing {args.data}")
        d = load_npz(args.data)
        source = str(args.data)
    log(f"data={source} device={dev} run={run}")
    if "4" in args.arms.split(","):
        if args.arms.strip() != "4":
            raise SystemExit("arm 4 has its own frame vocab + WCLS v2 bank: train it alone (--arms 4)")
        return run_arm4(args, d, run, log, dev)

    # vocab from train templates + bank clauses; tokenizer; WCLS layout
    n_tr = len(d["train_c_page"])
    if "train_riff_template" in d:
        templates = [str(x) for x in d["train_riff_template"]]
    else:
        templates = [PW.riff_to_template(str(d["train_riff"][i]), str(d["train_subject"][i]), str(d["train_role"][i]),
                                         str(d["train_echo_title"][i]), [str(c) for c in d["train_clause_text"][i]])
                     for i in range(n_tr)]
    bank_text = [str(x) for x in d["bank_text"]]
    vocab = PW.build_vocab(templates, bank_text)
    tok = PW.Tokenizer(vocab)
    layout = PW.build_bank_layout(d["bank_vec"], bank_text, d.get("bank_step_role", np.zeros(len(bank_text))), tok,
                                  seed=args.seed)
    n_bank = sum(len(t) > 0 for row in layout["text"] for t in row)
    tr = Split(d, "train", tok)
    va = Split(d, "val", tok, args.max_val)
    cov = float(np.mean([(tr.tokens[i] >= 0).sum() for i in range(tr.n)]))
    log(f"train={tr.n} val={va.n} vocab_words={sum(1 for w in vocab[8:] if w)} bank_slots={n_bank} "
        f"mean_target_len={cov:.1f}")
    json.dump(vocab, open(run / "vocab.json", "w"), ensure_ascii=False)
    (run / "writer-clauses.bin").write_bytes(PW.pack_wcls(layout))
    np.savez(run / "val_meta.npz", c_page=va.c_page.astype(np.int8), c_echo=va.c_echo.astype(np.int8),
             subject=np.array(va.subject), role=np.array(va.role), echo_title=np.array(va.echo))
    bank = bank_tensors(layout, dev)

    arms = [int(a) for a in args.arms.split(",")]
    metrics, states = {}, {}
    for arm in arms:
        init = None
        ref_model = None
        if arm == 3:
            if 2 not in states:
                raise SystemExit("arm 3 initializes from arm 2; include arm 2 in --arms")
            init = states[2]
        model = make_model(arm, layout, dev, init)
        select = None
        if arm == 3:
            ref_model = copy.deepcopy(model).eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)
            dev_split = Split(d, "train", tok, args.dev_n)
            floor = {}

            def select(m_, arm=arm):
                dm = evaluate(arm, *export_blobs(m_, tok, layout), dev_split, layout, dev, vocab)
                if not floor:  # step 0 = arm-2 weights: grounding / F1 / ROUGE-L floor
                    floor.update({k: dm[k] for k in ("grounding_precision_pct", "content_f1", "rouge_l")})
                ok = all(dm[k] >= v for k, v in floor.items())
                return ok, (dm["rep3_pct"], -dm["grounding_precision_pct"], -(dm["content_f1"] + dm["rouge_l"])), dm
        model.train()
        hist = train_arm(arm, model, tr, bank, args, dev, ref_model=ref_model, tok=tok, log=log, select=select)
        model.eval()
        states[arm] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        torch.save({"arm": arm, "cfg": ARMS[arm], "state": states[arm], "args": vars(args) | {
            "data": str(args.data), "run": str(run)}}, run / f"arm{arm}.pt")
        blob, wblob = export_blobs(model, tok, layout)
        m = evaluate(arm, blob, wblob, va, layout, dev, vocab)
        m["train_hist_tail"] = hist[-3:]
        metrics[arm] = m
        log(f"[eval arm{arm}] " + json.dumps({k: v for k, v in m.items() if k not in ("samples", "train_hist_tail")}))
        for s in m["samples"][:3]:
            log(f"   riff: {s['riff']!r}\n   teacher: {s['teacher']!r}")
    json.dump({"source": source, "args": vars(args) | {"data": str(args.data), "run": str(run)},
               "metrics": {str(k): v for k, v in metrics.items()}}, open(run / "metrics.json", "w"),
              indent=1, ensure_ascii=False, default=str)
    log("\n| arm | traj cos | R@1 | R@5 | content F1 | ROUGE-L | Rep-3 % | weave % | ungrounded % | grounded % |")
    log("|---|---|---|---|---|---|---|---|---|---|")
    for arm, m in metrics.items():
        log(f"| {arm} {m['name']} | {m['thought_traj_cos_mean']} | {m['clause_recall_at1_mean']} | "
            f"{m['clause_recall_at5_mean']} | {m['content_f1']} | {m['rouge_l']} | {m['rep3_pct']} | "
            f"{m['entity_echo_weave_pct']} | {m['ungrounded_name_pct']} | {m['grounding_precision_pct']} |")
    log(f"wrote {run}")


if __name__ == "__main__":
    main()
