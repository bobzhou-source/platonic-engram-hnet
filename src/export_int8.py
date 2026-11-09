"""Export a CS-PW v2 checkpoint (train_cspw.py ``arm4_full_cspw.pt``; a v1 ``arm3_platonic_engram_full.pt``
still exports the three PLHN-side files) to the on-device binaries.

    python src/export_int8.py --ckpt runs/<run>/arm4_full_cspw.pt \
        --models-dir export --golden-dir golden

Writes (atomically, temp + rename):
  platonic-hnet.bin     PLHN v1 header + int8 tensors (95,008 B)
  engram.bin            ENGR v1, 512-B header sector + 65,536 x 64-B slots (learned 24-d deltas,
                        span hashes: glossed words, then Bigram-Span hashes, then the dominant router
                        span; <=31-B UTF-8 glosses for corpus compounds/idioms)
  inbox-concepts.idx    ICPT v1 records for --books-dir (all) + --novels-dir (sample)
  cspw.bin              CSPW v2 (393,728 B): Chunk-GQA + Chrono-CfC + frame/phrase banks and strings
  story-graph.idx       SGRF v1 from cspw_timelines.jsonl / cspw_aliases.jsonl (event vectors = the
                        exported PLHN int8 scene vectors)
and goldens for the firmware unit tests: golden.jsonl (bigram on), golden_vectors.bin (PLGV),
cspw_golden.bin (CSGV) + copies of all five model files.
Every export is checked: torch QAT integer path == numpy integer reference == pure-Python reference
(bigram Engram on, as on the device) and torch CS-PW QAT == ``int8_cspw_weave``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html.parser
import json
import os
import random
import re
import struct
import sys
import time
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DICT_DIR = ROOT / "data" / "dictionaries"  # CC-CEDICT + Open English WordNet StarDict files

GOLDEN_TEXTS = [
    "",
    "a",
    "The bank raised interest rates again.",
    "We sat on the river bank and watched the boats.",
    "他在银行工作，每天很早出门。",
    "我们坐在河岸上看船。",
    "画蛇添足的做法只会适得其反。",
    "Mixed 中英 text: kick the bucket / 一石二鸟 / unhappiness.",
    "Emoji 🙂 and accents: café naïve façade — “quotes”.",
    "x" * 300,
    "长" * 120,
    "In the beginning was the Word, and the Word was with God, and the Word was God. " * 5,
]


# ----------------------------------------------------------------------------------------------
# Atomic I/O + packing
# ----------------------------------------------------------------------------------------------


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def quantize_group(arrays, scale: float):
    return [np.clip(np.floor(a / scale + 0.5), -127, 127).astype(np.int8) for a in arrays]


def quantize_model(net) -> tuple[dict, dict]:
    """Same power-of-two group scales and rounding as the QAT fake-quant."""
    scales = net.group_scales()
    params = {
        "byte_embed": net.byte_embed, "conv1_dw": net.conv1.dw, "conv1_pw": net.conv1.pw,
        "conv1_gate": net.conv1.gate, "conv2_dw": net.conv2.dw, "conv2_pw": net.conv2.pw,
        "conv2_gate": net.conv2.gate, "router_q": net.router_q, "router_k": net.router_k,
        "chunk_proj": net.chunk_proj, "core_conv_dw": net.core.dw, "core_conv_pw": net.core.pw,
        "core_gate": net.core.gate, "core_out_proj": net.core.out,
    }
    tensors = {}
    for name, shape, group in M.PLHN_LAYOUT:
        w = params[name].detach().cpu().double().numpy()
        assert w.shape == shape, (name, w.shape, shape)
        tensors[name] = quantize_group([w], scales[group])[0]
    return tensors, scales


def pack_plhn(tensors: dict, scales: dict) -> bytes:
    header = M.PLHN_HEADER.pack(M.PLHN_MAGIC, M.PLHN_VERSION, M.D_BYTE, M.D_CONCEPT, M.N_PRELUDE,
                                M.N_CORE, M.ROUTER_DIM, M.ENGRAM_LOG2, scales["embed"], scales["router"],
                                scales["proj"], scales["core"], 0)
    body = b"".join(np.ascontiguousarray(tensors[name], dtype=np.int8).tobytes() for name, _, _ in M.PLHN_LAYOUT)
    blob = header + body
    assert len(blob) == M.PLHN_FILE_BYTES
    return blob


def pack_engram(delta: np.ndarray, scale: float, hashes: np.ndarray, glosses: dict) -> bytes:
    n = M.ENGRAM_SLOTS
    assert delta.shape == (n, M.ENGRAM_DIMS) and delta.dtype == np.int8
    head = M.ENGR_HEADER.pack(M.ENGR_MAGIC, M.ENGR_VERSION, n, 64, scale)
    head += b"\0" * (M.ENGR_SECTOR - len(head))
    slots = np.zeros((n, 64), dtype=np.uint8)
    slots[:, 0:8] = hashes.astype("<u8").view(np.uint8).reshape(n, 8)
    slots[:, 8:32] = delta.view(np.uint8)
    for slot, gloss in glosses.items():
        g = gloss[:M.GLOSS_BYTES - 1]
        slots[slot, 32:32 + len(g)] = np.frombuffer(g, dtype=np.uint8)
    blob = head + slots.tobytes()
    assert len(blob) == M.ENGR_FILE_BYTES
    return blob


def pack_icpt(records) -> bytes:
    out = [M.ICPT_HEADER.pack(M.ICPT_MAGIC, M.ICPT_VERSION, len(records), M.D_CONCEPT)]
    for path_hash, chunk, stamp, label, vec in records:
        out.append(M.ICPT_RECORD.pack(path_hash, chunk, stamp, label, *vec))
    return b"".join(out)


def utf8_trunc(s: str, max_bytes: int) -> bytes:
    b = s.encode("utf-8")
    if len(b) <= max_bytes:
        return b
    b = b[:max_bytes]
    while b and M.is_cont(b[-1]):
        b = b[:-1]
    if b and b[-1] >= 0xC0:
        b = b[:-1]
    return b


def gloss_trunc(text: str, max_bytes: int = M.GLOSS_BYTES - 1) -> bytes:
    """UTF-8-safe truncation that prefers ending on a word boundary."""
    full = text.strip().encode("utf-8")
    b = utf8_trunc(text.strip(), max_bytes)
    if len(b) < len(full):
        cut = b.rfind(b" ")
        if cut >= 12:
            b = b[:cut]
        b = b.rstrip(b" ,;:(\"'")
    return b


# ----------------------------------------------------------------------------------------------
# Dictionaries + gloss selection
# ----------------------------------------------------------------------------------------------


def load_stardict(stem: Path) -> dict:
    idx = stem.with_suffix(".idx").read_bytes()
    data = stem.with_suffix(".dict").read_bytes()
    out, pos = {}, 0
    while pos < len(idx):
        end = idx.index(b"\0", pos)
        word = idx[pos:end].decode("utf-8", "replace")
        off, size = struct.unpack(">II", idx[end + 1:end + 9])
        pos = end + 9
        text = data[off:off + size].decode("utf-8", "replace")
        out[word] = out[word] + "\n" + text if word in out else text  # homograph entries merged
    return out


def first_sense_en(defn: str) -> str:
    for line in defn.splitlines():
        m = re.match(r"\s*1\.\s*(.+)", line)
        if m:
            return re.sub(r"\s*\[=.*?\]", "", m.group(1)).split(";")[0].strip()
    lines = [l for l in defn.splitlines() if l.strip() and not l.startswith("(")]
    return lines[0].strip() if lines else ""


ZH_SKIP = re.compile(r"^(variant of|old variant|see |used in|surname|CL:|abbr\. for|also written|erhua variant)", re.I)
ZH_FUNCTION = set("的了是一在有着个我你他她它们这那不就也都还又和与而之其被把给让从向对说")
ZH_PARTICLES = set("的了着过地得们吗呢吧啊么之")


def zh_senses(defn: str) -> list:
    """CEDICT StarDict entry(ies) = alternating 'pinyin' / 'sense; sense' lines."""
    lines = [l for l in defn.splitlines() if l.strip()]
    out = []
    for body in lines[1::2] if len(lines) > 1 else lines:
        body = re.sub(r"[^\s;|\[\]]+\|", "", body)  # 繁|简 references -> simplified only
        out += [s.strip() for s in body.split(";") if s.strip()]
    return out


def first_sense_zh(defn: str) -> str:
    senses = zh_senses(defn)
    pick = next((s for s in senses if not ZH_SKIP.match(s)), senses[0] if senses else "")
    sense = re.sub(r"\(.*?\)\s*|\[.*?\]\s*", "", pick).strip()
    return sense or pick


def load_syn_words(path: Path) -> set:
    raw = path.read_bytes() if path.exists() else b""
    out, pos = set(), 0
    while pos < len(raw):
        end = raw.index(b"\0", pos)
        out.add(raw[pos:end].decode("utf-8", "replace").lower())
        pos = end + 5
    return out


CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
STRIP = " \t\n\r.,;:!?\"'()[]{}<>“”‘’«»—–-…·、，。；：！？（）《》【】「」『』"
EN_SUFFIXES = [("iness", "y"), ("ness", ""), ("ingly", ""), ("ing", ""), ("ing", "e"), ("ied", "y"), ("ies", "y"),
               ("ed", ""), ("ed", "e"), ("es", ""), ("s", ""), ("ly", ""), ("ily", "y"), ("er", ""), ("est", ""),
               ("ful", ""), ("less", ""), ("ment", ""), ("able", ""), ("ish", "")]
EN_SKIP = {"http", "https", "www", "html", "com", "org", "pdf", "epub"}

# Web-novel / cultivation / fan-fiction vocabulary that CEDICT lacks or glosses with an unrelated
# everyday sense (斗气 "to have a grudge" vs. Doupo "battle qi"). Always exported, top priority.
NOVEL_TERMS = {
    "炼气": "Qi Refining (cultivation stage)", "练气": "Qi Refining (cultivation stage)",
    "筑基": "Foundation Establishment (stage)", "结丹": "Core Formation (stage)",
    "金丹": "Golden Core (cultivation stage)", "元婴": "Nascent Soul (cultivation stage)",
    "化神": "Spirit Transformation (stage)", "炼虚": "Void Refinement (stage)",
    "渡劫": "to cross a heavenly tribulation", "天劫": "heavenly tribulation", "雷劫": "lightning tribulation",
    "飞升": "to ascend to immortality", "修士": "cultivator", "修仙": "to cultivate immortality",
    "修为": "cultivation level", "灵根": "spiritual root (talent)", "灵石": "spirit stone (currency)",
    "灵力": "spiritual power", "灵识": "spiritual sense", "神识": "divine sense (perception)",
    "真元": "true essence (inner energy)", "元神": "primordial spirit", "神魂": "divine soul",
    "法宝": "magic treasure (artifact)", "法器": "magic artifact", "飞剑": "flying sword",
    "丹药": "elixir pill", "丹炉": "alchemy furnace", "炼丹": "to refine elixir pills",
    "炼器": "to forge magic artifacts", "阵法": "formation (magic array)", "阵盘": "formation disk",
    "储物袋": "storage pouch", "储物戒指": "storage ring", "纳戒": "storage ring", "宗门": "sect",
    "宗主": "sect master", "师尊": "master (honorific)", "道友": "fellow Daoist", "妖兽": "demonic beast",
    "魔兽": "magical beast", "魔修": "demonic cultivator", "功法": "cultivation technique",
    "道心": "Dao heart (resolve)", "气运": "fortune; destiny (luck)",
    "斗气": "battle qi (Doupo)", "斗之气": "Dou Qi (first rank)", "斗者": "Dou Zhe (Fighter rank)",
    "斗师": "Dou Shi (Fight Master rank)", "斗灵": "Dou Ling (Fight Spirit rank)",
    "斗王": "Dou Wang (Fight King rank)", "斗皇": "Dou Huang (Fight Emperor rank)",
    "斗宗": "Dou Zong (Fight Ancestor rank)", "斗尊": "Dou Zun (Fight Venerate rank)",
    "斗圣": "Dou Sheng (Fight Saint rank)", "斗帝": "Dou Di (Fight Emperor)", "异火": "Heavenly Flame",
    "萧炎": "Xiao Yan (Doupo hero)", "魂力": "soul power (Douluo)", "魂环": "soul ring (Douluo)",
    "魂骨": "soul bone (Douluo)", "魂师": "soul master (Douluo)", "武魂": "martial soul (Douluo)",
    "魂兽": "soul beast (Douluo)", "斗罗": "Douluo (title rank)", "封号斗罗": "Titled Douluo",
    "唐三": "Tang San (Douluo hero)", "写轮眼": "Sharingan (Naruto)", "轮回眼": "Rinnegan (Naruto)",
    "万花筒": "Mangekyo (Naruto)", "宇智波": "Uchiha (Naruto clan)", "千手": "Senju (Naruto clan)",
    "宇智波斑": "Uchiha Madara (Naruto)", "宇智波鼬": "Uchiha Itachi (Naruto)",
    "宇智波佐助": "Uchiha Sasuke (Naruto)", "千手柱间": "Senju Hashirama (Naruto)",
    "日向": "Hyuga (Naruto clan)", "木叶": "Konoha (Hidden Leaf)", "火影": "Hokage (Naruto)",
    "忍术": "ninjutsu", "尾兽": "tailed beast (Naruto)", "九尾": "Nine-Tails (fox)",
    "鸣人": "Naruto (Uzumaki)", "佐助": "Sasuke (Naruto)", "卡卡西": "Kakashi (Naruto)",
    "恶魔果实": "Devil Fruit (One Piece)", "漫威": "Marvel (comics)", "复仇者": "Avenger(s)",
    "神盾局": "S.H.I.E.L.D.", "主神": "Main God (Infinite Flow)", "主神空间": "Main God Space",
    "轮回者": "reincarnator (Infinite Flow)", "轮回小队": "reincarnation team", "诸天": "the myriad worlds",
    "诸天万界": "the myriad worlds", "位面": "plane (dimension)", "副本": "dungeon instance (game)",
    "金手指": "golden finger (cheat ability)", "穿越者": "transmigrator", "邪神": "evil god",
    "古神": "Old God (Lovecraftian)", "魔神": "demon god", "信仰之力": "power of faith",
    "神格": "divinity (god-spark)", "神国": "divine kingdom", "魅魔": "succubus", "异兽": "strange beast",
    "丧尸": "zombie", "末世": "the apocalypse", "废土": "wasteland (post-apocalypse)",
    "异能": "superpower", "异能者": "esper (ability user)", "超凡": "transcendent (extraordinary)",
    "超凡者": "Beyonder (transcendent)", "序列": "Sequence (Beyonder rank)",
}
# Speech-tag endings in narrative prose (笑道 "said with a smile"); longest match wins.
SPEECH_TAGS = sorted({
    "道": "said", "说道": "said", "问道": "asked", "答道": "answered", "喝道": "shouted",
    "喊道": "shouted", "叫道": "cried out", "骂道": "cursed", "叹道": "sighed", "怒道": "said angrily",
    "笑道": "said with a smile", "冷笑道": "sneered", "沉声道": "said in a deep voice",
    "低声道": "said in a low voice", "轻声道": "said softly", "大声道": "said loudly",
    "淡淡道": "said calmly", "开口道": "spoke up", "解释道": "explained", "回答道": "replied",
    "感叹道": "sighed", "嘀咕道": "muttered", "冷声道": "said coldly", "苦笑道": "said with a wry smile",
}.items(), key=lambda kv: -len(kv[0]))
COMPOUND_SURNAMES = set("欧阳 司马 上官 诸葛 宇文 慕容 东方 南宫 公孙 令狐 皇甫 尉迟 长孙 独孤 夏侯 端木 西门 轩辕 司徒 呼延".split())
# characters that dominate phonetic transliterations of foreign names (卡卡西, 宇智波, 克里斯蒂娜)
TRANSLIT = set("阿埃艾安奥巴拜班邦贝本比彼波伯布查达戴丹德迪蒂杜多厄恩法菲芬弗福盖冈戈格贡古哈海汉豪赫亨"
               "胡霍基吉加贾杰金卡凯坎康柯科克肯库拉莱兰朗劳勒雷里利莉列林琳卢鲁路伦罗洛马玛迈麦曼梅蒙米敏莫"
               "姆穆纳娜奈内尼宁纽诺欧帕佩彭皮普奇齐恰乔切琴丘萨塞桑瑟森沙莎尚舍什斯松苏索塔泰坦汤唐特滕提图"
               "托瓦万威韦维温沃乌西希夏谢辛休雅亚扬耶伊因尤约泽扎詹哲佐兹宇智")
# never part of a person's name span (particles, pronouns, copula/speech verbs, prepositions)
NAME_STOP = set("的了着过地得们吗呢吧啊么之我你他她它是在有这那就也都还又和与说把被给让从向对道")
NOVEL_NAMES = {"萧炎", "唐三", "鸣人", "佐助", "卡卡西"}
# fictional clan names used as surnames (宇智波斑 -> Uchiha Ban)
NOVEL_SURNAMES = {"宇智波": "Uchiha", "千手": "Senju", "日向": "Hyuga", "漩涡": "Uzumaki"}
# common single-character surnames (person-name detection is limited to these)
COMMON_SURNAMES = set(
    "王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈"
    "姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向"
    "汤常温康施文牛樊葛邢安齐易乔伍庞颜倪庄聂章鲁岳翟殷詹申欧耿关兰焦俞左柳甘祝包宁尚符舒阮柯纪梅童凌毕"
    "单季裴霍涂成苗谷盛曲翁冉骆蓝路游辛靳管柴蒙鲍华喻祁蒲房滕屈饶解牟艾尤阳穆农司卓古吉缪简车项连麦褚娄"
    "窦戚岑景党宫费卜冷晏席卫米柏宗瞿桂佟臧闵苟邬边卞姬师仇栾隋商刁沙荣巫寇桑郎甄丛仲虞敖巩佘池查麻苑迟"
    "邝萧楚云")
# family / clan suffixes: 慕容世家 -> the Murong family
CLAN_SUFFIXES = {"世家": "family", "家族": "clan", "一族": "clan", "山庄": "manor", "家": "family"}
DEVICE_PUNCT = {0x2019: "'", 0x2018: "'", 0x2010: "-", 0x2011: "-"}

# characters / words that make two-word compositions unreliable (question words, modals, pronouns)
COMPOSE_STOP = ZH_FUNCTION | ZH_PARTICLES | set("么怎什哪谁可能会要想该应")
COMPOSE_STOP_WORDS = {"自己", "已经", "完全", "一个", "这个", "那个", "没有", "可以", "就是", "还是", "不是",
                      "因为", "所以", "但是", "如果", "虽然", "时候", "起来", "出来", "下来", "下次", "本次"}


def strip_tones(p: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", p) if not unicodedata.combining(c))


class GlossBook:
    def __init__(self):
        zh_raw = load_stardict(DICT_DIR / "cedict" / "cedict")
        self.zh = {w: first_sense_zh(d) for w, d in zh_raw.items()}
        self.zh_surname = {}
        self.zh_pinyin = {}  # common (lower-case) reading, tone marks stripped
        for w, d in zh_raw.items():
            lines = [l.strip() for l in d.splitlines() if l.strip()]
            if len(w) == 1:
                # reading for romanisation: lower-case (non-surname) reading with the most senses
                pairs = [(lines[i], lines[i + 1]) for i in range(0, len(lines) - 1, 2) if not CJK.search(lines[i])]
                common = [p for p in pairs if p[0][:1].islower() and not ZH_SKIP.match(p[1])] or pairs
                if common:
                    best = max(common, key=lambda p: len(p[1].split(";")))[0]
                    self.zh_pinyin[w] = strip_tones(best).replace(" ", "").lower()
            sur = next((s for s in zh_senses(d) if re.match(r"(two-character )?surname ", s, re.I)), None)
            if sur and len(w) <= 2:
                self.zh_surname[w] = sur.split("surname ", 1)[1].split()[0].strip(",;")
        self.en = {w.lower(): first_sense_en(d) for w, d in load_stardict(DICT_DIR / "english" / "english").items()}
        # words the device's StarDict lookup already resolves (headwords + inflection synonyms)
        self.en_known = set(self.en) | load_syn_words(DICT_DIR / "english" / "english.syn")
        self.zh_max = 8

    def _zh_segment(self, word: str):
        """Fewest dictionary pieces; ties prefer standalone function characters (不+知道 over 不知+道)."""
        n = len(word)
        best = [None] * (n + 1)
        best[0] = (0, 0, [])
        for i in range(n):
            if best[i] is None:
                continue
            for L in range(1, min(self.zh_max, n - i) + 1):
                piece = word[i:i + L]
                if piece not in self.zh:
                    continue
                cnt, fn, parts = best[i]
                cand = (cnt + 1, fn - (L == 1 and piece in ZH_FUNCTION), parts + [piece])
                if best[i + L] is None or cand[:2] < best[i + L][:2]:
                    best[i + L] = cand
        return best[n][2] if best[n] else None

    def _short(self, w: str) -> str | None:
        """Compact sense for composition: novel term or CEDICT first sense, no 'to ', no parens."""
        g = NOVEL_TERMS.get(w) or self.zh.get(w)
        if not g or "surname" in g.lower():
            return None
        g = re.sub(r"\(.*?\)", "", g).split(",")[0].strip()
        return g[3:] if g.startswith("to ") else g

    def _romanize(self, chars: str) -> str | None:
        syl = [self.zh_pinyin.get(c) for c in chars]
        return "".join(syl).capitalize() if all(syl) else None

    def _surname(self, word: str):
        """-> (romanised surname, rest) for novel clans, compound surnames, common single surnames."""
        for clan, rom in NOVEL_SURNAMES.items():
            if word.startswith(clan):
                return rom, word[len(clan):]
        if word[:2] in COMPOUND_SURNAMES and word[:2] in self.zh_surname:
            return self.zh_surname[word[:2]], word[2:]
        if word[:1] in COMMON_SURNAMES and word[:1] in self.zh_surname and word[0] not in NAME_STOP:
            return self.zh_surname[word[0]], word[1:]
        return None

    def _person(self, word: str) -> str | None:
        """Surname + 1-2 given characters (3+ chars total) -> 'Gao Luocheng', 'Uchiha Ban'."""
        s = self._surname(word)
        if not s or len(word) < 3 or any(a == b for a, b in zip(word, word[1:])):
            return None
        sur, given = s
        if not 1 <= len(given) <= 2 or any(c in NAME_STOP for c in given):
            return None
        if len(word) - len(given) == 1 and word[:2] in self.zh:
            return None  # surname char starts an ordinary word (高手下, 林中间, 张开始): not a name
        rom = self._romanize(given)
        return f"{sur} {rom}" if rom else None

    def _clan(self, word: str) -> str | None:
        """慕容世家 -> 'the Murong family', 李家 -> 'the Li family'."""
        for suf, what in CLAN_SUFFIXES.items():
            if word.endswith(suf) and len(word) > len(suf):
                s = self._surname(word[: len(word) - len(suf)])
                if s and not s[1]:
                    return f"the {s[0]} {what}"
        return None

    def _speech(self, word: str) -> str | None:
        """Narrative speech tags: 笑道 -> 'said with a smile', 萧炎冷笑道 -> 'Xiao Yan sneered',
        淡然道 -> 'said (tranquil and calm)', 大笑道 -> 'said (laugh heartily)'."""
        if not word.endswith("道") or any(c in ZH_PARTICLES for c in word):
            return None
        tags = dict(SPEECH_TAGS)
        if any(word[i:] in self.zh and word[i:] not in tags for i in range(len(word) - 1)):
            return None  # ends in a 道-word (知道, 难道, 味道): not a speech tag
        for tag, verb in SPEECH_TAGS:
            if not word.endswith(tag):
                continue
            prefix = word[: len(word) - len(tag)]
            if not prefix:
                return verb if tag != "道" else None
            who = NOVEL_TERMS[prefix].split(" (")[0] if prefix in NOVEL_NAMES else self._person(prefix)
            if who:
                return f"{who} {verb}"
            if tag == "道" and len(prefix) == 1:
                return None  # 口道 / 声道 fragments: too ambiguous without the manner word
            manner = self._short(word[:-1])
            if manner:
                return f"said ({manner})"  # 大笑道: manner word + 道
            how = self._short(prefix) if not any(c in NAME_STOP for c in prefix) else None
            if not how:
                return None
            return f"said ({how})" if tag == "道" else f"{verb} ({how})"
        return None

    def gloss(self, word: str):
        """-> (gloss, is_headword) or None. Non-headword CJK glosses only come from patterns that
        are reliable (novel terms, speech tags, person names, transliterations, 不+verb, two
        multi-character words); character-by-character concatenation is never emitted."""
        if CJK.search(word):
            if word in NOVEL_TERMS:
                return NOVEL_TERMS[word], False
            if len(word) >= 2 and word in dict(SPEECH_TAGS):
                return dict(SPEECH_TAGS)[word], False  # 说道/问道 in narration, not 'to ask the way'
            if word in self.zh and len(word) >= 2:
                return self.zh[word], True
            if len(word) < 2 or word[0] in ZH_PARTICLES or word[-1] in ZH_PARTICLES:
                return None  # span glued to a particle (X的 / 了X): not a lexical unit
            g = self._speech(word)
            if g:
                return g, False
            g = self._clan(word)
            if g:
                return g, False
            g = self._person(word)
            if g:
                return f"(name) {g}", False
            if len(word) >= 3 and sum(c in TRANSLIT for c in word) * 3 >= 2 * len(word) and \
                    not any(c in NAME_STOP for c in word) and not any(a == b for a, b in zip(word, word[1:])):
                rom = self._romanize(word)
                if rom:
                    return f"(name) {rom}", False
            if word[0] == "不" and len(word) >= 3 and self.zh.get(word[1:], "").startswith("to "):
                return f"not {self._short(word[1:])}", False  # 不+verb only (不喜欢 -> not like)
            if any(c in COMPOSE_STOP for c in word):
                return None
            parts = self._zh_segment(word)
            if parts and len(parts) == 2 and all(len(p) >= 2 and p not in COMPOSE_STOP_WORDS for p in parts):
                a, b = self._short(parts[0]), self._short(parts[1])
                if a and b and "?" not in a + b:
                    return f"{a} + {b}", False
            return None
        lw = word.lower()
        if not re.fullmatch(r"[a-z][a-z'-]*[a-z]", lw) or lw in EN_SKIP:  # single token: the device holds one word
            return None
        if lw in self.en_known:
            return (self.en[lw], True) if lw in self.en and "-" in lw else None
        for suf, rep in EN_SUFFIXES:
            if lw.endswith(suf) and len(lw) - len(suf) >= 3:
                lemma = lw[: len(lw) - len(suf)] + rep
                if len(lemma) >= 4 and lemma in self.en and self.en[lemma]:
                    return f"~{lemma}: {self.en[lemma]}", False
        if "-" in lw:
            parts = lw.split("-")
            if all(p in self.en for p in parts):
                return "+".join(self.en[p].split(",")[0] for p in parts), False
        return None


def gloss_key(data: bytes, a: int, e: int) -> str | None:
    """Lookup word for router span data[a:e]; Latin words must be whole (word boundaries in context)."""
    try:
        s = data[a:e].decode("utf-8")
    except UnicodeDecodeError:
        return None
    s = unicodedata.normalize("NFC", s).strip(STRIP)
    if not s:
        return None
    if CJK.search(s):
        return s
    s = s.translate(DEVICE_PUNCT)  # firmware folds ’ ‘ -> ' and U+2010/2011 -> - before hashing
    if len(s) < 2:
        return None
    raw = data[a:e]
    if raw[:1].isalpha() and a > 0 and data[a - 1:a].isalpha():
        return None
    if raw[-1:].isalpha() and e < len(data) and data[e:e + 1].isalpha():
        return None
    return s.lower()


def build_engram_meta(ref: M.PlhnIntNp, texts, book: GlossBook, max_glosses: int, min_count: int = 2):
    """Per-slot dominant router span (hash) + glosses for compounds/idioms/inflections.

    Gloss slots are keyed by FNV1a64(word) where word = span stripped of whitespace/punctuation,
    lower-cased for Latin script (what the reader looks up on NotFound)."""
    slot_spans: dict[int, Counter] = {}
    words = Counter()
    n_bytes = n_chunks = 0
    for text in texts:
        data = M.text_bytes(text)
        if not data:
            continue
        h = ref.encode_bytes(data)
        spans = ref.segment(data, h)
        n_bytes += len(data)
        n_chunks += len(spans)
        for a, e in spans:
            span = data[a:e]
            slot = M.fnv1a64(span) % M.ENGRAM_SLOTS
            slot_spans.setdefault(slot, Counter())[span] += 1
            key = gloss_key(data, a, e)
            if key:
                words[key] += 1
    hashes = np.zeros(M.ENGRAM_SLOTS, dtype=np.uint64)
    for slot, cnt in slot_spans.items():
        hashes[slot] = M.fnv1a64(cnt.most_common(1)[0][0])
    # curated novel terms first (always), then pattern glosses for non-headword spans (the device
    # dictionaries already resolve headwords), then headwords; each tier by corpus frequency
    cands = [(-1, -words.get(w, 0), w, g) for w, g in NOVEL_TERMS.items()]
    for word, f in words.items():
        if f < min_count or word in NOVEL_TERMS:
            continue
        g = book.gloss(word)
        if g and g[0]:
            cands.append((0 if not g[1] else 1, -f, word, g[0]))
    cands.sort()
    glosses, gloss_words, tiers = {}, {}, {}
    for tier, negf, word, text in cands:
        if len(glosses) >= max_glosses:
            break
        wb = word.encode("utf-8")
        slot = M.fnv1a64(wb) % M.ENGRAM_SLOTS
        if slot in glosses:
            continue
        glosses[slot] = gloss_trunc(text)
        gloss_words[slot] = word
        tiers[slot] = tier
        hashes[slot] = M.fnv1a64(wb)
    by_tier = {t: [s for s in glosses if tiers[s] == t] for t in (-1, 0, 1)}
    examples = by_tier[-1][:8] + by_tier[0][:40] + by_tier[1][:8]
    stats = {"texts": len(texts), "bytes": n_bytes, "chunks": n_chunks,
             "bytes_per_chunk": n_bytes / max(1, n_chunks), "slots_seen": len(slot_spans),
             "gloss_candidates": len(cands), "glosses": len(glosses),
             "glosses_by_tier": {"novel_terms": len(by_tier[-1]), "patterns": len(by_tier[0]),
                                 "headwords": len(by_tier[1])},
             "gloss_examples": [(gloss_words[s], glosses[s].decode()) for s in examples]}
    return hashes, glosses, stats


# ----------------------------------------------------------------------------------------------
# EPUB text
# ----------------------------------------------------------------------------------------------


class _Text(html.parser.HTMLParser):
    BLOCK = {"p", "div", "br", "h1", "h2", "h3", "h4", "li", "tr", "blockquote", "section"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def epub_text(path: Path, max_chars: int = 400_000) -> tuple[str, str]:
    """-> (title, plain text in spine order)."""
    with zipfile.ZipFile(path) as z:
        container = ET.fromstring(z.read("META-INF/container.xml"))
        opf_path = next(e.get("full-path") for e in container.iter() if e.tag.endswith("rootfile"))
        opf = ET.fromstring(z.read(opf_path))
        base = opf_path.rsplit("/", 1)[0] + "/" if "/" in opf_path else ""
        title = next((e.text for e in opf.iter() if e.tag.endswith("}title") and e.text), None) or path.stem
        manifest = {e.get("id"): e.get("href") for e in opf.iter() if e.tag.endswith("}item")}
        spine = [e.get("idref") for e in opf.iter() if e.tag.endswith("}itemref")]
        names = set(z.namelist())
        out, total = [], 0
        for idref in spine:
            href = manifest.get(idref)
            if not href:
                continue
            from urllib.parse import unquote
            name = os.path.normpath(base + unquote(href.split("#")[0])).replace("\\", "/")
            if name not in names:
                continue
            p = _Text()
            p.feed(z.read(name).decode("utf-8", "replace"))
            txt = re.sub(r"[ \t\r\f\v]+", " ", "".join(p.parts))
            txt = re.sub(r"\n\s*", "\n", txt).strip()
            if txt:
                out.append(txt)
                total += len(txt)
            if total >= max_chars:
                break
    return title.strip(), "\n".join(out)


def passages(text: str, k: int, n_bytes: int = M.MAX_BYTES):
    """k passages of <= n_bytes UTF-8, evenly spaced through the substantive body."""
    paras = [p.strip() for p in text.split("\n") if len(p.strip().encode()) >= 80]
    if not paras:
        paras = [text.strip()] if text.strip() else []
    if not paras:
        return []
    lo = len(paras) // 20  # skip front matter
    body = paras[lo:] or paras
    picks = sorted({int(i * len(body) / k) for i in range(k)})
    return [utf8_trunc(body[i], n_bytes).decode("utf-8") for i in picks]


def build_icpt(ref: M.PlhnIntNp, books_dir: Path, novels_dir: Path, novel_sample: int, per_book: int):
    jobs = [(p, "/books/Books/" + p.name) for p in sorted(books_dir.glob("*.epub"))
            if not p.name.startswith(".") and ".pre-hooks." not in p.name]
    novels = [p for p in sorted(novels_dir.glob("*.epub")) if not p.name.startswith(".")]
    if novel_sample and len(novels) > novel_sample:
        step = len(novels) / novel_sample
        novels = [novels[int(i * step)] for i in range(novel_sample)]
    jobs += [(p, "/books/Novels/" + p.name) for p in novels]
    records, failures = [], []
    for path, card_path in jobs:
        try:
            title, text = epub_text(path)
        except Exception as exc:  # malformed EPUB: report, keep going
            failures.append((path.name, repr(exc)[:200]))
            continue
        ph = M.fnv1a64(card_path.encode("utf-8"))
        stamp = int(path.stat().st_mtime) & 0xFFFFFFFF
        label = utf8_trunc(title or path.stem, 47)
        for i, passage in enumerate(passages(text, per_book)):
            vec = ref.forward(passage)
            records.append((ph, i, stamp, label, vec))
    return records, failures, len(jobs)


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------


def load_corpus_texts(path: Path, limit: int, seed: int = 0):
    texts = []
    if path and path.exists():
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    texts.append(json.loads(line)["text"])
                except (json.JSONDecodeError, KeyError):
                    continue
        random.Random(seed).shuffle(texts)
    return texts[:limit]


def check_parity(net, ref_np, ref_py, texts, n_py: int):
    a = net.encode_int(texts)
    b = [ref_np.forward(t) for t in texts]
    mism = [i for i, (x, y) in enumerate(zip(a, b)) if list(x) != list(y)]
    py_mism = [i for i in range(min(n_py, len(texts))) if ref_py.forward(texts[i]) != b[i]]
    return {"n": len(texts), "torch_vs_numpy_mismatch": len(mism), "n_py": min(n_py, len(texts)),
            "py_vs_numpy_mismatch": len(py_mism), "first_mismatch": (mism + py_mism)[:5]}


def merge_bigram_hashes(hashes: np.ndarray, glosses: dict, spans) -> tuple[np.ndarray, dict]:
    """Bigram-Span hashes into the slot table, by descending count: a glossed slot keeps its word hash,
    the first (most frequent) span claims any other slot (replacing the informational dominant-span hash)."""
    out = hashes.copy()
    claimed, skipped = set(), 0
    for h, _ in sorted(spans, key=lambda kv: -kv[1]):
        h = int(h)
        s = h % M.ENGRAM_SLOTS
        if h == 0 or s in claimed:
            continue
        if s in glosses:
            skipped += out[s] != np.uint64(h)
            continue
        out[s] = np.uint64(h)
        claimed.add(s)
    return out, {"bigram_spans": len(spans), "bigram_slots": len(claimed), "bigram_skipped_gloss_slot": int(skipped)}


def load_spans(ck: dict, path: Path):
    if ck.get("bigram_spans"):
        return [(int(h), int(c)) for h, c in ck["bigram_spans"]], "checkpoint"
    if path.exists():
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return [(int(r["span_hash"]), int(r.get("count", 1))) for r in rows], str(path)
    return [], "none"


def pack_golden_vectors(items) -> bytes:
    """PLGV v1: magic, u32 version, u32 n, u32 0; per case u16 len + UTF-8 bytes + int8[128]."""
    out = [struct.pack("<4sIII", b"PLGV", 1, len(items), 0)]
    for data, vec in items:
        out.append(struct.pack("<H", len(data)) + data + np.asarray(vec, np.int8).tobytes())
    return b"".join(out)


def read_jsonl_rows(path: Path):
    if not path or not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv=None):
    import torch

    import cspw_model as CM

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--models-dir", type=Path, default=ROOT / "export")
    ap.add_argument("--golden-dir", type=Path, default=ROOT / "golden")
    ap.add_argument("--corpus", type=Path, default=ROOT / "data" / "train_corpus.jsonl")
    ap.add_argument("--span-texts", type=int, default=20000)
    ap.add_argument("--max-glosses", type=int, default=24000)
    ap.add_argument("--gloss-min-count", type=int, default=2, help="min corpus span frequency for a gloss")
    ap.add_argument("--books-dir", type=Path, default=ROOT / "data" / "books")
    ap.add_argument("--novels-dir", type=Path, default=ROOT / "data" / "novels")
    ap.add_argument("--novel-sample", type=int, default=120)
    ap.add_argument("--per-book", type=int, default=4)
    ap.add_argument("--bigram-spans", type=Path, default=ROOT / "data" / "cspw_bigram_spans.jsonl",
                    help="used when the checkpoint carries no bigram_spans")
    ap.add_argument("--timelines", type=Path, default=None, help="default: data/cspw_timelines.jsonl, else the "
                    "checkpoint's story source")
    ap.add_argument("--aliases", type=Path, default=None)
    ap.add_argument("--cspw-golden-cases", type=int, default=64)
    ap.add_argument("--manifest", type=Path, default=ROOT / "export_manifest.json")
    args = ap.parse_args(argv)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "encoder" in ck:
        state = ck["encoder"]
    elif ck.get("arm") == "arm3_platonic_engram_full":
        state = ck["state_dict"]
    else:
        raise SystemExit(f"{args.ckpt}: expected a train_cspw checkpoint or arm3_platonic_engram_full, "
                         f"got {ck.get('arm')}")
    has_cspw = "cspw" in ck
    # the device always applies the Bigram-Span Engram, so the parity graph does too
    net = M.PlatonicEngramHNet(use_engram=True, use_bigram=True)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if set(missing) - {"engram_hash"} or unexpected:
        raise SystemExit(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    net.eval()

    t0 = time.time()
    tensors, scales = quantize_model(net)
    plhn = pack_plhn(tensors, scales)
    escale = scales["engram"]
    edelta = np.clip(np.floor(net.engram.detach().double().numpy() / escale + 0.5), -127, 127).astype(np.int8)
    # segmentation is upstream of the Engram, so span statistics do not depend on the hash table
    ref_nogloss = M.PlhnIntNp(plhn, pack_engram(edelta, escale, np.zeros(M.ENGRAM_SLOTS, np.uint64), {}))

    texts = load_corpus_texts(args.corpus, args.span_texts)
    if not texts:
        texts = [p for b in sorted(args.books_dir.glob("*.epub")) for p in passages(epub_text(b)[1], 200)]
    book = GlossBook()
    hashes, glosses, gstats = build_engram_meta(ref_nogloss, texts, book, args.max_glosses, args.gloss_min_count)
    spans, span_src = load_spans(ck, args.bigram_spans)
    hashes, bstats = merge_bigram_hashes(hashes, glosses, spans)
    gstats.update(bstats, bigram_source=span_src)
    engr = pack_engram(edelta, escale, hashes, glosses)
    net.set_engram_hashes(hashes)  # the bigram gate reads the exported table

    ref_np = M.PlhnIntNp(plhn, engr)
    ref_py = M.PlhnIntRef(plhn, engr)
    parity = check_parity(net, ref_np, ref_py, GOLDEN_TEXTS + texts[:256], n_py=len(GOLDEN_TEXTS) + 8)
    if parity["torch_vs_numpy_mismatch"] or parity["py_vs_numpy_mismatch"]:
        raise SystemExit(f"integer parity failed: {parity}")
    ref_uni = M.PlhnIntNp(plhn, engr, bigram=False)
    bigram_texts = [t for t in texts[:2000] if ref_np.forward(t) != ref_uni.forward(t)]
    parity["bigram_changed_of_2000"] = len(bigram_texts)

    records, failures, n_books = build_icpt(ref_np, args.books_dir, args.novels_dir, args.novel_sample, args.per_book)
    icpt = pack_icpt(records)

    # ---- CS-PW + story graph
    out_files = {"platonic-hnet.bin": plhn, "engram.bin": engr, "inbox-concepts.idx": icpt}
    cspw_info, story_info, golden_cases = {}, {}, []
    if has_cspw:
        cfg = ck["cspw_cfg"]
        cs = CM.ChronoSlotPointerWeaver(d_model=cfg["d_model"], pool=cfg["pool"])
        cs.load_state_dict(ck["cspw"])
        cs.quant = True
        cs.eval()
        ctensors, sal_b = cs.export_tensors()
        cblob = CM.pack_cspw(ctensors, sal_b, ck["frames"], ck["phrases"])
        w = CM.CspwWeights(cblob)

        story_src = ck.get("story", {})
        tpath = args.timelines or (ROOT / "data" / "cspw_timelines.jsonl")
        if not tpath.exists() and story_src.get("timelines"):
            tpath = Path(story_src["timelines"])
        apath = args.aliases or (tpath.parent / "cspw_aliases.jsonl")
        tls = [t for t in read_jsonl_rows(tpath) if t.get("teacher_valid", True) is not False]
        als = read_jsonl_rows(apath)
        if not tls:
            raise SystemExit(f"no story timelines at {tpath}")
        vcache = {}

        def vec_of(t):
            if t not in vcache:
                vcache[t] = list(ref_np.forward(t))
            return vcache[t]

        ents = CM.timeline_entities(tls, vec_of)
        sblob = CM.pack_sgrf(ents, CM.alias_rows(tls, als))
        graph = CM.StoryGraph(sblob)
        out_files["cspw.bin"] = cblob
        out_files["story-graph.idx"] = sblob

        # goldens: device scene selections from the exported graph, then random int8 scene sets
        rng = np.random.default_rng(7)
        sel = []
        for e in ents:
            evs = sorted(e["events"], key=lambda x: x["cutoff"])
            for cur in sorted({x["cutoff"] for x in evs})[-2:]:
                hit = graph.lookup(e["term_hash"], e["book_path_hash"], cur)
                if hit is not None:
                    sc, ec = CM.select_scenes(hit[1])
                    if sc:
                        sel.append(([x["vec"] for x in sc], [x["chapter_or_page"] for x in sc],
                                    ec["vec"] if ec is not None else None))
        sel = sel[: args.cspw_golden_cases // 2]
        pool = [v for v in vcache.values() if any(v)]
        while len(sel) < args.cspw_golden_cases:
            m = int(rng.integers(1, CM.MAX_SCENES + 1))
            sc = [pool[int(rng.integers(len(pool)))] if pool and rng.random() < 0.7
                  else rng.integers(-127, 128, CM.D_SLOT).tolist() for _ in range(m)]
            pg = sorted(int(x) for x in rng.integers(0, 0x10000, m))
            ec = pool[int(rng.integers(len(pool)))] if pool and rng.random() < 0.5 else None
            sel.append((sc, pg, ec))
        for sc, pg, ec in sel:
            golden_cases.append(CM.pack_golden_case(sc, pg, ec, CM.int8_cspw_weave(w, sc, pg, ec)))

        # torch QAT CS-PW == integer reference on the golden inputs
        cmism = 0
        for sc, pg, ec in sel:
            r = CM.int8_cspw_weave(w, sc, pg, ec)
            S = torch.zeros(1, CM.MAX_SCENES, CM.D_SLOT, dtype=torch.float64)
            S[0, :len(sc)] = torch.tensor(sc, dtype=torch.float64)
            P = torch.zeros(1, CM.MAX_SCENES, dtype=torch.long)
            P[0, :len(pg)] = torch.tensor(pg)
            E = torch.zeros(1, CM.D_SLOT, dtype=torch.float64) if ec is None else \
                torch.tensor([ec], dtype=torch.float64)
            with torch.no_grad():
                o = cs.double()(S, P, torch.tensor([len(sc)]), E, torch.tensor([ec is not None]))
            if not np.array_equal(o["slots"][0].round().long().numpy(), np.asarray(r["slots"])):
                cmism += 1
        if cmism:
            raise SystemExit(f"CS-PW torch QAT vs int8 reference: {cmism}/{len(sel)} mismatches")
        texts_ok = sum(1 for e in ents[:200]
                       if (h := graph.lookup(e["term_hash"], e["book_path_hash"], max(x["cutoff"] for x in e["events"])))
                       and CM.weave_refresher(w, h[0], h[1])[0])
        cspw_info = {"cfg": cfg, "arm": ck.get("arm"), "salience_b": int(sal_b), "torch_vs_int_mismatch": cmism,
                     "golden_cases": len(golden_cases), "golden_from_story_graph": min(len(sel), args.cspw_golden_cases // 2)}
        story_info = {"timelines": str(tpath), "aliases": str(apath), "synthetic": bool(story_src.get("synthetic")),
                      "entities": len(ents), "events": sum(len(e["events"]) for e in ents),
                      "alias_rows": len(CM.alias_rows(tls, als)), "refreshers_ok_of_200": texts_ok}

    for name, blob in out_files.items():
        atomic_write(args.models_dir / name, blob)

    golden = []
    for t in GOLDEN_TEXTS + bigram_texts[:8]:
        golden.append({"text": t, "hex": M.text_bytes(t).hex(), "vec": ref_np.forward(t),
                       "spans": [list(s) for s in ref_np.chunk_states(t)[1]]})
    args.golden_dir.mkdir(parents=True, exist_ok=True)
    for name, blob in out_files.items():
        if name != "inbox-concepts.idx":
            atomic_write(args.golden_dir / name, blob)
    atomic_write(args.golden_dir / "golden.jsonl",
                 "".join(json.dumps(g, ensure_ascii=False) + "\n" for g in golden).encode())
    gv_texts = GOLDEN_TEXTS + bigram_texts[:64] + texts[:192]
    gv = pack_golden_vectors([(M.text_bytes(t), ref_np.forward(t)) for t in gv_texts])
    atomic_write(args.golden_dir / "golden_vectors.bin", gv)
    if golden_cases:
        atomic_write(args.golden_dir / "cspw_golden.bin", CM.pack_golden(golden_cases))
    gl = [{"word": w_, "gloss": g} for w_, g in gstats["gloss_examples"]]
    atomic_write(args.golden_dir / "engram_gloss_examples.json", json.dumps(gl, ensure_ascii=False, indent=1).encode())

    sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    files = {n: {"bytes": len(b), "sha256": sha(b)} for n, b in out_files.items()}
    files["inbox-concepts.idx"].update(records=len(records), books=n_books, failures=failures)
    manifest = {
        "ckpt": str(args.ckpt), "arm": ck.get("arm"), "scales": scales, "files": files,
        "engram": gstats, "parity": parity, "cspw": cspw_info, "story_graph": story_info,
        "goldens": {"golden.jsonl": len(golden), "golden_vectors.bin": len(gv_texts),
                    "cspw_golden.bin": len(golden_cases)},
        "seconds": round(time.time() - t0, 1),
        "gloss_key_rule": "FNV1a64(word) where word = UTF-8 span stripped of whitespace/punct, NFC, "
                          "lower-cased if no CJK",
    }
    atomic_write(args.manifest, json.dumps(manifest, ensure_ascii=False, indent=1).encode())
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
