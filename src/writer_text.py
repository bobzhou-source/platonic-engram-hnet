"""Device text helpers shared by PE-Writer training data prep and the public benchmark.

* ``utf8_fit`` / ``word_fit``: fit subject / role / echo-title strings into the device's ``char[32]`` fields.
* ``page_words`` / ``context_around`` / ``held_index``: rebuild the firmware's word boxes and the held-word
  context window exactly as the reader's dictionary page does.
"""
from __future__ import annotations

import re

CONTEXT_WORDS = 8
FIELD_BYTES = 31  # subject / role / echo_title fit char[32]
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
WS = re.compile(r"\s+")


def utf8_fit(s: str, n: int) -> str:
    s = WS.sub(" ", s or "").strip()
    b = s.encode("utf-8")
    if len(b) <= n:
        return s
    return b[:n].decode("utf-8", "ignore").rstrip()


PHRASE_BREAK = {"in", "of", "and", "or", "that", "who", "which", "whose", "with", "for", "to", "by", "from", "at",
                "on", "as", "while", "but", "through", "during", "within", "against", "across", "under", "over",
                "between", "into", "amid", "after", "before", "prone", "responsible", "capable", "able", "known",
                "like", "whom", "where", "when", "seen", "set", "about", "toward", "towards", "among"}
ADJ_ENDS = ("al", "ous", "ed", "ing", "ful", "ible", "able", "ic", "ive", "less")
NOUN_OK = {"rival", "general", "official", "criminal", "ritual", "capital", "principal", "cardinal", "admiral",
           "animal", "heretic", "mystic", "cleric", "critic", "catholic", "detective", "operative", "relative",
           "representative", "captive", "native", "fugitive", "individual", "intellectual", "fanatic", "magic",
           "republic", "logic", "music", "rhetoric", "relic", "being", "king", "thing", "offspring", "ring",
           "kingdom", "creed", "seed", "need", "deed", "arsenal", "tribunal", "festival", "survival", "ideal",
           "manual", "journal", "signal", "symbol", "potential", "minimal"}
PARTITIVE = {"type", "kind", "group", "member", "form", "part", "sort", "piece", "set", "body", "school", "branch",
             "class", "pair", "series", "variety", "source", "symbol", "state", "act", "mode", "method"}


def _is_break(w: str) -> bool:
    w = w.lower().strip(",;:")
    return w in PHRASE_BREAK or (len(w) > 4 and w.endswith(("ing", "ed")) and w not in NOUN_OK)


def _np_end_ok(w: str) -> bool:
    w = w.lower().strip(",;:")
    return (w not in {"a", "an", "the"} and w not in PHRASE_BREAK
            and (w in NOUN_OK or not w.endswith(ADJ_ENDS)))


def word_fit(s: str, n: int) -> str:
    """<= n UTF-8 bytes. Latin text keeps a complete noun phrase: cut before a phrase break (preposition,
    conjunction, participle) after a noun-like word, else article + the coordinated head ("a historical and
    political concept" -> "a political concept"), else a plain word-boundary cut. CJK: character boundary."""
    s = WS.sub(" ", s or "").strip().rstrip(".;,")
    if len(s.encode()) <= n:
        return s
    if CJK.search(s):
        return utf8_fit(s, n)
    words = s.split(" ")
    fit = 0
    while fit < len(words) and len(" ".join(words[:fit + 1]).encode()) <= n:
        fit += 1
    for k in range(fit, 1, -1):
        coord = words[k].lower() in ("and", "or")
        content = [w for w in words[:k] if w.lower() not in ("a", "an", "the")]
        if coord and len(content) < 2:
            continue
        if words[k].lower() == "of" and words[k - 1].lower() in PARTITIVE:
            continue
        if (_is_break(words[k]) or words[k - 1].endswith((",", ";", ":"))) and _np_end_ok(words[k - 1]):
            return " ".join(words[:k]).rstrip(",;:-–— ")
    head = next((i for i in range(1, len(words)) if _is_break(words[i]) and words[i].lower() not in ("and", "or")),
                len(words))
    np_words = words[:head]
    art = [np_words[0]] if np_words[0].lower() in ("a", "an", "the") else []
    for j in range(len(np_words) - 1, 0, -1):
        if np_words[j].lower() in ("and", "or"):
            tail = np_words[j + 1:]
            if not tail:
                break
            if art and art[0].lower() in ("a", "an"):
                art = ["an" if tail[0][:1].lower() in "aeiou" else "a"]
            cand = " ".join(art + tail)
            if tail and len(cand.encode()) <= n and _np_end_ok(tail[-1]):
                return cand
            break
    out = " ".join(words[:fit]).rstrip(",;:-–— ")
    for tail in (" of", " the", " a", " an", " and", " to", " in", " by", " for", " with", " as"):
        if out.endswith(tail):
            out = out[: -len(tail)]
    return out or utf8_fit(s, n)


# ------------------------------------------------------------------ device word boxes / held-word context

def page_words(text: str) -> list[tuple[str, bool, int]]:
    """(word, is_cjk, char offset) boxes as DictionaryPageWords::appendToken builds them."""
    out = []
    for m in re.finditer(r"\S+", text):
        tok = m.group(0)
        if CJK.search(tok):
            for j, ch in enumerate(tok):
                if CJK.match(ch):
                    out.append((ch, True, m.start() + j))
        elif re.search(r"[^\W_]", tok):
            out.append((tok, False, m.start()))
    return out


def context_around(words, sel: int, radius: int = CONTEXT_WORDS) -> str:
    first, last = max(0, sel - radius), min(len(words) - 1, sel + radius)
    text = ""
    for i in range(first, last + 1):
        w, cjk, _ = words[i]
        if text and not (cjk and i > first and words[i - 1][1]):
            text += " "
        text += w
    return text


def held_index(words, pos: int) -> int:
    for i, (w, _, off) in enumerate(words):
        if off + len(w) > pos:
            return i
    return len(words) - 1
