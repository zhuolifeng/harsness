"""
solution.py — 考生唯一需要提交的文件

规则
----
1. 只能修改 MyHarness 类内部；其余部分不可改动。考生可以先行查看 harness_base.py 以了解可用接口和调用约定。
2. 只允许 import Python 标准库（re, math, random, json, collections 等）、numpy
   以及 harness_base（已提供）。
3. 禁止 import 其他第三方库（openai, sklearn, torch …）。
4. 禁止通过任何途径读写磁盘文件。
5. call_llm 每次调用的 prompt token 数若超过 max_prompt_tokens，
   会被自动截断至预算上限后再发送，
   可用 count_tokens（计算单条消息的 token 数） 和 count_messages_tokens（计算消息列表的总 token 数）预先控制 prompt 长度。
6. predict() 只接收 text，任何绕过接口获取 label 的行为将导致得分归零。
"""

from harness_base import Harness
from typing import Optional

# ============================================================
# 考生实现区（考生只能修改 MyHarness 类里的内容）
# ============================================================
class MyHarness(Harness):
    import math as _math
    import re as _re
    import threading as _threading
    from collections import Counter as _Counter, defaultdict as _DefaultDict

    _WORD_RE = _re.compile(r"[a-z0-9]+")
    _STOPWORDS = frozenset({
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "do",
        "does", "for", "from", "get", "got", "have", "has", "had", "how", "i",
        "in", "is", "it", "its", "me", "my", "of", "on", "or", "please", "the",
        "this", "that", "to", "was", "were", "what", "when", "where", "why",
        "will", "with", "would", "you", "your", "could", "should", "shall",
        "may", "might", "must", "need", "not", "no", "so", "if", "but", "just",
        "also", "very", "much", "more", "most", "some", "any", "all", "each",
        "every", "other", "about", "than", "then", "there", "here", "out",
        "up", "down", "into", "over", "after", "before", "between", "through",
    })

    def __init__(self, call_llm, count_tokens, count_messages_tokens, max_prompt_tokens: int):
        super().__init__(call_llm, count_tokens, count_messages_tokens, max_prompt_tokens)
        self._by_label = self._DefaultDict(list)
        self._labels = []
        self._label_tokens = {}
        self._token_df = self._Counter()
        self._doc_count = 0
        self._llm_status = 0
        self._llm_probe_lock = self._threading.Lock()
        self._is_mcq = None

    def update(self, text: str, label: str) -> None:
        super().update(text, label)
        tokens = self._tokenize(text)
        entry = {
            "text": text,
            "toks": set(tokens),
            "chars": self._char_ngrams(tokens),
            "norm": " ".join(tokens),
        }
        if label not in self._by_label:
            self._labels.append(label)
            self._label_tokens[label] = set(self._tokenize(label.replace("_", " ")))
        self._by_label[label].append(entry)
        self._token_df.update(entry["toks"])
        self._doc_count += 1
        self._is_mcq = None

    def predict(self, text: str) -> str:
        if not self._labels:
            return ""

        if self._is_mcq is None:
            self._is_mcq = self._detect_mcq()

        ranked = self._rank(text)
        best = ranked[0]["label"]

        # MCQ: always use LLM
        if self._is_mcq:
            r = self._llm_predict(text, ranked)
            return r or best

        # High confidence lexical: skip LLM
        margin = ranked[0]["score"] - ranked[1]["score"] if len(ranked) > 1 else 1.0
        if ranked[0]["score"] >= 0.60 and margin >= 0.15:
            return best

        r = self._llm_predict(text, ranked)
        return r or best

    # ---- Task detection ----
    def _detect_mcq(self) -> bool:
        if len(self._labels) > 10:
            return False
        return all(self._re.match(r'^[A-Za-z]$', l) for l in self._labels)

    # ---- Tokenization ----
    def _tokenize(self, text: str) -> list[str]:
        return self._WORD_RE.findall((text or "").lower())

    def _char_ngrams(self, tokens: list[str]) -> set[str]:
        s = " " + " ".join(tokens) + " "
        grams = set()
        for n in (3, 4):
            for i in range(len(s) - n + 1):
                grams.add(s[i:i+n])
        return grams

    # ---- Scoring ----
    def _idf(self, t: str) -> float:
        return self._math.log((self._doc_count + 1) / (self._token_df.get(t, 0) + 1)) + 1.0

    def _cosine(self, a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        common = a & b
        if not common:
            return 0.0
        num = sum(self._idf(t) for t in common)
        na = self._math.sqrt(sum(self._idf(t)**2 for t in a))
        nb = self._math.sqrt(sum(self._idf(t)**2 for t in b))
        return num / (na * nb) if na and nb else 0.0

    def _jaccard(self, a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        u = a | b
        return len(a & b) / len(u) if u else 0.0

    def _rank(self, text: str) -> list[dict]:
        qtoks = self._tokenize(text)
        qset = set(qtoks)
        qchars = self._char_ngrams(qtoks)
        qnorm = " ".join(qtoks)
        ranked = []

        for label, examples in self._by_label.items():
            lt_score = self._jaccard(qset, self._label_tokens[label])
            scores = []
            for e in examples:
                w = self._cosine(qset, e["toks"])
                c = self._jaccard(qchars, e["chars"])
                p = 0.06 if qnorm and (qnorm in e["norm"] or e["norm"] in qnorm) else 0.0
                scores.append(0.55 * w + 0.28 * c + 0.12 * lt_score + p)
            scores.sort(reverse=True)
            top = scores[0]
            avg2 = sum(scores[:2]) / min(2, len(scores))
            ranked.append({
                "label": label,
                "score": 0.7 * top + 0.3 * avg2,
                "examples": [e["text"] for e in examples[:3]],
            })

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    # ---- LLM ----
    def _llm_predict(self, text: str, ranked: list[dict]) -> Optional[str]:
        if self._llm_status < 0:
            return None
        if self._llm_status == 0:
            with self._llm_probe_lock:
                if self._llm_status == 0:
                    r = self._do_llm(text, ranked)
                    self._llm_status = 1 if r else -1
                    return r
        return self._do_llm(text, ranked) if self._llm_status > 0 else None

    def _do_llm(self, text: str, ranked: list[dict]) -> Optional[str]:
        msg = self._build_prompt(text, ranked)
        try:
            resp = self.call_llm(msg)
        except Exception:
            return None
        return self._parse(resp)

    def _build_prompt(self, text: str, ranked: list[dict]) -> list[dict]:
        if self._is_mcq:
            return self._prompt_mcq(text)
        return self._prompt_cls(text, ranked)

    def _prompt_mcq(self, text: str) -> list[dict]:
        # Few-shot for MCQ
        demos = []
        for label in self._labels:
            exs = self._by_label[label][:2]
            for e in exs:
                demos.append(f"{e['text']}\n{label}")
        all_labels = "/".join(self._labels)
        prompt = f"Classify. Answer only {all_labels}.\n\n" + "\n\n".join(demos) + f"\n\n{text}\n"
        return [{"role": "user", "content": prompt}]

    def _prompt_cls(self, text: str, ranked: list[dict]) -> list[dict]:
        # Compact prompt: top candidates with 1 example each
        n_cands = min(25, len(ranked))
        lines = []
        for i, item in enumerate(ranked[:n_cands]):
            ex = item["examples"][0][:70] if item["examples"] else ""
            lines.append(f"{item['label']}: {ex}")

        cand_block = "\n".join(lines)
        # Remaining labels as a list
        shown = set(item["label"] for item in ranked[:n_cands])
        others = [l for l in self._labels if l not in shown]
        other_str = ", ".join(others) if others else ""

        prompt = (
            f"Classify into one label. Reply with the label only.\n\n"
            f"Labels:\n{cand_block}\n"
        )
        if other_str:
            prompt += f"\nOther: {other_str}\n"
        prompt += f"\nInput: {text}\nLabel:"

        messages = [{"role": "user", "content": prompt}]
        # Trim candidates if over budget
        while self.count_messages_tokens(messages) > self.max_prompt_tokens and n_cands > 10:
            n_cands -= 3
            lines = lines[:n_cands]
            cand_block = "\n".join(lines)
            shown = set(item["label"] for item in ranked[:n_cands])
            others = [l for l in self._labels if l not in shown]
            other_str = ", ".join(others) if others else ""
            prompt = f"Classify into one label. Reply with the label only.\n\nLabels:\n{cand_block}\n"
            if other_str:
                prompt += f"\nOther: {other_str}\n"
            prompt += f"\nInput: {text}\nLabel:"
            messages = [{"role": "user", "content": prompt}]
        return messages

    def _parse(self, resp: str) -> Optional[str]:
        if not resp:
            return None
        # Take first line, strip formatting
        line = resp.strip().split("\n")[0].strip()
        line = self._re.sub(r'^[`\'"*]+|[`\'"*.,!]+$', '', line).strip()
        line = self._re.sub(r'^(?:label|answer)\s*[:：]\s*', '', line, flags=self._re.IGNORECASE).strip()
        if not line:
            return None
        low = line.lower()
        # Exact
        for l in self._labels:
            if l == line or l.lower() == low:
                return l
        # Contained
        matches = [l for l in self._labels if l.lower() in low]
        if len(matches) == 1:
            return matches[0]
        if matches:
            return max(matches, key=len)
        # Canonical
        canon = self._canon(line)
        for l in self._labels:
            if self._canon(l) == canon:
                return l
        # Fuzzy: label with spaces instead of underscores
        for l in self._labels:
            if l.replace("_", " ").lower() == low:
                return l
        return None

    def _canon(self, s: str) -> str:
        return "".join(c for c in s.lower() if c.isalnum())
