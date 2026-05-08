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
    _SPELLING = {
        "recognised": "recognized",
        "recognise": "recognize",
        "recognising": "recognizing",
        "cheque": "check",
        "cheques": "checks",
        "cancelled": "canceled",
        "cancelling": "canceling",
    }
    _STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "do",
        "does", "for", "from", "get", "got", "have", "has", "how", "i", "in",
        "is", "it", "me", "my", "of", "on", "or", "please", "the", "this",
        "to", "what", "when", "why", "with", "you", "your",
    }

    def __init__(self, call_llm, count_tokens, count_messages_tokens, max_prompt_tokens: int):
        super().__init__(call_llm, count_tokens, count_messages_tokens, max_prompt_tokens)
        self._examples = []
        self._by_label = self._DefaultDict(list)
        self._labels = []
        self._label_tokens = {}
        self._token_df = self._Counter()
        self._doc_count = 0
        self._llm_status = 0  # 0=unknown, 1=available, -1=unavailable
        self._llm_probe_lock = self._threading.Lock()
        self._label_hints = {}

    def update(self, text: str, label: str) -> None:
        super().update(text, label)
        tokens = self._tokens(text)
        entry = {
            "text": text,
            "tokens": set(tokens),
            "chars": self._char_grams(tokens),
            "norm": " ".join(tokens),
        }
        if label not in self._by_label:
            self._labels.append(label)
            self._label_tokens[label] = set(self._tokens(label.replace("_", " ")))
        self._examples.append((label, entry))
        self._by_label[label].append(entry)
        self._token_df.update(entry["tokens"])
        self._doc_count += 1
        self._label_hints.pop(label, None)

    def predict(self, text: str) -> str:
        if not self._labels:
            return self._fallback_llm(text) or ""

        ranked = self._rank_labels(text)
        best_label = ranked[0]["label"]
        margin = ranked[0]["score"] - ranked[1]["score"] if len(ranked) > 1 else ranked[0]["score"]

        # High-margin matches are usually exact lexical intent matches; avoid an unnecessary LLM call.
        if ranked[0]["score"] >= 0.55 and margin >= 0.12:
            return best_label

        llm_label = self._predict_with_llm(text, ranked)
        return llm_label or best_label

    def _tokens(self, text: str) -> list[str]:
        tokens = self._WORD_RE.findall((text or "").lower())
        return [self._SPELLING.get(token, token) for token in tokens]

    def _char_grams(self, tokens: list[str]) -> set[str]:
        text = " " + " ".join(tokens) + " "
        grams = set()
        for n in (3, 4):
            if len(text) >= n:
                grams.update(text[i:i + n] for i in range(len(text) - n + 1))
        return grams

    def _idf(self, token: str) -> float:
        return self._math.log((self._doc_count + 1) / (self._token_df.get(token, 0) + 1)) + 1.0

    def _weighted_cosine(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        common = left & right
        if not common:
            return 0.0
        numerator = sum(self._idf(token) for token in common)
        left_norm = self._math.sqrt(sum(self._idf(token) ** 2 for token in left))
        right_norm = self._math.sqrt(sum(self._idf(token) ** 2 for token in right))
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def _jaccard(self, left: set, right: set) -> float:
        if not left and not right:
            return 1.0
        union = left | right
        return len(left & right) / len(union) if union else 0.0

    def _rank_labels(self, text: str) -> list[dict]:
        query_tokens = self._tokens(text)
        query_set = set(query_tokens)
        query_chars = self._char_grams(query_tokens)
        query_norm = " ".join(query_tokens)
        ranked = []

        for label, examples in self._by_label.items():
            label_scores = []
            label_token_score = self._jaccard(query_set, self._label_tokens[label])
            for entry in examples:
                word_score = self._weighted_cosine(query_set, entry["tokens"])
                char_score = self._jaccard(query_chars, entry["chars"])
                phrase_bonus = 0.05 if query_norm and (query_norm in entry["norm"] or entry["norm"] in query_norm) else 0.0
                score = 0.57 * word_score + 0.28 * char_score + 0.15 * label_token_score + phrase_bonus
                label_scores.append((score, entry))

            label_scores.sort(key=lambda item: item[0], reverse=True)
            top = label_scores[0][0]
            avg = sum(score for score, _ in label_scores[:2]) / min(2, len(label_scores))
            ranked.append({
                "label": label,
                "score": 0.72 * top + 0.28 * avg,
                "examples": [entry for _, entry in label_scores[:2]],
            })

        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked

    def _predict_with_llm(self, text: str, ranked: list[dict]) -> Optional[str]:
        if self._llm_status < 0:
            return None

        if self._llm_status == 0:
            with self._llm_probe_lock:
                if self._llm_status == 0:
                    label = self._call_llm_classifier(text, ranked)
                    self._llm_status = 1 if label else -1
                    return label

        return self._call_llm_classifier(text, ranked) if self._llm_status > 0 else None

    def _call_llm_classifier(self, text: str, ranked: list[dict]) -> Optional[str]:
        messages = self._build_messages(text, ranked)
        try:
            response = self.call_llm(messages)
        except Exception:
            return None
        return self._extract_label(response)

    def _build_messages(self, text: str, ranked: list[dict]) -> list[dict]:
        candidate_count = min(30, len(ranked))

        while True:
            candidate_lines = []
            candidate_labels = set()
            for idx, item in enumerate(ranked[:candidate_count]):
                candidate_labels.add(item["label"])
                example_limit = 2 if idx < 12 else 1
                examples = " | ".join(example["text"] for example in item["examples"][:example_limit])
                candidate_lines.append(
                    f"{idx + 1}. {item['label']} ; keywords: {self._label_hint(item['label'])} ; examples: {examples}"
                )
            other_labels = ", ".join(label for label in self._labels if label not in candidate_labels)

            prompt = (
                "You are classifying a banking support message.\n"
                "Choose exactly one label. Prefer the detailed candidates, but use an other valid label if none fit.\n"
                "Return only the exact label string.\n"
                "Use the keywords and examples to distinguish close labels such as fees, unrecognized payments, "
                "refunds, transfers, top-ups, and card issues.\n\n"
                "Candidate labels:\n" + "\n".join(candidate_lines) + "\n\n"
                f"Other valid labels:\n{other_labels}\n\n"
                f"Message:\n{text}\n\nLabel:"
            )
            messages = [{"role": "user", "content": prompt}]
            if len(prompt) <= 7600 or candidate_count <= 18:
                return messages
            candidate_count -= 4

    def _label_hint(self, label: str) -> str:
        cached = self._label_hints.get(label)
        if cached:
            return cached

        counts = self._Counter()
        counts.update(token for token in self._label_tokens.get(label, set()) if token not in self._STOPWORDS)
        for entry in self._by_label[label]:
            counts.update(token for token in entry["tokens"] if token not in self._STOPWORDS and len(token) > 1)

        scored = []
        for token, count in counts.items():
            scored.append((count * self._idf(token), token))
        scored.sort(reverse=True)
        words = [token.replace("_", " ") for _, token in scored[:8]]
        hint = ", ".join(words) if words else label.replace("_", " ")
        self._label_hints[label] = hint
        return hint

    def _extract_label(self, response: str) -> Optional[str]:
        if not response:
            return None
        cleaned = response.strip()
        lowered = cleaned.lower()

        for label in self._labels:
            if label == cleaned or label.lower() == lowered:
                return label
        for label in self._labels:
            if label.lower() in lowered:
                return label

        canonical_response = self._canonical_label(cleaned)
        lookup = {self._canonical_label(label): label for label in self._labels}
        return lookup.get(canonical_response)

    def _canonical_label(self, text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalnum())

    def _fallback_llm(self, text: str) -> Optional[str]:
        try:
            return (self.call_llm([
                {"role": "user", "content": f"Classify this banking support message. Return only the label.\n\n{text}\n\nLabel:"}
            ]) or "").strip()
        except Exception:
            return None
