import argparse
import json

from llm_client import call_llm, count_messages_tokens, count_tokens
from solution import MyHarness


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train_dev.jsonl")
    parser.add_argument("--dev", default="data/test_dev.jsonl")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    train = load_jsonl(args.train)
    dev = load_jsonl(args.dev)

    calls = {"n": 0}

    def traced_llm(messages):
        calls["n"] += 1
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        print(f"\n[LLM CALL #{calls['n']}] prompt_chars={prompt_chars}")
        return call_llm(messages)

    harness = MyHarness(traced_llm, count_tokens, count_messages_tokens, 2048)
    for item in train:
        harness.update(item["text"], item["label"])

    for idx, item in enumerate(dev[: args.limit]):
        pred = harness.predict(item["text"])
        print(f"{idx} {pred} -> {item['label']}")

    print(f"total_llm_calls = {calls['n']}")


if __name__ == "__main__":
    main()
