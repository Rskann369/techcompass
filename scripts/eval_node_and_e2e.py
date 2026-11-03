"""TechCompass — 节点级 + 端到端评估入口（真实 LLM）。"""
import json

from eval.node_eval import run_node_eval
from eval.end_to_end_eval import run_end_to_end_eval


def main():
    result = {
        "node_eval": run_node_eval(),
        "end_to_end": run_end_to_end_eval(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
