"""TechCompass — 真实 LLM 主链路入口。

流程（编排与分析均用真实 LLM）：
    用户 query
      → LangGraph create_react_agent（真实 LLM 编排：检索 / 现场拉取 / 分析 / 报告）
      → analyze_consumer_sentiment（真实 LLM 开放域结构化分析）
      → 结论解读（headline + summary + 带链接依据 + 置信度）
      → 周报 / 告警 / 归档 + trace 持久化

用法：
    PYTHONPATH=src ./.venv311/bin/python scripts/run_main_pipeline.py "smart home camera privacy"
    PYTHONPATH=src ./.venv311/bin/python scripts/run_main_pipeline.py "..." --max-steps 10
"""

import argparse
import json
from pathlib import Path

from agents.langgraph_react_agent import LangGraphReactAgent


def main():
    parser = argparse.ArgumentParser(description="TechCompass real-LLM main pipeline")
    parser.add_argument("question")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    print("=" * 72, flush=True)
    print("TechCompass Main Pipeline (real LLM)", flush=True)
    print(f"Question: {args.question}", flush=True)
    print("=" * 72, flush=True)

    agent = LangGraphReactAgent(max_steps=args.max_steps)
    result = agent.run(args.question)

    conclusion = result.get("conclusion")
    output = {
        "ok": result.get("ok"),
        "agent_type": result.get("agent_type"),
        "question": result.get("question"),
        "answer": result.get("answer"),
        "conclusion": conclusion,
        "evidence": [
            {
                "title": d.get("title"),
                "source": d.get("source"),
                "source_type": d.get("source_type"),
                "similarity": d.get("_semantic_similarity"),
                "url": d.get("url"),
                "content": (d.get("content") or d.get("text") or "")[:500],
            }
            for d in result.get("relevant_evidence", [])
        ],
        "analysis": result.get("analysis"),
        "metrics": result.get("metrics"),
        "run_id": result.get("run_id"),
    }

    if conclusion:
        print("\n" + "=" * 72)
        print("舆情结论（运营可读）")
        print("=" * 72)
        print(f"结论：{conclusion['headline']}")
        print(f"正文：{conclusion['summary']}")
        print(f"情绪：{conclusion['sentiment']} | 风险：{conclusion['risk_level']} | 影响：{conclusion['impact_level']}")
        print(f"置信度：{conclusion['confidence']['level']}（{conclusion['confidence']['score']}）")
        print("依据（含链接）：")
        for e in conclusion["evidence_basis"]:
            print(f"  [{e['ref']}] {e['title']}")
            print(f"      {e['source_label']} · 相关性 {e['relevance']}（{e['similarity']}）")
            if e["url"]:
                print(f"      {e['url']}")
        if conclusion["actions"]:
            print("行动建议：")
            for a in conclusion["actions"]:
                print(f"  - {a}")
    else:
        print("\n（未生成结构化结论：Agent 未调用 analyze_consumer_sentiment。）")

    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"\nSaved result to: {path}")
    else:
        print("\n" + text)


if __name__ == "__main__":
    main()
