"""TechCompass — 多源实时采集 + 语义重排序演示/入口。

流程：
    用户 query
      → MultiSourceFetcher 多源拉取（HackerNews / GNews / Reddit）
      → 统一 schema + 去重
      → SemanticReranker 语义排序 + 阈值过滤
      → 输出 top N（含来源 + 链接 + 相似度）

用法：
    PYTHONPATH=src ./.venv311/bin/python scripts/fetch_multi_source.py "wireless earbuds charging" --top-k 5 --threshold 0.5
"""

import argparse
import json
from pathlib import Path

from adapters.multi_source import build_default_fetcher
from rag.semantic_rerank import SemanticReranker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--limit-per-source", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--json", action="store_true", help="输出完整 JSON")
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    fetcher = build_default_fetcher(env_path=str(PROJECT_ROOT / ".env"))
    result = fetcher.search(args.query, limit_per_source=args.limit_per_source)

    print("=" * 72)
    print("TechCompass 多源采集")
    print(f"query: {args.query}")
    print(f"原始抓取: {result.total_raw} 条 | 去重后: {result.total_deduped} 条")
    for name, count in result.per_source.items():
        print(f"  - {name}: {count}")
    for name, err in result.errors.items():
        print(f"  - {name} 错误: {err}")

    reranker = SemanticReranker()
    top = reranker.rerank(
        args.query,
        [d.to_dict() for d in result.documents],
        top_k=args.top_k,
        threshold=args.threshold,
    )

    print("=" * 72)
    print(f"语义重排序后 top {len(top)}（阈值 {args.threshold}）:")
    for i, d in enumerate(top, 1):
        print(f"[{i}] {d.get('title')}")
        print(f"    source={d.get('source')} | sim={d.get('_semantic_similarity')}")
        if d.get("url"):
            print(f"    {d.get('url')}")

    if args.json:
        print(json.dumps({
            "query": args.query,
            "total_raw": result.total_raw,
            "total_deduped": result.total_deduped,
            "per_source": result.per_source,
            "errors": result.errors,
            "top": top,
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
