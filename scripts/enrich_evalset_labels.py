"""TechCompass — 基于真实文档生成高质量评测标注。

目标：
- 分类标注 >= 30
- 风险正样本 >= 15
- 实体标注 >= 30
- 每条 case 的 expected 来自可解释规则，不是手写臆造。

规则：
- category 根据标题中的产品词推导
- entities 根据标题中的品牌/产品词推导
- risk 根据隐私/安全/召回/质量问题词推导
- impact 根据 risk level 推导
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rag.store import RagStore

OUTPUT = Path("data/evalsets/enriched_pipeline_cases.jsonl")

CATEGORY_RULES = [
    ("headphones", re.compile(r"earbud|headphone|airpods|buds|linkbuds|momentum", re.I)),
    ("smartwatch", re.compile(r"smartwatch|watch|garmin|fitbit", re.I)),
    ("speaker", re.compile(r"speaker|soundlink|emberton|xboom", re.I)),
    ("keyboard", re.compile(r"keyboard|hyperx|logitech g|xtrfy", re.I)),
    ("mouse", re.compile(r"mouse|superstrike", re.I)),
    ("laptop", re.compile(r"laptop|cpu|battery problem", re.I)),
    ("smart_tv", re.compile(r"smart tv|roku|lg tv", re.I)),
    ("smart_home", re.compile(r"smart home|homekit|adt|security|camera|display", re.I)),
    ("battery", re.compile(r"battery|charging", re.I)),
]

BRAND_RULES = [
    ("Sennheiser", re.compile(r"sennheiser|momentum", re.I)),
    ("Sony", re.compile(r"sony|linkbuds|wh-1000", re.I)),
    ("JBL", re.compile(r"jbl", re.I)),
    ("Garmin", re.compile(r"garmin", re.I)),
    ("Samsung", re.compile(r"samsung|galaxy", re.I)),
    ("Fitbit", re.compile(r"fitbit", re.I)),
    ("Bose", re.compile(r"bose|quietcomfort", re.I)),
    ("Logitech", re.compile(r"logitech", re.I)),
    ("HyperX", re.compile(r"hyperx", re.I)),
    ("LG", re.compile(r"lg", re.I)),
    ("Roku", re.compile(r"roku", re.I)),
    ("Apple", re.compile(r"apple|airpods|homekit", re.I)),
    ("ADT", re.compile(r"adt", re.I)),
    ("Marshall", re.compile(r"marshall", re.I)),
]

RISK_POS_RULES = [
    re.compile(r"privacy|data collection|data leak|unauthorized|security risk|surveillance", re.I),
    re.compile(r"recall|safety|hazard|fire|shock|injury", re.I),
    re.compile(r"defect|defective|flawed|broken|not working|bricked", re.I),
    re.compile(r"smuggling|counterfeit|fake goods", re.I),
]


def infer_category(text: str) -> str:
    for category, pattern in CATEGORY_RULES:
        if pattern.search(text):
            return category
    return "unknown"


def infer_entities(text: str) -> list[str]:
    entities = []
    for brand, pattern in BRAND_RULES:
        if pattern.search(text):
            entities.append(brand)
    # 常见品类实体
    if re.search(r"earbud|headphone|buds", text, re.I):
        entities.append("earbuds")
    if re.search(r"smartwatch|watch", text, re.I):
        entities.append("smartwatch")
    if re.search(r"speaker", text, re.I):
        entities.append("speaker")
    if re.search(r"keyboard", text, re.I):
        entities.append("keyboard")
    if re.search(r"mouse", text, re.I):
        entities.append("mouse")
    if re.search(r"battery|charging", text, re.I):
        entities.append("battery")
    if re.search(r"privacy", text, re.I):
        entities.append("privacy")
    return sorted(set(entities))


def infer_risk(text: str) -> str:
    return "pos" if any(pattern.search(text) for pattern in RISK_POS_RULES) else "neg"


def infer_impact(risk: str) -> str:
    return "medium" if risk == "pos" else "low"


def main() -> None:
    store = RagStore()
    rows = store.conn.execute(
        "SELECT id, title, content FROM documents ORDER BY id"
    ).fetchall()
    store.close()

    seen = set()
    cases = []

    for row in rows:
        title = row["title"] or ""
        content = row["content"] or ""
        text = f"{title}\n{content}"

        # 去重
        key = (title + "|" + content[:200]).lower()
        if key in seen:
            continue
        seen.add(key)

        category = infer_category(text)
        entities = infer_entities(text)
        risk = infer_risk(text)
        sentiment = "negative" if risk == "pos" else "neutral"
        impact = infer_impact(risk)

        # question 使用标题的前几个关键词，避免过长
        keywords = re.findall(r"[A-Za-z0-9]+", title)
        question = " ".join(keywords[:6]) or title

        cases.append({
            "question": question,
            "expected": {
                "category": category,
                "sentiment": sentiment,
                "impact_level": impact,
                "entities": entities,
                "risk": risk,
            },
            "source_document_id": row["id"],
            "source_title": title,
        })

    with OUTPUT.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    valid_category = sum(c["expected"]["category"] != "unknown" for c in cases)
    risk_pos = sum(c["expected"]["risk"] == "pos" for c in cases)
    entity_annotated = sum(bool(c["expected"]["entities"]) for c in cases)

    print(f"total cases: {len(cases)}")
    print(f"valid category annotations: {valid_category}")
    print(f"risk positive cases: {risk_pos}")
    print(f"entity annotated cases: {entity_annotated}")
    print(f"output: {OUTPUT}")


if __name__ == "__main__":
    main()
