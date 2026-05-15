"""Rule-based intent and entity parser for the structured RAG fast path."""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedQuestion:
    question: str
    intent: Optional[str]
    entity: str
    entity_type_hint: Optional[str] = None


INTENT_PATTERNS = [
    ("query_input_tensor", ["输入张量", "输入数据", "输入"]),
    ("query_output_tensor", ["输出张量", "输出结果", "输出"]),
    ("query_enname", ["英文名称", "英文名", "enname", "english name"]),
    ("query_author", ["创建者", "贡献者", "作者", "author"]),
    ("query_components", ["由哪些算法组成", "由哪些流程组成", "由哪些组件组成", "需要哪些算法", "使用哪些算法", "由什么组成", "组成"]),
    ("query_description", ["描述", "功能", "是干什么的", "有什么作用", "说明"]),
]

QUESTION_WORDS = ["是什么", "有哪些", "有哪几种", "多少", "谁", "吗", "呢", "请问", "帮我", "一下", "的"]
INTENT_WORDS = [word for _, words in INTENT_PATTERNS for word in words]
PUNCT_RE = re.compile(r"[，。！？；：、,.!?;:\(\)（）\[\]【】\"'“”‘’]")


def parse_question(question: str) -> ParsedQuestion:
    q = (question or "").strip()
    lower_q = q.lower()
    intent = None
    for name, keywords in INTENT_PATTERNS:
        if any(keyword.lower() in lower_q for keyword in keywords):
            intent = name
            break

    entity_type_hint = None
    if "流程" in q:
        entity_type_hint = "Flow"
    elif "算法" in q:
        entity_type_hint = "Algorithm"

    return ParsedQuestion(
        question=q,
        intent=intent,
        entity=_extract_entity(q),
        entity_type_hint=entity_type_hint,
    )


def _extract_entity(question: str) -> str:
    matches = re.findall(r"([\u4e00-\u9fa5A-Za-z0-9_\-]+?)(?:算法|流程)", question or "")
    if matches:
        return _clean_entity(matches[0])

    cleaned = question or ""
    for word in sorted(INTENT_WORDS + QUESTION_WORDS, key=len, reverse=True):
        cleaned = re.sub(re.escape(word), " ", cleaned, flags=re.IGNORECASE)
    cleaned = PUNCT_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return _clean_entity(cleaned)


def _clean_entity(value: str) -> str:
    value = PUNCT_RE.sub("", (value or "").strip())
    return re.sub(r"\s+", "", value)
