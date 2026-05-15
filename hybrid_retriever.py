"""Structured-first hybrid retriever."""

from typing import Any, Dict

from structured_retriever import StructuredRetriever


class HybridRetriever:
    def __init__(self):
        self.structured = StructuredRetriever()

    def retrieve(self, question: str) -> Dict[str, Any]:
        structured = self.structured.retrieve(question)
        if structured.get("hit"):
            return {
                "route": "structured",
                "structured_results": structured,
                "fallback_required": False,
            }
        return {
            "route": "legacy_rag",
            "structured_results": structured,
            "fallback_required": True,
        }

    def inspect_schema(self) -> Dict[str, Any]:
        return self.structured.inspect_schema()
