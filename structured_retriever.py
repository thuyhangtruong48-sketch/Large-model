"""Neo4j-first retriever for structured questions."""

from typing import Any, Dict, List, Optional, Tuple

from neo4j import GraphDatabase

from db_schema import INTENT_FIELDS, NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, RELATION_TYPES
from intent_parser import ParsedQuestion, parse_question


class StructuredRetriever:
    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
        database: str = NEO4J_DATABASE,
    ):
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def retrieve(self, question: str) -> Dict[str, Any]:
        parsed = parse_question(question)
        empty = {
            "hit": False,
            "source": "structured_neo4j",
            "parsed": parsed.__dict__,
            "entity": None,
            "rows": [],
            "answer": "",
            "cypher": "",
            "error": "",
        }
        if not parsed.intent or not parsed.entity:
            return empty

        try:
            entity = self._resolve_entity(parsed)
            if not entity:
                return empty
            rows, cypher = self._run_intent_query(parsed.intent, entity)
            if not rows:
                empty.update({"entity": entity, "cypher": cypher})
                return empty
            return {
                "hit": True,
                "source": "structured_neo4j",
                "parsed": parsed.__dict__,
                "entity": entity,
                "rows": rows,
                "answer": self._format_answer(parsed, entity, rows),
                "cypher": cypher,
                "error": "",
            }
        except Exception as exc:
            empty["error"] = str(exc)
            return empty

    def inspect_schema(self) -> Dict[str, Any]:
        return {
            "labels": self._run_read("CALL db.labels() YIELD label RETURN label ORDER BY label"),
            "relationship_types": self._run_read("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType"),
            "sample_keys": self._run_read("MATCH (n) RETURN labels(n) AS labels, keys(n) AS keys LIMIT 20"),
        }

    def _resolve_entity(self, parsed: ParsedQuestion) -> Optional[Dict[str, Any]]:
        labels = ["Algorithm", "Flow", "InputTensor", "OutputTensor"]
        if parsed.intent == "query_components":
            labels = ["Flow", "Algorithm"]
        elif parsed.entity_type_hint:
            labels = [parsed.entity_type_hint] + [label for label in labels if label != parsed.entity_type_hint]

        rows = self._search_nodes(parsed.entity, labels)
        if not rows:
            return None
        return self._choose_entity(rows, parsed)

    def _search_nodes(self, keyword: str, labels: List[str]) -> List[Dict[str, Any]]:
        cypher = """
        MATCH (n)
        WHERE any(label IN labels(n) WHERE label IN $labels)
          AND (
            coalesce(n.name, '') CONTAINS $keyword
            OR toLower(coalesce(n.enname, '')) CONTAINS toLower($keyword)
            OR toLower(coalesce(n.componentid, '')) CONTAINS toLower($keyword)
            OR toLower(coalesce(n.uuid, '')) CONTAINS toLower($keyword)
            OR toLower(coalesce(n.tensorname, '')) CONTAINS toLower($keyword)
          )
        RETURN elementId(n) AS element_id, labels(n) AS labels, properties(n) AS props
        LIMIT 10
        """
        return self._run_read(cypher, {"keyword": keyword, "labels": labels})

    def _choose_entity(self, rows: List[Dict[str, Any]], parsed: ParsedQuestion) -> Dict[str, Any]:
        if parsed.entity_type_hint:
            for row in rows:
                if parsed.entity_type_hint in row.get("labels", []):
                    return row
        if parsed.intent == "query_components":
            for row in rows:
                if "Flow" in row.get("labels", []):
                    return row
        return rows[0]

    def _run_intent_query(self, intent: str, entity: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
        if intent == "query_input_tensor":
            return self._query_related_tensors(entity, "InputTensor", RELATION_TYPES["input_tensor"])
        if intent == "query_output_tensor":
            return self._query_related_tensors(entity, "OutputTensor", RELATION_TYPES["output_tensor"])
        if intent == "query_components":
            return self._query_components(entity)
        if intent in INTENT_FIELDS:
            return self._query_property(entity, INTENT_FIELDS[intent])
        return [], ""

    def _query_related_tensors(self, entity: Dict[str, Any], tensor_label: str, rel_types: List[str]) -> Tuple[List[Dict[str, Any]], str]:
        cypher = f"""
        MATCH (n)-[r]->(t:{tensor_label})
        WHERE elementId(n) = $element_id AND type(r) IN $rel_types
        RETURN type(r) AS relationship, labels(t) AS labels, properties(t) AS props
        ORDER BY coalesce(t.name, t.tensorname, t.uuid)
        """
        return self._run_read(cypher, {"element_id": entity["element_id"], "rel_types": rel_types}), cypher

    def _query_components(self, entity: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
        cypher = """
        MATCH (n)-[r]->(child)
        WHERE elementId(n) = $element_id AND type(r) IN $rel_types
        RETURN type(r) AS relationship, labels(child) AS labels, properties(child) AS props
        ORDER BY coalesce(child.name, child.componentid, child.uuid)
        """
        return self._run_read(cypher, {"element_id": entity["element_id"], "rel_types": RELATION_TYPES["components"]}), cypher

    def _query_property(self, entity: Dict[str, Any], field: str) -> Tuple[List[Dict[str, Any]], str]:
        cypher = f"""
        MATCH (n)
        WHERE elementId(n) = $element_id
        RETURN n.{field} AS value
        """
        rows = [row for row in self._run_read(cypher, {"element_id": entity["element_id"]}) if row.get("value")]
        return rows, cypher

    def _format_answer(self, parsed: ParsedQuestion, entity: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
        props = entity.get("props", {}) or {}
        entity_name = props.get("name") or props.get("componentid") or props.get("uuid") or parsed.entity

        if parsed.intent in ("query_input_tensor", "query_output_tensor", "query_components"):
            items = []
            for row in rows:
                item_props = row.get("props", {}) or {}
                name = item_props.get("name") or item_props.get("tensorname") or item_props.get("componentid") or item_props.get("uuid")
                detail = item_props.get("tensorname") or item_props.get("componentid") or item_props.get("enname")
                desc = item_props.get("description")
                if detail and detail != name:
                    name = f"{name}（{detail}）"
                if desc:
                    name = f"{name}：{desc}"
                if name:
                    items.append(name)
            label = {"query_input_tensor": "输入张量", "query_output_tensor": "输出张量", "query_components": "组成组件"}[parsed.intent]
            return f"{entity_name}的{label}包括：{_join_cn(_dedupe(items))}。"

        label = {"query_enname": "英文名", "query_author": "作者", "query_description": "描述"}.get(parsed.intent, "属性")
        return f"{entity_name}的{label}是：{rows[0].get('value')}。"

    def _run_read(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        params = params or {}
        try:
            with self.driver.session(database=self.database) as session:
                return [record.data() for record in session.run(cypher, **params)]
        except Exception:
            with self.driver.session() as session:
                return [record.data() for record in session.run(cypher, **params)]


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _join_cn(values: List[str]) -> str:
    return "、".join(values) if values else "未查询到结果"
