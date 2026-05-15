"""Schema hints for database-aware RAG."""

import os


NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "RSDprocessingKG")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "Neo4j")


NODE_SCHEMAS = {
    "Algorithm": {
        "display": "算法",
        "fields": ["name", "uuid", "componentid", "enname", "description", "proname", "prochsname", "author", "componentclass"],
    },
    "Flow": {
        "display": "流程",
        "fields": ["name", "uuid", "componentid", "enname", "description", "proname", "prochsname", "author", "componentclass"],
    },
    "InputTensor": {
        "display": "输入张量",
        "fields": ["name", "uuid", "tensorclassid", "tensorid", "tensorname", "description", "component_c_uuid"],
    },
    "OutputTensor": {
        "display": "输出结果",
        "fields": ["name", "uuid", "tensorclassid", "tensorid", "tensorname", "description", "component_c_uuid"],
    },
}


RELATION_TYPES = {
    "input_tensor": ["输入张量"],
    "output_tensor": ["输出结果"],
    "components": ["使用"],
}


INTENT_FIELDS = {
    "query_enname": "enname",
    "query_author": "author",
    "query_description": "description",
}
