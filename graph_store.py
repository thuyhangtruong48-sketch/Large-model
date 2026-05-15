# import re
# import ollama
# from neo4j import GraphDatabase
# import json
#
# # === Step 1: Prompt 模板 ===
# PROMPT_TEMPLATE = '''
# 你是一个专业的知识图谱问答助手。以下是一些例子，展示了如何将中文自然语言问题转换为Cypher语句，用于在Neo4j中查询图谱。
# 注意：产品不是一个节点，而是算法（Algorithm）或流程（Flow）的属性，如 prochsname 表示产品的中文名，proname表示产品的英文名。
# 每个例子都很重要，请全面参考。
#
# 例子1：
# 问题：生产融合产品需要用到什么流程？
# Cypher：MATCH (f:Flow) WHERE f.prochsname = "融合" OR f.proname = "ABC" RETURN DISTINCT f.name
#
# 例子1.1：
# 问题：生产归一化植被指数产品需要用到什么算法？
# Cypher：MATCH (a:Algorithm) WHERE a.prochsname = "归一化植被指数" OR a.proname = "ABC" RETURN DISTINCT a.name
#
# 例子1.2：
# 问题：实现融合流程需要用到什么算法?
# Cypher：MATCH (f:Flow {{name: '融合'}})-[:使用]->(a:Algorithm) RETURN DISTINCT a.name
#
# 例子2：
# 问题：请帮我搭建一个融合流程
# Cypher：MATCH (f:Flow {{name: '融合'}})-[:使用]->(f:Flow) RETURN DISTINCT f.name
#
# 例子3：
# 问题：融合算法的输入张量/输入数据有哪些？
# Cypher：MATCH (a:Algorithm {{name: '融合'}})-[:输入张量]->(t:InputTensor) RETURN DISTINCT t
#
# 例子4：
# 问题：融合流程的输入张量/输入数据有哪些？
# Cypher：MATCH (f:Flow {{name: '融合'}})-[:输入张量]->(t:InputTensor) RETURN DISTINCT t
#
# 例子5：
# 问题：融合算法的输出张量是什么？
# Cypher：MATCH (a:Algorithm {{name: '融合'}})-[:输出结果]->(t:OutputTensor) RETURN DISTINCT t
#
# 例子6：
# 问题：融合流程的输出张量是什么？
# Cypher：MATCH (f:Flow {{name: '融合'}})-[:输出结果]->(t:OutputTensor) RETURN DISTINCT t
#
# 例子7：
# 问题：融合流程由哪些算法组成？
# Cypher：MATCH (f:Flow {{name: '融合'}})-[:使用]->(a:Algorithm) RETURN DISTINCT a.name
#
# 例子8：
# 问题：融合算法的英文名/英文名称是什么？
# Cypher：MATCH (a:Algorithm {{name: '融合'}}) RETURN DISTINCT a.enname
#
# 例子9：
# 问题：融合流程的英文名/英文名称是什么？
# Cypher：MATCH (f:Flow {{name: '融合'}}) RETURN DISTINCT f.enname
#
# 例子10：
# 问题：融合算法的算法描述是什么？
# Cypher：MATCH (a:Algorithm {{name: '融合'}}) RETURN DISTINCT a.description
#
# 例子11：
# 问题：融合流程的算法描述是什么？
# Cypher：MATCH (f:Flow {{name: '融合'}}) RETURN DISTINCT f.description
#
# 例子12：
# 问题：融合算法生产的产品的产品名称是什么？
# Cypher：MATCH (a:Algorithm {{name: '融合'}}) RETURN DISTINCT a.prochsname
#
# 例子13：
# 问题：融合流程生产的产品的产品名称是什么？
# Cypher：MATCH (f:Flow {{name: '融合'}}) RETURN DISTINCT f.prochsname
#
# 例子14：
# 问题：融合算法生产的产品的产品英文名称是什么？
# Cypher：MATCH (a:Algorithm {{name: '融合'}}) RETURN DISTINCT a.proname
#
# 例子15：
# 问题：融合流程生产的产品的产品英文名称是什么？
# Cypher：MATCH (f:Flow {{name: '融合'}}) RETURN DISTINCT f.proname
#
# 例子16：
# 问题：融合算法是由谁贡献的？
# Cypher：MATCH (a:Algorithm {{name: '融合'}}) RETURN DISTINCT a.author
#
# 例子17：
# 问题：融合流程是由谁贡献的？
# Cypher：MATCH (f:Flow {{name: '融合'}}) RETURN DISTINCT f.author
#
# 例子18：
# 问题：融合算法属于什么类型？
# Cypher：MATCH (a:Algorithm {{name: '融合'}}) RETURN DISTINCT a.componentclass
#
# 例子19：
# 问题：融合流程属于什么类型？
# Cypher：MATCH (f:Flow {{name: '融合'}}) RETURN DISTINCT f.componentclass
#
# 例子20：
# 问题：辐射定标输入张量的英文名是什么？
# Cypher：MATCH (i:InputTensor {{name: '辐射定标'}}) RETURN DISTINCT i.tensorname
#
# 例子21：
# 问题：辐射定标输出张量的英文名是什么？
# Cypher：MATCH (o:OutputTensor {{name: '辐射定标'}}) RETURN DISTINCT o.tensorname
#
# 例子22：
# 问题：辐射定标输入张量的描述是什么？
# Cypher：MATCH (i:InputTensor {{name: '辐射定标'}}) RETURN DISTINCT i.description
#
# 例子23：
# 问题：辐射定标输出张量的描述是什么？
# Cypher：MATCH (o:OutputTensor {{name: '辐射定标'}}) RETURN DISTINCT o.description
#
# 例子24：
# 问题：融合流程由哪些流程组成？
# Cypher：MATCH (f:Flow {{name: '融合'}})-[:使用]->(f:Flow) RETURN DISTINCT f.name
#
# 现在请根据用户的问题，输出对应的Cypher查询语句。
# 问题：{user_question}
# Cypher：
# '''
#
#
# # === Step 2: 将中文问题转为 Cypher ===
# def question_to_cypher(user_question: str) -> str:
#     prompt = PROMPT_TEMPLATE.format(user_question=user_question)
#     try:
#         response = ollama.chat(
#             model='qwen2.5:14b',
#             messages=[{'role': 'user', 'content': prompt}]
#         )
#         answer = response['message']['content'].strip()
#         print("answer=", answer)
#         # 用正则提取 Cypher 代码块或返回内容
#         match = re.search(r"```cypher\s*(.*?)\s*```", answer, re.DOTALL)
#         if match:
#             return match.group(1).strip()
#         else:
#             # 尝试提取可能直接返回的Cypher语句
#             match = re.search(r"MATCH.*?RETURN.*", answer)
#             if match:
#                 return match.group(0).strip()
#             # 没有代码块，直接返回原始内容
#             return answer
#     except Exception as e:
#         print(f"❌ 模型推理失败: {e}")
#         return None
#
# # === 执行 Cypher 查询并返回两种格式的结果 ===
# def run_cypher_query(cypher_query: str):
#     uri = "neo4j://localhost:7687"
#     user = "neo4j"
#     password = "RSDprocessingKG"
#     database_name = "Neo4j"
#
#     # 从Cypher查询中提取算法名称
#     algorithm_name = None
#     match = re.search(r"{name: '(.+?)'}", cypher_query)
#     if match:
#         algorithm_name = match.group(1)
#
#     # 从Cypher查询中提取关系类型
#     relationship_type = None
#     rel_match = re.search(r'\[:?(\w+)\]', cypher_query)
#     if rel_match:
#         relationship_type = rel_match.group(1)
#
#     # 创建三元组列表和简易列表
#     triplets = []
#     object_list = []
#
#     try:
#         driver = GraphDatabase.driver(uri, auth=(user, password))
#         with driver.session(database=database_name) as session:
#             result = session.run(cypher_query)
#
#             seen_values = set()  # 用于去重
#
#             for record in result:
#                 record_data = record.data()
#                 value = None
#
#                 # 查找有效的结果值
#                 if 't' in record_data and isinstance(record_data['t'], dict):
#                     node = record_data['t']
#                     if 'name' in node:
#                         value = node['name']
#                     elif 'tensorname' in node:
#                         value = node['tensorname']
#                     elif 'enname' in node:
#                         value = node['enname']
#                     elif 'proname' in node:
#                         value = node['proname']
#
#                 # 处理直接返回值
#                 elif record_data:
#                     for key, val in record_data.items():
#                         if val and val not in (None, "", "N/A", "null"):
#                             value = str(val)
#                             break
#
#                 # 添加三元组格式（去除重复）
#                 if value and value not in seen_values:
#                     seen_values.add(value)
#
#                     # 创建三元组格式
#                     triplet_str = f"({algorithm_name}, {relationship_type}, {value})"
#                     triplets.append({
#                         "content": triplet_str,
#                         "score": 1.0,
#                         "source": "kgsearch"
#                     })
#
#                     # 同时添加到简化列表
#                     object_list.append(value)
#
#             return {
#                 "triplets": triplets,
#                 "objects": object_list
#             }
#
#     except Exception as e:
#         print(f"❌ 查询执行失败: {e}")
#         return {
#             "triplets": [],
#             "objects": []
#         }
#
#
# # === 新函数：执行知识图谱查询 ===
# def query_knowledge_graph(user_question: str):
#     # 生成Cypher查询
#     cypher = question_to_cypher(user_question)
#     if not cypher:
#         print(f"⚠️ 无法为问题生成Cypher查询: {user_question}")
#         return {
#             "question": user_question,
#             "cypher": "N/A",
#             # "results": []
#             "triplets": [],  # 三元组格式结果
#             "objects": []  # 简化列表格式结果
#         }
#
#     # 执行查询并获取格式化结果
#     results = run_cypher_query(cypher)
#
#     return {
#         "question": user_question,
#         "cypher": cypher,
#         # "results": results
#         "triplets": results["triplets"],  # 三元组格式
#         "objects": results["objects"]  # 简化列表格式
#     }


import re
import requests
import json
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
# ✅ 不强依赖 python-ollama 包：优先用它；没有就走 HTTP 调用本机 Ollama(11434)
try:
    import ollama  # 可能不存在
except Exception:
    ollama = None

import requests
import json

OLLAMA_BASE_URL = "http://127.0.0.1:11434"

def ollama_generate(model: str, prompt: str) -> str:
    """
    兼容两种方式：
    1) 安装了 python-ollama 包：直接调用
    2) 没安装 python-ollama：走 HTTP /api/generate
    """
    if ollama is not None:
        # python-ollama 的返回结构可能不同，这里做一个最常见的兼容
        resp = ollama.generate(model=model, prompt=prompt)
        return resp.get("response", "") if isinstance(resp, dict) else str(resp)

    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

from neo4j import GraphDatabase
from collections import defaultdict

# === Neo4j 连接配置 ===
uri = "neo4j://localhost:7687"
user = "neo4j"
password = "RSDprocessingKG"
database_name = "Neo4j"
driver = GraphDatabase.driver(uri, auth=(user, password))


# === 函数1: 知识图谱属性查询 ===
def query_knowledge_graph(user_question: str):
    """实现文档1的功能：根据自然语言问题查询知识图谱"""

    # === Step 1: Prompt 模板 ===
    PROMPT_TEMPLATE = '''
    你是一个专业的知识图谱问答助手。以下是一些例子，展示了如何将中文自然语言问题转换为Cypher语句，用于在Neo4j中查询图谱。
    注意：产品不是一个节点，而是算法（Algorithm）或流程（Flow）的属性，如 prochsname 表示产品的中文名，proname表示产品的英文名。
    每个例子都很重要，请全面参考。

    例子1：
    问题：生产融合产品需要用到什么流程？
    Cypher：MATCH (f:Flow) WHERE f.prochsname = "融合" OR f.proname = "ABC" RETURN DISTINCT f.name

    例子1.1：
    问题：生产归一化植被指数产品需要用到什么算法？
    Cypher：MATCH (a:Algorithm) WHERE a.prochsname = "归一化植被指数" OR a.proname = "ABC" RETURN DISTINCT a.name

    例子1.2：
    问题：实现融合流程需要用到什么算法?
    Cypher：MATCH (f:Flow {{name: '融合'}})-[:使用]->(a:Algorithm) RETURN DISTINCT a.name

    例子2：
    问题：请帮我搭建一个融合流程
    Cypher：MATCH (f:Flow {{name: '融合'}})-[:使用]->(f:Flow) RETURN DISTINCT f.name

    例子3：
    问题：融合算法的输入张量/输入数据有哪些？
    Cypher：MATCH (a:Algorithm {{name: '融合'}})-[:输入张量]->(t:InputTensor) RETURN DISTINCT t

    例子4：
    问题：融合流程的输入张量/输入数据有哪些？
    Cypher：MATCH (f:Flow {{name: '融合'}})-[:输入张量]->(t:InputTensor) RETURN DISTINCT t

    例子5：
    问题：融合算法的输出张量是什么？
    Cypher：MATCH (a:Algorithm {{name: '融合'}})-[:输出结果]->(t:OutputTensor) RETURN DISTINCT t

    例子6：
    问题：融合流程的输出张量是什么？
    Cypher：MATCH (f:Flow {{name: '融合'}})-[:输出结果]->(t:OutputTensor) RETURN DISTINCT t

    例子7：
    问题：融合流程由哪些算法组成？
    Cypher：MATCH (f:Flow {{name: '融合'}})-[:使用]->(a:Algorithm) RETURN DISTINCT a.name

    例子8：
    问题：融合算法的英文名/英文名称是什么？
    Cypher：MATCH (a:Algorithm {{name: '融合'}}) RETURN DISTINCT a.enname

    例子9：
    问题：融合流程的英文名/英文名称是什么？
    Cypher：MATCH (f:Flow {{name: '融合'}}) RETURN DISTINCT f.enname

    例子10：
    问题：融合算法的算法描述是什么？
    Cypher：MATCH (a:Algorithm {{name: '融合'}}) RETURN DISTINCT a.description

    例子11：
    问题：融合流程的算法描述是什么？
    Cypher：MATCH (f:Flow {{name: '融合'}}) RETURN DISTINCT f.description

    例子12：
    问题：融合算法生产的产品的产品名称是什么？
    Cypher：MATCH (a:Algorithm {{name: '融合'}}) RETURN DISTINCT a.prochsname

    例子13：
    问题：融合流程生产的产品的产品名称是什么？
    Cypher：MATCH (f:Flow {{name: '融合'}}) RETURN DISTINCT f.prochsname

    例子14：
    问题：融合算法生产的产品的产品英文名称是什么？
    Cypher：MATCH (a:Algorithm {{name: '融合'}}) RETURN DISTINCT a.proname

    例子15：
    问题：融合流程生产的产品的产品英文名称是什么？
    Cypher：MATCH (f:Flow {{name: '融合'}}) RETURN DISTINCT f.proname

    例子16：
    问题：融合算法是由谁贡献的？
    Cypher：MATCH (a:Algorithm {{name: '融合'}}) RETURN DISTINCT a.author

    例子17：
    问题：融合流程是由谁贡献的？
    Cypher：MATCH (f:Flow {{name: '融合'}}) RETURN DISTINCT f.author

    例子18：
    问题：融合算法属于什么类型？
    Cypher：MATCH (a:Algorithm {{name: '融合'}}) RETURN DISTINCT a.componentclass

    例子19：
    问题：融合流程属于什么类型？
    Cypher：MATCH (f:Flow {{name: '融合'}}) RETURN DISTINCT f.componentclass

    例子20：
    问题：辐射定标输入张量的英文名是什么？
    Cypher：MATCH (i:InputTensor {{name: '辐射定标'}}) RETURN DISTINCT i.tensorname

    例子21：
    问题：辐射定标输出张量的英文名是什么？
    Cypher：MATCH (o:OutputTensor {{name: '辐射定标'}}) RETURN DISTINCT o.tensorname

    例子22：
    问题：辐射定标输入张量的描述是什么？
    Cypher：MATCH (i:InputTensor {{name: '辐射定标'}}) RETURN DISTINCT i.description

    例子23：
    问题：辐射定标输出张量的描述是什么？
    Cypher：MATCH (o:OutputTensor {{name: '辐射定标'}}) RETURN DISTINCT o.description

    例子24：
    问题：融合流程由哪些流程组成？
    Cypher：MATCH (f:Flow {{name: '融合'}})-[:使用]->(f:Flow) RETURN DISTINCT f.name

    现在请根据用户的问题，输出对应的Cypher查询语句。
    问题：{user_question}
    Cypher：
    '''

    def question_to_cypher(user_question: str) -> str:
        """将中文问题转换为Cypher查询语句"""
        prompt = PROMPT_TEMPLATE.format(user_question=user_question)
        try:
            url = f"{OLLAMA_BASE_URL}/api/chat"
            payload = {
                "model": "qwen2.5:14b",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
            r = requests.post(url, json=payload, timeout=600)
            r.raise_for_status()
            response = r.json()

            # Ollama /api/chat 返回结构一般是 {"message":{"content":...}}
            answer = response.get("message", {}).get("content", "").strip()

            # 用正则提取 Cypher 代码块或返回内容
            match = re.search(r"```cypher\s*(.*?)\s*```", answer, re.DOTALL)
            if match:
                return match.group(1).strip()
            else:
                # 尝试提取可能直接返回的Cypher语句
                match = re.search(r"MATCH.*?RETURN.*", answer)
                if match:
                    return match.group(0).strip()
                # 没有代码块，直接返回原始内容
                return answer
        except Exception as e:
            print(f"❌❌ 模型推理失败: {e}")
            return None

    def run_cypher_query(cypher_query: str):
        """执行Cypher查询并返回结果"""
        # 从Cypher查询中提取算法名称
        algorithm_name = None
        match = re.search(r"{name: '(.+?)'}", cypher_query)
        if match:
            algorithm_name = match.group(1)

        # 从Cypher查询中提取关系类型
        relationship_type = None
        rel_match = re.search(r'\[:?(\w+)\]', cypher_query)
        if rel_match:
            relationship_type = rel_match.group(1)

        # 创建三元组列表和简易列表
        triplets = []
        object_list = []

        try:
            with driver.session(database=database_name) as session:
                result = session.run(cypher_query)

                seen_values = set()  # 用于去重

                for record in result:
                    record_data = record.data()
                    value = None

                    # 查找有效的结果值
                    if 't' in record_data and isinstance(record_data['t'], dict):
                        node = record_data['t']
                        if 'name' in node:
                            value = node['name']
                        elif 'tensorname' in node:
                            value = node['tensorname']
                        elif 'enname' in node:
                            value = node['enname']
                        elif 'proname' in node:
                            value = node['proname']

                    # 处理直接返回值
                    elif record_data:
                        for key, val in record_data.items():
                            if val and val not in (None, "", "N/A", "null"):
                                value = str(val)
                                break

                    # 添加三元组格式（去除重复）
                    if value and value not in seen_values:
                        seen_values.add(value)

                        # 创建三元组格式
                        triplet_str = f"({algorithm_name}, {relationship_type}, {value})"
                        triplets.append({
                            "content": triplet_str,
                            "score": 1.0,
                            "source": "kgsearch"
                        })

                        # 同时添加到简化列表
                        object_list.append(value)

                return {
                    "triplets": triplets,
                    "objects": object_list
                }

        except Exception as e:
            print(f"❌❌ 查询执行失败: {e}")
            return {
                "triplets": [],
                "objects": []
            }

    # 主逻辑
    cypher = question_to_cypher(user_question)
    if not cypher:
        print(f"⚠️ 无法为问题生成Cypher查询: {user_question}")
        return {
            "question": user_question,
            "cypher": "N/A",
            "triplets": [],
            "objects": []
        }

    results = run_cypher_query(cypher)
    return {
        "question": user_question,
        "cypher": cypher,
        "triplets": results["triplets"],
        "objects": results["objects"]
    }


# === 函数2: 流程层级查询 ===
def get_flow_hierarchy_natural_language(flow_name: str):
    """实现文档2的功能：获取流程层级结构并转换为自然语言描述"""

    def _get_flow_hierarchy(flow_name):
        """获取流程的层级结构"""
        query = """
        MATCH (root:Flow {name:$flow_name})
        RETURN DISTINCT root.componentid as node_id, root.name as name
        """
        hierarchies = []
        seen = set()  # 去重

        with driver.session(database=database_name) as session:
            results = session.run(query, flow_name=flow_name)
            for record in results:
                node_id = record["node_id"]
                name = record["name"]
                if node_id in seen:
                    continue
                seen.add(node_id)

                hierarchy = _expand_hierarchy(session, node_id)
                hierarchies.append((node_id, name, hierarchy))
        return hierarchies

    def _expand_hierarchy(session, root_id):
        """扩展层级结构"""
        query = """
            MATCH path=(root {componentid:$root_id})-[:使用*1..]->(n)
            RETURN [x IN nodes(path) | {id:x.componentid, name:x.name}] as nodes
            """
        hierarchy = defaultdict(list)
        results = session.run(query, root_id=root_id)

        for record in results:
            nodes = record["nodes"]
            for i in range(len(nodes) - 1):
                parent = (nodes[i]["id"], nodes[i]["name"])
                child = (nodes[i + 1]["id"], nodes[i + 1]["name"])
                if child not in hierarchy[parent]:
                    hierarchy[parent].append(child)
        return hierarchy

    def _hierarchy_to_natural_language(hierarchy, root):
        """将层级结构转换为自然语言描述"""
        node_id, name = root
        direct_children = hierarchy.get(root, [])
        if not direct_children:
            return f"流程{name}（{node_id}）没有子组件"

        flow_children = []
        algorithm_children = []

        for child in direct_children:
            child_id, child_name = child
            if child in hierarchy and hierarchy[child]:
                flow_children.append((child_id, child_name))
            else:
                algorithm_children.append((child_id, child_name))

        parts = []
        direct_desc = f"流程{name}（{node_id}）包含"

        if flow_children:
            flow_items = [f"流程{child_name}（{child_id}）" for child_id, child_name in flow_children]
            direct_desc += "、".join(flow_items)

            if algorithm_children:
                direct_desc += "和"

        if algorithm_children:
            algo_items = [f"算法{child_name}（{child_id}）" for child_id, child_name in algorithm_children]
            direct_desc += "、".join(algo_items)

        parts.append(direct_desc)

        for child_id, child_name in flow_children:
            grand_children = hierarchy.get((child_id, child_name), [])
            if not grand_children:
                continue

            flow_grand_children = []
            algo_grand_children = []

            for grand_child in grand_children:
                grand_id, grand_name = grand_child
                if grand_child in hierarchy and hierarchy[grand_child]:
                    flow_grand_children.append((grand_id, grand_name))
                else:
                    algo_grand_children.append((grand_id, grand_name))

            grand_desc = f"流程{child_name}（{child_id}）由"

            if flow_grand_children:
                flow_grand_items = [f"流程{grand_name}（{grand_id}）" for grand_id, grand_name in flow_grand_children]
                grand_desc += "、".join(flow_grand_items)

                if algo_grand_children:
                    grand_desc += "和"

            if algo_grand_children:
                algo_grand_items = [f"算法{grand_name}（{grand_id}）" for grand_id, grand_name in algo_grand_children]
                grand_desc += "、".join(algo_grand_items)

            grand_desc += "组成"
            parts.append(grand_desc)

        return "，".join(parts) + "。"

    # 主逻辑
    hierarchies = _get_flow_hierarchy(flow_name)
    descriptions = []
    for node_id, name, hierarchy in hierarchies:
        desc = _hierarchy_to_natural_language(hierarchy, (node_id, name))
        descriptions.append(desc)
    return descriptions


# # === 测试代码 ===
# if __name__ == "__main__":
#     # 测试知识图谱查询
#     test_question = "RPC校正算法的输入张量有哪些？"
#     kg_result = query_knowledge_graph(test_question)
#     print(f"知识图谱查询结果: {kg_result['objects']}")
#
#     # 测试流程层级查询
#     flow_name = "图像融合"
#     flow_hierarchy = get_flow_hierarchy_natural_language(flow_name)
#     for desc in flow_hierarchy:
#         print(f"流程搭建步骤描述: {desc}")
