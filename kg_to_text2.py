###将知识图谱转换成聚合式自然语言文本语料
from neo4j import GraphDatabase
import json
import os

# === 配置你的 Neo4j 连接参数 ===
uri = "neo4j://localhost:7687"
user = "neo4j"
password = "RSDprocessingKG" # <-- 修改为你的密码
database_name = "Neo4j"

def s(x, default="未知"):
    """None -> default, 其它 -> str(x)"""
    return default if x is None else str(x)
# === 输出路径 ===
save_dir = r"D:\lihao\data\kg_to_text.txt"
os.makedirs(save_dir, exist_ok=True)


def get_node_type(labels):
    if "Algorithm" in labels:
        return "算法"
    elif "Flow" in labels:
        return "流程"
    elif "InputTensor" in labels:
        return "输入数据"
    elif "OutputTensor" in labels:
        return "输出结果"
    else:
        return "实体"


def node_attributes_to_text(node, node_type):
    props = dict(node)
    text_lines = []

    # 统一提取字段
    zh_name = props.get("name") or ""##算法/流程元件中文名；张量中文名
    en_name = props.get("enname") or props.get("tensorname") or ""##算法/流程元件英文名；张量英文名
    description = props.get("description") or ""##对算法/流程/张量的描述
    parent_zh = props.get("prochsname") or ""##算法/流程的产品中文名
    parent_en = props.get("proname") or ""##算法/流程的产品名称
    category = props.get("componentclass") or ""##算法/流程所属类别
    author = props.get("author") or ""  ##算法或流程的贡献者
    uuid = props.get("componentid") or ""  ##算法或流程的唯一id

    if node_type in ["算法", "流程"]:
        if zh_name and zh_name != "None":
            text_lines.append(f"{zh_name}（{uuid}）是一个{category}类型的{node_type}")
            if en_name and en_name != "None":
                text_lines.append(f"中文名为{zh_name}，英文名为{en_name}")
            if description:
                text_lines.append(f"其功能描述为：{description}")
            if author:
                text_lines.append(f"由{author}创建/贡献")
            if parent_zh or parent_en:
                text_lines.append(f"利用{zh_name}{node_type}生产出的产品的产品名称为{parent_zh}（{parent_en}）")
        else:
            if en_name and en_name != "None":
                text_lines.append(f"{en_name}（{uuid}）是一个{category}类型的{node_type}")
                if description:
                    text_lines.append(f"功能描述：{description}")
                if author:
                    text_lines.append(f"由{author}创建/贡献")
                if parent_zh or parent_en:
                    text_lines.append(f"利用{en_name}{node_type}生产出的产品的产品名称为{parent_zh}（{parent_en}）")

    elif node_type == "输入数据":
        if zh_name and zh_name != "None":
            text_lines.append(f"输入张量：{zh_name}")
            if en_name and en_name != "None":
                text_lines.append(f"英文名：{en_name}")
            if description:
                text_lines.append(f"说明描述：{description}")
        else:
            if en_name and en_name != "None":
                text_lines.append(f"输入张量：{en_name}")
                if description:
                    text_lines.append(f"说明描述：{description}")

    elif node_type == "输出结果":
        if zh_name and zh_name != "None":
            text_lines.append(f"输出结果：{zh_name}")
            if en_name and en_name != "None":
                text_lines.append(f"英文名：{en_name}")
            if description:
                text_lines.append(f"说明描述：{description}")
        else:
            if en_name and en_name != "None":
                text_lines.append(f"输出结果：{en_name}")
                if description:
                    text_lines.append(f"说明描述：{description}")

    return text_lines


def generate_aggregated_text(node, node_type, relations):
    """为单个实体生成聚合描述文本"""
    props = dict(node)
    zh_name = props.get("name") or props.get("tensorname") or "未知实体"
    uuid = props.get("componentid") or ""  ##算法或流程的唯一id

    # 基础属性文本
    attribute_lines = node_attributes_to_text(node, node_type)

    # 收集关系信息
    inputs = set()
    outputs = set()
    used_by = set()
    uses = set()

    for rel in relations:
        rel_type = rel["rel_type"]
        target_node = rel["target_node"]
        target_props = dict(target_node)
        target_name = target_props.get("name") or target_props.get("tensorname") or "未知实体"
        target_enname = target_props.get("enname") or target_props.get("tensorname") or target_props.get("")
        target_type = get_node_type(target_node.labels)
        terget_uuid = target_props.get("componentid") or target_props.get("")

        if rel_type == "输入张量":
            inputs.add(target_name)
        elif rel_type == "输出结果":
            outputs.add(target_name)
        elif rel_type == "使用":
            if target_type == "算法" or target_type == "流程":
                uses.add(s(target_type, "") + s(target_name, "") + "（" + s(terget_uuid) + "）")
        # 处理反向关系
        elif rel_type == "被使用":
            if target_type == "算法" or target_type == "流程":
                used_by.add(target_type+target_name+"（"+terget_uuid+"）")

    # 构建关系描述
    relation_lines = []
    if inputs:
        inputs_list = "、".join(inputs)
        relation_lines.append(f"{zh_name}（{uuid}）的输入张量包括：{inputs_list}")
    if outputs:
        outputs_list = "、".join(outputs)
        relation_lines.append(f"{zh_name}（{uuid}）的输出结果为：{outputs_list}")
    if uses:
        uses_list = "、".join(uses)
        relation_lines.append(f"实现/搭建{zh_name}{node_type}（{uuid}）需要使用以下组件（算法或流程）：{uses_list}")
    if used_by:
        used_by_list = "、".join(used_by)
        relation_lines.append(f"在搭建/实现{used_by_list}时使用了{zh_name}{node_type}（{uuid}）")

    # 合并所有文本
    all_lines = attribute_lines + relation_lines
    return "。".join(all_lines) + "。"


# === 主逻辑：从 Neo4j 拉取数据、生成聚合文本 ===
def export_aggregated_corpus():
    driver = GraphDatabase.driver(uri, auth=(user, password))
    corpus = []

    with driver.session(database=database_name) as session:
        # 查询所有节点及其关系
        query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->(m)
        WITH n, COLLECT({rel_type: TYPE(r), target_node: m}) AS out_relations
        OPTIONAL MATCH (p)-[r2]->(n)
        WITH n, out_relations, COLLECT({rel_type: "被使用", target_node: p}) AS in_relations
        RETURN n, out_relations + in_relations AS all_relations
        """
        results = session.run(query)

        for record in results:
            node = record["n"]
            relations = record["all_relations"]

            # 过滤空关系
            relations = [rel for rel in relations if rel["rel_type"] and rel["target_node"]]

            node_type = get_node_type(node.labels)
            text = generate_aggregated_text(node, node_type, relations)
            corpus.append(text)

    # 去重语料
    corpus = list(set(corpus))

    # 保存语料文本
    with open(os.path.join(save_dir, "bykg2508_text2.txt"), "w", encoding="utf-8") as f:
        for text in corpus:
            f.write(text + "\n")

    print(f"✅ 导出完成，生成聚合描述文本：{len(corpus)}条")


# === 运行 ===
if __name__ == "__main__":
    export_aggregated_corpus()