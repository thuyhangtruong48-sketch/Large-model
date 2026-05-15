# ###将知识图谱转换成RAG所需的自然语言文本语料,一条一条的
from neo4j import GraphDatabase
import json
import os

# === 配置你的 Neo4j 连接参数 ===
uri = "neo4j://localhost:7687"
user = "neo4j"
password = "RSDprocessingKG" # <-- 修改为你的密码
database_name = "neo4j"

# === 输出路径 ===
save_dir = r"D:\lihao\SystemonRSDprocessingKG202510\output\kg_to_text"
os.makedirs(save_dir, exist_ok=True)


# === 添加的新函数：格式化节点名称（带uuid） ===
def format_node_name_with_uuid(node, node_type):
    """在算法和流程节点名称后添加uuid"""
    name = node.get("name", "未知节点")
    if node_type in ["算法", "流程"]:
        uuid = node.get("componentid", "未知ID")
        return (
            f"{name}{node_type}（{uuid}）",
            f"{node_type}{name}（{uuid}）"
        )
    return (name, name)


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


# === 构建自然语言模板 ===
# 多模板规则：每个关系类型对应多种语言表达方式
relation_templates = {
    "输入张量": {
        ("算法", "输入数据"): [
            "{source}的输入张量是{target}。",
            "{source}的输入数据之一是{target}。",
            "{target}是{source}的输入数据之一。",
            "{source}处理的原始数据之一包括{target}。",
            "{source}的输入张量数据之一包括{target}。",
            "{source}的输入张量之一包括{target}。",
            "{target}是{source}的输入张量数据之一。",
            "{source_alt}的输入张量是{target}。",
            "{source_alt}的输入数据之一是{target}。",
            "{target}是{source_alt}的输入数据之一。",
            "{source_alt}处理的原始数据之一包括{target}。",
            "{source_alt}的输入张量数据之一包括{target}。",
            "{source_alt}的输入张量之一包括{target}。",
            "{target}是{source_alt}的输入张量数据之一。"
        ],
        ("流程", "输入数据"): [
            "{source}的输入张量是{target}。",
            "{source}需要输入{target}。",
            "{target}是{source}的输入内容之一。",
            "{source}处理的原始数据之一包括{target}。",
            "{source}的输入张量数据之一包括{target}。",
            "{target}是{source}的输入张量数据之一。",
            "{source_alt}的输入张量是{target}。",
            "{source_alt}需要输入{target}。",
            "{target}是{source_alt}的输入内容之一。",
            "{source_alt}处理的原始数据之一包括{target}。",
            "{source_alt}的输入张量数据之一包括{target}。",
            "{target}是{source_alt}的输入张量数据之一。"
        ]
    },
    "输出结果": {
        ("算法", "输出结果"): [
            "{source}会生成{target}。",
            "{target}是{source}的输出文件。",
            "{source}的输出结果是{target}。",
            "{target}是由{source}处理之后产生的结果。",
            "{source}的处理结果为{target}。",
            "{source}的输出张量为{target}。",
            "{source_alt}会生成{target}。",
            "{target}是{source_alt}的输出文件。",
            "{source_alt}的输出结果是{target}。",
            "{target}是由{source_alt}处理之后产生的结果。",
            "{source_alt}的处理结果为{target}。",
            "{source_alt}的输出张量为{target}。"
        ],
        ("流程", "输出结果"): [
            "{source}会生成{target}。",
            "{target}是{source}的输出文件之一。",
            "{source}的输出结果之一是{target}。",
            "{target}是由{source}处理之后产生的结果之一。",
            "{source}的处理结果之一为{target}。",
            "{source}的输出张量为{target}。",
            "{source_alt}会生成{target}。",
            "{target}是{source_alt}的输出文件之一。",
            "{source_alt}的输出结果之一是{target}。",
            "{target}是由{source_alt}处理之后产生的结果之一。",
            "{source_alt}的处理结果之一为{target}。",
            "{source_alt}的输出张量为{target}。"
        ]
    },
    "使用": {
        ("流程", "算法"): [
            "{source}使用{target}。",
            "{target}组成{source}。",
            "{source}使用{target}完成遥感数据处理。",
            "{target}是{source}的引用。",
            "{target}是{source}中的一个组件。",
            "{target}被{source}所调用。",
            "{target}是{source}中的一个处理模块。",
            "{target}的执行是{source}成功运行的关键步骤。",
            "{target}是实现{source}所必需的处理模块之一。",
            "搭建{source}需要用到{target}。",

            "{source}的搭建需要使用{target}。",
            "实现{source}需要用到{target}。",

            "{source_alt}使用{target}。",
            "{target}组成{source_alt}。",
            "{source_alt}使用{target}完成遥感数据处理。",
            "{target}是{source_alt}的引用。",
            "{target}是{source_alt}中的一个组件。",
            "{target}被{source_alt}所调用。",
            "{target}是{source_alt}中的一个处理模块。",
            "{target}的执行是{source_alt}成功运行的关键步骤。",
            "{target}是实现{source_alt}所必需的处理模块之一。",
            "搭建{source_alt}需要用到{target}。",

            "{source_alt}的搭建需要使用{target}。",
            "实现{source_alt}需要用到{target}。"

        ],
        ("流程", "流程"): [
            "{source}使用{target}。",
            "{target}是{source}的组成部分之一。",
            "{target}组成{source}。",
            "{target}被{source}所调用。",
            "{target}的执行是{source}成功运行的关键步骤。",
            "{target}是实现{source}所必需的处理模块之一。",
            "搭建{source}需要用到{target}。",

            "{source}的搭建需要使用{target}。",
            "实现{source}需要用到{target}。",

            "{source_alt}使用{target}。",
            "{target}是{source_alt}的组成部分之一。",
            "{target}组成{source_alt}。",
            "{target}被{source_alt}所调用。",
            "{target}的执行是{source_alt}成功运行的关键步骤。",
            "{target}是实现{source_alt}所必需的处理模块之一。",
            "搭建{source_alt}需要用到{target}。",

            "{source_alt}的搭建需要使用{target}。",
            "实现{source_alt}需要用到{target}。"
        ]
    },
    # 其他关系可按需添加
}


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
        if zh_name and zh_name != "None" and en_name and en_name != "None":
            text_lines.append(f"{node_type}{zh_name}的ID为{uuid}。")
            text_lines.append(f"{zh_name}{node_type}的ID为{uuid}。")
            text_lines.append(f"{node_type}{zh_name}（{uuid}）的英文名为{en_name}。")
            text_lines.append(f"{node_type}{zh_name}（{uuid}）的英文名称为{en_name}。")
            text_lines.append(f"{zh_name}{node_type}（{uuid}）的英文名为{en_name}。")
            text_lines.append(f"{zh_name}{node_type}（{uuid}）的英文名称为{en_name}。")
            text_lines.append(f"{node_type}{en_name}（{uuid}）的中文名为{zh_name}。")
            text_lines.append(f"{zh_name}（{uuid}）的enname是{en_name}。")
            text_lines.append(f"{en_name}（{uuid}）的name是{zh_name}。")
        if description:
            text_lines.append(f"{node_type}{zh_name}（{uuid}）的功能描述为：{description}。")
            text_lines.append(f"{zh_name}{node_type}（{uuid}）的功能描述为：{description}。")
            text_lines.append(f"{node_type}{en_name}（{uuid}）的功能描述为：{description}。")
            text_lines.append(f"{zh_name}（{uuid}）的description是{description}。")
        if parent_zh or parent_en:
            text_lines.append(f"利用{zh_name}{node_type}（{uuid}）生产的产品中文名为{parent_zh}。")
            text_lines.append(f"利用{zh_name}{node_type}（{uuid}）生产的产品英文名称为{parent_en}。")
            text_lines.append(f"利用{node_type}{zh_name}（{uuid}）生产的产品中文名为{parent_zh}。")
            text_lines.append(f"利用{node_type}{zh_name}（{uuid}）生产的产品英文名称为{parent_en}。")
            text_lines.append(f"利用{zh_name}（{uuid}）的prochsname是{parent_zh}。")
            text_lines.append(f"利用{zh_name}（{uuid}）的proname是{parent_en}。")
        if category:
            text_lines.append(f"{node_type}{zh_name}（{uuid}）的类别为{category}。")
            text_lines.append(f"{node_type}{zh_name}（{uuid}）的类型为{category}。")
            text_lines.append(f"{zh_name}{node_type}（{uuid}）的类别为{category}。")
            text_lines.append(f"{zh_name}{node_type}（{uuid}）的类型为{category}。")
            text_lines.append(f"{node_type}{en_name}（{uuid}）的类别为{category}。")
            text_lines.append(f"{node_type}{en_name}（{uuid}）的类型为{category}。")
            text_lines.append(f"{zh_name}（{uuid}）的componentclass是{category}。")
        if author:
            text_lines.append(f"{node_type}{zh_name}（{uuid}）的贡献者为{author}。")
            text_lines.append(f"{node_type}{zh_name}（{uuid}）的创建者为{author}。")
            text_lines.append(f"{zh_name}{node_type}（{uuid}）的贡献者为{author}。")
            text_lines.append(f"{zh_name}{node_type}（{uuid}）的创建者为{author}。")
            text_lines.append(f"{node_type}{en_name}（{uuid}）的贡献者为{author}。")
            text_lines.append(f"{node_type}{en_name}（{uuid}）的创建者为{author}。")
            text_lines.append(f"{author}创建了{node_type}{zh_name}（{uuid}）。")
            text_lines.append(f"{author}贡献了{node_type}{zh_name}（{uuid}）。")
            text_lines.append(f"{author}创建了{zh_name}{node_type}（{uuid}）。")
            text_lines.append(f"{author}贡献了{zh_name}{node_type}（{uuid}）。")
            text_lines.append(f"{author}创建了{node_type}{en_name}（{uuid}）。")
            text_lines.append(f"{author}贡献了{node_type}{en_name}（{uuid}）。")
            text_lines.append(f"{zh_name}（{uuid}）的author是{author}。")

    elif node_type == "输入数据":
        if zh_name and zh_name != "None" and en_name and en_name != "None":
            text_lines.append(f"输入张量{zh_name}的英文名为{en_name}。")
            text_lines.append(f"输入张量{zh_name}的英文名称为{en_name}。")
            text_lines.append(f"输入数据{zh_name}的英文名为{en_name}。")
            text_lines.append(f"输入数据{zh_name}的英文名称为{en_name}。")
            text_lines.append(f"{zh_name}输入张量的英文名为{en_name}。")
            text_lines.append(f"{zh_name}输入张量的英文名称为{en_name}。")
            text_lines.append(f"{zh_name}输入数据的英文名为{en_name}。")
            text_lines.append(f"{zh_name}输入数据的英文名称为{en_name}。")
            text_lines.append(f"输入张量{en_name}的中文名为{zh_name}。")
            text_lines.append(f"输入张量{en_name}的中文名称为{zh_name}。")
        if description:
            text_lines.append(f"关于输入张量{zh_name}的说明：{description}。")
            text_lines.append(f"关于输入张量{zh_name}的描述：{description}。")
            text_lines.append(f"关于{zh_name}输入张量的说明：{description}。")
            text_lines.append(f"关于{zh_name}输入张量的描述：{description}。")
            text_lines.append(f"关于输入张量{en_name}的说明：{description}。")
            text_lines.append(f"关于输入张量{en_name}的描述：{description}。")

    elif node_type == "输出结果":
        if zh_name and zh_name != "None" and en_name and en_name != "None":
            text_lines.append(f"输出结果{zh_name}的英文名为{en_name}。")
            text_lines.append(f"输出结果{zh_name}的英文名称为{en_name}。")
            text_lines.append(f"输出张量{zh_name}的英文名为{en_name}。")
            text_lines.append(f"输出张量{zh_name}的英文名称为{en_name}。")
            text_lines.append(f"{zh_name}输出结果的英文名为{en_name}。")
            text_lines.append(f"{zh_name}输出结果的英文名称为{en_name}。")
            text_lines.append(f"{zh_name}输出张量的英文名为{en_name}。")
            text_lines.append(f"{zh_name}输出张量的英文名称为{en_name}。")
            text_lines.append(f"输出张量{en_name}的中文名为{zh_name}。")
            text_lines.append(f"输出张量{en_name}的中文名称为{zh_name}。")
        if description:
            text_lines.append(f"关于输出结果{zh_name}的说明：{description}。")
            text_lines.append(f"关于输出张量{zh_name}的描述：{description}。")
            text_lines.append(f"关于{zh_name}输出张量的说明：{description}。")
            text_lines.append(f"关于{zh_name}输出结果的描述：{description}。")
            text_lines.append(f"关于输出结果{en_name}的说明：{description}。")
            text_lines.append(f"关于输出结果{en_name}的描述：{description}。")

    return text_lines


def to_natural_language(source_node, source_type, relation, target_node, target_type):
    # 使用新函数格式化节点名称（添加uuid）
    source_style1, source_style2 = format_node_name_with_uuid(source_node, source_type)
    target_style1, target_style2 = format_node_name_with_uuid(target_node, target_type)

    key = (source_type, target_type)
    sentences = []
    if relation in relation_templates and key in relation_templates[relation]:
        templates = relation_templates[relation][key]
        for tpl in templates:
            formatted = tpl.format(
                source=source_style1,
                source_alt=source_style2,
                target=target_style1,
                target_alt=target_style2)
            sentences.append(formatted)
    else:
        # 没有匹配模板时默认模板
        sentences.append(f"{source_style1}的{relation}是{target_style1}。")
        sentences.append(f"{source_style2}的{relation}是{target_style1}。")
        sentences.append(f"{target_style1}是{source_style1}的{relation}。")
        sentences.append(f"{target_style1}是{source_style2}的{relation}。")
    return sentences


def get_composition_hierarchy(session, node_id):
    """
    递归函数，从一个给定的节点ID开始，沿着“使用”关系构建层级结构。
    """
    # 获取当前节点的信息
    result = session.run("MATCH (n) WHERE id(n) = $id RETURN n", id=node_id).single()
    if not result:
        return None
    node = result['n']
    node_name = node.get("name", "未知节点")
    node_type = get_node_type(node.labels)

    # 获取uuid（算法/流程节点）
    uuid = node.get("componentid", "") if node_type in ["算法", "流程"] else ""

    hierarchy = {
        "name": node_name,
        "type": node_type,
        "uuid": uuid,  # 保存uuid
        "children": []
    }

    # 查找并递归处理所有被“使用”的子节点
    children_results = session.run("""
        MATCH (parent)-[:使用]->(child)
        WHERE id(parent) = $id
        RETURN id(child) AS child_id
    """, id=node_id)

    for record in children_results:
        child_hierarchy = get_composition_hierarchy(session, record["child_id"])
        if child_hierarchy:
            hierarchy["children"].append(child_hierarchy)

    return hierarchy


def generate_chained_descriptions(hierarchy):
    """
    【核心新函数】
    根据层级结构，生成能够连接子层级的、链式的复合描述语句。
    """

    def format_node_display(node_info, style=1):
        """格式化节点显示（添加uuid）"""
        name = node_info["name"]
        node_type = node_info["type"]
        if node_type in ["算法", "流程"]:
            uuid = node_info["uuid"]
            if style == 1:
                return f'{name}{node_type}（{uuid}）'  # 格式1
            else:
                return f'{node_type}{name}（{uuid}）'  # 格式2
        return f'{node_type}{name}'

    # 内部递归函数，负责生成描述片段
    def build_clauses(sub_hierarchy):
        # 如果当前节点是叶子节点（没有子节点），则没有构成描述
        if not sub_hierarchy or not sub_hierarchy.get("children"):
            return ""

        # 获取直接子节点的描述列表
        child_descs = [f'{child["type"]}{child["name"]}({child["uuid"]})' for child in sub_hierarchy["children"]]
        children_str = "、".join(child_descs)

        # 构建当前层级的核心构成描述（例如“由...组成”）
        current_level_clause = f"由{children_str}搭建组成"

        # 递归地为有更深层结构的子节点生成“从句”
        follow_up_clauses = []
        for child in sub_hierarchy["children"]:
            # 对有孙子节点的子节点进行递归
            if child.get("children"):
                # 递归调用，获取子节点的构成描述
                child_composition_clause = build_clauses(child)
                # 拼接成一个完整的从句，例如“其中流程B又由算法a和算法b组成”
                follow_up_clauses.append(f'其中{child["type"]}{child["name"]}{child_composition_clause}')

        # 将当前层级的描述和所有从句连接起来
        if follow_up_clauses:
            return f"{current_level_clause}，" + "，".join(follow_up_clauses)
        else:
            return current_level_clause

    # 如果顶层没有子节点，直接返回空
    if not hierarchy or not hierarchy.get("children"):
        return []

    # 调用内部函数，为整个层级树生成完整的构成描述
    full_composition_str = build_clauses(hierarchy)

    if not full_composition_str:
        return []

    all_sentences = []
    for style in [1, 2]:  # 两种格式
        top_level_name = format_node_display(hierarchy, style)
        sentences = [
            f"{top_level_name}的构成是：{full_composition_str}。",
            f"搭建{top_level_name}需要：{full_composition_str}。",
            f"要实现{top_level_name}，其具体构成是：{full_composition_str}。",
            f"要搭建{top_level_name}，其具体搭建是：{full_composition_str}。",
        ]
        all_sentences.extend(sentences)

    return all_sentences


# === 主逻辑：从 Neo4j 拉取数据、保存 JSON、生成文本语料 ===
def export_graph_and_generate_corpus():
    driver = GraphDatabase.driver(uri, auth=(user, password))
    records = []
    corpus = []
    processed_nodes = set()

    with driver.session(database=database_name) as session:
        # 拉取所有三元组
        results = session.run("""
            MATCH (n)
            OPTIONAL MATCH (n)-[r]->(m)
            RETURN n, collect(r) AS rels, collect(m) AS targets
        """)

        for record in results:
            node = record["n"]
            rels = record["rels"]
            targets = record["targets"]
            node_id = node.id
            node_type = get_node_type(node.labels)
            node_name = node.get("name", "未知节点")
            node_id = node.get("componentid")

            if node_id not in processed_nodes:
                corpus.extend(node_attributes_to_text(node, node_type))
                processed_nodes.add(node_id)

            for rel, target in zip(rels, targets):
                if rel is None or target is None:
                    continue
                rel_type = rel.type
                target_name = target.get("name", "未知节点")
                target_type = get_node_type(target.labels)
                # 构造语料
                sentences = to_natural_language(node, node_type, rel_type, target, target_type)
                corpus.extend(sentences)

                # 保存为 JSON 格式的结构化数据（可选）
                records.append({
                    "name": node_name,
                    "source_type": node_type,
                    "id": node_id,
                    "relation": rel_type,
                    "target": target_name,
                    "target_type": target_type
                })

        # 步骤2: 查找顶层流程并生成层级描述语料
        print("🔍 正在查找顶层流程并生成层级描述...")
        hierarchical_corpus = []

        # Cypher 查询：找到所有类型为“流程”的节点，且这些节点没有指向它的“使用”关系
        top_level_flows = session.run("""
            MATCH (n:Flow)
            WHERE NOT ()-[:使用]->(n)
            RETURN id(n) AS node_id
        """)

        for record in top_level_flows:
            node_id = record["node_id"]
            # 为每个顶层流程构建其完整的层级结构
            hierarchy_tree = get_composition_hierarchy(session, node_id)
            # 根据层级结构生成描述性语句
            composition_sentences = generate_chained_descriptions(hierarchy_tree)
            hierarchical_corpus.extend(composition_sentences)

        # 步骤3: 将新生成的层级语料合并到总语料中
        corpus.extend(hierarchical_corpus)
        print("hierarchical_corpus：\n", hierarchical_corpus)
        print(f"✨ 已生成 {len(hierarchical_corpus)} 条层级描述语句。")

    # 去重语料
    corpus = list(set(corpus))

    # 保存 JSON 和语料文本
    with open(os.path.join(save_dir, "bykg2508.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    with open(os.path.join(save_dir, "bykg2508_text1.txt"), "w", encoding="utf-8") as f:
        for line in corpus:
            f.write(line + "\n")

    print(f"✅ 导出完成，结构化三元组数：{len(records)}，语料数：{len(corpus)}")


# === 运行 ===
if __name__ == "__main__":
    export_graph_and_generate_corpus()

