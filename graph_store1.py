from neo4j import GraphDatabase
from collections import defaultdict

# 连接 Neo4j
uri = "neo4j://localhost:7687"
database_name = "Neo4j"
driver = GraphDatabase.driver(uri, auth=("neo4j", "RSDprocessingKG"))


def get_flow_hierarchy(flow_name):
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

            hierarchy = expand_hierarchy(session, node_id)
            hierarchies.append((node_id, name, hierarchy))
    return hierarchies


def expand_hierarchy(session, root_id):
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


def hierarchy_to_natural_language(hierarchy, root):
    """
    将树结构转换为自然语言描述
    """
    node_id, name = root
    # 1. 获取直接子节点
    direct_children = hierarchy.get(root, [])
    if not direct_children:
        return f"流程{name}（{node_id}）没有子组件"

    # 2. 对直接子节点进行分类（流程/算法）
    flow_children = []
    algorithm_children = []

    for child in direct_children:
        child_id, child_name = child
        # 判断子节点是否是流程（根据是否有子节点）
        if child in hierarchy and hierarchy[child]:
            flow_children.append((child_id, child_name))
        else:
            algorithm_children.append((child_id, child_name))

    # 3. 构建描述语句
    parts = []

    # 3.1 描述直接子节点
    if flow_children or algorithm_children:
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

    # 3.2 描述每个流程子节点的组成
    for child_id, child_name in flow_children:
        # 获取子节点的子节点（孙子节点）
        grand_children = hierarchy.get((child_id, child_name), [])
        if not grand_children:
            continue

        # 对孙子节点进行分类（流程/算法）
        flow_grand_children = []
        algo_grand_children = []

        for grand_child in grand_children:
            grand_id, grand_name = grand_child
            # 判断孙子节点是否是流程（根据是否有子节点）
            if grand_child in hierarchy and hierarchy[grand_child]:
                flow_grand_children.append((grand_id, grand_name))
            else:
                algo_grand_children.append((grand_id, grand_name))

        # 构建子流程的描述
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


if __name__ == "__main__":
    flow_name = "图像融合"
    hierarchies = get_flow_hierarchy(flow_name)
    for idx, (node_id, name, hierarchy) in enumerate(hierarchies, 1):
        desc = hierarchy_to_natural_language(hierarchy, (node_id, name))
        # print(f"\n【搭建{name}（{node_id}）所需的算法或流程】")
        print(desc)
