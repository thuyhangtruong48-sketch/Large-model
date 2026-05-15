# ###备份修改前的代码
# ###数据库更改后、只保留共有元件数据的程序，修改了关系类型的名称
import os
import json
import pymysql
from py2neo import Graph, Node
from collections import defaultdict
from dotenv import load_dotenv  # 推荐使用python-dotenv管理配置


class KnowledgeGraphBuilder:
    def __init__(self):
        # 初始化Neo4j连接
        self.g = Graph("neo4j://localhost:7687", auth=("neo4j", "RSDprocessingKG"), name="Neo4j")
        self.g.delete_all()
        self.uuid_mapping = {}  # 存储所有UUID到节点标签的映射

        # 初始化MySQL连接
        load_dotenv()  # 从.env文件加载配置
        self.mysql_conn = pymysql.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DATABASE"),
            charset='utf8mb4'
        )

    def build(self):
        try:
            components_data = self._fetch_mysql_data("component")
            print(f"[MYSQL] component rows = {len(components_data)}")
            if components_data:
                print("[MYSQL] component sample keys:", list(components_data[0].keys()))

            self._process_components(components_data)

            tensors_data = self._fetch_mysql_data("tensor")
            print(f"[MYSQL] tensor rows = {len(tensors_data)}")
            if tensors_data:
                print("[MYSQL] tensor sample keys:", list(tensors_data[0].keys()))

            self._process_tensors(tensors_data)
        finally:
            self.mysql_conn.close()

    def _fetch_mysql_data(self, table_name):
        """从MySQL指定表获取数据"""
        with self.mysql_conn.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = f"SELECT * FROM {table_name}"
            cursor.execute(sql)
            return cursor.fetchall()

    def _process_components(self, components_data):
        """处理组件数据"""
        relationships = defaultdict(list)

        for data in components_data:
            uuid = data["c_uuid"]

            # 确定节点类型
            node_type = self._get_component_type(uuid)
            self.uuid_mapping[uuid] = node_type

            # 创建节点
            self._create_component_node(node_type, data)

            # 收集关系
            if data.get("container_uuid"):
                container_type = self._get_component_type(data["container_uuid"])
                relationships[(container_type, node_type)].append((data["container_uuid"], uuid))

            # 创建组件间关系
        for (source_type, target_type), pairs in relationships.items():
            self._create_relationships(source_type, target_type, pairs, "使用")

    def _process_tensors(self, tensors_data):
        """处理张量数据"""
        for data in tensors_data:
            tensor_type = "InputTensor" if data["t_uuid"].startswith("I") else "OutputTensor"
            component_uuid = data["component_c_uuid"]

            # 创建张量节点
            self._create_tensor_node(tensor_type, data)

            # 创建与组件的关系
            if component_uuid in self.uuid_mapping:
                if tensor_type == "InputTensor":
                    rel_type = "输入张量"
                    self._create_relationship(self.uuid_mapping[component_uuid], tensor_type, component_uuid, data["t_uuid"], rel_type)
                else:
                    rel_type = "输出结果"
                    self._create_relationship(self.uuid_mapping[component_uuid], tensor_type, component_uuid, data["t_uuid"], rel_type)


    def _get_component_type(self, uuid):
        """根据UUID前缀确定组件类型"""
        prefix_map = {
            "Alg": "Algorithm",
            "Flow": "Flow",
            "S": "Algorithm",
            "F": "Flow",
        }
        for prefix, label in prefix_map.items():
            if uuid.startswith(prefix):
                return label
        return "Component"

    def _create_component_node(self, label, data):
        """创建组件节点"""
        node = Node(
            label,
            name=data.get("chsname", data["c_uuid"]),
            uuid=data["c_uuid"],
            container_uuid=data.get("container_uuid", ""),
            componentid=data.get("componentid", ""),
            enname=data.get("enname", ""),
            description=data.get("description", ""),
            proname=data.get("proname", ""),
            prochsname=data.get("prochsname", ""),
            author=data.get("author", ""),
            componentclass=data.get("componentclass", "")
        )
        self.g.create(node)

    def _create_tensor_node(self, label, data):
        """创建张量节点"""
        node = Node(
            label,
            name=data.get("tensorchsname", data["t_uuid"]),
            uuid=data["t_uuid"],
            tensorclassid=data.get("tensorclassid", ""),
            tensorid=data.get("tensorid", ""),
            tensorname=data.get("tensorname", ""),
            description=data.get("description", ""),
            component_c_uuid=data.get("component_c_uuid", "")
        )
        self.g.create(node)

    def _create_relationships(self, source_type, target_type, pairs, rel_type):
        """批量创建关系，跳过自循环"""
        # 过滤掉自循环的关系
        filtered_pairs = [pair for pair in pairs if pair[0] != pair[1]]
        if not filtered_pairs:
            return  # 没有有效关系则跳过

        query = """
        UNWIND $pairs AS pair
        MATCH (a:{source} {{uuid: pair[0]}}), (b:{target} {{uuid: pair[1]}})
        MERGE (a)-[:{rel}]->(b)
        """.format(source=source_type, target=target_type, rel=rel_type)

        self.g.run(query, parameters={"pairs": filtered_pairs})

    def _create_relationship(self, source_type, target_type, source_uuid, target_uuid, rel_type):
        """创建单个关系（跳过自循环）"""
        if source_uuid == target_uuid:
            return  # 跳过自循环

        query = """
        MATCH (a:{source} {{uuid: $source_uuid}})
        MATCH (b:{target} {{uuid: $target_uuid}})
        MERGE (a)-[:{rel}]->(b)
        """.format(source=source_type, target=target_type, rel=rel_type)

        self.g.run(query, source_uuid=source_uuid, target_uuid=target_uuid)

    def create_indexes(self):
        """为指定标签和属性创建索引"""
        index_specs = {
            "Algorithm": ["name", "enname", "proname", "prochsname"],
            "Flow": ["name", "enname", "proname", "prochsname"],
            "InputTensor": ["name", "tensorname"],
            "OutputTensor": ["name", "tensorname"],
        }

        for label, props in index_specs.items():
            for prop in props:
                index_name = f"{label.lower()}_{prop}_index"
                cypher = f""" CREATE INDEX {index_name} IF NOT EXISTS FOR (n:`{label}`) ON (n.{prop})"""
                try:
                    self.g.run(cypher)
                    print(f"✅ Created index: {index_name}")
                except Exception as e:
                    print(f"❌ Failed to create index {index_name}: {e}")


if __name__ == '__main__':
    builder = KnowledgeGraphBuilder()
    builder.build()
    builder.create_indexes()
