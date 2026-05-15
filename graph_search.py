from graph_store import query_knowledge_graph
import json

# 示例问题测试
queries = [
    # "数据类型转换流程的输入张量有些什么？"
    # "数据类型转换流程的输入张量包括哪些数据？",
    # "基于多边形影像裁切算法的英文名是什么？",
    # "影像无效值填充算法的英文名称是什么？",
    # "基于多边形影像裁切算法是由谁贡献的？",
    # "空间提取是什么算法生产的产品的中文名称？",
    # "4波段影像归一化植被指数流程是由哪些算法组成的？"
    # "生产图像融合产品需要用到什么流程？",
    # "实现数据类型转换流程需要用到什么算法?",
    # "搭建图像融合流程需要用到什么算法？",
    # "搭建图像融合流程需要用到什么流程？",
    # "搭建2波段算法归一化植被指数流程需要用到什么算法？",
    "请帮我搭建一个图像融合流程"

]

for query in queries:
    print(f"\n问题: {query}")

    # 调用知识图谱查询函数
    kg_result = query_knowledge_graph(query)

    # 打印Cypher查询
    print(f"生成的Cypher查询:\n{kg_result['cypher']}")

    # 打印结果
    if kg_result['triplets']:
        print("\n知识图谱结果:")
        print(json.dumps(kg_result['triplets'], ensure_ascii=False, indent=2))
        print(json.dumps(kg_result['objects'], ensure_ascii=False, indent=2))
    else:
        print("⚠️ 知识图谱未返回结果")

