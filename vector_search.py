from vector_store import MilvusVectorStore
import os
# 1. 加载文本
with open(r"D:\lihao\data\bykg2508_text1.txt", "r", encoding="utf-8") as f:
    corpus = [line.strip() for line in f if line.strip()]
# C:\Users\mayij\Desktop\kg_to_text\aggregated_corpus.txt
# C:\Users\mayij\Desktop\kg_to_text\bykg2505_text.txt
# 2. 初始化并构建 Milvus 向量数据库
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "bge-large-zh")

store = MilvusVectorStore(
    collection_name="my_collection1",
    model_path=model_path
)

store.create_collection(corpus)  # 向量化 + 存储 + 建索引

# 示例问题测试
queries = [
    # "波段计算算法的输入张量有哪些？",
    # "波段计算算法的输入张量有哪些？"
    # "基于多边形影像裁切算法的输入张量有哪些？",
    # "基于多边形影像裁切算法的英文名是什么？",
    # "波段融合算法的英文名称是什么？",
    # "空间提取是什么算法生产的产品的中文名称？",
    # "4波段影像归一化植被指数流程是由哪些算法组成的？",
    # "生产重采样产品需要用到什么算法？",
    # "实现数据转换类型流程需要用到什么算法?",
    # "搭建图像融合流程需要用到什么流程和算法？",
    "请帮我搭建一个图像融合流程"
]
for query in queries:
    result = store.search(query, corpus=corpus, top_k=15)
    print("问题：", query)
## 正确答案：待定影像，产品名称，矢量图层名，空间参考类型，空间分辨率，待定自然波段，参考矢量，边界类型，
## 检索答案：参考矢量，边界类型，空间参考类型，矢量图层名，空间分辨率
    for item in result:
        # print(item)
        print(item["score"], item["content"])



