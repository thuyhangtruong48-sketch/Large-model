import time
import os
import numpy as np

from inverted_store import build_inverted_index, search_bm25_query
from vector_store import MilvusVectorStore
from collections import defaultdict

# 加载文本路径
file_path = r"D:\lihao\data\bykg2508_text1.txt"
# C:\Users\mayij\Desktop\kg_to_text\aggregated_corpus.txt
# C:\Users\mayij\Desktop\kg_to_text\bykg2505_text.txt


# 1. 加载文本
def load_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# 2. 初始化并构建 Milvus 向量数据库
def initialize_vector_store(collection_name, model_path, corpus_lines, recreate=False):
    store = MilvusVectorStore(collection_name=collection_name, model_path=model_path, recreate=recreate)
    # 仅在需要时创建集合
    if not store.collection_exists or recreate:
        store.create_collection(corpus_lines)
    else:
        store.load_collection()
    return store


# 3.构建倒排索引
def initialize_inverted_index(corpus_dict):
    return build_inverted_index(corpus_dict)


def calculate_entropy_weights(vector_results, keyword_results, kg_results=None, es_results=None):
    # 收集所有唯一文档内容
    all_contents = set()
    for result in vector_results:
        all_contents.add(result['content'])
    for result in keyword_results:
        all_contents.add(result['content'])

    n = len(all_contents)
    if n == 0:
        # 默认权重
        return {'vectorsearch': 0.34, 'keywordsearch': 0.33, 'essearch': 0.33}

    score_matrix = np.zeros((n, 3))  # 列: [vectorsearch, keywordsearch]
    content_list = list(all_contents)
    content_to_idx = {content: idx for idx, content in enumerate(content_list)}

    # 填充评分矩阵
    for result in vector_results:
        idx = content_to_idx[result['content']]
        score_matrix[idx, 0] = result['score']  # 向量检索分数

    for result in keyword_results:
        idx = content_to_idx[result['content']]
        score_matrix[idx, 1] = result['score']  # 关键词检索分数

    # 处理0值问题
    epsilon = 1e-4
    score_matrix = np.where(score_matrix == 0, epsilon, score_matrix)

    # Min-Max归一化
    min_vals = np.min(score_matrix, axis=0)
    max_vals = np.max(score_matrix, axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1  # 处理全0列
    normalized_matrix = (score_matrix - min_vals) / ranges

    # 计算比重矩阵
    p_matrix = normalized_matrix / np.sum(normalized_matrix, axis=0)

    # 计算信息熵
    p_matrix_safe = np.where(p_matrix == 0, epsilon, p_matrix)
    entropy_vals = -np.sum(p_matrix_safe * np.log(p_matrix_safe), axis=0) / np.log(n)

    # 计算权重
    diff_coeffs = 1 - entropy_vals
    weights = diff_coeffs / np.sum(diff_coeffs)

    print("weights[0]=", weights[0], "weights[1]=", weights[1])

    # 返回权重字典
    return {
        'vectorsearch': weights[0],
        'keywordsearch': weights[1]
    }


def weighted_fusion_results(vector_results, keyword_results, source_weights):
    # 融合2类检索结果，基于 content 去重 + 加权得分
    fusion_dict = defaultdict(lambda: {
        'content': '',
        'scores': {'vectorsearch': 0, 'keywordsearch': 0},
        'final_score': 0.0,
        'sources': []
    })

    for result in vector_results:
        content = result['content']
        fusion_dict[content]['content'] = content
        fusion_dict[content]['scores']['vectorsearch'] = result['score']
        fusion_dict[content]['sources'].append('vectorsearch')

    # 添加关键字检索结果
    for result in keyword_results:
        content = result['content']
        # 确保同一个内容不会被重复添加来源标签
        if content in fusion_dict:
            if 'keywordsearch' not in fusion_dict[content]['sources']:
                fusion_dict[content]['sources'].append('keywordsearch')
            # 更新分数
            fusion_dict[content]['scores']['keywordsearch'] = max(
                fusion_dict[content]['scores'].get('keywordsearch', 0),
                result['score']
            )
        else:
            fusion_dict[content]['content'] = content
            fusion_dict[content]['scores']['keywordsearch'] = result['score']
            fusion_dict[content]['sources'].append('keywordsearch')

    # 计算加权融合分数
    for key, entry in fusion_dict.items():
        # 确保每种检索类型的分数都在0-1之间
        vector_score = min(max(entry['scores']['vectorsearch'], 0.0), 1.0)
        keyword_score = min(max(entry['scores'].get('keywordsearch', 0.0), 0.0), 1.0)

        # 计算加权分数
        entry['final_score'] = (
                source_weights['vectorsearch'] * vector_score +
                source_weights['keywordsearch'] * keyword_score
        )

    # 转换为列表并排序
    sorted_results = sorted(fusion_dict.values(), key=lambda x: x['final_score'], reverse=True)

    # 只返回前 top_k 个结果
    return sorted_results


def query_and_fuse(query, corpus_lines, corpus_dict, store, bm25_params):
    vector_results = store.search(query, corpus=corpus_lines, top_k=15)
    keyword_results = search_bm25_query(query, *bm25_params)[:15]
    source_weights = calculate_entropy_weights(vector_results, keyword_results)
    fused_results = weighted_fusion_results(vector_results, keyword_results, source_weights)
    return vector_results, keyword_results, source_weights, fused_results


def display_fusion_results(query, vector_results, keyword_results, fused_results):
    print(f"\n【问题】：{query}")

    print("\n向量检索结果前15条:")
    for i, item in enumerate(vector_results[:15], 1):
        print(f"{i}. {item['content'][:80]}... (score: {item['score']:.4f})")

    print("\n关键字检索结果前15条:")
    for i, item in enumerate(keyword_results[:15], 1):
        print(f"{i}. {item['content'][:80]}... (score: {item['score']:.4f})")

    print("\n融合结果前20条:")
    for i, item in enumerate(fused_results[:20], 1):
        sources = "+".join(item['sources'])
        print(f"{i}. [{sources}] {item['content']}  (score: {item['final_score']:.4f})")


if __name__ == "__main__":
    file_path = r"D:\lihao\data\bykg2508_text123.txt"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "bge-large-zh")
    collection_name = "my_collection1"

    start_time = time.time()

    recreate_collection = True
    corpus_lines = load_corpus(file_path)
    corpus_dict = {i: text for i, text in enumerate(corpus_lines)}
    store = initialize_vector_store(collection_name, model_path, corpus_lines, recreate=recreate_collection)
    bm25_params = initialize_inverted_index(corpus_dict)

    # 示例问题测试
    queries = [
        # "波段计算算法的输入张量有哪些？",
        # "波段提取算法的输入张量有哪些？",
        # "基于多边形影像裁切算法的输入张量包括哪些？",
        # "基于多边形影像裁切算法的英文名是什么？",
        # "波段融合算法的英文名称是什么？",
        # "空间提取是什么算法生产的产品的中文名称？",
        # "4波段影像归一化植被指数流程是由哪些算法组成的？"
        # "生产重采样产品需要用到什么算法？",
        # "实现数据转换类型流程需要用到什么算法?",
        # "搭建图像融合流程需要用到什么流程？",
        "搭建2波段算法归一化植被指数流程需要用到什么算法？",
        # "请帮我搭建一个图像融合流程"
    ]
    ##正确答案：待定影像，产品名称，矢量图层名，空间参考类型，空间分辨率，待定自然波段，参考矢量，边界类型，
    ##待定自然波段没有检索出来
    for query in queries:
        vector_results, keyword_results, source_weights, fused_results= query_and_fuse(
            query, corpus_lines, corpus_dict, store, bm25_params)
        display_fusion_results(query, vector_results, keyword_results, fused_results)
        print(f"⏱️ 耗时: {round(time.time() - start_time, 2)} 秒")
