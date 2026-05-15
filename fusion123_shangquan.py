import time
import math
from collections import defaultdict
import numpy as np

# ✅ 加上 es_search_query
from inverted_store import build_inverted_index, search_bm25_query, es_search_query
from vector_store import MilvusVectorStore
from graph_store import query_knowledge_graph  # 确保你有这个模块


def load_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def initialize_vector_store(collection_name, model_path, corpus_lines, recreate=False):
    store = MilvusVectorStore(collection_name=collection_name, model_path=model_path, recreate=recreate)
    # 仅在需要时创建集合
    if not store.collection_exists or recreate:
        store.create_collection(corpus_lines)
    else:
        store.load_collection()
    return store


def initialize_inverted_index(corpus_dict):
    return build_inverted_index(corpus_dict)


def calculate_entropy_weights(kg_results, vector_results, keyword_results, es_results):
    # 收集所有唯一文档内容
    all_contents = set()

    # 从知识图谱结果中收集内容
    for triplet in kg_results.get('triplets', []):
        all_contents.add(triplet['content'])

    # 从向量检索结果中收集内容
    for result in vector_results:
        all_contents.add(result['content'])

    # 从关键词检索结果中收集内容
    for result in keyword_results:
        all_contents.add(result['content'])

    # ✅ 新增：从ES检索结果中收集内容
    for result in es_results:
        all_contents.add(result['content'])

    n = len(all_contents)
    if n == 0:
        # ✅ 默认权重：4路
        return {'kgsearch': 0.25, 'vectorsearch': 0.25, 'keywordsearch': 0.25, 'essearch': 0.25}

    # ✅ 3列 -> 4列
    score_matrix = np.zeros((n, 4))  # 列: [kg, vector, keyword, es]
    content_list = list(all_contents)
    content_to_idx = {content: idx for idx, content in enumerate(content_list)}

    # 填充评分矩阵
    for triplet in kg_results.get('triplets', []):
        idx = content_to_idx[triplet['content']]
        score_matrix[idx, 0] = triplet['score']  # 知识图谱分数

    for result in vector_results:
        idx = content_to_idx[result['content']]
        score_matrix[idx, 1] = result['score']  # 向量检索分数

    for result in keyword_results:
        idx = content_to_idx[result['content']]
        score_matrix[idx, 2] = result['score']  # 关键词检索分数

    # ✅ 新增：ES分数填充到第4列
    for result in es_results:
        idx = content_to_idx[result['content']]
        score_matrix[idx, 3] = result['score']  # ES检索分数

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
    col_sums = np.sum(normalized_matrix, axis=0)
    col_sums[col_sums == 0] = 1
    p_matrix = normalized_matrix / col_sums

    # 计算信息熵
    p_matrix_safe = np.where(p_matrix == 0, epsilon, p_matrix)
    entropy_vals = -np.sum(p_matrix_safe * np.log(p_matrix_safe), axis=0) / np.log(n)

    # 计算权重
    diff_coeffs = 1 - entropy_vals
    if np.sum(diff_coeffs) == 0:
        # 极端情况：全一样
        return {'kgsearch': 0.25, 'vectorsearch': 0.25, 'keywordsearch': 0.25, 'essearch': 0.25}

    weights = diff_coeffs / np.sum(diff_coeffs)

    # 返回权重字典
    return {
        'kgsearch': float(weights[0]),
        'vectorsearch': float(weights[1]),
        'keywordsearch': float(weights[2]),
        'essearch': float(weights[3]),
    }


def weighted_fusion_results(kg_results, vector_results, keyword_results, es_results, source_weights):
    # 融合4类检索结果，基于 content 去重 + 加权得分
    fusion_dict = defaultdict(lambda: {
        'content': '',
        'scores': {'kgsearch': 0, 'vectorsearch': 0, 'keywordsearch': 0, 'essearch': 0},
        'final_score': 0.0,
        'sources': []
    })

    # 添加知识图谱结果
    for triplet in kg_results.get('triplets', []):
        content = triplet['content']
        fusion_dict[content]['content'] = content
        fusion_dict[content]['scores']['kgsearch'] = triplet['score']
        if 'kgsearch' not in fusion_dict[content]['sources']:
            fusion_dict[content]['sources'].append('kgsearch')

    # 添加向量检索结果
    for result in vector_results:
        content = result['content']
        fusion_dict[content]['content'] = content
        fusion_dict[content]['scores']['vectorsearch'] = result['score']
        if 'vectorsearch' not in fusion_dict[content]['sources']:
            fusion_dict[content]['sources'].append('vectorsearch')

    # 添加关键字检索结果
    for result in keyword_results:
        content = result['content']
        fusion_dict[content]['content'] = content
        fusion_dict[content]['scores']['keywordsearch'] = result['score']
        if 'keywordsearch' not in fusion_dict[content]['sources']:
            fusion_dict[content]['sources'].append('keywordsearch')

    # ✅ 新增：添加ES检索结果
    for result in es_results:
        content = result['content']
        fusion_dict[content]['content'] = content
        fusion_dict[content]['scores']['essearch'] = result['score']
        if 'essearch' not in fusion_dict[content]['sources']:
            fusion_dict[content]['sources'].append('essearch')

    # 计算加权融合分数
    for _, entry in fusion_dict.items():
        # 确保每种检索类型的分数都在0-1之间
        kg_score = min(max(entry['scores']['kgsearch'], 0.0), 1.0)
        vector_score = min(max(entry['scores']['vectorsearch'], 0.0), 1.0)
        keyword_score = min(max(entry['scores']['keywordsearch'], 0.0), 1.0)
        es_score = min(max(entry['scores']['essearch'], 0.0), 1.0)

        entry['final_score'] = (
            source_weights['kgsearch'] * kg_score +
            source_weights['vectorsearch'] * vector_score +
            source_weights['keywordsearch'] * keyword_score +
            source_weights['essearch'] * es_score
        )

    # 转换为列表并排序
    sorted_results = sorted(fusion_dict.values(), key=lambda x: x['final_score'], reverse=True)
    return sorted_results


def query_and_fuse(query, corpus_lines, corpus_dict, store, bm25_params):
    kg_results = query_knowledge_graph(query)
    vector_results = store.search(query, corpus=corpus_lines, top_k=15)
    keyword_results = search_bm25_query(query, *bm25_params)[:15]

    # ✅ 新增：ES 召回（索引名 docs，你如果不是 docs 这里改一下）
    es_results = es_search_query(query, index_name="docs", top_k=15)

    # ✅ 熵权法计算权重（4路）
    source_weights = calculate_entropy_weights(kg_results, vector_results, keyword_results, es_results)

    # ✅ 加权融合（4路）
    fused_results = weighted_fusion_results(kg_results, vector_results, keyword_results, es_results, source_weights)

    return kg_results, vector_results, keyword_results, es_results, source_weights, fused_results


def display_fusion_results(query, kg_results, vector_results, keyword_results, es_results, weights, fused_results):
    print(f"\n【问题】：{query}")
    print(f"知识图谱Cypher: {kg_results.get('cypher', '')}")

    # 打印动态权重
    print("\n动态权重分配:")
    print(f"  知识图谱检索权重: {weights['kgsearch']:.4f}")
    print(f"  向量检索权重: {weights['vectorsearch']:.4f}")
    print(f"  关键词检索权重: {weights['keywordsearch']:.4f}")
    print(f"  ES检索权重:     {weights['essearch']:.4f}")

    print("\n知识图谱检索结果前15条:")
    for i, item in enumerate(kg_results.get('triplets', [])[:15], 1):
        print(f"{i}. {item['content'][:80]}... (score: {item['score']:.4f})")

    print("\n向量检索结果前15条:")
    for i, item in enumerate(vector_results[:15], 1):
        print(f"{i}. {item['content'][:80]}... (score: {item['score']:.4f})")

    print("\n关键字检索结果前15条:")
    for i, item in enumerate(keyword_results[:15], 1):
        print(f"{i}. {item['content'][:80]}... (score: {item['score']:.4f})")

    print("\nES检索结果前15条:")
    for i, item in enumerate(es_results[:15], 1):
        print(f"{i}. {item['content'][:80]}... (score: {item['score']:.4f})")

    print("\n融合结果前20条:")
    for i, item in enumerate(fused_results[:20], 1):
        sources = "+".join(item['sources'])
        print(f"{i}. [{sources}] {item['content'][:80]}...  (score: {item['final_score']:.4f})")


if __name__ == "__main__":
    file_path = r"D:\lihao\data\bykg2508_text123.txt"
    model_path = r"D:\lihao\SystemonRSDprocessingKG202510\bge-large-zh"
    collection_name = "my_collection1"

    start_time = time.time()

    recreate_collection = False
    corpus_lines = load_corpus(file_path)
    corpus_dict = {i: text for i, text in enumerate(corpus_lines)}
    store = initialize_vector_store(collection_name, model_path, corpus_lines, recreate=recreate_collection)
    bm25_params = initialize_inverted_index(corpus_dict)

    queries = [
        "请帮我搭建一个图像融合流程"
    ]

    for query in queries:
        kg_results, vector_results, keyword_results, es_results, weights, fused_results = query_and_fuse(
            query, corpus_lines, corpus_dict, store, bm25_params
        )
        display_fusion_results(query, kg_results, vector_results, keyword_results, es_results, weights, fused_results)
        print(f"⏱⏱⏱️ 耗时: {round(time.time() - start_time, 2)} 秒")
