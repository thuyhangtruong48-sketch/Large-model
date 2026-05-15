import time
import numpy as np
from collections import defaultdict

from inverted_store import build_inverted_index, search_bm25_query, es_search_query
from vector_store import MilvusVectorStore
from graph_store import query_knowledge_graph


def load_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def initialize_vector_store(collection_name, model_path, corpus_lines, recreate=False):
    store = MilvusVectorStore(collection_name=collection_name, model_path=model_path, recreate=recreate)
    if not store.collection_exists or recreate:
        store.create_collection(corpus_lines)
    else:
        store.load_collection()
    return store


def initialize_inverted_index(corpus_dict):
    return build_inverted_index(corpus_dict)


def calculate_entropy_weights(kg_results, vector_results, keyword_results, es_results):
    all_contents = set()

    for triplet in kg_results.get('triplets', []):
        all_contents.add(triplet['content'])
    for result in vector_results:
        all_contents.add(result['content'])
    for result in keyword_results:
        all_contents.add(result['content'])
    for result in es_results:
        all_contents.add(result['content'])

    n = len(all_contents)
    if n == 0:
        return {'kgsearch': 0.25, 'keywordsearch': 0.25, 'vectorsearch': 0.25, 'essearch': 0.25}

    score_matrix = np.zeros((n, 4))  # [kg, vector, keyword, es]
    content_list = list(all_contents)
    content_to_idx = {c: i for i, c in enumerate(content_list)}

    for triplet in kg_results.get('triplets', []):
        score_matrix[content_to_idx[triplet['content']], 0] = triplet['score']
    for result in vector_results:
        score_matrix[content_to_idx[result['content']], 1] = result['score']
    for result in keyword_results:
        score_matrix[content_to_idx[result['content']], 2] = result['score']
    for result in es_results:
        score_matrix[content_to_idx[result['content']], 3] = result['score']

    epsilon = 1e-4
    score_matrix = np.where(score_matrix == 0, epsilon, score_matrix)

    min_vals = np.min(score_matrix, axis=0)
    max_vals = np.max(score_matrix, axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1
    normalized = (score_matrix - min_vals) / ranges

    col_sums = np.sum(normalized, axis=0)
    col_sums[col_sums == 0] = 1
    p = normalized / col_sums

    p_safe = np.where(p == 0, epsilon, p)
    entropy = -np.sum(p_safe * np.log(p_safe), axis=0) / np.log(n)

    diff = 1 - entropy
    if np.sum(diff) == 0:
        return {'kgsearch': 0.25, 'keywordsearch': 0.25, 'vectorsearch': 0.25, 'essearch': 0.25}

    w = diff / np.sum(diff)
    return {'kgsearch': float(w[0]), 'vectorsearch': float(w[1]), 'keywordsearch': float(w[2]), 'essearch': float(w[3])}


def weighted_fusion_results(kg_results, vector_results, keyword_results, es_results, source_weights):
    fusion_dict = defaultdict(lambda: {
        'content': '',
        'scores': {'kgsearch': 0.0, 'keywordsearch': 0.0, 'vectorsearch': 0.0, 'essearch': 0.0},
        'final_score': 0.0,
        'sources': []
    })

    for triplet in kg_results.get('triplets', []):
        c = triplet['content']
        fusion_dict[c]['content'] = c
        fusion_dict[c]['scores']['kgsearch'] = triplet['score']
        if 'kgsearch' not in fusion_dict[c]['sources']:
            fusion_dict[c]['sources'].append('kgsearch')

    for result in vector_results:
        c = result['content']
        fusion_dict[c]['content'] = c
        fusion_dict[c]['scores']['vectorsearch'] = result['score']
        if 'vectorsearch' not in fusion_dict[c]['sources']:
            fusion_dict[c]['sources'].append('vectorsearch')

    for result in keyword_results:
        c = result['content']
        fusion_dict[c]['content'] = c
        fusion_dict[c]['scores']['keywordsearch'] = result['score']
        if 'keywordsearch' not in fusion_dict[c]['sources']:
            fusion_dict[c]['sources'].append('keywordsearch')

    for result in es_results:
        c = result['content']
        fusion_dict[c]['content'] = c
        fusion_dict[c]['scores']['essearch'] = result['score']
        if 'essearch' not in fusion_dict[c]['sources']:
            fusion_dict[c]['sources'].append('essearch')

    for _, entry in fusion_dict.items():
        # 如果你希望严格归一化到 0-1，可以加 min/max；这里先保持原分数
        entry['final_score'] = (
            source_weights['kgsearch'] * entry['scores']['kgsearch'] +
            source_weights['keywordsearch'] * entry['scores']['keywordsearch'] +
            source_weights['vectorsearch'] * entry['scores']['vectorsearch'] +
            source_weights['essearch'] * entry['scores']['essearch']
        )

    return sorted(fusion_dict.values(), key=lambda x: x['final_score'], reverse=True)


def query_and_fuse(query, corpus_lines, store, bm25_params, use_entropy=True, fixed_weights=None):
    kg_results = query_knowledge_graph(query)
    vector_results = store.search(query, corpus=corpus_lines, top_k=15)
    keyword_results = search_bm25_query(query, *bm25_params)[:15]
    es_results = es_search_query(query, index_name="docs", top_k=15)

    if use_entropy:
        source_weights = calculate_entropy_weights(kg_results, vector_results, keyword_results, es_results)
    else:
        source_weights = fixed_weights or {'kgsearch': 0.25, 'keywordsearch': 0.25, 'vectorsearch': 0.25, 'essearch': 0.25}

    fused_results = weighted_fusion_results(kg_results, vector_results, keyword_results, es_results, source_weights)
    return kg_results, vector_results, keyword_results, es_results, source_weights, fused_results


def display_fusion_results(query, kg_results, vector_results, keyword_results, es_results, weights, fused_results):
    print(f"\n【问题】：{query}")
    print(f"知识图谱Cypher: {kg_results.get('cypher', '')}")

    print("\n动态权重分配:")
    print(f"  kgsearch:      {weights['kgsearch']:.4f}")
    print(f"  vectorsearch:  {weights['vectorsearch']:.4f}")
    print(f"  keywordsearch: {weights['keywordsearch']:.4f}")
    print(f"  essearch:      {weights['essearch']:.4f}")

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
    file_path = r"D:\lihao\data\bykg2508_text1.txt"
    model_path = r"D:\lihao\SystemonRSDprocessingKG202510\bge-large-zh"
    collection_name = "my_collection1"

    start_time = time.time()

    recreate_collection = False
    corpus_lines = load_corpus(file_path)
    corpus_dict = {i: text for i, text in enumerate(corpus_lines)}  # 你若还要build_inverted_index可用
    store = initialize_vector_store(collection_name, model_path, corpus_lines, recreate=recreate_collection)
    bm25_params = initialize_inverted_index(corpus_dict)

    queries = [
        "有几个名为图像融合的流程？"
    ]

    fixed_weights = {
        'kgsearch': 0.5,
        'keywordsearch': 0.2,
        'vectorsearch': 0.2,
        'essearch': 0.1
    }

    for query in queries:
        kg_results, vector_results, keyword_results, es_results, weights, fused_results = query_and_fuse(
            query, corpus_lines, store, bm25_params,
            use_entropy=True,      # 改 False 就用 fixed_weights
            fixed_weights=fixed_weights
        )
        display_fusion_results(query, kg_results, vector_results, keyword_results, es_results, weights, fused_results)
        print(f"⏱️ 耗时: {round(time.time() - start_time, 2)} 秒")
