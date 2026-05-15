import time

from inverted_store import build_inverted_index, search_bm25_query
from vector_store import MilvusVectorStore
from graph_store import query_knowledge_graph  # 确保你有这个模块
from collections import defaultdict
import json
import re


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
    return store


def initialize_inverted_index(corpus_dict):
    return build_inverted_index(corpus_dict)


def weighted_fusion_results(kg_results, vector_results, keyword_results, source_weights):
    fusion_dict = defaultdict(lambda: {
        'content': '',
        'scores': {'kgsearch': 0, 'keywordsearch': 0, 'vectorsearch': 0},
        'final_score': 0.0,
        'sources': []
    })

    for triplet in kg_results.get('triplets', []):
        content = triplet['content']
        fusion_dict[content]['content'] = content
        fusion_dict[content]['scores']['kgsearch'] = triplet['score']
        if 'kgsearch' not in fusion_dict[content]['sources']:
            fusion_dict[content]['sources'].append('kgsearch')

    for result in vector_results:
        content = result['content']
        fusion_dict[content]['content'] = content
        fusion_dict[content]['scores']['vectorsearch'] = result['score']
        if 'vectorsearch' not in fusion_dict[content]['sources']:
            fusion_dict[content]['sources'].append('vectorsearch')

    for result in keyword_results:
        content = result['content']
        fusion_dict[content]['content'] = content
        fusion_dict[content]['scores']['keywordsearch'] = result['score']
        if 'keywordsearch' not in fusion_dict[content]['sources']:
            fusion_dict[content]['sources'].append('keywordsearch')

    for key, entry in fusion_dict.items():
        entry['final_score'] = (
            source_weights['kgsearch'] * entry['scores']['kgsearch'] +
            source_weights['keywordsearch'] * entry['scores']['keywordsearch'] +
            source_weights['vectorsearch'] * entry['scores']['vectorsearch']
        )

    sorted_results = sorted(fusion_dict.values(), key=lambda x: x['final_score'], reverse=True)
    return sorted_results

def query_and_fuse(query, corpus_lines, corpus_dict, store, bm25_params, source_weights):
    kg_results = query_knowledge_graph(query)
    vector_results = store.search(query, corpus=corpus_lines, top_k=15)
    keyword_results = search_bm25_query(query, *bm25_params)[:15]
    fused_results = weighted_fusion_results(kg_results, vector_results, keyword_results, source_weights)
    return kg_results, vector_results, keyword_results, fused_results

def display_fusion_results(query, kg_results, vector_results, keyword_results, fused_results):
    print(f"\n【问题】：{query}")
    print(f"知识图谱Cypher: {kg_results['cypher']}")

    print("\n知识图谱检索结果前10条:")
    for i, item in enumerate(kg_results['triplets'][:10], 1):
        print(f"{i}. {item['content'][:80]}... (score: {item['score']:.4f})")

    print("\n向量检索结果前10条:")
    for i, item in enumerate(vector_results[:15], 1):
        print(f"{i}. {item['content'][:80]}... (score: {item['score']:.4f})")

    print("\n关键字检索结果前10条:")
    for i, item in enumerate(keyword_results[:15], 1):
        print(f"{i}. {item['content'][:80]}... (score: {item['score']:.4f})")

    print("\n融合结果前15条:")
    for i, item in enumerate(fused_results[:15], 1):
        sources = "+".join(item['sources'])
        print(f"{i}. [{sources}] {item['content']}  (score: {item['final_score']:.4f})")

if __name__ == "__main__":
    file_path = r"D:\lihao\data\bykg2508_text1.txt"
    model_path = r"D:/Project_code/SystemonRSDprocessingKG/bge-large-zh"
    collection_name = "my_collection1"
    source_weights = {
        'kgsearch': 0.5,
        'keywordsearch': 0.25,
        'vectorsearch': 0.25
    }

    start_time = time.time()

    recreate_collection = False
    corpus_lines = load_corpus(file_path)
    corpus_dict = {i: text for i, text in enumerate(corpus_lines)}
    store = initialize_vector_store(collection_name, model_path, corpus_lines, recreate=recreate_collection)
    bm25_params = initialize_inverted_index(corpus_dict)

    queries = [
        "波段计算算法的输入张量有哪些？",
    ]

    for query in queries:
        kg_results, vector_results, keyword_results, fused_results = query_and_fuse(
            query, corpus_lines, corpus_dict, store, bm25_params, source_weights)
        display_fusion_results(query, kg_results, vector_results, keyword_results, fused_results)
        print(f"⏱️ 耗时: {round(time.time() - start_time, 2)} 秒")