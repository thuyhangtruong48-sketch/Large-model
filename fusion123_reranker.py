import time
import math
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from inverted_store import build_inverted_index, search_bm25_query
from vector_store import MilvusVectorStore
from graph_store import query_knowledge_graph  # 确保你有这个模块


class ReRanker:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def rerank(self, query, candidates, top_k=30):
        """使用交叉编码器对候选结果进行重排"""
        inputs = [(query, item['content']) for item in candidates]
        encodings = self.tokenizer.batch_encode_plus(
            inputs, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            logits = self.model(**encodings).logits
            scores = logits.squeeze(-1).tolist()

        # 添加重排分数到候选结果
        for i, item in enumerate(candidates):
            item['rerank_score'] = scores[i]

        # 按重排分数排序并返回top_k
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        return reranked[:top_k]


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
        # 从ES检索结果中收集内容（新增）
    for result in es_results:
        all_contents.add(result['content'])

    n = len(all_contents)
    if n == 0:
        # 默认权重
        return {'kgsearch': 0.333, 'vectorsearch': 0.333, 'keywordsearch': 0.334}

    score_matrix = np.zeros((n, 3))  # 列: [kg, vector, keyword]
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

    # 返回权重字典
    return {
        'kgsearch': weights[0],
        'vectorsearch': weights[1],
        'keywordsearch': weights[2]
    }


def weighted_fusion_results(kg_results, vector_results, keyword_results, source_weights):
    # 融合3类检索结果，基于 content 去重 + 加权得分
    fusion_dict = defaultdict(lambda: {
        'content': '',
        'scores': {'kgsearch': 0, 'vectorsearch': 0, 'keywordsearch': 0},
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

    # 计算加权融合分数
    for key, entry in fusion_dict.items():
        # 确保每种检索类型的分数都在0-1之间
        kg_score = min(max(entry['scores']['kgsearch'], 0.0), 1.0)
        vector_score = min(max(entry['scores']['vectorsearch'], 0.0), 1.0)
        keyword_score = min(max(entry['scores']['keywordsearch'], 0.0), 1.0)

        # 计算加权分数
        entry['final_score'] = (
                source_weights['kgsearch'] * kg_score +
                source_weights['vectorsearch'] * vector_score +
                source_weights['keywordsearch'] * keyword_score
        )

    # 转换为列表并排序
    sorted_results = sorted(fusion_dict.values(), key=lambda x: x['final_score'], reverse=True)

    return sorted_results


def query_and_fuse(query, corpus_lines, corpus_dict, store, bm25_params, reranker=None):
    kg_results = query_knowledge_graph(query)

    # 将三元组转换为自然语言描述
    transformed_triplets = []
    for triplet in kg_results['triplets']:
        # 将 (头实体, 关系, 尾实体) 转换为自然语言描述
        # 示例: (基于多边形影像裁切, 输入张量, 空间分辨率)
        # -> "基于多边形影像裁切算法的输入张量包括空间分辨率"
        parts = triplet['content'].strip('()').split(', ')
        if len(parts) == 3:
            head, relation, tail = parts
            # 根据关系类型选择合适的模板
            if relation == "输入张量":
                content = f"{head}的{relation}包括{tail}"
            elif relation == "输出张量":
                content = f"{head}的{relation}包括{tail}"
            else:
                content = f"{head},{relation},{tail}"
            transformed_triplets.append({
                'content': content,
                'score': triplet['score']
            })

    # 替换原始的三元组格式
    kg_results['triplets'] = transformed_triplets

    vector_results = store.search(query, corpus=corpus_lines, top_k=15)
    keyword_results = search_bm25_query(query, *bm25_params)[:15]

    # 使用熵权法计算权重
    source_weights = calculate_entropy_weights(kg_results, vector_results, keyword_results)

    # 加权融合结果
    fused_results = weighted_fusion_results(kg_results, vector_results, keyword_results, source_weights)

    # 使用重排模型优化结果
    if reranker:
        fused_results = reranker.rerank(query, fused_results)

    return kg_results, vector_results, keyword_results, source_weights, fused_results


def display_fusion_results(query, kg_results, vector_results, keyword_results, weights, fused_results):
    print(f"\n【问题】：{query}")
    print(f"知识图谱Cypher: {kg_results['cypher']}")

    # 打印动态权重
    print("\n动态权重分配:")
    print(f"  知识图谱检索权重: {weights['kgsearch']:.4f}")
    print(f"  向量检索权重: {weights['vectorsearch']:.4f}")
    print(f"  关键词检索权重: {weights['keywordsearch']:.4f}")

    # 其他显示逻辑保持不变...
    print("\n知识图谱检索结果前15条:")
    for i, item in enumerate(kg_results['triplets'][:15], 1):
        print(f"{i}. {item['content'][:80]}... (score: {item['score']:.4f})")

    print("\n向量检索结果前15条:")
    for i, item in enumerate(vector_results[:15], 1):
        print(f"{i}. {item['content'][:80]}... (score: {item['score']:.4f})")

    print("\n关键字检索结果前15条:")
    for i, item in enumerate(keyword_results[:15], 1):
        print(f"{i}. {item['content'][:80]}... (score: {item['score']:.4f})")

    print("\n融合结果前20条:")
    for i, item in enumerate(fused_results[:25], 1):
        sources = "+".join(item['sources'])
        print(f"{i}. [{sources}] {item['content']}  (score: {item['final_score']:.4f})")


if __name__ == "__main__":
    # 初始化代码保持不变...
    file_path = r"D:\lihao\data\bykg2508_text123.txt"
    model_path = r"D:\lihao\SystemonRSDprocessingKG202510\bge-large-zh"
    reranker_path = r"D:\lihao\SystemonRSDprocessingKG202510\bge-reranker-base"
    collection_name = "my_collection1"

    start_time = time.time()

    # 初始化重排模型
    reranker = ReRanker(reranker_path) if reranker_path else None

    recreate_collection = True
    corpus_lines = load_corpus(file_path)
    corpus_dict = {i: text for i, text in enumerate(corpus_lines)}
    store = initialize_vector_store(collection_name, model_path, corpus_lines, recreate=recreate_collection)
    bm25_params = initialize_inverted_index(corpus_dict)

    queries = [
        # "波段计算算法的输入张量有哪些？",
        # "数据类型转换流程的输入张量有哪些？",
        # "基于多边形影像裁切算法的输入张量包括哪些数据？",
        # "数据类型转换流程由哪些算法组成？",
        # "影像无效值填充算法的英文名称是什么？",
        # "4波段影像归一化植被指数流程是由哪些流程组成的？"
        # "生产空间提取产品需要用到什么算法？",
        # "实现数据类型转换流程需要用到什么算法?",
        # "搭建图像融合流程需要用到什么算法？",
        # "搭建图像融合流程需要用到什么流程？",
        # "搭建2波段算法归一化植被指数流程需要用到什么算法？",
        # "请帮我搭建一个图像融合流程"
        # "波段擦除算法的输入张量有些什么？"
        "归一化植被指数算法适用于什么卫星影像？"
    ]

    for query in queries:
        # 修改调用参数，移除固定权重
        kg_results, vector_results, keyword_results, weights, fused_results = query_and_fuse(
            query, corpus_lines, corpus_dict, store, bm25_params, reranker)

        display_fusion_results(query, kg_results, vector_results, keyword_results, weights, fused_results)
        print(f"⏱⏱⏱️ 耗时: {round(time.time() - start_time, 2)} 秒")