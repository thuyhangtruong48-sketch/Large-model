import jieba
from collections import Counter, defaultdict
import string
import math
from elasticsearch import Elasticsearch, helpers
import numpy as np


def build_inverted_index(documents):
    stopwords = {"的", "了", "是", "在", "为", "之一", "包括", "如何", "一个", "和", "中", "其", "被", "从", "组成", "哪些",
                 "有", "什么", "谁", "属于", "这个", "句子", "生成", "表示", "请", "帮", "我"}
    chinese_punctuation = "。！？＂＃＄％＆＇（）＊＋，－．／：；＜＝＞＠［＼］＾＿｀｛｜｝～￥“”‘’、《》【】（）——……"
    all_punctuation = set(string.punctuation + chinese_punctuation)
    custom_terms = ["英文", "精校正", "名"]
    forced_splits = [('英文', '名称'), ('英文', '名'), ('名', '是'), ('名', '为')]

    tokenized_docs = tokenize_documents(documents, stopwords, all_punctuation, custom_terms, forced_splits)

    try:
        es = Elasticsearch("http://localhost:9200", request_timeout=3, verify_certs=False)
        if es.ping():
            create_es_index(es, "inverted_index", tokenized_docs)
        else:
            print("[inverted_store] Elasticsearch unavailable, using in-memory BM25 only")
    except Exception as exc:
        print(f"[inverted_store] Elasticsearch unavailable, using in-memory BM25 only: {exc}")

    inverted_index, df, doc_lengths, avgdl = build_in_memory_index(tokenized_docs)

    return inverted_index, tokenized_docs, documents, stopwords, all_punctuation, df, doc_lengths, avgdl


def tokenize_documents(documents, stopwords, all_punctuation, custom_terms=None, forced_splits=None):
    if custom_terms:
        for term in custom_terms:
            jieba.add_word(term, freq=1000)
    if forced_splits:
        for pair in forced_splits:
            jieba.suggest_freq(pair, True)

    tokenized_documents = {}
    for doc_id, text in documents.items():
        tokens = jieba.lcut(text)
        clean_tokens = [
            token for token in tokens
            if token not in stopwords and token not in all_punctuation and token.strip()
        ]
        tokenized_documents[doc_id] = clean_tokens
    # # 打印结果
    # for doc_id, tokens in tokenized_documents.items():
    #     print(f"文档 {doc_id} 的分词结果：{tokens}")
    return tokenized_documents


def create_es_index(es, index_name, tokenized_documents):
    if not es.indices.exists(index=index_name):
        mapping = {
            "mappings": {
                "properties": {
                    "doc_id": {"type": "integer"},
                    "token": {"type": "text"},
                    "frequency": {"type": "integer"}
                }
            }
        }
        es.indices.create(index=index_name, mappings=mapping["mappings"])

        actions = []
        for doc_id, tokens in tokenized_documents.items():
            token_freq = Counter(tokens)
            for token, freq in token_freq.items():
                actions.append({
                    "_index": index_name,
                    "_source": {
                        "doc_id": doc_id,
                        "token": token,
                        "frequency": freq
                    }
                })
        helpers.bulk(es, actions)
        print("[ES] index created")
    else:
        print("[ES] using existing index")


def build_in_memory_index(tokenized_documents):
    inverted_index = defaultdict(set)
    doc_lengths = {}
    avgdl = 0

    for doc_id, tokens in tokenized_documents.items():
        doc_lengths[doc_id] = len(tokens)
        avgdl += len(tokens)
        for token in tokens:
            inverted_index[token].add(doc_id)

    avgdl /= len(tokenized_documents)
    df = {token: len(doc_ids) for token, doc_ids in inverted_index.items()}

    # print("✅ 倒排索引构建完成。")
    # # 打印结果
    # for word, doc_ids in inverted_index.items():
    #     print(f"{word} → {list(doc_ids)}")
    return inverted_index, df, doc_lengths, avgdl


def bm25_score(query_tokens, tokenized_documents, df, doc_lengths, avgdl, total_docs, k1=1.8, b=0.5):
    scores = defaultdict(float)
    for token in query_tokens:
        if token not in df:
            continue
        idf = math.log((total_docs + 1) / (df[token] + 0.5))  # 加1防止负数
        for doc_id, tokens in tokenized_documents.items():
            tf = tokens.count(token)
            if tf == 0:
                continue
            dl = doc_lengths[doc_id]
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            score = idf * tf * (k1 + 1) / denom
            scores[doc_id] += score
    return scores


def search_bm25_query(query, inverted_index, tokenized_documents, documents,
                      stopwords, all_punctuation, df, doc_lengths, avgdl, top_k=15, min_match_ratio=0.6, min_match_floor=2):
    # 添加相同的自定义词典
    custom_terms = ["英文", "精校正"]
    for term in custom_terms:
        jieba.add_word(term, freq=1000)

    # 添加强制拆分规则
    jieba.suggest_freq(('英文', '名称'), True)  # 关键拆分设置
    jieba.suggest_freq(('英文', '名'), True)
    jieba.suggest_freq(('名', '是'), True)
    jieba.suggest_freq(('名', '为'), True)

    query_tokens = [
        token for token in jieba.lcut(query)
        if token not in stopwords and token not in all_punctuation and token.strip()
    ]
    # print("🔍 查询分词：", query_tokens)
    for token in query_tokens:
        if token not in inverted_index:
            print(f"[BM25] token not found in inverted index: {token}")

    # 收集包含任意查询词的文档
    candidate_doc_ids = set()
    for token in query_tokens:
        if token in inverted_index:
            candidate_doc_ids |= inverted_index[token]  # 使用并集

    # 如果没有候选文档，返回空结果
    if not candidate_doc_ids:
        return [{"content": "", "score": 0.0, "source": "keywordsearch"}]

    # 计算每个文档匹配的查询词数量
    doc_match_count = {}
    for doc_id in candidate_doc_ids:
        tokens_in_doc = set(tokenized_documents[doc_id])
        # 计算该文档包含的查询词数量
        count = sum(1 for token in query_tokens if token in tokens_in_doc)
        doc_match_count[doc_id] = count

    # 设置最低匹配阈值
    min_matches = max(min_match_floor, int(len(query_tokens) * min_match_ratio))

    # 筛选出符合最低匹配要求的文档
    matched_doc_ids = [
        doc_id for doc_id in candidate_doc_ids
        if doc_match_count[doc_id] >= min_matches
    ]

    if not matched_doc_ids:
        return [{"content": "", "score": 0.0, "source": "keywordsearch"}]

    bm25_scores = bm25_score(query_tokens, tokenized_documents, df, doc_lengths, avgdl, len(documents))
    filtered_scores = {doc_id: bm25_scores[doc_id] for doc_id in matched_doc_ids}

    results = []
    if filtered_scores:
        # Min-Max 归一化
        scores_values = list(filtered_scores.values())
        min_score = min(scores_values)
        max_score = max(scores_values)

        normalized_scores = []
        for doc_id, score in filtered_scores.items():
            if max_score != min_score:  # 避免除以0
                normalized = (score - min_score) / (max_score - min_score)
                # 只保留分数大于0的结果
                if normalized > 0:
                    normalized_scores.append((doc_id, normalized))

        # 按得分降序排序并取前top_k条
        sorted_results = sorted(normalized_scores, key=lambda x: x[1], reverse=True)[:top_k]

        if sorted_results:
            for doc_id, norm_score in sorted_results:
                result = {
                    "content": documents[doc_id],
                    "score": round(norm_score, 3),
                    "source": "keywordsearch"
                }
                results.append(result)
        else:
            # 所有文档归一化得分为0的情况
            results.append({
                "content": "",
                "score": 0.0,
                "source": "keywordsearch"
            })
    else:
        # 没有BM25得分大于0的匹配文档
        results.append({
            "content": "",
            "score": 0.0,
            "source": "keywordsearch"
        })

    return results
def es_search_query(query: str, index_name: str = "docs", top_k: int = 15,
                    es_url: str = "http://localhost:9200"):
    es = Elasticsearch(es_url, request_timeout=30, verify_certs=False)

    body = {
        "size": top_k,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["content^2", "title"],
                "type": "best_fields"
            }
        }
    }
    resp = es.search(index=index_name, body=body)
    hits = resp.get("hits", {}).get("hits", [])

    results = []
    for h in hits:
        src = h.get("_source", {})
        results.append({
            "content": src.get("content", ""),
            "score": float(h.get("_score", 0.0)),
            "source": "essearch"
        })

    if not results:
        return [{"content": "", "score": 0.0, "source": "essearch"}]
    return results



