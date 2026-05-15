from inverted_store import build_inverted_index, search_bm25_query

# 加载文本路径
file_path = r"D:\lihao\data\bykg2508_text1.txt"
# C:\Users\mayij\Desktop\kg_to_text\aggregated_corpus.txt
# C:\Users\mayij\Desktop\kg_to_text\bykg2505_text.txt
# 构建倒排索引

# 加载文本
with open(file_path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
    documents = {i + 1: line for i, line in enumerate(lines)}

inverted_index, tokenized_documents, documents, stopwords, all_punctuation, term_doc_freq, doc_lengths, avg_doc_len = build_inverted_index(documents)

# 查询
queries = [
    # "波段计算算法的输入张量有哪些？",
    # "基于多边形影像裁切算法的输入张量包括哪些？",
    # "数据类型转换算法的输出张量是什么？",
    # "数据类型转换算法的输出结果是什么？",
    # "影像无效值填充算法的英文名称是什么？"
    # "产品名称输入张量的英文名是什么？"
    # "基于多边形影像裁切算法的英文名是什么？",
    # "波段融合算法的英文名是什么？",
    # "空间提取是什么算法生产的产品的中文名称？",
    # "4波段影像归一化植被指数流程是由哪些算法组成的？",
    # "生产重采样产品需要用到什么算法？",
    # "实现数据转换类型流程需要用到什么算法?",
    # "搭建图像融合流程需要用到什么流程和算法？",
    "请帮我搭建一个图像融合流程"
]
for query in queries:
##正确答案：待定影像，产品名称，矢量图层名，空间参考类型，空间分辨率，待定自然波段，参考矢量，边界类型，
##空间参考类型、待定自然波段没有检索出来
    result = search_bm25_query(query, inverted_index, tokenized_documents, documents, stopwords, all_punctuation, term_doc_freq, doc_lengths, avg_doc_len, top_k=15)
    print("问题：", query)
    for item in result:
        print(item)

