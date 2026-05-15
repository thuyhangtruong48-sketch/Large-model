import time

from fusion12 import query_and_fuse
from vector_store import MilvusVectorStore
from inverted_store import build_inverted_index
import requests


def load_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def initialize_vector_store(collection_name, model_path, corpus_lines):
    store = MilvusVectorStore(collection_name=collection_name, model_path=model_path)
    store.create_collection(corpus_lines)
    return store


def initialize_inverted_index(corpus_dict):
    return build_inverted_index(corpus_dict)


def summarize_with_qwen(query: str, top_k=15):
    # 1. 获取融合后的结果（只取fused部分）
    _, _, fused_results = query_and_fuse(query, corpus_lines, corpus_dict, store, bm25_params, source_weights)
    context = "\n".join([r["content"] for r in fused_results[:top_k]])
    # print("检索内容:\n", context)

    # 2. 构造 prompt

    prompt_template = """
    你是一个遥感数据处理方面的算法专家，你擅长分析和总结算法相关的信息，能基于用户的问题来回答算法和流程相关的问题。
    用户输入问题：{query}

    【检索内容】：
    {context}

    【例子说明】
    比如：用户提问的问题是：A算法的输入数据有哪些？检索到的内容有：["A算法的输入张量是a。", "A算法的输入张量之一包括a。", "A算法的输入张量数据之一包括a。", "B算法的输入张量是g。",
    "B算法的输入张量是h。", "A算法的输入张量是b。", "A算法的输入张量是c。", "A算法的输入张量是d。", "B算法的输入张量之一包括y。", "A算法的输入张量数据之一包括e。",
    "A算法的输入张量数据之一包括b。", "f是A算法的输入张量数据之一。", "B算法的输入张量是y。", "A算法的输入张量之一包括d。", "a输入张量的英文名称为a。"]
    你需要通过整理合并表示相同意思的内容，之后得出正确答案，输出：A算法的输入张量数据有a、b、c、d、e以及f。

    【注意事项】：
    1. 回答必须围绕“{query}”。
    2. 如果多个句子都在讲输入张量，合并相同含义但表述不同的输入张量，请整合成“算法的输入张量有xxx、yyy、zzz”这种形式。
    3. 内容冗余请合并，不重复。将检索到的重复、分散的信息整合成一句完整的话，使用中文回答。如果两个或多个句子表达的意思一样，则将其合并，只整理出一个答案。
    4. 请准确捕获用户提的问题是关于算法还是流程的，不要混淆。
    5. 记住：Algorithm代表算法，Flow代表流程，InputTensor表示算法或流程的输入数据，OutputTensor表示算法或流程的输出结果。
    6. 若出现流程AA由流程BB和流程CC组成，且流程BB由算法A、算法B、算法C组成，流程CC由算法D和算法E组成，则进行合并，最终形成：算法A、算法B、算法C构成流程BB，算法D和算法E构成流程CC，最后流程BB和流程CC构成流程AA，这就是流程AA的整体组件。

    【回答】：
    """

    prompt = prompt_template.format(query=query, context=context)
    # 3. 请求本地大语言模型（Ollama 默认在 http://localhost:11434）
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "deepseek-r1:32b",
            "prompt": prompt,
            "stream": False
        }
    )

    result_text = response.json()["response"]
    return result_text

if __name__ == "__main__":
    file_path = r"D:\lihao\data\bykg2508_text1.txt"
    model_path = r"D:\lihao\SystemonRSDprocessingKG202510\bge-large-zh"
    collection_name = "my_collection1"
    source_weights = {
            'kgsearch': 0.5,
            'keywordsearch': 0.25,
            'vectorsearch': 0.25
        }

    start_time = time.time()

    corpus_lines = load_corpus(file_path)
    corpus_dict = {i: text for i, text in enumerate(corpus_lines)}
    store = initialize_vector_store(collection_name, model_path, corpus_lines)
    bm25_params = initialize_inverted_index(corpus_dict)

    query = ("为这个句子生成表示：波段计算算法的输入张量有哪些？")
    answer = summarize_with_qwen(query)
    print("总结答案：", answer)
    print(f"⏱️ 耗时: {round(time.time() - start_time, 2)} 秒")


