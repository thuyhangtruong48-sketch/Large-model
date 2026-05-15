import time

from fusion123_shangquan import query_and_fuse
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


def summarize_with_model(query: str, corpus_lines, corpus_dict, store, bm25_params, top_k=30):
    # 1. 获取融合后的结果（只取fused部分）
    _, _, _, _, fused_results = query_and_fuse(query, corpus_lines, corpus_dict, store, bm25_params)
    context = "\n".join([r["content"] for r in fused_results[:top_k]])
    # print("检索内容:\n", context)

    # 2. 构造 prompt
    prompt_template = f"""
        你是一个遥感数据处理方面的算法专家，你擅长分析和总结算法相关的信息，能基于用户的问题来回答算法和流程相关的问题。
        用户输入问题：{query}

        【检索内容】：
        {context}

        【例子说明】
        1.问题：A算法的输入数据有哪些?
        检索到的内容有：["A算法的输入张量是a。", "A算法的输入张量之一包括a。", "A算法的输入张量数据之一包括a。", "B算法的输入张量是g。",
        "B算法的输入张量是h。", "A算法的输入张量是b。", "A算法的输入张量是c。", "A算法的输入张量是d。", "B算法的输入张量之一包括y。", "A算法的输入张量数据之一包括e。",
        "A算法的输入张量数据之一包括b。", "f是A算法的输入张量数据之一。", "B算法的输入张量是y。", "A算法的输入张量之一包括d。", "a输入张量的英文名称为a。"]
        输出答案：A算法的输入张量数据有a、b、c、d、e以及f。
        
        2.请帮我搭建一个B流程。
        检索到的内容有：["搭建流程B需要用到a流程。", "搭建流程B需要用到b流程。", "B是一个...类型的流程。...实现/搭建B流程需要使用以下组件（算法或流程）：流程b、流程a",
        "b是一个...流程。...实现/搭建b流程需要使用以下组件（算法或流程）：算法aa、算法bb、算法cc、算法dd、算法ee。在搭建/实现流程B时使用了b流程。"]
        输出答案：搭建B流程需要用到a流程和b流程，而b流程又由算法aa、算法bb、算法cc、算法dd、算法ee搭建而成。

        【注意事项】：
        1. 回答必须围绕“{query}”。
        2. 记住：Algorithm代表算法，Flow代表流程，InputTensor表示算法或流程的输入数据，OutputTensor表示算法或流程的输出结果。
        3. 若有两个名称相同的算法或流程，请分别进行回答。
        4. 内容冗余请合并，不重复。将检索到的重复、分散的信息整合成一句完整的话，使用中文回答。如果两个或多个句子表达的意思一样，则将其合并，只整理出一个答案。
        5. 请准确捕获用户提的问题是关于算法还是流程的，不要混淆。
        6. 若出现流程AA由流程BB和流程CC组成，且流程BB由算法A、算法B、算法C组成，流程CC由算法D和算法E组成，则进行合并，最终形成：算法A、算法B、算法C构成流程BB，算法D和算法E构成流程CC，最后流程BB和流程CC构成流程AA，这就是流程AA的整体组件。

        【回答】：
        """

    prompt = prompt_template.format(query=query, context=context)
    # 3. 请求本地大语言模型（Ollama 默认在 http://localhost:11434）
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:14b",
            "prompt": prompt,
            "stream": False
        }
    )

    result_text = response.json()["response"]
    return result_text, fused_results

if __name__ == "__main__":
    file_path = r"D:\lihao\data\bykg2508_text123.txt"
    model_path = r"D:\lihao\SystemonRSDprocessingKG202510\bge-large-zh"
    collection_name = "my_collection1"

    start_time = time.time()

    corpus_lines = load_corpus(file_path)
    corpus_dict = {i: text for i, text in enumerate(corpus_lines)}
    store = initialize_vector_store(collection_name, model_path, corpus_lines)
    bm25_params = initialize_inverted_index(corpus_dict)

    query = ("请帮我搭建一个图像融合流程")
    answer = summarize_with_model(query, corpus_lines, corpus_dict, store, bm25_params)
    print("总结答案：", answer)
    print(f"⏱️ 耗时: {round(time.time() - start_time, 2)} 秒")

