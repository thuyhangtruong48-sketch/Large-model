from fusion123 import query_and_fuse
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
    _, _, _, fused_results = query_and_fuse(query, corpus_lines, corpus_dict, store, bm25_params, source_weights)
    context = "\n".join([r["content"] for r in fused_results[:top_k]])
    print("content:\n", context)

    # 2. 构造 prompt
    prompt = f"""
你是一个遥感数据处理方面的算法专家，你能基于用户的问题来回答算法和流程相关的问题。
用户输入问题：{query} 

【相关内容】：
{context} 

【例子说明】
比如：波段融合算法是由高分全色影像、低分自然波段、低分多光谱影像以及分块大小四个输入张量数据组成的。用户提问的问题是：波段融合算法的输入数据有哪些？检索到的内容有：['波段融合算法的输入张量之一包括分块大小。', 
'分块大小是波段融合算法的输入张量数据之一。', '波段融合算法的输入张量数据之一包括低分多光谱影像。', '波段融合算法的输入张量之一包括高分全色影像。', '波段融合算法算法处理的原始数据之一包括高分全色影像。', 
'波段融合算法的输入张量数据之一包括低分自然波段。', '波段融合算法的输入张量之一包括低分自然波段。', '波段融合算法的输入张量数据之一包括分块大小。', '波段融合算法的输入张量数据之一包括低分多光谱影像。', 
'波段提取算法的输入张量之一包括波段计算Py代码。', '波段擦除算法的输入张量数据之一包括擦除影像波段值。', '波段融合算法的输入张量数据之一包括高分全色影像。', '波段提取算法的输入张量之一包括产品名称。', 
'低分多光谱影像是波段融合算法的输入张量数据之一。', '低分自然波段是波段融合算法的输入数据之一。']。你需要通过整理合并之后得出正确答案答案：波段融合算法的输入数据有高分全色影像、低分自然波段、低分多光谱影像以及分块大小。

【注意事项】：
1. 回答必须围绕“{query}”。
2. 如果多个句子都在讲输入张量，请整合成“算法的输入张量有xxx、yyy”这种形式。
3. 内容冗余请合并，不重复。将检索到的重复、分散的信息整合成一句完整的话，使用中文回答。如果两个或多个句子表达的意思一样，则将其合并，只整理出一个答案。
4. 请准确捕获用户提的问题是关于算法还是流程的，不要混淆。
5. 记住：Algorithm代表算法，Flow代表流程，InputTensor表示算法或流程的输入数据，OutputTensor表示算法或流程的输出结果。
6. 若出现流程A由流程B和流程C组成，且流程B由算法a、算法b、算法c组成，流程C由算法d和算法e组成，则进行合并，最终形成：算法a、算法b、算法c构成流程B，算法d和算法e构成流程C，最后流程B和流程C构成流程A，这就是流程A的整体组件。
7.当检索结果中出现“(波段镶嵌, 输入张量, 背景值)”，表明它是由知识图谱检索得到的结果。这句话表达的意思与“波段镶嵌算法的输入数据之一是背景值”、"背景值是波段镶嵌算法的输入数据之一"、
“波段镶嵌算法的输入张量数据之一包括背景值”是一致的，只是不同的表达方式。所以其中的“背景值”就是输入的数据之一。 
8.content中的“(波段镶嵌, 输入张量, 背景值)、(波段镶嵌, 输入张量, 输出影像分辨率)、(波段镶嵌, 输入张量, 公共区域处理)、(波段镶嵌, 输入张量, 参考影像)、(波段镶嵌, 输入张量, 目标自然波段编号)、
(波段镶嵌, 输入张量, 目标影像列表)”，表明的意思就是波段镶嵌算法的输入张量包括背景值、 输出影像分辨率、公共区域处理、参考影像、目标自然波段编号、目标影像列表。
9.输出影像分辨率也是属于波段镶嵌算法的输入张量数据之一，输出内容中包含“输出”的不一定就是输出结果，包含“输入”的不一定就是输入张量，要仔细甄别。


【回答】：
"""

    # 3. 请求本地 Qwen 模型（Ollama 默认在 http://localhost:11434）
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:32b",
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

    corpus_lines = load_corpus(file_path)
    corpus_dict = {i: text for i, text in enumerate(corpus_lines)}
    store = initialize_vector_store(collection_name, model_path, corpus_lines)
    bm25_params = initialize_inverted_index(corpus_dict)

    query = ("为这个句子生成表示：波段计算算法的输入张量有哪些？")
    answer = summarize_with_qwen(query)
    print("总结答案：", answer)