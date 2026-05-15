from fusion12_shangquan import query_and_fuse
from graph_store import query_knowledge_graph, get_flow_hierarchy_natural_language  # 确保你有这个模块
from vector_store import MilvusVectorStore
from inverted_store import build_inverted_index
import requests
import time
import re
import json
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
from collections import Counter


def load_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def initialize_vector_store(collection_name, model_path, corpus_lines):
    store = MilvusVectorStore(collection_name=collection_name, model_path=model_path)
    store.create_collection(corpus_lines)
    return store


def initialize_inverted_index(corpus_dict):
    return build_inverted_index(corpus_dict)


# === 新增函数：问题类型分类器 ===
def classify_question_type(user_question: str) -> dict:
    """
    使用大模型判断问题类型：
    - "kg_query": 知识图谱查询类问题
    - "flow_hierarchy": 流程层级结构类问题
    - "other": 其他类型问题
    """
    prompt = f"""
    你是一个专业的问题分类助手，需要判断用户问题的类型。以下是分类规则：

    1. 知识图谱查询类问题（kg_query）特点：
       - 询问特定实体（算法、流程、产品等）的属性（如名称、描述、作者、类型等）
       - 询问实体之间的关系（如输入/输出张量、使用的算法等）
       - 示例："融合算法的输入张量有哪些？", "辐射定标流程的作者是谁？", "归一化植被指数产品的英文名是什么？", "生产元数据更新产品需要是用什么算法？"

    2. 流程层级结构类问题（flow_hierarchy）特点：
       - 明确提到"搭建"、"层级"、"结构"、"组成"等关键词
       - 询问流程的组成关系（如包含哪些子流程或算法）
       - 示例："请搭建图像融合流程", "如何搭建辐射定标流程", "地形校正流程由哪些部分组成"

    3. 其他类型问题（other）：
       - 不属于以上两类的任何问题
       - 示例："什么是辐射定标？", "如何使用这个系统？"

    请分析以下用户问题，并返回JSON格式结果：
    {{
        "type": "kg_query" | "flow_hierarchy" | "other",
        "target_flow": "流程名称"  // 仅当type为flow_hierarchy时提供
    }}

    用户问题：{user_question}
    """

    try:
        url = f"{OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": "qwen2.5:14b",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        r = requests.post(url, json=payload, timeout=600)
        r.raise_for_status()
        response = r.json()

        # Ollama /api/chat 返回结构一般是 {"message":{"content":...}}
        answer = response.get("message", {}).get("content", "").strip()

        # 尝试提取JSON格式的响应
        match = re.search(r'```json\s*({.*?})\s*```', answer, re.DOTALL)
        if match:
            result = json.loads(match.group(1))
            # 如果分类为流程层级问题，且target_flow包含"流程"二字，则去除
            if result.get("type") == "flow_hierarchy" and result.get("target_flow"):
                # 去除末尾的"流程"二字
                if result["target_flow"].endswith("流程"):
                    result["target_flow"] = result["target_flow"][:-2]
                # 如果去除后为空，使用默认值
                if not result["target_flow"]:
                    result["target_flow"] = "图像融合"
            return result
        else:
            # 尝试直接解析
            try:
                return json.loads(answer)
            except:
                # 默认返回知识图谱查询
                print(f"⚠️ 无法解析分类结果，默认使用kg_query: {answer}")
                return {"type": "kg_query", "target_flow": None}
    except Exception as e:
        print(f"❌❌ 问题分类失败: {e}")
        return {"type": "kg_query", "target_flow": None}


def summarize_with_model(query: str, corpus_lines, corpus_dict, store, bm25_params, top_k=15):
    # 1. 获取融合后的结果（只取fused部分）
    _, _, _, fused_results = query_and_fuse(query, corpus_lines, corpus_dict, store, bm25_params)
    print("fused_results=", fused_results)
    # 2. 判断问题类型并获取相应结果
    classification = classify_question_type(query)
    question_type = classification["type"]
    kg_response = ""

    # 根据问题类型调用不同处理函数
    if question_type == "kg_query":
        kg_result = query_knowledge_graph(query)
        kg_response = "知识图谱查询结果：\n"

        # 处理知识图谱结果
        if kg_result["objects"]:
            kg_response += ", ".join(kg_result["objects"])
        else:
            kg_response += "知识图谱中未找到相关信息"

    elif question_type == "flow_hierarchy":
        flow_name = classification["target_flow"]
        if not flow_name:
            # 尝试从问题中提取流程名称
            flow_match = re.search(r'(?:搭建|展示|查询|的)*([\u4e00-\u9fa5a-zA-Z0-9]+?)(?:流程|的流程|流程的组成)?$', query)
            if flow_match:
                flow_name = flow_match.group(1)
                # 再次检查并去除可能存在的"流程"二字
                if flow_name.endswith("流程"):
                    flow_name = flow_name[:-2]
            else:
                flow_name = "图像融合"
            print(f"⚠️ 未明确指定流程名称，使用默认值: {flow_name}")

        hierarchy_descriptions = get_flow_hierarchy_natural_language(flow_name)
        kg_response = "流程层级结构：\n" + "\n".join(hierarchy_descriptions)

    else:  # other类型问题
        # 尝试用知识图谱回答
        kg_result = query_knowledge_graph(query)
        kg_response = "知识图谱查询结果：\n"
        if kg_result["objects"]:
            kg_response += ", ".join(kg_result["objects"])
        else:
            kg_response += "无法找到相关信息"

    # 3. 合并检索内容和知识图谱结果
    context = "\n".join([r["content"] for r in fused_results[:top_k]])
    full_context = f"【文本检索内容】：\n{context}\n\n{kg_response}"
    print("完整上下文内容:\n", full_context)

    # 4. 构造 prompt
    prompt = f"""
    你是一个遥感数据处理方面的算法专家。请先基于【检索内容】进行严谨的逐步推理，再给出最终答案。
    注意：严格使用下面的输出格式，两段都必须出现。

    【用户问题】：
    {query}

    【检索内容】：
    {full_context}

    【输出格式（务必遵守）】：
    【推理过程】逐步说明你如何从检索内容里抽取、去重、合并信息；如存在“流程由子流程组成、子流程又由算法组成”，请递归整合，并避免臆想未在检索中出现的信息。
    【最终答案】只保留结论，但必须完整保留层级关系，逐层展开说明。禁止将层级压缩到一句话。

    【例子说明】
    1. 问题：A算法的输入数据有哪些?
       检索：["A算法的输入张量是a。", "A算法的输入张量之一包括a。", "A算法的输入张量是b。", "A算法的输入张量是c。"]
       【推理过程】……（合并去重，确认均指向A算法的输入）
       【最终答案】A算法的输入张量有a、b、c。

    2. 问题：请帮我搭建一个B流程。
       检索：["B流程需要a流程、b流程", "b流程由aa、bb、cc、dd、ee算法组成"]
       【推理过程】……（先得到B包含a、b；再展开b→aa~ee）
       【最终答案】B流程由a流程和b流程组成；其中b流程由aa、bb、cc、dd、ee算法组成。

    【规范与约束】
    1) 若问题类型属于"kg_query"和"flow_hierarchy"，则仅使用【检索内容】中的信息，禁止编造。若问题类型不属于这两类，则大模型可推理回答。
    2) Algorithm=算法，Flow=流程；InputTensor=输入数据，OutputTensor=输出结果。
    3) 若存在同名流程/算法，需分别给出组成。
    4) 将重复、同义、分散的信息合并成一句完整话术。
    5) 出现“AA由A与B组成，A由a,b,c，B由d,e,f,g”，需整合为：a,b,c构成A；d,e,f,g构成B；最终A与B构成AA。

    现在开始。
    """
    prompt += "\n请一步一步推理，并严格按照【推理过程】和【最终答案】的格式输出。"

    # 5. 请求本地大语言模型（Ollama 默认在 http://localhost:11434）
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:14b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.9,  # 适度提高，促使展开思考
                "top_p": 0.9,
                "num_ctx": 8192,  # 根据显存与模型上下文能力调整
                "repeat_penalty": 1.05
            }

        }
    )

    result_text = response.json()["response"]
    return result_text, fused_results


# === Self-Consistency 封装 ===
def summarize_with_self_consistency(query: str, corpus_lines, corpus_dict, store, bm25_params,
                                    num_samples=10, top_k=15):
    all_answers = []
    all_raw = []

    for i in range(num_samples):
        result_text, fused_results = summarize_with_model(query, corpus_lines, corpus_dict, store, bm25_params, top_k)
        all_raw.append(result_text)

        match = re.search(r'【最终答案】([\s\S]*)', result_text)
        if match:
            final_answer = match.group(1).strip()
            all_answers.append(final_answer)

        print(f"✅ 第 {i+1}/{num_samples} 次采样完成")

    counter = Counter(all_answers)
    most_common_answer, count = counter.most_common(1)[0]

    print("\n=== Self-Consistency 投票统计 ===")
    for ans, cnt in counter.most_common():
        print(f"{cnt} 次: {ans[:80]}...")

    return most_common_answer, all_raw


if __name__ == "__main__":
    file_path = r"D:\lihao\data\bykg2508_text123.txt"
    model_path = r"D:\lihao\SystemonRSDprocessingKG202510\bge-large-zh"
    collection_name = "my_collection1"

    corpus_lines = load_corpus(file_path)
    corpus_dict = {i: text for i, text in enumerate(corpus_lines)}
    store = initialize_vector_store(collection_name, model_path, corpus_lines)
    bm25_params = initialize_inverted_index(corpus_dict)

    print("系统已初始化完成，您可以开始提问（输入‘exit’退出）：")
    start_time = time.time()
    # 添加交互式循环
    while True:
        # 从终端获取用户输入
        query = input("\n请输入您的问题：")

        # 检查是否退出
        if query.lower() in ["exit", "quit", "q"]:
            print("感谢使用，再见！")
            break

        # 处理问题并显示答案
        if query.strip():
            answer, _ = summarize_with_self_consistency(query, corpus_lines, corpus_dict, store, bm25_params, num_samples=10)
            print("\n答案：", answer)
            print(f"⏱️ 本次查询耗时: {round(time.time() - start_time, 2)} 秒")
        else:
            print("问题不能为空，请重新输入。")

        # 重置计时器
        start_time = time.time()


#Zero-shot CoT + self-consistency