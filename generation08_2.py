from fusion12_shangquan import query_and_fuse
from graph_store import query_knowledge_graph, get_flow_hierarchy_natural_language  # 确保你有这个模块
from vector_store import MilvusVectorStore
from inverted_store import build_inverted_index
import requests
import time
import re
import json
OLLAMA_BASE_URL = "http://127.0.0.1:11434"


# ============ 全局对话历史 ============
conversation_history = []  # 存储多轮对话


def load_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def initialize_vector_store(collection_name, model_path, corpus_lines):
    store = MilvusVectorStore(collection_name=collection_name, model_path=model_path)
    store.create_collection(corpus_lines)
    return store


def initialize_inverted_index(corpus_dict):
    return build_inverted_index(corpus_dict)


# === 指代消解函数 ===
def resolve_reference(user_question: str, conversation_history: list) -> str:
    """
    简单指代消解：如果问题里包含 '这个算法'、'该流程'，则替换为上一次提到的实体
    """
    if not conversation_history:
        return user_question

    last_answer = conversation_history[-1]["assistant"]
    last_question = conversation_history[-1]["user"]

    # 查找上轮问题或答案里提到的算法/流程名
    match = re.search(r'([\u4e00-\u9fa5a-zA-Z0-9_]+)(?:算法|流程)', last_question + last_answer)
    if match:
        entity = match.group(0)  # 保留 "xxx算法" / "yyy流程"
        if "这个算法" in user_question:
            user_question = user_question.replace("这个算法", entity)
        if "该算法" in user_question:
            user_question = user_question.replace("该算法", entity)
        if "该流程" in user_question:
            user_question = user_question.replace("该流程", entity)
        if "这个流程" in user_question:
            user_question = user_question.replace("这个流程", entity)

    return user_question


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
       - 示例："请搭建图像融合流程", "如何搭建辐射定标流程", "地形校正流程由哪些部分组成", "搭建处理图像流程的步骤是什么？"

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
    global conversation_history

    # 先做指代消解
    query = resolve_reference(query, conversation_history)

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
    history_text = ""
    for turn in conversation_history[-5:]:
        history_text += f"用户：{turn['user']}\n助手：{turn['assistant']}\n"

    full_context = f"【对话历史】:\n{history_text}\n\n【检索内容】：\n{context}\n\n{kg_response}"
    print("完整上下文内容:\n", full_context)

    # 4. 构造 Few-shot CoT prompt
    prompt = f"""
你是一个遥感数据处理方面的算法专家。请基于【检索内容】进行逐步推理，再给出最终答案。
注意：必须按照示例的【推理过程】+【最终答案】格式输出，并且【最终答案】必须完整保留层级关系，逐层展开说明。禁止将层级压缩到一句话。

下面是几个示例（请模仿这种风格来回答）：
示例1：
【用户问题】A算法的输入数据/输入张量有哪些?
【检索内容】["A算法的输入张量是a。", "A算法的输入数据之一包括a。", "A算法的输入张量是b。", "A算法的输入张量是c。"]
【推理过程】从检索中提取到A算法的输入有a、a、b、c。去重后得到a、b、c。
【最终答案】A算法的输入张量有a、b、c。

示例2：
【用户问题】请帮我搭建一个B流程。
【检索内容】["B流程需要a流程、b流程", "b流程又由aa、bb、cc、dd、ee算法组成", "a流程又由ff、gg算法组成"]
【推理过程】检索表明B流程由a流程和b流程组成；再展开b流程，包含aa、bb、cc、dd、ee算法；再展开a流程，包含ff、gg算法。
【最终答案】B流程由a流程和b流程组成；其中b流程由aa、bb、cc、dd、ee算法组成，a流程由ff、gg算法组成。

示例3：
【用户问题】食堂有23个苹果，如果他们用掉20个后又买了6个。他们现在有多少个苹果？
【检索内容】[无]
【推理过程】食堂原来有23个苹果，他们用掉20个，所以还有23-20=3个。他们又买了6个，所以现在有6+3=9个。
【最终答案】他们现在有9个苹果。

现在，请回答以下问题：

【用户问题】：
{query}

【检索内容】：
{full_context}

【输出要求】：
1. 必须先输出【推理过程】，再输出【最终答案】。
2. 严格模仿示例的风格。
3. 若问题类型属于"kg_query"和"flow_hierarchy"，则仅使用【检索内容】中的信息，禁止编造。若问题类型不属于这两类，则大模型可推理回答。
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
                "temperature": 0.7,  # 适度提高，促使展开思考
                "top_p": 0.9,
                "num_ctx": 8192,  # 根据显存与模型上下文能力调整
                "repeat_penalty": 1.05
            }

        }
    )

    result_text = response.json()["response"]
    # 保存到对话历史
    conversation_history.append({"user": query, "assistant": result_text})
    return result_text, fused_results


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
            answer, _ = summarize_with_model(query, corpus_lines, corpus_dict, store, bm25_params)
            print("\n答案：", answer)
            print(f"⏱️ 本次查询耗时: {round(time.time() - start_time, 2)} 秒")
        else:
            print("问题不能为空，请重新输入。")

        # 重置计时器
        start_time = time.time()
#Few-shot CoT