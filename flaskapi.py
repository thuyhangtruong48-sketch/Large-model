import base64
import os
import re
import time
from typing import List, Tuple, Dict, Any

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

try:
    from hybrid_retriever import HybridRetriever
except Exception as e:
    HybridRetriever = None
    HYBRID_IMPORT_ERROR = f"{type(e).__name__}: {e}"
else:
    HYBRID_IMPORT_ERROR = ""

# ========= 兼容导入：generation 模块 =========
GEN_SRC = None
try:
    from generation08_2 import (
        summarize_with_model,
        load_corpus,
        initialize_vector_store,
        initialize_inverted_index,
    )
    GEN_SRC = "generation08_2"
except Exception as e1:
    try:
        from generation_llm import (
            summarize_with_model,
            load_corpus,
            initialize_vector_store,
            initialize_inverted_index,
        )
        GEN_SRC = "generation_llm"
    except Exception as e2:
        summarize_with_model = None
        load_corpus = None
        initialize_vector_store = None
        initialize_inverted_index = None
        GEN_SRC = f"IMPORT_FAILED: {type(e1).__name__}: {e1} | {type(e2).__name__}: {e2}"

# ========= 兼容导入：agent 模块（避免缺包导致 Flask 起不来） =========
AGENT_SRC = None
MyAgent = None
try:
    from agent import MyAgent
    AGENT_SRC = "agent"
except Exception as e:
    MyAgent = None
    AGENT_SRC = f"AGENT_IMPORT_FAILED: {type(e).__name__}: {e}"

current_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=current_dir)
CORS(app)


# -----------------------------
#  RAG 命中：强确定性“原句证据”
# -----------------------------
ALG_ID_RE = re.compile(r"(Alg[_\s]?(\d{14,20}))", re.IGNORECASE)
FLOW_ID_RE = re.compile(r"(Flow[_\s]?[0-9a-fA-F]{16,64})", re.IGNORECASE)

def _extract_alg_flow_ids(q: str) -> Dict[str, str]:
    q = q or ""
    m1 = ALG_ID_RE.search(q)
    m2 = FLOW_ID_RE.search(q)

    alg_id = ""
    flow_id = ""
    if m1:
        raw = m1.group(1)
        # 统一成 Alg_XXXXXXXX...
        digits = m1.group(2)
        alg_id = f"Alg_{digits}"
    if m2:
        raw = m2.group(1)
        # 统一成 Flow_xxx
        s = raw.replace(" ", "").replace("FLOW", "Flow").replace("flow", "Flow")
        s = s.replace("Flow_", "Flow_")
        flow_id = s if s.startswith("Flow_") else ("Flow_" + s.split("Flow")[-1].lstrip("_"))

    return {"alg_id": alg_id, "flow_id": flow_id}

def _extract_alg_name(q: str) -> str:
    """
    从问题中尽量提取“算法名”：
    例如：'归一化水体指数算法（Alg_...）的输入张量有哪些？'
    取 '归一化水体指数算法'
    """
    if not q:
        return ""
    # 优先取中文括号前：XXX（Alg_...）
    m = re.search(r"([\u4e00-\u9fa5A-Za-z0-9_]+?)\s*（\s*Alg[_\s]?\d{14,20}\s*）", q)
    if m:
        return m.group(1).strip()

    # 再尝试取 “XXX算法” 前缀
    m2 = re.search(r"([\u4e00-\u9fa5A-Za-z0-9_]+算法)", q)
    if m2:
        return m2.group(1).strip()

    return ""

def _normalize_text(s: str) -> str:
    # 去空白、统一中英文括号，方便匹配
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"\s+", "", s)
    return s

def _rag_first_exact_hits(corpus_lines: List[str], question: str, topk: int = 3) -> Tuple[List[str], str]:
    """
    这是你要的“硬命中”：保证证据来自 txt 原句。
    命中策略（从强到弱）：
    1) Alg_ID 精确包含
    2) Flow_ID 精确包含
    3) 算法名精确包含
    4) 关键词（从问题里抽几个词）包含
    """
    if not corpus_lines:
        return [], "corpus_empty"

    q_norm = _normalize_text(question)
    ids = _extract_alg_flow_ids(question)
    alg_id = _normalize_text(ids.get("alg_id", ""))
    flow_id = _normalize_text(ids.get("flow_id", ""))
    alg_name = _normalize_text(_extract_alg_name(question))

    # 先构造匹配 key（按优先级）
    keys = []
    if alg_id:
        keys.append(("alg_id", alg_id))
    if flow_id:
        keys.append(("flow_id", flow_id))
    if alg_name and len(alg_name) >= 2:
        keys.append(("alg_name", alg_name))

    # 额外：从问题里取一些“中文连续词/数字串”当弱关键词
    # 例如 “输入张量/输出/创建者/author/description”等
    weak_terms = []
    for w in ["输入张量", "输出", "输入数据", "原始数据", "创建者", "贡献者", "author", "description", "componentclass", "类别", "类型"]:
        if _normalize_text(w) in q_norm:
            weak_terms.append(_normalize_text(w))
    # 如果问题里没有这些词，也不强行加

    hits = []
    hit_reason = "no_hit"

    # 逐级匹配：先强匹配（Alg/Flow/Name），拿到就返回
    corpus_norm = [(_normalize_text(x), x) for x in corpus_lines]

    for ktype, key in keys:
        tmp = []
        for cn, raw in corpus_norm:
            if key and key in cn:
                tmp.append(raw)
                if len(tmp) >= topk:
                    break
        if tmp:
            hit_reason = f"hit_by_{ktype}:{key}"
            return tmp[:topk], hit_reason

    # 弱匹配：要求同时包含“算法名/AlgID之一” + “弱词” 更可靠
    # 若啥也没有，就只用弱词匹配（可能会泛）
    if weak_terms:
        tmp = []
        for cn, raw in corpus_norm:
            ok = True
            for wt in weak_terms:
                if wt not in cn:
                    ok = False
                    break
            if ok:
                tmp.append(raw)
                if len(tmp) >= topk:
                    break
        if tmp:
            hit_reason = "hit_by_weak_terms:" + ",".join(weak_terms)
            return tmp[:topk], hit_reason

    return [], "no_hit"


def _safe_top_evidence_from_fused(fused_results, topk=3, max_len=260) -> List[str]:
    """
    summarize_with_model 的 fused_results 结构不稳定，做鲁棒提取；
    但注意：这部分不是“硬证据”，只当补充。硬证据来自 corpus_lines 原句。
    """
    ev = []
    try:
        if not fused_results:
            return ev

        if isinstance(fused_results, list):
            items = fused_results[:topk]
        else:
            items = [fused_results]

        for item in items:
            if item is None:
                continue
            if isinstance(item, dict):
                text = (
                    item.get("text")
                    or item.get("content")
                    or item.get("chunk")
                    or item.get("passage")
                    or item.get("doc")
                    or ""
                )
                s = str(text).strip()
                if not s:
                    continue
                s = (s[:max_len] + "…") if len(s) > max_len else s
                ev.append(s)
            else:
                s = str(item).strip()
                if not s:
                    continue
                s = (s[:max_len] + "…") if len(s) > max_len else s
                ev.append(s)
    except Exception:
        return ev
    return ev


class UnifiedSystem:
    def __init__(self):
        # QA/RAG
        self.corpus_lines = None
        self.corpus_dict = None
        self.store = None
        self.bm25_params = None
        self.qa_system_initialized = False

        # Agent
        self.agent_system = None
        self.agent_system_initialized = False

        # Structured-first database-aware retrieval
        self.hybrid_retriever = None
        self.hybrid_retriever_initialized = False

    def initialize_qa_system(self, file_path, model_path, collection_name):
        try:
            print("正在初始化问答系统...")

            if summarize_with_model is None:
                raise Exception(f"generation 模块导入失败：{GEN_SRC}")
            if not os.path.exists(file_path):
                raise Exception(f"语料文件不存在: {file_path}")
            if not os.path.exists(model_path):
                raise Exception(f"模型路径不存在: {model_path}")

            self.corpus_lines = load_corpus(file_path)
            self.corpus_dict = {i: text for i, text in enumerate(self.corpus_lines)}
            self.store = initialize_vector_store(collection_name, model_path, self.corpus_lines)
            self.bm25_params = initialize_inverted_index(self.corpus_dict)

            self.qa_system_initialized = True
            print("问答系统初始化完成！")
            return True

        except Exception as e:
            self.qa_system_initialized = False
            print(f"问答系统初始化失败: {e}")
            return False

    def initialize_agent_system(self):
        try:
            if MyAgent is None:
                raise Exception(f"智能体依赖未就绪：{AGENT_SRC}")

            print("正在初始化智能体系统...")
            self.agent_system = MyAgent()
            self.agent_system_initialized = True
            print("智能体系统初始化完成！")
            return True

        except Exception as e:
            self.agent_system_initialized = False
            print(f"智能体系统初始化失败: {e}")
            return False

    def initialize_hybrid_retriever(self):
        try:
            if HybridRetriever is None:
                raise Exception(HYBRID_IMPORT_ERROR)
            self.hybrid_retriever = HybridRetriever()
            self.hybrid_retriever_initialized = True
            print("结构化检索层初始化完成！")
            return True
        except Exception as e:
            self.hybrid_retriever = None
            self.hybrid_retriever_initialized = False
            print(f"结构化检索层初始化失败: {e}")
            return False

    # ==========================
    # 你要的：所有问题先走 RAG
    # ==========================
    def process_question(self, question: str, mode: str = "auto"):
        """
        mode 保留，但默认你图里的链路：RAG-first -> Agent -> Tools -> Final
        """
        result: Dict[str, Any] = {
            "success": True,
            "type": "",
            "mode_used": "",
            "reasoning": "",
            "final_answer": "",
            "tool_calls": [],
            "image_data": None,

            # ✅ 强制 RAG 反馈（你要截图证明用）
            "rag_checked": True,
            "rag_hit": False,
            "rag_hit_reason": "",
            "rag_evidence_top3": [],  # 来自 txt 原句

            # ✅ 调试信息
            "debug": {
                "generation_source": GEN_SRC,
                "agent_source": AGENT_SRC,
                "qa_ready": self.qa_system_initialized,
                "agent_ready": self.agent_system_initialized,
                "hybrid_ready": self.hybrid_retriever_initialized,
            },

            "response_time": 0,
        }

        t0 = time.time()

        try:
            q = (question or "").strip()
            if not q:
                raise Exception("问题不能为空")

            # ---- Step RAG(强制): 从txt原句中命中证据 ----
            if self.hybrid_retriever_initialized and self.hybrid_retriever is not None:
                hybrid = self.hybrid_retriever.retrieve(q)
                result["debug"]["hybrid_probe"] = hybrid
                if hybrid.get("route") == "structured":
                    structured = hybrid.get("structured_results", {})
                    result.update(
                        {
                            "type": "structured",
                            "mode_used": "structured_neo4j_first",
                            "reasoning": "命中结构化查询层，已直接从 Neo4j 节点、关系或属性返回结果。",
                            "final_answer": structured.get("answer", ""),
                            "tool_calls": [
                                {
                                    "tool_name": "structured_neo4j",
                                    "observation": structured.get("cypher", ""),
                                }
                            ],
                            "rag_checked": False,
                            "rag_hit": True,
                            "rag_hit_reason": "structured_neo4j_hit",
                            "rag_evidence_top3": [],
                            "response_time": round(time.time() - t0, 2),
                        }
                    )
                    return result

            evidence, hit_reason = _rag_first_exact_hits(self.corpus_lines or [], q, topk=3)
            result["rag_evidence_top3"] = evidence
            result["rag_hit"] = bool(evidence)
            result["rag_hit_reason"] = hit_reason

            # （可选）补跑 summarize_with_model 仅用于给更丰富的 reasoning/answer，
            # 但“命中证据”仍以 txt 原句为准。
            qa_reasoning = ""
            qa_answer = ""
            qa_fused_debug = {}
            if self.qa_system_initialized and summarize_with_model is not None:
                try:
                    qa_result = summarize_with_model(
                        q, self.corpus_lines, self.corpus_dict, self.store, self.bm25_params
                    )
                    if isinstance(qa_result, tuple) and len(qa_result) == 3:
                        qa_reasoning, qa_answer, fused = qa_result
                    elif isinstance(qa_result, tuple) and len(qa_result) == 2:
                        qa_reasoning, qa_answer, fused = "", qa_result[0], qa_result[1]
                    else:
                        qa_reasoning, qa_answer, fused = "", str(qa_result), []
                    qa_fused_debug = {
                        "qa_ran": True,
                        "fused_empty": (not fused),
                        "fused_len": (len(fused) if isinstance(fused, list) else None),
                        "fused_preview_top3": _safe_top_evidence_from_fused(fused, topk=3),
                    }
                except Exception as e:
                    qa_fused_debug = {"qa_ran": True, "qa_error": str(e)}
            else:
                qa_fused_debug = {"qa_ran": False}

            result["debug"]["qa_probe"] = qa_fused_debug

            # ---- Step Agent: 统一走 agent（你图里的核心）----
            if not self.agent_system_initialized:
                # Agent没起来也要返回RAG结果
                result["success"] = False
                result["type"] = "rag_only"
                result["mode_used"] = "rag_only"
                result["final_answer"] = (
                    "Agent 未初始化，但已完成 RAG 检索。\n"
                    + ("✅ 命中证据已返回。" if result["rag_hit"] else "❌ 未命中任何原句。")
                )
                result["response_time"] = round(time.time() - t0, 2)
                return result

            agent_payload = self._process_with_agent(q, qa_fallback_answer=qa_answer, qa_fallback_reasoning=qa_reasoning)
            result.update(agent_payload)
            if "LLM 调用失败" in (result.get("final_answer") or "") and result.get("rag_evidence_top3"):
                result["final_answer"] = "本地 Ollama 未连接，先返回 RAG 命中的原句证据：\n" + "\n".join(
                    f"{i}. {text}" for i, text in enumerate(result["rag_evidence_top3"], 1)
                )
            result["type"] = "agent"
            result["mode_used"] = "rag_first_agent"

            result["response_time"] = round(time.time() - t0, 2)
            return result

        except Exception as e:
            result["success"] = False
            result["type"] = "error"
            result["final_answer"] = f"处理过程中出错: {str(e)}"
            result["reasoning"] = f"错误原因: {str(e)}"
            result["response_time"] = round(time.time() - t0, 2)
            return result

    def _process_with_agent(self, question: str, qa_fallback_answer: str = "", qa_fallback_reasoning: str = ""):
        """
        Agent 负责工具调用；但如果 Agent 最终没给答案，可以用 QA 的答案兜底（不影响你要的RAG证据展示）
        """
        agent_result = self.agent_system.run_with_details(question)
        tool_calls = agent_result.get("tool_calls", []) or []

        image_data = None
        for call in tool_calls:
            tname = call.get("tool_name")
            obs = (call.get("observation") or "").strip()
            if tname == "算法或流程结构图展示" and "IMAGE_PATH:" in obs:
                try:
                    image_path = obs.split("IMAGE_PATH:")[1].strip()
                    if os.path.exists(image_path):
                        with open(image_path, "rb") as img_file:
                            b64_data = base64.b64encode(img_file.read()).decode("utf-8")
                            ext = os.path.splitext(image_path)[1].lower().replace(".", "")
                            if ext == "jpg":
                                ext = "jpeg"
                            image_data = f"data:image/{ext};base64,{b64_data}"
                except Exception as e:
                    print(f"读取图片失败: {e}")

        final_answer = (agent_result.get("final_answer") or "").strip()
        reasoning = (agent_result.get("reasoning") or "").strip()

        # 如果 agent 没答出来，用 QA 兜底（你也可以不要）
        if not final_answer and qa_fallback_answer:
            final_answer = qa_fallback_answer
        if not reasoning and qa_fallback_reasoning:
            reasoning = qa_fallback_reasoning

        return {
            "reasoning": reasoning,
            "final_answer": final_answer,
            "tool_calls": tool_calls,
            "image_data": image_data,
        }


unified_system = UnifiedSystem()
system_initialized = False


def initialize_system():
    global system_initialized

    # 你自己的路径（保持你现在的）
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(project_root, "data", "bykg2508_text123.txt")
    model_path = os.path.join(current_dir, "bge-large-zh")
    collection_name = "my_collection1"

    qa_success = unified_system.initialize_qa_system(file_path, model_path, collection_name)
    agent_success = unified_system.initialize_agent_system()
    hybrid_success = unified_system.initialize_hybrid_retriever()

    system_initialized = bool(qa_success or agent_success or hybrid_success)
    print(f"初始化结果 - QA: {qa_success}, Agent: {agent_success}, Hybrid: {hybrid_success}, Overall: {system_initialized}")


@app.route("/")
def index():
    return render_template("RSDP.html")


@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        data = request.get_json() or {}
        question = (data.get("question") or "").strip()
        mode = (data.get("mode") or "auto").strip()

        if not question:
            return jsonify({"success": False, "error": "问题不能为空"}), 400

        if not system_initialized:
            return jsonify(
                {
                    "success": False,
                    "error": "系统未初始化完成，请检查语料/模型/向量库/智能体依赖",
                    "final_answer": "系统暂时不可用",
                    "debug": {
                        "generation_source": GEN_SRC,
                        "agent_source": AGENT_SRC,
                    },
                }
            ), 503

        print(f"收到问题: {question}, mode={mode}")
        result = unified_system.process_question(question, mode)
        return jsonify(result)

    except Exception as e:
        print(f"处理问题时出错: {e}")
        return jsonify(
            {
                "success": False,
                "error": f"系统处理出错: {str(e)}",
                "final_answer": "请稍后重试",
                "debug": {
                    "generation_source": GEN_SRC,
                    "agent_source": AGENT_SRC,
                },
            }
        ), 500


@app.route("/status")
def status():
    return jsonify(
        {
            "status": "running",
            "initialized": system_initialized,
            "qa_system_ready": unified_system.qa_system_initialized,
            "agent_system_ready": unified_system.agent_system_initialized,
            "hybrid_retriever_ready": unified_system.hybrid_retriever_initialized,
            "generation_source": GEN_SRC,
            "agent_source": AGENT_SRC,
            "hybrid_import_error": HYBRID_IMPORT_ERROR,
        }
    )


@app.route("/tools")
def get_available_tools():
    try:
        if not unified_system.agent_system_initialized or unified_system.agent_system is None:
            return jsonify({"tools": [], "error": "智能体系统未初始化/依赖未安装"})

        tools_obj = getattr(unified_system.agent_system, "tools", None)
        if tools_obj is None:
            return jsonify({"tools": [], "error": "MyAgent.tools 不存在"})

        tools = []
        iterable = tools_obj.values() if isinstance(tools_obj, dict) else tools_obj

        for t in iterable:
            if isinstance(t, dict):
                name = t.get("name", "")
                desc = t.get("description", "")
            else:
                name = getattr(t, "name", str(t))
                desc = getattr(t, "description", "")
            tools.append({"name": name, "description": desc})

        return jsonify({"tools": tools})

    except Exception as e:
        return jsonify({"tools": [], "error": str(e)})


if __name__ == "__main__":
    initialize_system()
    app.run(host="127.0.0.1", port=5001, debug=False)
