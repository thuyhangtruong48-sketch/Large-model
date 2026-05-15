import os
import re
import random
import json
from typing import Dict, List, Any, Optional

from langchain_core.tools import Tool

# ========= 兼容导入：generation 模块（修复你当前的 generation08_2liushi 不存在） =========
GEN_SRC = None
try:
    # 你项目里确实有 generation08_2.py（优先）
    from generation08_2 import (
        summarize_with_model,
        load_corpus,
        initialize_vector_store,
        initialize_inverted_index,
    )
    GEN_SRC = "generation08_2"
except Exception as e1:
    try:
        # 备用：generation_llm.py
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


def _safe_trim(s: str, n: int = 180) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\r", " ").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "..."


def _format_evidence_topk(fused_results: Any, k: int = 3) -> str:
    """
    尝试把 fused_results 里前 k 条内容抽出来，作为“RAG确实检索到”的证据。
    兼容 fused_results 为 list[dict]/list[str]/None 的情况。
    """
    if not fused_results:
        return "EVIDENCE_TOP3: (empty)"

    items = fused_results
    if not isinstance(items, list):
        items = [items]

    lines = []
    for idx, it in enumerate(items[:k], 1):
        if isinstance(it, dict):
            # 常见字段名做兼容
            txt = (
                it.get("text")
                or it.get("chunk")
                or it.get("content")
                or it.get("doc")
                or it.get("passage")
                or it.get("sentence")
                or ""
            )
            src = it.get("source") or it.get("file") or it.get("id") or ""
            score = it.get("score") or it.get("sim") or it.get("bm25") or ""
            line = f"{idx}. src={_safe_trim(src, 60)} score={score} | { _safe_trim(txt, 220) }"
        else:
            line = f"{idx}. { _safe_trim(it, 240) }"
        lines.append(line)

    return "EVIDENCE_TOP3:\n" + "\n".join(lines)


class AdvancedRemoteSensingDataTool:
    """高级遥感数据检索工具类"""

    def __init__(self) -> None:
        # 你的路径配置（保持不动）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.current_dir = current_dir
        self.project_root = project_root
        self.images_dir = os.path.join(current_dir, "images")
        self.base_path = os.path.join(current_dir, "test2")

        self.satellite_mapping = {
            "高分一号": "GF1",
            "高分二号": "GF2",
            "高分三号": "GF3",
            "高分七号": "GF7",
            "GF1": "GF1",
            "GF2": "GF2",
            "GF3": "GF3",
            "GF7": "GF7",
            "资源一号": "ZY1",
            "资源二号": "ZY2",
            "ZY1": "ZY1",
            "ZY2": "ZY2",
        }
        self.sensor_mapping = {
            "全色多光谱相机1": "PMS1",
            "全色多光谱相机2": "PMS2",
            "PMS1": "PMS1",
            "PMS2": "PMS2",
            "立体相机": "DLC",
            "DLC": "DLC",
            "多光谱相机": "MUX",
            "MUX": "MUX",
            "红外相机": "IRS",
            "IRS": "IRS",
        }
        self.product_levels = ["L1A", "L1B", "L2A", "L2B"]

        # 初始化问答RAG
        self._initialize_qa_rag()

    def _initialize_qa_rag(self):
        """初始化TCHR_RAG组件"""
        self.corpus_lines = []
        self.corpus_dict = {}
        self.store = None
        self.bm25_params = None
        self.rag_ready = False

        try:
            if summarize_with_model is None:
                raise Exception(f"generation 模块导入失败：{GEN_SRC}")

            # 你自己的配置
            corpus_file_path = os.path.join(self.project_root, "data", "bykg2508_text1.txt")
            model_path = os.path.join(self.current_dir, "bge-large-zh")
            collection_name = "my_collection1"

            if not os.path.exists(corpus_file_path):
                raise Exception(f"语料文件不存在: {corpus_file_path}")
            if not os.path.exists(model_path):
                raise Exception(f"模型路径不存在: {model_path}")

            self.corpus_lines = load_corpus(corpus_file_path)
            self.corpus_dict = {i: text for i, text in enumerate(self.corpus_lines)}

            self.store = initialize_vector_store(collection_name, model_path, self.corpus_lines)
            self.bm25_params = initialize_inverted_index(self.corpus_dict)

            self.rag_ready = True
            print(f"[RAG_INIT_OK] source={GEN_SRC} corpus_lines={len(self.corpus_lines)} collection={collection_name}")

        except Exception as e:
            print(f"[RAG_INIT_FAIL] {e}")
            self.rag_ready = False


class MyAgentTool(AdvancedRemoteSensingDataTool):
    def tools(self):
        return [
            Tool(
                name="算法或流程结构图展示",
                description=(
                    "仅在知识问答助手执行完毕后，且用户问题明确涉及某个具体流程或算法名称时调用。"
                    "Action Input 不要是整句话，应该是提取出的算法或流程名称。"
                ),
                func=self.algpic,
            ),
            Tool(
                name="知识问答助手",
                description=(
                    "专门用于回答用户输入的关于遥感数据处理相关的问题。"
                    "Action Input 必须是用户完整的原始问题字符串。"
                    "拿到文字答案后，如问题包含具体算法/流程名称，必须继续调用“算法或流程结构图展示”。"
                ),
                func=self.TCHRRAG_qa,
            ),
            Tool(
                name="归一化植被指数",
                description=(
                    "计算归一化植被指数。输入应该是影像数据检索返回的数据文件名称列表或单个文件名。"
                    "将为每个文件生成一个-1到1之间的随机NDVI值。"
                ),
                func=self.calculate_ndvi,
            ),
            Tool(
                name="影像数据检索",
                description="根据多种条件智能查找遥感数据文件。支持按日期/卫星/传感器/经纬度/产品级别等条件搜索。",
                func=self.find_data,
            ),
            Tool(
                name="RPC校正",
                description="进行rpc校正。输入为影像数据检索返回的数据文件名称列表，或者单个文件名。",
                func=self.rpc,
            ),
        ]

    # ===================== RAG 工具：强制带“证据” =====================
    def _safe_top_evidence_from_fused(fused_results, topk=3, max_len=300):
        """
        从 fused_results 里尽可能抽取原文证据。结构不确定也能兼容。
        返回 list[str]，每条尽量是“原句/片段”。
        """
        ev = []
        if not fused_results:
            return ev

        try:
            for item in fused_results:
                if item is None:
                    continue

                if isinstance(item, dict):
                    text = (
                            item.get("text") or item.get("content") or item.get("chunk")
                            or item.get("passage") or item.get("doc")
                    )
                    if text:
                        s = str(text).strip()
                        if s:
                            ev.append(s[:max_len])

                elif isinstance(item, (list, tuple)):
                    # 常见 (text, score) / (id, text, score)
                    for x in item:
                        if isinstance(x, str) and len(x.strip()) > 0:
                            ev.append(x.strip()[:max_len])
                            break

                elif isinstance(item, str):
                    s = item.strip()
                    if s:
                        ev.append(s[:max_len])

                if len(ev) >= topk:
                    break
        except Exception:
            pass

        # 去重
        dedup = []
        seen = set()
        for s in ev:
            key = s.strip()
            if key and key not in seen:
                seen.add(key)
                dedup.append(key)
            if len(dedup) >= topk:
                break
        return dedup

    def TCHRRAG_qa(self, question: str) -> str:
        """
        ✅ 强制：必须给出命中的原句证据（top3）
        ✅ 如果没命中：明确返回“未命中”，禁止 LLM 编造句子1/2/3
        """
        try:
            if not question or question.strip() == "":
                return "【RAG】问题为空。"

            if not getattr(self, "rag_ready", False) or self.store is None or self.bm25_params is None:
                return "【RAG】系统未就绪：语料/模型/向量库未初始化成功。"

            qa_result = summarize_with_model(
                question, self.corpus_lines, self.corpus_dict, self.store, self.bm25_params
            )
            if isinstance(qa_result, tuple) and len(qa_result) == 3:
                reasoning, final_answer, fused_results = qa_result
            elif isinstance(qa_result, tuple) and len(qa_result) == 2:
                reasoning, final_answer, fused_results = "", qa_result[0], qa_result[1]
            else:
                reasoning, final_answer, fused_results = "", str(qa_result), []

            evidence = _safe_top_evidence_from_fused(fused_results, topk=3)

            # ✅ 关键：没证据就直接判定“未命中”，不给 LLM 编的空间
            if not evidence:
                return (
                    "【RAG】未命中任何原句证据（fused_results为空或无法抽取文本）。\n"
                    "【EVIDENCE_TOP3】\n"
                    "- （无）\n"
                    "【FINAL】未在语料中检索到与问题直接匹配的原句，请换一个更贴近语料原句的问法（例如带Alg_ID）。"
                )

            # ✅ 有证据：把原句明确吐出来（这就是你要的“可证明走RAG”）
            out = []
            out.append("【RAG】命中证据如下（来自语料/检索融合结果）：")
            out.append("【EVIDENCE_TOP3】")
            for i, s in enumerate(evidence, 1):
                out.append(f"- 证据{i}: {s}")
            out.append("【FINAL】" + (final_answer or "").strip())
            return "\n".join(out)

        except Exception as e:
            return f"【RAG】问答工具异常: {str(e)}"

    # ===================== 结构图工具 =====================
    def algpic(self, input_str: str) -> str:
        """用于展示用户查询中涉及到的算法或者流程的结构图。"""
        try:
            key = (input_str or "").strip()
            if not key:
                return "未提供算法/流程名称。"

            if not os.path.exists(self.images_dir):
                return f"错误：无法找到图片存储目录: {self.images_dir}"

            files = os.listdir(self.images_dir)
            target_image_path = None
            best_score = 0.0

            for file_name in files:
                if not file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                    continue
                name_stem = os.path.splitext(file_name)[0]

                file_chars = set(name_stem)
                input_chars = set(key)
                common_chars = file_chars.intersection(input_chars)
                match_ratio = len(common_chars) / max(len(file_chars), 1)

                if match_ratio >= 0.75 and len(name_stem) > 1:
                    if match_ratio > best_score:
                        best_score = match_ratio
                        target_image_path = os.path.join(self.images_dir, file_name)

            if target_image_path:
                return f"已找到结构图，路径标记: IMAGE_PATH:{target_image_path}"
            return f"未搜索到与‘{key}’匹配的图片。"

        except Exception as e:
            return f"结构图匹配错误：{e}"

    # ===================== NDVI / RPC 示例工具 =====================
    def calculate_ndvi(self, input: str) -> str:
        try:
            if not input or input.strip() == "":
                return "未找到任何数据文件，无法计算NDVI值。"

            file_names = self._parse_input_files(input)
            if not file_names:
                return "错误：未找到有效的数据文件名称，无法计算NDVI值。"

            results = []
            for i, file_name in enumerate(file_names, 1):
                ndvi_value = round(random.uniform(-1, 1), 3)
                results.append(f"{i}. {file_name}: NDVI = {ndvi_value}")

            return f"NDVI计算结果（共{len(results)}个文件）:\n" + "\n".join(results)
        except Exception as e:
            return f"NDVI计算错误: {e}"

    def rpc(self, input: str) -> str:
        try:
            if not input or input.strip() == "":
                return "未找到任何数据文件，无法进行rpc校正。"

            file_names = self._parse_input_files(input)
            if not file_names:
                return "未找到有效的数据文件，无法进行rpc校正。"

            results = [fn[::-1] for fn in file_names]

            if len(results) == 1:
                return f"数据进行rpc校正后的结果为: {results[0]}"
            formatted_results = [f"{i}. {r}" for i, r in enumerate(results, 1)]
            return f"rpc校正结果（共{len(results)}个）:\n" + "\n".join(formatted_results)
        except Exception as e:
            return f"无法进行rpc校正：{str(e)}"

    def _parse_input_files(self, input: str) -> List[str]:
        file_names = []
        if input.startswith("[") and input.endswith("]"):
            input_content = input[1:-1]
            file_names = [name.strip().strip("\"'") for name in input_content.split(",") if name.strip()]
        else:
            file_lines = input.strip().split("\n")
            file_names = [line.strip() for line in file_lines if line.strip()]

        if len(file_names) == 1 and "," in file_names[0]:
            file_names = [name.strip().strip("\"'") for name in file_names[0].split(",") if name.strip()]

        return [name for name in file_names if name]

    # ===================== 影像数据检索（你原逻辑保留） =====================
    def find_data(self, input: str) -> str:
        try:
            if not os.path.exists(self.base_path):
                return f"数据目录不存在: {self.base_path}"

            search_text = input
            try:
                if "{" in input and "}" in input:
                    json_str = input[input.find("{") : input.rfind("}") + 1]
                    data = json.loads(json_str)
                    values = [str(v) for v in data.values()]
                    search_text = " ".join(values)
            except Exception:
                pass

            conditions = self.parse_search_conditions(search_text)

            all_folders = self.get_all_data_folders()
            if not all_folders:
                return "数据目录为空。"

            matching_folders = self.filter_folders(all_folders, conditions)
            if not matching_folders:
                return f"未找到数据。解析到的条件: {conditions}。请尝试更简单的查询，如'2022年7月'。"

            return self.format_search_results(matching_folders, conditions)

        except Exception as e:
            return f"数据检索出错: {str(e)}"

    def parse_search_conditions(self, query: str) -> Dict[str, Any]:
        conditions = {}
        conditions.update(self._parse_date_conditions(query))
        conditions.update(self._parse_satellite_sensor_conditions(query))
        conditions.update(self._parse_geolocation_conditions(query))
        conditions.update(self._parse_product_level_conditions(query))
        conditions.update(self._parse_serial_number_conditions(query))
        return conditions

    def _parse_date_conditions(self, query: str) -> Dict[str, Any]:
        conditions = {}
        hyphen_ym_match = re.search(r"(\d{4})-(\d{1,2})", query)
        if hyphen_ym_match:
            year, month = hyphen_ym_match.groups()
            conditions["year_month"] = f"{year}{month.zfill(2)}"

        full_date_match = re.search(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日", query)
        if full_date_match:
            year, month, day = full_date_match.groups()
            conditions["exact_date"] = f"{year}{month.zfill(2)}{day.zfill(2)}"
            return conditions

        year_month_match = re.search(r"(\d{4})年\s*(\d{1,2})月", query)
        if year_month_match:
            year, month = year_month_match.groups()
            conditions["year_month"] = f"{year}{month.zfill(2)}"
            return conditions

        year_match = re.search(r"(\d{4})年", query)
        if year_match:
            conditions["year"] = year_match.group(1)
            return conditions

        return conditions

    def _parse_satellite_sensor_conditions(self, query: str) -> Dict[str, Any]:
        conditions = {}
        for cn, code in self.satellite_mapping.items():
            if cn in query:
                conditions["satellite"] = code
                break
        for cn, code in self.sensor_mapping.items():
            if cn in query:
                conditions["sensor"] = code
                break
        return conditions

    def _parse_geolocation_conditions(self, query: str) -> Dict[str, Any]:
        conditions = {}
        lon_match = re.search(r"东?经\s*(\d+\.?\d*)度?", query)
        lat_match = re.search(r"北?纬\s*(\d+\.?\d*)度?", query)

        if lon_match and lat_match:
            conditions["longitude"] = float(lon_match.group(1))
            conditions["latitude"] = float(lat_match.group(1))
            conditions["coordinate_type"] = "single_point"

        lon_range_match = re.search(r"东?经\s*(\d+\.?\d*)\s*到\s*(\d+\.?\d*)度?", query)
        lat_range_match = re.search(r"北?纬\s*(\d+\.?\d*)\s*到\s*(\d+\.?\d*)度?", query)

        if lon_range_match and lat_range_match:
            conditions["longitude_range"] = {
                "min": float(lon_range_match.group(1)),
                "max": float(lon_range_match.group(2)),
            }
            conditions["latitude_range"] = {
                "min": float(lat_range_match.group(1)),
                "max": float(lat_range_match.group(2)),
            }
            conditions["coordinate_type"] = "range"

        return conditions

    def _parse_product_level_conditions(self, query: str) -> Dict[str, Any]:
        conditions = {}
        for level in self.product_levels:
            if level in query.upper():
                conditions["product_level"] = level
                break

        if "L1A" in query or "1A" in query or "一级A" in query:
            conditions["product_level"] = "L1A"
        elif "L1B" in query or "1B" in query or "一级B" in query:
            conditions["product_level"] = "L1B"
        elif "L2A" in query or "2A" in query or "二级A" in query:
            conditions["product_level"] = "L2A"
        elif "L2B" in query or "2B" in query or "二级B" in query:
            conditions["product_level"] = "L2B"

        return conditions

    def _parse_serial_number_conditions(self, query: str) -> Dict[str, Any]:
        conditions = {}
        serial_match = re.search(r"(\d{10})", query)
        if serial_match:
            conditions["serial_number"] = serial_match.group(1)
        return conditions

    def get_all_data_folders(self) -> List[Dict[str, Any]]:
        folders = []
        if not os.path.exists(self.base_path):
            return folders
        for item in os.listdir(self.base_path):
            item_path = os.path.join(self.base_path, item)
            if os.path.isdir(item_path):
                folder_info = self.parse_folder_name(item)
                if folder_info:
                    folders.append(folder_info)
        return folders

    def parse_folder_name(self, folder_name: str) -> Optional[Dict[str, Any]]:
        parts = folder_name.split("_")
        if len(parts) < 6:
            return None
        try:
            folder_info = {
                "satellite": parts[0],
                "sensor": parts[1],
                "longitude_str": parts[2],
                "latitude_str": parts[3],
                "date_str": parts[4],
                "product_info": parts[5],
                "full_name": folder_name,
            }

            lon_match = re.search(r"[EW](\d+\.?\d*)", parts[2])
            if lon_match:
                folder_info["longitude"] = float(lon_match.group(1))
                if "W" in parts[2]:
                    folder_info["longitude"] = -folder_info["longitude"]

            lat_match = re.search(r"[NS](\d+\.?\d*)", parts[3])
            if lat_match:
                folder_info["latitude"] = float(lat_match.group(1))
                if "S" in parts[3]:
                    folder_info["latitude"] = -folder_info["latitude"]

            date_match = re.search(r"(\d{4})(\d{2})(\d{2})", parts[4])
            if date_match:
                folder_info["year"] = date_match.group(1)
                folder_info["month"] = date_match.group(2)
                folder_info["day"] = date_match.group(3)
                folder_info["full_date"] = parts[4]
                folder_info["year_month"] = date_match.group(1) + date_match.group(2)

            level_match = re.search(r"([Ll]\d+[A-Za-z])", parts[5])
            if level_match:
                folder_info["product_level"] = level_match.group(1).upper()

            serial_match = re.search(r"(\d{10})", parts[5])
            if serial_match:
                folder_info["serial_number"] = serial_match.group(1)

            return folder_info
        except Exception:
            return None

    def filter_folders(self, folders: List[Dict[str, Any]], conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [f for f in folders if self._matches_conditions(f, conditions)]

    def _matches_conditions(self, folder: Dict[str, Any], conditions: Dict[str, Any]) -> bool:
        if "exact_date" in conditions and folder.get("full_date") != conditions["exact_date"]:
            return False
        if "year_month" in conditions and folder.get("year_month") != conditions["year_month"]:
            return False
        if "year" in conditions and folder.get("year") != conditions["year"]:
            return False

        if "satellite" in conditions and folder.get("satellite") != conditions["satellite"]:
            return False
        if "sensor" in conditions and folder.get("sensor") != conditions["sensor"]:
            return False

        if conditions.get("coordinate_type") == "single_point":
            lon_diff = abs(folder.get("longitude", 0) - conditions.get("longitude", 0))
            lat_diff = abs(folder.get("latitude", 0) - conditions.get("latitude", 0))
            if lon_diff > 0.1 or lat_diff > 0.1:
                return False

        if conditions.get("coordinate_type") == "range":
            lon = folder.get("longitude", 0)
            lat = folder.get("latitude", 0)
            lon_range = conditions["longitude_range"]
            lat_range = conditions["latitude_range"]
            if not (lon_range["min"] <= lon <= lon_range["max"] and lat_range["min"] <= lat <= lat_range["max"]):
                return False

        if "product_level" in conditions and folder.get("product_level") != conditions["product_level"]:
            return False
        if "serial_number" in conditions and folder.get("serial_number") != conditions["serial_number"]:
            return False

        return True

    def format_search_results(self, folders: List[Dict[str, Any]], conditions: Dict[str, Any]) -> str:
        folder_names = [folder["full_name"] for folder in folders]
        return "\n".join(folder_names)
