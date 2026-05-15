# agent.py  (稳定版：不依赖 AgentExecutor / LLMSingleActionAgent)
import re
import json
from typing import List, Union, Any, Dict

from langchain_ollama import OllamaLLM  # 只要这个能导入就行
from agent_tools import MyAgentTool


class MyOutputParser:
    """
    兼容你原来的输出格式：
    - Final Answer: ...
    - Action: xxx
      Action Input: yyy
    """
    def parse(self, output: str) -> Dict[str, Any]:
        if not output:
            return {"type": "finish", "final": ""}

        if "Final Answer:" in output:
            return {
                "type": "finish",
                "final": output.split("Final Answer:")[-1].strip(),
                "raw": output
            }

        # 允许 Action 1/2/3 这种写法
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\s*:(.*)"
        match = re.search(regex, output, re.DOTALL)
        if not match:
            # 找不到 Action，但有内容：当成最终回答，增强鲁棒性
            if "Action:" not in output and len(output.strip()) > 5:
                return {"type": "finish", "final": output.strip(), "raw": output}
            return {"type": "error", "error": f"无法解析 LLM 输出: `{output}`", "raw": output}

        action = match.group(1).strip()
        action_input = match.group(2).strip()

        # 清理 Observation 之后的内容
        if "\nObservation" in action_input:
            action_input = action_input.split("\nObservation")[0].strip()

        # 去掉引号
        action_input = action_input.strip().strip('"').strip("'")

        # 如果是 JSON，确保可解析（不强制）
        if action_input.startswith("{") and action_input.endswith("}"):
            try:
                json.loads(action_input)
            except Exception:
                # 不是标准 JSON 就保持字符串
                pass

        return {
            "type": "action",
            "tool": action,
            "tool_input": action_input,
            "raw": output
        }


class MyAgent:
    def __init__(self) -> None:
        # 你的提示词模板（保留）
        self.template = """
你是一个专业的遥感数据处理智能体助手。
你必须严格遵守以下规则：
1. **信息完整性原则**。如果用户询问的是关于算法、流程的详细信息，在获取文字答案后，必须尝试调用‘算法或流程结构图展示‘工具。
2. **不要编造参数**。如果用户没有指定卫星（如 Landsat），不要自己添加卫星条件，只搜索时间。
3. **不要死循环**。如果 Find_Data 没找到数据，请直接停止并告诉用户没找到，不要反复尝试不同的参数。
4. **不要重复使用/调用工具**。不循环、重复调用同一个工具。

可用工具：
{tools}

示例：
Question: 大气校正算法的输入张量有些什么？
Thought: 用户询问算法的知识。首先我需要用知识问答助手获取文字答案。
Action: 知识问答助手
Action Input: 大气校正算法的输入张量有些什么？
Observation: 大气校正算法的输入张量...
Thought: 我已经有了文字介绍，但这是一个算法问题，我还需要调用结构图工具来展示其结构。
Action: 算法或流程结构图展示
Action Input: 大气校正算法
Observation: [图表链接/路径]
Thought: 我已经提供了文字介绍和结构图。
Final Answer: 大气校正算法的输入张量...（附带结构图）

**重要：必须严格按照以下格式进行思考和回复（不要遗漏 Action Input）：**
Question: 用户的问题
Thought: 分析用户需求。如果是算法查询，记得要文字+图片双重回答。
Action: 工具名称 (必须是 [{tool_names}] 中的一个)
Action Input: 工具的输入内容 (如果是问答工具，这里填入用户的问题)
Observation: 工具返回的结果 (不要自己填写，等待工具返回)
Thought: (检查是否需要补充结构图？如果需要，继续Action；如果不需要，则总结)
... (可以重复 Thought/Action/Action Input/Observation 步骤)
Final Answer: 最终结果总结

**警告**：
- 决定调用工具时，**Action** 和 **Action Input** 必须同时出现。
- **绝对不要**在 `Action:` 之后直接写 `Final Answer:`。
- 如果使用 [知识问答助手]工具，Action Input 必须是用户完整的问题内容。

现在开始! 必须使用中文回答。
Question: {input}
{agent_scratchpad}
""".strip()

        # LLM：保持你原来的 Ollama 配置
        self.llm = OllamaLLM(
            model="qwen2.5:14b",
            base_url="http://localhost:11434",
            temperature=0.2
        )

        # 工具列表（你原来的 MyAgentTool）
        self.tools = MyAgentTool().tools()
        self.tool_map = {t.name: t for t in self.tools}
        self.tool_names = list(self.tool_map.keys())

        self.parser = MyOutputParser()

    def _render_prompt(self, user_input: str, intermediate_steps: List[Dict[str, str]]) -> str:
        # 拼 agent_scratchpad
        scratchpad = ""
        for step in intermediate_steps:
            scratchpad += step["log"].rstrip() + "\n"
            scratchpad += f"Observation: {step['observation']}\nThought: "

        tools_text = "\n".join([f"{t.name}: {t.description}" for t in self.tools])

        return self.template.format(
            input=user_input,
            agent_scratchpad=scratchpad,
            tools=tools_text,
            tool_names=", ".join(self.tool_names)
        )

    def _call_tool(self, tool_name: str, tool_input: str) -> str:
        if tool_name not in self.tool_map:
            return f"[工具不存在] {tool_name} 不在可用工具列表中：{self.tool_names}"

        tool = self.tool_map[tool_name]

        # langchain tool 兼容：优先 invoke，其次 run，再其次 __call__
        try:
            if hasattr(tool, "invoke"):
                return str(tool.invoke(tool_input))
            if hasattr(tool, "run"):
                return str(tool.run(tool_input))
            return str(tool(tool_input))
        except Exception as e:
            return f"[工具执行异常] {tool_name}: {e}"

    def run(self, input: str) -> str:
        res = self.run_with_details(input)
        return res.get("final_answer", "")

    def run_with_details(self, input: str, max_iterations: int = 5) -> dict:
        """
        返回：
        - final_answer
        - tool_calls: [{tool_name, tool_input, observation, step}]
        - reasoning: 推理过程（从 log 拼出来）
        """
        intermediate_steps: List[Dict[str, str]] = []
        tool_calls = []
        reasoning_logs = []

        used_tools = set()  # 防止重复调用同一工具死循环（符合你的规则）

        for it in range(1, max_iterations + 1):
            prompt = self._render_prompt(input, intermediate_steps)

            # 调 LLM（OllamaLLM 在不同版本有 invoke / predict / __call__）
            try:
                if hasattr(self.llm, "invoke"):
                    llm_out = self.llm.invoke(prompt)
                elif hasattr(self.llm, "predict"):
                    llm_out = self.llm.predict(prompt)
                else:
                    llm_out = self.llm(prompt)
            except Exception as e:
                return {
                    "final_answer": f"LLM 调用失败: {e}",
                    "tool_calls": tool_calls,
                    "reasoning": f"错误原因: {e}",
                    "success": False
                }

            llm_text = str(llm_out).strip()
            parsed = self.parser.parse(llm_text)

            # 记录推理 log（用于前端展示）
            reasoning_logs.append(f"[第{it}轮] LLM输出:\n{llm_text}")

            if parsed["type"] == "finish":
                final = parsed.get("final", "")
                return {
                    "final_answer": final,
                    "tool_calls": tool_calls,
                    "reasoning": self._build_reasoning_text(reasoning_logs, final),
                    "success": True
                }

            if parsed["type"] == "error":
                err = parsed.get("error", "解析失败")
                return {
                    "final_answer": f"解析失败：{err}",
                    "tool_calls": tool_calls,
                    "reasoning": self._build_reasoning_text(reasoning_logs, f"解析失败：{err}"),
                    "success": False
                }

            # action
            tool_name = parsed["tool"]
            tool_input = parsed["tool_input"]

            # 防止重复调用同一工具
            if tool_name in used_tools:
                observation = f"[已阻止重复调用] 工具 {tool_name} 已经用过一次，按规则不再重复调用。"
            else:
                used_tools.add(tool_name)
                observation = self._call_tool(tool_name, tool_input)

            # 记录工具调用
            tool_calls.append({
                "tool_name": tool_name,
                "tool_input": str(tool_input),
                "observation": str(observation),
                "step": it
            })

            # intermediate_steps 用于下一轮 prompt 的 scratchpad
            intermediate_steps.append({
                "log": llm_text,
                "observation": str(observation)
            })

        # 超过最大轮次
        final = "已达到最大推理轮次，仍未得到 Final Answer。建议缩小问题或检查工具返回内容。"
        return {
            "final_answer": final,
            "tool_calls": tool_calls,
            "reasoning": self._build_reasoning_text(reasoning_logs, final),
            "success": False
        }

    def _build_reasoning_text(self, reasoning_steps: List[str], final_answer: str) -> str:
        if not reasoning_steps:
            return f"智能体直接给出了最终答案：{final_answer}"

        text = "智能体推理过程：\n"
        for i, step in enumerate(reasoning_steps, 1):
            text += f"{i}. {step}\n\n"
        text += f"最终结论: {final_answer}"
        return text


if __name__ == "__main__":
    myagent = MyAgent()
    q = "波段计算算法的输出张量是什么?"
    r = myagent.run_with_details(q)
    print(r["final_answer"])
    print(r["reasoning"])
    print(r["tool_calls"])
