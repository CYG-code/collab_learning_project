import asyncio
import os
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

load_dotenv()

async def main():
    api_key = os.environ.get("LINGYA_API_KEY")
    base_url = "https://api.lingyaai.cn/v1"

    # 告诉 AutoGen：这是第三方中转模型，不要用严格的官方 Token 校验逻辑报错
    custom_model_info = ModelInfo(
        vision=False, 
        function_calling=True, 
        json_output=False, 
        family="unknown"
    )

    # ==========================================
    # 1. 挂载灵芽 API 上的真实可用模型
    # ==========================================
    
    # Planner 使用最新最强的 Gemini 3.1 Pro Preview (依据截图)
    planner_model = OpenAIChatCompletionClient(
        model="gemini-3.1-pro-preview", 
        api_key=api_key,
        base_url=base_url,
        model_info=custom_model_info 
    )
    
    # Facilitator 使用高性价比的 gpt-4o-mini (依据截图)
    facilitator_model = OpenAIChatCompletionClient(
        model="gpt-4o-mini", 
        api_key=api_key,
        base_url=base_url,
        model_info=custom_model_info 
    )

    # ==========================================
    # 2. 定义角色 (逻辑保持不变)
    # ==========================================
    planner = AssistantAgent(
        name="Planner",
        model_client=planner_model,
        system_message="""你是一个项目规划师。
        你的职责是帮助学生将复杂的难题拆解为可执行的步骤。
        你要逻辑清晰，直接给出结构化的建议。
        如果任务已经完成，请回复 'TERMINATE' 来结束讨论。"""
    )

    facilitator = AssistantAgent(
        name="Facilitator",
        model_client=facilitator_model,
        system_message="""你是一个协作学习的引导者。
        注意：尽量保持静默！只有当学生（Student）感到困惑、思路卡壳，或者主动向你求助时，你才发言。
        绝对不要直接给出答案。你需要用苏格拉底式的提问，启发学生自己思考。"""
    )

    student = UserProxyAgent(
        name="Student",
        description="参与协作学习的真实人类学生。"
    )

    # ==========================================
    # 3. 运行群聊
    # ==========================================
    termination_condition = TextMentionTermination("TERMINATE")

    team = SelectorGroupChat(
        participants=[student, planner, facilitator],
        model_client=planner_model, # 群聊的路由判断也交给最聪明的 Gemini 来做
        termination_condition=termination_condition,
    )

    print("🚀 协作学习群聊已启动！(输入 'TERMINATE' 结束)\n")
    initial_task = "我们需要设计一个方案：如何利用废旧的安卓手机搭建一个校园局域网内的个人博客服务器？大家有什么思路？"
    
    await Console(team.run_stream(task=initial_task))

if __name__ == "__main__":
    asyncio.run(main())