import asyncio
import os
import chainlit as cl
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

load_dotenv()

# ==========================================
# 1. 基础配置 (从 .env 读取灵芽配置)
# ==========================================
api_key = os.environ.get("LINGYA_API_KEY")
base_url = "https://api.lingyaai.cn/v1"
custom_model_info = ModelInfo(vision=False, function_calling=True, json_output=False, family="unknown")

planner_model = OpenAIChatCompletionClient(
    model="gemini-3.1-pro-preview", api_key=api_key, base_url=base_url, model_info=custom_model_info
)
facilitator_model = OpenAIChatCompletionClient(
    model="gpt-4o-mini", api_key=api_key, base_url=base_url, model_info=custom_model_info
)

# ==========================================
# 2. 网页刷新时的初始化动作
# ==========================================
@cl.on_chat_start
async def start_chat():
    # 定义 AI 角色
    planner = AssistantAgent(
        name="Planner",
        model_client=planner_model,
        system_message="""你是一个项目规划师。
        你的职责是帮助学生拆解复杂的难题。你要逻辑清晰，直接给出结构化的建议。
        当你觉得当前阶段的规划已经给完，需要等待学生思考或反馈时，请在回复的最末尾加上 'WAIT' 结束本轮发言。"""
    )

    facilitator = AssistantAgent(
        name="Facilitator",
        model_client=facilitator_model,
        system_message="""你是一个协作学习的引导者。
        注意：尽量保持静默！只有当学生思路卡壳，或者主动向你求助时，你才发言。
        绝对不要直接给出答案。用苏格拉底式的提问，启发学生自己思考。
        当你的提问结束后，请在回复的最末尾加上 'WAIT' 结束本轮发言。"""
    )

    # 终止条件：当 AI 说 WAIT，或者对话超过3轮（防止 AI 之间没完没了地互聊）
    termination = TextMentionTermination("WAIT") | MaxMessageTermination(max_messages=3)

    # 创建群聊 (剔除 UserProxyAgent，由网页输入接管用户发言)
    team = SelectorGroupChat(
        participants=[planner, facilitator],
        model_client=planner_model, 
        termination_condition=termination,
    )
    
    # 将创建好的团队存入当前用户的网页 Session 中
    cl.user_session.set("team", team)
    
    # 在网页端发送欢迎语
    await cl.Message(
        content="🚀 **协作学习多智能体系统已启动！**\n\n我是后台系统。现在 `Planner` 和 `Facilitator` 已经进入了聊天室。你可以把你的课题发出来了（比如：*如何用废旧安卓手机搭服务器？*）",
        author="System"
    ).send()

# ==========================================
# 3. 处理用户在网页端的输入
# ==========================================
@cl.on_message
async def main(message: cl.Message):
    # 从 Session 中取出刚才初始化的团队
    team = cl.user_session.get("team")
    
    # 将用户在网页的输入发给 AI 群聊，并实时捕获它们的讨论流
    async for msg in team.run_stream(task=message.content):
        
        # 过滤出包含真实文本内容的回复
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            # 忽略系统回显
            if msg.source != "user" and msg.content.strip():
                # 清理掉触发词 WAIT，让网页界面更美观
                display_text = msg.content.replace("WAIT", "").strip()
                
                # 将 AI 的回复推送到网页上显示，并标注是哪个角色的发言
                await cl.Message(content=display_text, author=msg.source).send()