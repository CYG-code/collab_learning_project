import asyncio
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

load_dotenv()

app = FastAPI()

# ==========================================
# 1. 核心大平层：WebSocket 连接管理器 (实现广播机制)
# ==========================================
class ConnectionManager:
    def __init__(self):
        # 存储所有在线的客户端网页
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str, sender: str):
        # 将消息瞬间推送到所有人的屏幕上
        for connection in self.active_connections:
            await connection.send_json({"sender": sender, "message": message})

manager = ConnectionManager()

# ==========================================
# 2. 初始化 AutoGen 全局大脑 (Mix-and-Match)
# ==========================================
api_key = os.environ.get("LINGYA_API_KEY")
base_url = "https://api.lingyaai.cn/v1"
custom_model_info = ModelInfo(vision=False, function_calling=True, json_output=False, family="unknown", structured_output=False)

planner_model = OpenAIChatCompletionClient(
    model="gemini-3.1-pro-preview", api_key=api_key, base_url=base_url, model_info=custom_model_info
)
facilitator_model = OpenAIChatCompletionClient(
    model="gpt-4o-mini", api_key=api_key, base_url=base_url, model_info=custom_model_info
)

planner = AssistantAgent(
    name="Planner",
    model_client=planner_model,
    system_message="""你是一个项目规划师。帮助学生拆解难题。给出结构化建议。
    结束发言时，请在末尾加上 'WAIT'。"""
)

facilitator = AssistantAgent(
    name="Facilitator",
    model_client=facilitator_model,
    system_message="""你是一个协作学习的引导者。
    注意：静默观察！只有当学生卡壳时才用苏格拉底式提问。绝对不给直接答案。
    结束发言时，请在末尾加上 'WAIT'。"""
)

termination = TextMentionTermination("WAIT") | MaxMessageTermination(max_messages=3)
team = SelectorGroupChat(participants=[planner, facilitator], model_client=planner_model, termination_condition=termination)

# AI 思考锁：防止多个人同时触发导致 AI 精神分裂
ai_lock = asyncio.Lock()

# ==========================================
# 3. 后台异步任务：处理 AI 逻辑 (完全不阻塞前端)
# ==========================================
async def process_ai_response(user_msg: str):
    async with ai_lock:
        try:
            # 监听 AI 的思考流
            async for msg in team.run_stream(task=user_msg):
                if hasattr(msg, 'content') and isinstance(msg.content, str):
                    if msg.source != "user" and msg.content.strip():
                        display_text = msg.content.replace("WAIT", "").strip()
                        if display_text:
                            # AI 一出结果，立刻广播给所有人
                            await manager.broadcast(display_text, msg.source)
        except Exception as e:
            await manager.broadcast(f"AI 思考出错: {str(e)}", "System")

# ==========================================
# 4. 路由定义：前端网页与 WebSocket 接口
# ==========================================
# 访问主页时，直接返回我们写好的 HTML 前端页面
@app.get("/")
async def get():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# WebSocket 通道：处理实时聊天
@app.websocket("/ws/{username}")
async def websocket_endpoint(websocket: WebSocket, username: str):
    await manager.connect(websocket)
    await manager.broadcast(f"👋 【{username}】 加入了协作学习室", "System")
    try:
        while True:
            # 1. 接收人类发来的消息
            data = await websocket.receive_text()
            
            # 2. 瞬间广播给全房间的人（解除回合制，实现微信体验）
            await manager.broadcast(data, username)
            
            # 3. 告诉 AI 这句话是谁说的，并把任务扔到后台让 AI 去慢慢想
            formatted_msg_for_ai = f"人类学生 [{username}] 说: {data}"
            asyncio.create_task(process_ai_response(formatted_msg_for_ai))
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"🏃 【{username}】 离开了房间", "System")