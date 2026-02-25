import asyncio
import os
import json
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
# 1. 核心大平层：基于 JSON 的 WebSocket 管理器
# ==========================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        # 将 JSON 字典推送到所有人的屏幕上
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# ==========================================
# 2. 初始化 AutoGen 全局大脑
# ==========================================
api_key = os.environ.get("LINGYA_API_KEY")
base_url = "https://api.lingyaai.cn/v1"
custom_model_info = ModelInfo(vision=False, function_calling=True, json_output=False, family="unknown", structured_output=False)

planner_model = OpenAIChatCompletionClient(model="gemini-3.1-pro-preview", api_key=api_key, base_url=base_url, model_info=custom_model_info)
facilitator_model = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=api_key, base_url=base_url, model_info=custom_model_info)

planner = AssistantAgent(
    name="Planner",
    model_client=planner_model,
    system_message="你是一个项目规划师。帮助学生拆解难题。给出结构化建议。结束发言时请加 'WAIT'。"
)
facilitator = AssistantAgent(
    name="Facilitator",
    model_client=facilitator_model,
    system_message="你是一个协作学习的引导者。只有学生卡壳时才用苏格拉底式提问。结束发言时请加 'WAIT'。"
)

termination = TextMentionTermination("WAIT") | MaxMessageTermination(max_messages=3)
team = SelectorGroupChat(participants=[planner, facilitator], model_client=planner_model, termination_condition=termination)

ai_lock = asyncio.Lock()

# ==========================================
# 3. 后台任务：处理 AI 逻辑并发送状态
# ==========================================
async def process_ai_response(user_msg: str):
    async with ai_lock:
        # 任务开始：广播 Planner 正在思考
        await manager.broadcast({"type": "typing", "sender": "Planner", "is_typing": True})
        try:
            async for msg in team.run_stream(task=user_msg):
                if hasattr(msg, 'content') and isinstance(msg.content, str):
                    if msg.source != "user" and msg.content.strip():
                        display_text = msg.content.replace("WAIT", "").strip()
                        if display_text:
                            # 发送真实消息
                            await manager.broadcast({"type": "message", "sender": msg.source, "message": display_text})
        except Exception as e:
            await manager.broadcast({"type": "message", "sender": "System", "message": f"AI 思考出错: {str(e)}"})
        finally:
            # 任务结束：取消思考状态
            await manager.broadcast({"type": "typing", "sender": "Planner", "is_typing": False})

# ==========================================
# 4. WebSocket 路由：处理前端发来的 JSON
# ==========================================
@app.get("/")
async def get():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws/{username}")
async def websocket_endpoint(websocket: WebSocket, username: str):
    await manager.connect(websocket)
    await manager.broadcast({"type": "message", "sender": "System", "message": f"👋 【{username}】 加入了协作空间"})
    try:
        while True:
            # 接收前端发来的 JSON 数据
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "message":
                content = data.get("content")
                # 1. 广播给所有人显示
                await manager.broadcast({"type": "message", "sender": username, "message": content})
                # 2. 扔给 AI 处理
                formatted_msg_for_ai = f"人类学生 [{username}] 说: {content}"
                asyncio.create_task(process_ai_response(formatted_msg_for_ai))
                
            elif msg_type == "typing":
                # 将某人的“正在输入”状态广播给其他人
                await manager.broadcast({"type": "typing", "sender": username, "is_typing": data.get("is_typing")})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast({"type": "message", "sender": "System", "message": f"🏃 【{username}】 离开了"})