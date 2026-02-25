import asyncio
import os
import json
import base64
import fitz  # PyMuPDF，用于极速解析 PDF
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

# 引入 AutoGen 的多模态组件
from autogen_core import Image
# 👑 修复点 1：将 MultimodalMessage 改为官方正确的 MultiModalMessage
from autogen_agentchat.messages import MultiModalMessage

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

load_dotenv()
app = FastAPI()

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
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

api_key = os.environ.get("LINGYA_API_KEY")
base_url = "https://api.lingyaai.cn/v1"

# 激活多模态能力
custom_model_info = ModelInfo(vision=True, function_calling=True, json_output=False, family="unknown", structured_output=False)

planner_model = OpenAIChatCompletionClient(model="gemini-3.1-pro-preview", api_key=api_key, base_url=base_url, model_info=custom_model_info)
facilitator_model = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=api_key, base_url=base_url, model_info=custom_model_info)

planner = AssistantAgent(
    name="Planner",
    model_client=planner_model,
    system_message="你是一个项目规划师。帮助学生拆解难题。结束发言时请加 'WAIT'。"
)
facilitator = AssistantAgent(
    name="Facilitator",
    model_client=facilitator_model,
    system_message="你是一个协作学习的引导者。用苏格拉底式提问。结束发言时请加 'WAIT'。"
)

termination = TextMentionTermination("WAIT") | MaxMessageTermination(max_messages=3)
team = SelectorGroupChat(participants=[planner, facilitator], model_client=planner_model, termination_condition=termination)
ai_lock = asyncio.Lock()

async def process_ai_response(user_msg: str, username: str, files: list):
    async with ai_lock:
        await manager.broadcast({"type": "typing", "sender": "Planner", "is_typing": True})
        try:
            # 准备任务内容载体，第一项永远是文本
            task_content = [user_msg]
            
            # 处理附件
            for f in files:
                mime = f.get("mime", "")
                b64_data = f.get("data", "")
                name = f.get("name", "附件")
                
                if not b64_data or "," not in b64_data:
                    continue
                pure_b64 = b64_data.split(",")[1]
                
                if mime.startswith("image/"):
                    # 如果是图片，直接转换为 AutoGen 视觉对象
                    task_content.append(Image.from_base64(pure_b64))
                elif mime == "application/pdf":
                    # 如果是 PDF，提取文本并塞回文字消息中
                    try:
                        pdf_bytes = base64.b64decode(pure_b64)
                        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                        pdf_text = f"\n\n--- 📎 附件 PDF ({name}) 内容 ---\n"
                        for page in doc:
                            pdf_text += page.get_text()
                        pdf_text += "\n--- PDF 结束 ---\n"
                        task_content[0] += pdf_text
                    except Exception as e:
                        task_content[0] += f"\n[无法解析 PDF {name}: {str(e)}]\n"

            # 👑 修复点 2：实例化正确的 MultiModalMessage
            if len(task_content) > 1:
                task = MultiModalMessage(content=task_content, source=username)
            else:
                task = task_content[0]

            async for msg in team.run_stream(task=task):
                if hasattr(msg, 'content') and isinstance(msg.content, str):
                    if msg.source != "user" and msg.content.strip():
                        display_text = msg.content.replace("WAIT", "").strip()
                        if display_text:
                            await manager.broadcast({"type": "message", "sender": msg.source, "message": display_text})
        except Exception as e:
            await manager.broadcast({"type": "message", "sender": "System", "message": f"AI 思考出错: {str(e)}"})
        finally:
            await manager.broadcast({"type": "typing", "sender": "Planner", "is_typing": False})

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
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "message":
                content = data.get("content")
                files = data.get("files", [])
                
                display_msg = content
                if files:
                    file_names = [f["name"] for f in files]
                    display_msg += f"\n\n*(📎 附带文件: {', '.join(file_names)})*"

                await manager.broadcast({"type": "message", "sender": username, "message": display_msg})
                
                formatted_msg = f"人类学生 [{username}] 说: {content}"
                asyncio.create_task(process_ai_response(formatted_msg, username, files))
                
            elif msg_type == "typing":
                await manager.broadcast({"type": "typing", "sender": username, "is_typing": data.get("is_typing")})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast({"type": "message", "sender": "System", "message": f"🏃 【{username}】 离开了"})