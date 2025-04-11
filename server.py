import sys
import asyncio
import json
import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from agents import Agent, Runner
from agents.mcp import MCPServerStdio
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# Windows 호환성 설정
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# FastAPI 앱 생성
app = FastAPI(title="중고 구매 에이전트 API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI 클라이언트 초기화
client = OpenAI()

# 저장소
chat_histories = {}  # user_id -> chat_history

# mcp.json 파일 확인
MCP_CONFIG_FILE = './mcp.json'
if not os.path.exists(MCP_CONFIG_FILE):
    # mcp.json 파일이 없는 경우, 기본 설정 생성
    default_config = {
        "mcpServers": {
            "mainServer": {
                "command": "python",
                "args": ["-m", "mcp.server.cli"]
            }
        }
    }
    with open(MCP_CONFIG_FILE, 'w') as f:
        json.dump(default_config, f, indent=2)
    print(f"기본 {MCP_CONFIG_FILE} 파일이 생성되었습니다.")

# 데이터 모델
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    tool_calls: List[str] = []

# MCP 서버 설정
async def setup_mcp_servers():
    servers = []
    
    try:
        # mcp.json 파일에서 설정 읽기
        with open(MCP_CONFIG_FILE, 'r') as f:
            config = json.load(f)
        
        # 구성된 MCP 서버들을 순회
        for server_name, server_config in config.get('mcpServers', {}).items():
            try:
                mcp_server = MCPServerStdio(
                    params={
                        "command": server_config.get("command"),
                        "args": server_config.get("args", [])
                    },
                    cache_tools_list=True
                )
                await mcp_server.connect()
                servers.append(mcp_server)
                print(f"MCP 서버 '{server_name}' 연결 성공")
            except Exception as e:
                print(f"MCP 서버 '{server_name}' 연결 실패: {str(e)}")
    except Exception as e:
        print(f"MCP 서버 설정 로드 실패: {str(e)}")
        
    if not servers:
        print("경고: 연결된 MCP 서버가 없습니다. 에이전트가 도구를 사용할 수 없습니다.")
        
    return servers

# 에이전트 설정
async def setup_agent():
    # MCP 서버 설정
    mcp_servers = await setup_mcp_servers()
    
    agent = Agent(
        name="Assistant",
        instructions="""
    너는 중고 구매를 도와주는 에이전트야.
    """,
        model="gpt-4o-mini",
        mcp_servers=mcp_servers
    )
    return agent, mcp_servers

# 전역 MCP 서버 관리
mcp_servers_global = None

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 MCP 서버 초기화"""
    global mcp_servers_global
    _, mcp_servers_global = await setup_agent()

@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 MCP 서버 정리"""
    global mcp_servers_global
    if mcp_servers_global:
        for server in mcp_servers_global:
            try:
                await server.__aexit__(None, None, None)
            except:
                pass

# REST API 엔드포인트 - 채팅 메시지 처리
@app.post("/api/chat", response_model=ChatResponse)
async def process_chat(chat_request: ChatRequest, background_tasks: BackgroundTasks):
    user_id = chat_request.user_id
    user_message = chat_request.message
    
    # 사용자 채팅 기록이 없으면 초기화
    if user_id not in chat_histories:
        chat_histories[user_id] = []
    
    # 사용자 메시지 추가
    chat_histories[user_id].append({"role": "user", "content": user_message})
    
    # 에이전트 설정 및 응답 생성
    try:
        agent, local_mcp_servers = await setup_agent()
        
        # 응답 생성
        response_text = ""
        tool_calls = []
        
        result = Runner.run_streamed(agent, input=chat_histories[user_id])
        
        async for event in result.stream_events():
            # LLM 응답 토큰 스트리밍
            if event.type == "raw_response_event":
                response_text += event.data.content or ""
            
            # 도구 이벤트 처리
            elif event.type == "run_item_stream_event":
                item = event.item
                if item.type == "tool_call_item":
                    tool_name = item.raw_item.name
                    tool_calls.append(tool_name)
        
        # 응답 메시지 추가
        chat_histories[user_id].append({"role": "assistant", "content": response_text})
        
        # 로컬 MCP 서버 정리 (전역 서버는 유지)
        background_tasks.add_task(close_mcp_servers, local_mcp_servers)
        
        return ChatResponse(response=response_text, tool_calls=tool_calls)
    
    except Exception as e:
        print(f"처리 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"메시지 처리 중 오류 발생: {str(e)}")

# WebSocket 연결 관리를 위한 클래스
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: str):
        self.active_connections.pop(user_id, None)

    async def send_message(self, user_id: str, message: str):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)

manager = ConnectionManager()

# WebSocket API 엔드포인트 - 실시간 스트리밍 채팅
@app.websocket("/ws/chat/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    
    try:
        if user_id not in chat_histories:
            chat_histories[user_id] = []
            
        # 클라이언트에게 기존 채팅 기록 전송
        await websocket.send_json({
            "type": "history",
            "data": chat_histories[user_id]
        })
        
        while True:
            try:
                # 클라이언트로부터 메시지 수신
                data = await websocket.receive_text()
                message_data = json.loads(data)
                user_message = message_data.get("message", "")
                print(user_message)
                
                # 사용자 메시지 추가 및 클라이언트에게 전송
                chat_histories[user_id].append({"role": "user", "content": user_message})
                await websocket.send_json({
                    "type": "message",
                    "data": {"role": "user", "content": user_message}
                })
                
                # 에이전트 설정
                try:
                    agent, local_mcp_servers = await setup_agent()
                    
                    # 에이전트 응답 생성 및 스트리밍
                    result = Runner.run_streamed(agent, input=chat_histories[user_id])
                    
                    response_text = ""
                    tool_calls = []
                    
                    async for event in result.stream_events():
                        try:
                            # LLM 응답 토큰 스트리밍
                            if event.type == "raw_response_event":
                                if hasattr(event.data, 'content'):
                                    content = event.data.content or ""
                                else:
                                    content = str(event.data) or ""
                                response_text += content
                                
                                # 클라이언트에게 토큰 스트리밍
                                await websocket.send_json({
                                    "type": "token",
                                    "data": content
                                })
                            
                            # 도구 이벤트 처리
                            elif event.type == "run_item_stream_event":
                                item = event.item
                                if item.type == "tool_call_item":
                                    tool_name = item.raw_item.name
                                    tool_calls.append(tool_name)
                                    
                                    # 클라이언트에게 도구 호출 정보 전송
                                    await websocket.send_json({
                                        "type": "tool_call",
                                        "data": tool_name
                                    })
                        except Exception as e:
                            print(f"이벤트 처리 중 오류: {str(e)}")
                            continue
                    
                    # 응답 완료 후 전체 메시지 저장
                    chat_histories[user_id].append({"role": "assistant", "content": response_text})
                    
                    # 응답 완료 알림
                    await websocket.send_json({
                        "type": "complete",
                        "data": {
                            "message": {"role": "assistant", "content": response_text},
                            "tool_calls": tool_calls
                        }
                    })
                    
                    # 로컬 MCP 서버 정리
                    for server in local_mcp_servers:
                        await server.__aexit__(None, None, None)
                        
                except Exception as e:
                    error_message = f"에이전트 처리 중 오류 발생: {str(e)}"
                    print(error_message)
                    await websocket.send_json({
                        "type": "error",
                        "data": error_message
                    })
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "data": "잘못된 메시지 형식입니다. JSON 형식으로 보내주세요."
                })
            except Exception as e:
                error_message = f"메시지 처리 중 오류 발생: {str(e)}"
                print(error_message)
                await websocket.send_json({
                    "type": "error",
                    "data": error_message
                })
    
    except WebSocketDisconnect:
        manager.disconnect(user_id)
    except Exception as e:
        print(f"WebSocket 오류: {e}")
        manager.disconnect(user_id)

# MCP 서버 종료 함수
async def close_mcp_servers(servers):
    for server in servers:
        try:
            await server.__aexit__(None, None, None)
        except:
            pass

# 상태 확인 엔드포인트
@app.get("/health")
async def health_check():
    return {"status": "ok", "mcp_servers_connected": mcp_servers_global is not None and len(mcp_servers_global) > 0}

# 유저 채팅 기록 조회
@app.get("/api/chat/{user_id}", response_model=List[Message])
async def get_chat_history(user_id: str):
    if user_id not in chat_histories:
        return []
    return chat_histories[user_id]

# 유저 채팅 기록 삭제
@app.delete("/api/chat/{user_id}")
async def clear_chat_history(user_id: str):
    if user_id in chat_histories:
        chat_histories[user_id] = []
    return {"message": "채팅 기록이 삭제되었습니다."}

# 서버 실행 코드
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)