import sys
import asyncio
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException, WebSocketException, status
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from agents import Agent, Runner
from agents.mcp import MCPServerStdio
from dotenv import load_dotenv
from jwt import PyJWT
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

jwt_decoder = PyJWT()

# 환경 변수 로드
load_dotenv()
print("환경 변수 로드 완료")

# JWT 설정
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
if not JWT_SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY 환경 변수가 설정되지 않았습니다.")

# MongoDB 설정
MONGODB_URI = os.getenv('MONGODB_URI')
if not MONGODB_URI:
    raise ValueError("MONGODB_URI 환경 변수가 설정되지 않았습니다.")
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'chat_db')  # 기본값 설정
print(f"MongoDB 설정: {MONGODB_URI}, DB: {MONGODB_DB_NAME}")

# MongoDB 클라이언트 초기화
mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client[MONGODB_DB_NAME]
solo_chat_collection = db['solo_chats']
group_chat_collection = db['group_chats']

# Windows 호환성 설정
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# 전역 MCP 서버 관리
mcp_servers_global = []

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

# MCP 서버 설정 (타임아웃 추가)
async def setup_mcp_servers():
    servers = []
    
    try:
        # mcp.json 파일에서 설정 읽기
        with open(MCP_CONFIG_FILE, 'r') as f:
            config = json.load(f)
        
        # 구성된 MCP 서버들을 순회
        for server_name, server_config in config.get('mcpServers', {}).items():
            try:
                print(f"MCP 서버 '{server_name}' 연결 시도 중...")
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
    print("에이전트 설정 시작...")
    # MCP 서버 설정
    try:
        mcp_servers = await setup_mcp_servers()
        
        agent = Agent(
            name="Assistant",
            instructions="""
        너는 중고 구매를 도와주는 에이전트야.
        """,
            model="gpt-4o-mini",
            mcp_servers=mcp_servers
        )
        print("에이전트 설정 완료")
        return agent, mcp_servers
    except Exception as e:
        print(f"에이전트 설정 실패: {str(e)}")
        return None, []

# MongoDB 연결 테스트 함수
async def test_mongodb_connection():
    try:
        # 서버 정보 요청을 통한 연결 테스트
        await mongo_client.admin.command('ping')
        print("MongoDB 연결 성공")
        return True
    except Exception as e:
        print(f"MongoDB 연결 실패: {str(e)}")
        return False

# lifespan 이벤트 핸들러
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 실행
    global mcp_servers_global
    
    # MongoDB 연결 테스트
    mongo_connected = await test_mongodb_connection()
    if not mongo_connected:
        print("경고: MongoDB에 연결할 수 없습니다. 일부 기능이 작동하지 않을 수 있습니다.")
    
    # 에이전트 및 MCP 서버 설정
    try:
        print("MCP 서버 초기화 시작...")
        agent, mcp_servers = await setup_agent()
        if agent:
            mcp_servers_global = mcp_servers
            print(f"MCP 서버 초기화 완료: {len(mcp_servers_global)}개 서버 연결됨")
        else:
            print("MCP 서버 초기화 실패: 에이전트가 None입니다")
            mcp_servers_global = []
    except Exception as e:
        print(f"MCP 서버 초기화 중 예외 발생: {str(e)}")
        mcp_servers_global = []
    
    yield
    
    # 종료 시 실행
    print("서버 종료 중...")
    if mcp_servers_global:
        for server in mcp_servers_global:
            try:
                await server.__aexit__(None, None, None)
                print("MCP 서버 연결 종료")
            except Exception as e:
                print(f"MCP 서버 종료 중 오류: {str(e)}")

# FastAPI 앱 생성
app = FastAPI(title="중고 구매 에이전트 API", lifespan=lifespan)

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
        if not agent:
            raise HTTPException(status_code=500, detail="에이전트 초기화 실패")
            
        # 응답 생성
        response_text = ""
        tool_calls = []
        
        print(f"챗봇 응답 생성 시작: {user_id}")
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
        print(f"챗봇 응답 생성 완료: {user_id}")
        
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
        print(f"WebSocket 연결 수락: {user_id}")

    def disconnect(self, user_id: str):
        self.active_connections.pop(user_id, None)
        print(f"WebSocket 연결 종료: {user_id}")

    async def send_message(self, user_id: str, message: str):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)

manager = ConnectionManager()

async def get_token_from_header(websocket: WebSocket) -> str:
    """웹소켓 헤더에서 JWT 토큰을 추출합니다."""
    headers = dict(websocket.headers)
    auth_header = headers.get('authorization')
    
    if not auth_header or not auth_header.startswith('Bearer '):
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="인증 토큰이 없습니다.")
    
    return auth_header.split(' ')[1]

async def verify_token(token: str) -> str:
    """JWT 토큰을 검증하고 user_id를 반환합니다."""
    try:
        # PyJWT 최신 버전 사용
        payload = jwt_decoder.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get('user_id')
        if not user_id:
            raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="유효하지 않은 토큰입니다.")
        return user_id
    except Exception as e:  # 더 구체적인 예외 처리
        print(f"토큰 검증 오류: {str(e)}")
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="유효하지 않은 토큰입니다.")

# WebSocket API 엔드포인트 - 실시간 스트리밍 채팅(1:AI)
@app.websocket("/ws/solchat")
async def websocket_endpoint(websocket: WebSocket):
    # JWT 토큰 검증
    token = await get_token_from_header(websocket)
    token_user_id = await verify_token(token)
    print(token_user_id)
    
    client_user_id = token_user_id
    # URL의 user_id와 토큰의 user_id가 일치하는지 확인
    
    await manager.connect(websocket, client_user_id)
    
    try:
        # 기존 채팅 기록 조회
        try:
            chat_history = await solo_chat_collection.find(
                {"user_id": client_user_id}
            ).sort("timestamp", 1).to_list(length=None)
            
            # 클라이언트에게 기존 채팅 기록 전송
            formatted_history = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]
            await websocket.send_json({
                "type": "history",
                "data": formatted_history
            })
            print(f"채팅 기록 전송 완료: {client_user_id}, {len(formatted_history)}개 메시지")
        except Exception as e:
            print(f"채팅 기록 조회 중 오류: {str(e)}")
            await websocket.send_json({
                "type": "error",
                "data": f"채팅 기록 조회 중 오류: {str(e)}"
            })
        
        while True:
            try:
                # 클라이언트로부터 메시지 수신
                data = await websocket.receive_text()
                message_data = json.loads(data)
                user_message = message_data.get("message", "")
                print(f"사용자 메시지 수신: {client_user_id}, 내용: {user_message[:30]}...")
                
                # 사용자 메시지를 MongoDB에 저장
                try:
                    await solo_chat_collection.insert_one({
                        "user_id": client_user_id,
                        "role": "user",
                        "content": user_message,
                        "timestamp": datetime.utcnow()
                    })
                except Exception as e:
                    print(f"메시지 저장 중 오류: {str(e)}")
                
                # 클라이언트에게 메시지 전송
                await websocket.send_json({
                    "type": "message",
                    "data": {"role": "user", "content": user_message}
                })
                
                # 에이전트 설정 및 응답 생성
                try:
                    print(f"에이전트 응답 생성 시작: {client_user_id}")
                    agent, local_mcp_servers = await setup_agent()
                    if not agent:
                        raise Exception("에이전트 초기화 실패")
                        
                    chat_history = await solo_chat_collection.find(
                        {"user_id": client_user_id}
                    ).sort("timestamp", 1).to_list(length=None)
                    formatted_history = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]
                    
                    result = Runner.run_streamed(agent, input=formatted_history)
                    response_text = ""
                    tool_calls = []
                    
                    async for event in result.stream_events():
                        try:
                            if event.type == "raw_response_event":
                                content = event.data.content if hasattr(event.data, 'content') else str(event.data)
                                response_text += event.data.delta or ""
                                await websocket.send_json({
                                    "type": "token",
                                    "data": content
                                })
                            elif event.type == "run_item_stream_event":
                                item = event.item
                                if item.type == "tool_call_item":
                                    tool_name = item.raw_item.name
                                    tool_calls.append(tool_name)
                                    await websocket.send_json({
                                        "type": "tool_call",
                                        "data": tool_name
                                    })
                        except Exception as e:
                            print(f"이벤트 처리 중 오류: {str(e)}")
                            continue
                    
                    # 응답 메시지를 MongoDB에 저장
                    try:
                        await solo_chat_collection.insert_one({
                            "user_id": client_user_id,
                            "role": "assistant",
                            "content": response_text,
                            "timestamp": datetime.utcnow()
                        })
                    except Exception as e:
                        print(f"응답 저장 중 오류: {str(e)}")
                    
                    # 응답 완료 알림
                    await websocket.send_json({
                        "type": "complete",
                        "data": {
                            "message": {"role": "assistant", "content": response_text},
                            "tool_calls": tool_calls
                        }
                    })
                    print(f"에이전트 응답 생성 완료: {client_user_id}")
                    
                    # 로컬 MCP 서버 정리
                    for server in local_mcp_servers:
                        try:
                            await server.__aexit__(None, None, None)
                        except Exception as e:
                            print(f"MCP 서버 정리 중 오류: {str(e)}")
                        
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
        manager.disconnect(client_user_id)
    except Exception as e:
        print(f"WebSocket 오류: {e}")
        manager.disconnect(client_user_id)

# 실시간 스트리밍 채팅(2:AI)
@app.websocket("/ws/groupchat/{room_num}")
async def group_websocket_endpoint(websocket: WebSocket, room_num: str):
    # JWT 토큰 검증
    token = await get_token_from_header(websocket)
    token_user_id = await verify_token(token)
    
    await manager.connect(websocket, room_num)
    
    try:
        # 기존 채팅 기록 조회
        try:
            chat_history = await group_chat_collection.find(
                {"room_num": room_num}
            ).sort("timestamp", 1).to_list(length=None)
            
            # 클라이언트에게 기존 채팅 기록 전송
            formatted_history = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]
            await websocket.send_json({
                "type": "history",
                "data": formatted_history
            })
            print(f"그룹 채팅 기록 전송 완료: {room_num}, {len(formatted_history)}개 메시지")
        except Exception as e:
            print(f"그룹 채팅 기록 조회 중 오류: {str(e)}")
            await websocket.send_json({
                "type": "error",
                "data": f"채팅 기록 조회 중 오류: {str(e)}"
            })
        
        while True:
            try:
                # 클라이언트로부터 메시지 수신
                data = await websocket.receive_text()
                message_data = json.loads(data)
                user_message = message_data.get("message", "")
                print(f"그룹 메시지 수신: 방 {room_num}, 사용자 {token_user_id}, 내용: {user_message[:30]}...")
                
                # 사용자 메시지를 MongoDB에 저장
                try:
                    await group_chat_collection.insert_one({
                        "user_id": token_user_id,
                        "room_num": room_num,
                        "role": "user",
                        "content": user_message,
                        "timestamp": datetime.utcnow()
                    })
                except Exception as e:
                    print(f"그룹 메시지 저장 중 오류: {str(e)}")
                
                # 클라이언트에게 메시지 전송
                await websocket.send_json({
                    "type": "message",
                    "data": {"role": "user", "content": user_message}
                })
                
                # 에이전트 설정 및 응답 생성
                try:
                    print(f"그룹 에이전트 응답 생성 시작: 방 {room_num}")
                    agent, local_mcp_servers = await setup_agent()
                    if not agent:
                        raise Exception("에이전트 초기화 실패")
                        
                    chat_history = await group_chat_collection.find(
                        {"room_num": room_num}
                    ).sort("timestamp", 1).to_list(length=None)
                    formatted_history = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]
                    
                    result = Runner.run_streamed(agent, input=formatted_history)
                    response_text = ""
                    tool_calls = []
                    
                    async for event in result.stream_events():
                        try:
                            if event.type == "raw_response_event":
                                content = event.data.content if hasattr(event.data, 'content') else str(event.data)
                                response_text += event.data.delta or ""
                                await websocket.send_json({
                                    "type": "token",
                                    "data": content
                                })
                            elif event.type == "run_item_stream_event":
                                item = event.item
                                if item.type == "tool_call_item":
                                    tool_name = item.raw_item.name
                                    tool_calls.append(tool_name)
                                    await websocket.send_json({
                                        "type": "tool_call",
                                        "data": tool_name
                                    })
                        except Exception as e:
                            print(f"그룹 이벤트 처리 중 오류: {str(e)}")
                            continue
                    
                    # 응답 메시지를 MongoDB에 저장
                    try:
                        await group_chat_collection.insert_one({
                            "user_id": "AI",
                            "room_num": room_num,
                            "role": "assistant",
                            "content": response_text,
                            "timestamp": datetime.utcnow()
                        })
                    except Exception as e:
                        print(f"그룹 응답 저장 중 오류: {str(e)}")
                    
                    # 응답 완료 알림
                    await websocket.send_json({
                        "type": "complete",
                        "data": {
                            "message": {"role": "assistant", "content": response_text},
                            "tool_calls": tool_calls
                        }
                    })
                    print(f"그룹 에이전트 응답 생성 완료: 방 {room_num}")
                    
                    # 로컬 MCP 서버 정리
                    for server in local_mcp_servers:
                        try:
                            await server.__aexit__(None, None, None)
                        except Exception as e:
                            print(f"그룹 MCP 서버 정리 중 오류: {str(e)}")
                        
                except Exception as e:
                    error_message = f"그룹 에이전트 처리 중 오류 발생: {str(e)}"
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
                error_message = f"그룹 메시지 처리 중 오류 발생: {str(e)}"
                print(error_message)
                await websocket.send_json({
                    "type": "error",
                    "data": error_message
                })
    
    except WebSocketDisconnect:
        manager.disconnect(room_num)
    except Exception as e:
        print(f"그룹 WebSocket 오류: {e}")
        manager.disconnect(room_num)

# MCP 서버 종료 함수
async def close_mcp_servers(servers):
    for server in servers:
        try:
            await server.__aexit__(None, None, None)
        except Exception as e:
            print(f"MCP 서버 종료 오류: {str(e)}")

# 상태 확인 엔드포인트
@app.get("/health")
async def health_check():
    # MongoDB 연결 상태 확인
    mongo_connected = False
    try:
        await mongo_client.admin.command('ping')
        mongo_connected = True
    except:
        pass

    return {
        "status": "ok", 
        "mongodb_connected": mongo_connected,
        "mcp_servers_connected": mcp_servers_global is not None and len(mcp_servers_global) > 0
    }

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