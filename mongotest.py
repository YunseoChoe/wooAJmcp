import os
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, field_serializer
from typing import Optional, List, ClassVar, Annotated
from datetime import datetime
from bson import ObjectId
import uvicorn
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# MongoDB 설정
MONGODB_URI = os.getenv('MONGODB_URI')
if not MONGODB_URI:
    raise ValueError("MONGODB_URI 환경 변수가 설정되지 않았습니다.")
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME')
if not MONGODB_DB_NAME:
    MONGODB_DB_NAME = 'user_db'  # 기본값 설정

# FastAPI 앱 생성
app = FastAPI(title="사용자 관리 API")

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB 연결 설정
client = AsyncIOMotorClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]
users_collection = db["users"]

# PyObjectId 클래스 - MongoDB ObjectId 변환 처리
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("유효하지 않은 ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, _schema_generator):
        return {'type': 'string'}

# 데이터 모델 정의
class UserBase(BaseModel):
    username: str
    full_name: Optional[str] = None
    phone: Optional[str] = None
    is_active: bool = True

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    username: Optional[str] = None
    full_name: Optional[str] = None
    phone: Optional[str] = None
    is_active: Optional[bool] = None
    password: Optional[str] = None

class UserInDB(UserBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {
        "json_encoders": {ObjectId: str},
        "from_attributes": True,
        "json_schema_extra": {
            "example": {
                "_id": "60d5ec2af87e2d3b3c5c6d9b",
                "username": "johndoe",
                "full_name": "John Doe",
                "phone": "010-1234-5678",
                "is_active": True,
                "created_at": "2023-06-25T08:36:15.126Z",
                "updated_at": "2023-06-25T08:36:15.126Z"
            }
        }
    }
    
    @field_serializer('id')
    def serialize_id(self, id: PyObjectId):
        return str(id)

class UserResponse(BaseModel):
    id: str = Field(..., alias="_id")
    username: str
    full_name: Optional[str] = None
    phone: Optional[str] = None
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "example": {
                "_id": "60d5ec2af87e2d3b3c5c6d9b",
                "username": "johndoe",
                "full_name": "John Doe",
                "phone": "010-1234-5678",
                "is_active": True,
                "created_at": "2023-06-25T08:36:15.126Z",
                "updated_at": "2023-06-25T08:36:15.126Z"
            }
        }
    }

# 의존성 함수 - 사용자 존재 여부 확인
async def get_user_or_404(user_id: str):
    if not ObjectId.is_valid(user_id):
        raise HTTPException(status_code=400, detail="유효하지 않은 사용자 ID 형식")
    
    user = await users_collection.find_one({"_id": ObjectId(user_id)})
    if user is None:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")
    return user

# API 엔드포인트
@app.post("/users/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    # 중복 사용자 검사
    existing_user = await users_collection.find_one({"username": user.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="이미 존재하는 사용자명입니다")
    
    # 보안을 위해 실제 구현에서는 패스워드 해싱 처리가 필요합니다
    # 예: hashed_password = pwd_context.hash(user.password)
    
    user_dict = user.model_dump()
    password = user_dict.pop('password')  # 해싱된 패스워드로 대체해야 함
    
    # 실제 구현에서는 아래와 같이 해싱된 패스워드를 저장
    # user_dict['hashed_password'] = hashed_password
    user_dict['password'] = password  # 테스트용 (실제로는 이렇게 저장하면 안 됨)
    
    # 타임스탬프 추가
    now = datetime.utcnow()
    user_dict['created_at'] = now
    user_dict['updated_at'] = now
    
    # 데이터베이스에 저장
    result = await users_collection.insert_one(user_dict)
    
    # 생성된 사용자 조회
    created_user = await users_collection.find_one({"_id": result.inserted_id})
    
    return created_user

@app.get("/users/", response_model=List[UserResponse])
async def read_users(skip: int = 0, limit: int = 100):
    users = await users_collection.find().skip(skip).limit(limit).to_list(length=limit)
    return users

@app.get("/users/{user_id}", response_model=UserResponse)
async def read_user(user: dict = Depends(get_user_or_404)):
    return user

@app.put("/users/{user_id}", response_model=UserResponse)
async def update_user(user_update: UserUpdate, user: dict = Depends(get_user_or_404)):
    user_id = user["_id"]
    
    # 업데이트할 필드만 포함
    update_data = {k: v for k, v in user_update.model_dump(exclude_unset=True).items() if v is not None}
    
    # 사용자명 중복 확인
    if "username" in update_data:
        existing_username = await users_collection.find_one({
            "username": update_data["username"],
            "_id": {"$ne": user_id}
        })
        if existing_username:
            raise HTTPException(status_code=400, detail="이미 존재하는 사용자명입니다")
    
    # 패스워드 필드가 있는 경우 해싱 처리 (실제 구현에서)
    if "password" in update_data:
        # 실제 구현에서는 아래와 같이 해싱
        # update_data["hashed_password"] = pwd_context.hash(update_data.pop("password"))
        pass
    
    # 업데이트 시간 갱신
    update_data["updated_at"] = datetime.utcnow()
    
    # 업데이트 실행
    await users_collection.update_one(
        {"_id": user_id},
        {"$set": update_data}
    )
    
    # 업데이트된 사용자 조회
    updated_user = await users_collection.find_one({"_id": user_id})
    
    return updated_user

@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user: dict = Depends(get_user_or_404)):
    user_id = user["_id"]
    await users_collection.delete_one({"_id": user_id})
    return None

# 서버 실행 코드
if __name__ == "__main__":
    uvicorn.run("mongotest:app", host="0.0.0.0", port=8000, reload=True)