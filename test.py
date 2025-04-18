from dotenv import load_dotenv
import os

load_dotenv()

mongo_uri = os.getenv("MONGODB_URI")
print(f"Mongo URI: {mongo_uri}")  # 디버깅용 출력