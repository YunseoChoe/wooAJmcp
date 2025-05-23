# Python 3.13 기반 이미지 사용
FROM python:3.13-slim

ARG OPENAI_API_KEY
ARG JWT_SECRET_KEY
ARG MONGODB_URI
ARG MONGODB_DB_NAME
ARG KAKAO_API_KEY

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Playwright 설치 및 브라우저 설치
RUN playwright install chromium
RUN playwright install-deps

# 소스 코드 복사
COPY . .

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV JWT_SECRET_KEY=${JWT_SECRET_KEY}
ENV MONGODB_URI=${MONGODB_URI}
ENV MONGODB_DB_NAME=${MONGODB_DB_NAME}
ENV KAKAO_API_KEY=${KAKAO_API_KEY}

# 포트 노출
EXPOSE 8000

# 서버 실행
#CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["python", "server.py"]