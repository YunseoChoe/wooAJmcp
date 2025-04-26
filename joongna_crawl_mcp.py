from mcp.server.fastmcp import FastMCP
import subprocess
import json
import tempfile
import os
import time
from typing import Dict, List, Optional
from collections import defaultdict
from dotenv import load_dotenv
import requests
import math
from pydantic import BaseModel

# .env 파일에서 환경 변수 불러오기
load_dotenv()

KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")

# 캐시 데이터 관리를 위한 클래스
class CacheManager:
    def __init__(self, ttl=3600):  # 1시간 캐시 유효기간
        self.cache = defaultdict(dict)
        self.last_accessed = defaultdict(float)
        self.ttl = ttl

    def get(self, keyword, page):
        current_time = time.time()
        if keyword in self.cache and page in self.cache[keyword]:
            # 마지막 접근 시간 업데이트
            self.last_accessed[keyword] = current_time
            return self.cache[keyword][page]
        return None

    def set(self, keyword, page, data):
        current_time = time.time()
        self.cache[keyword][page] = data
        self.last_accessed[keyword] = current_time

    def clean_expired(self):
        current_time = time.time()
        expired_keywords = []
        for keyword, last_time in self.last_accessed.items():
            if current_time - last_time > self.ttl:
                expired_keywords.append(keyword)
        
        for keyword in expired_keywords:
            del self.cache[keyword]
            del self.last_accessed[keyword]

# 캐시 매니저 인스턴스 생성
cache_manager = CacheManager()

# Create an MCP server
mcp = FastMCP("joongo_server")

# 외부 크롤러 스크립트를 위한 코드
# 이 스크립트는 별도 파일로 저장하여 subprocess로 실행합니다
CRAWLER_SCRIPT = """
import asyncio
from playwright.async_api import async_playwright
import json
import sys

async def crawl_joongna(keyword, page=1):
    results = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        page_obj = await context.new_page()
        
        url = f"https://web.joongna.com/search/{keyword}?page={page}"
        print(f"URL 접속: {url}", file=sys.stderr)
        
        try:
            await page_obj.goto(url, wait_until="networkidle")
            await page_obj.wait_for_selector("a[href^='/product/']", timeout=30000)
            
            for _ in range(5):
                await page_obj.evaluate("window.scrollBy(0, 800)")
                await asyncio.sleep(1)
            
            product_items = await page_obj.query_selector_all("a[href^='/product/']")
            
            print(f"상품 항목 수: {len(product_items)}", file=sys.stderr)
            
            for item in product_items:
                try:
                    relative_url = await item.get_attribute("href")
                    product_url = f"https://web.joongna.com{relative_url}" if relative_url else ""
                    
                    name_element = await item.query_selector("h2")
                    product_name = await name_element.inner_text() if name_element else "제목 없음"
                    
                    price_element = await item.query_selector("div.font-semibold")
                    price = await price_element.inner_text() if price_element else "가격 정보 없음"
                    
                    img_element = await item.query_selector("img")
                    image_url = await img_element.get_attribute("src") if img_element else ""
                    
                    time_element = await item.query_selector("span.text-gray-400")
                    upload_time = await time_element.inner_text() if time_element else ""
                    
                    results.append({
                        'product_name': product_name,
                        'price': price,
                        'image_url': image_url,
                        'product_url': product_url,
                        'upload_time': upload_time
                    })
                except Exception as e:
                    print(f"항목 처리 중 오류 발생: {e}", file=sys.stderr)
            
        except Exception as e:
            print(f"크롤링 중 오류 발생: {e}", file=sys.stderr)
        finally:
            await browser.close()
    
    return results

async def get_product_detail(product_url):
    if not product_url.startswith("https://web.joongna.com/product/"):
        return {"error": "유효하지 않은 중고나라 상품 URL입니다."}
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        page_obj = await context.new_page()
        
        try:
            await page_obj.goto(product_url, wait_until="networkidle")
            await page_obj.wait_for_selector("h1", timeout=30000)
            
            product_name = await page_obj.eval_on_selector("h1", "el => el.textContent")
            
            price_element = await page_obj.query_selector("div.text-2xl")
            price = await price_element.inner_text() if price_element else "가격 정보 없음"
            
            description_element = await page_obj.query_selector("div.text-sm")
            description = await description_element.inner_text() if description_element else "설명 없음"
            
            image_elements = await page_obj.query_selector_all("img")
            image_urls = []
            for img in image_elements:
                src = await img.get_attribute("src")
                if src and "joongna.com" in src and not src in image_urls:
                    image_urls.append(src)
            
            seller_element = await page_obj.query_selector("div.text-xs")
            seller_info = await seller_element.inner_text() if seller_element else "판매자 정보 없음"
            
            return {
                'product_name': product_name,
                'price': price,
                'description': description,
                'image_urls': image_urls[:5],
                'seller_info': seller_info,
                'product_url': product_url
            }
            
        except Exception as e:
            print(f"상세 정보 크롤링 중 오류 발생: {e}", file=sys.stderr)
            return {"error": f"상품 정보를 가져오는 중 오류가 발생했습니다: {str(e)}"}
        finally:
            await browser.close()

async def main():
    import argparse
    parser = argparse.ArgumentParser(description='중고나라 크롤러')
    parser.add_argument('--type', choices=['search', 'detail'], required=True, help='크롤링 유형 (검색 또는 상세)')
    parser.add_argument('--keyword', help='검색 키워드')
    parser.add_argument('--page', type=int, default=1, help='검색 페이지')
    parser.add_argument('--url', help='상품 URL')
    args = parser.parse_args()
    
    if args.type == 'search':
        if not args.keyword:
            print(json.dumps({"error": "검색을 위한 키워드가 필요합니다."}))
            return
        
        results = await crawl_joongna(args.keyword, args.page)
        print(json.dumps(results))
    
    elif args.type == 'detail':
        if not args.url:
            print(json.dumps({"error": "상세 정보를 위한 URL이 필요합니다."}))
            return
        
        result = await get_product_detail(args.url)
        print(json.dumps(result))

if __name__ == "__main__":
    asyncio.run(main())
"""

def create_crawler_script():
    """크롤러 스크립트를 임시 파일로 생성합니다."""
    fd, path = tempfile.mkstemp(suffix='.py', prefix='joongna_crawler_')
    os.write(fd, CRAWLER_SCRIPT.encode('utf-8'))
    os.close(fd)
    return path

@mcp.tool()
def search_joongna_items(keyword: str, page: int = 1) -> Dict:
    """ 중고나라에서 검색어를 기반으로 상품을 검색하고 결과를 반환합니다. 
    
    Args:
        keyword (str): 검색할 키워드
        page (int): 검색할 페이지 번호 (1부터 시작)
        
    Returns:
        Dict: 검색 결과 (상품명, 가격, 이미지URL, 제품URL, 올린시간 등)
    """
    # 캐시에서 데이터 확인
    cached_data = cache_manager.get(keyword, page)
    if cached_data:
        print(f"캐시된 데이터 사용: {keyword} - 페이지 {page}")
        return cached_data
    
    print(f"중고나라 크롤링 시작: {keyword} - 페이지 {page}")
    
    # 크롤러 스크립트 생성
    script_path = create_crawler_script()
    
    try:
        # 별도 프로세스로 크롤러 실행
        cmd = ['python', script_path, '--type', 'search', '--keyword', keyword, '--page', str(page)]
        process = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # 결과 파싱
        results = json.loads(process.stdout)
        
        # 응답 데이터 구성
        response_data = {
            'keyword': keyword,
            'page': page,
            'total_items': len(results),
            'has_more': len(results) > 0,  # 결과가 있으면 다음 페이지 존재 가능성
            'items': results
        }
        
        # 캐시에 데이터 저장
        cache_manager.set(keyword, page, response_data)
        
        return response_data
    
    except subprocess.CalledProcessError as e:
        print(f"크롤링 프로세스 오류: {e}")
        print(f"표준 오류: {e.stderr}")
        return {
            'keyword': keyword,
            'page': page,
            'total_items': 0,
            'has_more': False,
            'items': [],
            'error': f"크롤링 중 오류가 발생했습니다: {e.stderr}"
        }
    except Exception as e:
        print(f"처리 중 오류 발생: {e}")
        return {
            'keyword': keyword,
            'page': page,
            'total_items': 0,
            'has_more': False,
            'items': [],
            'error': str(e)
        }
    finally:
        # 임시 파일 삭제
        try:
            os.remove(script_path)
        except:
            pass

@mcp.tool()
def get_joongna_product_detail(product_url: str) -> Dict:
    """ 중고나라 상품 URL을 기반으로 상세 정보를 가져옵니다.
    
    Args:
        product_url (str): 상품 URL (https://web.joongna.com/product/숫자)
        
    Returns:
        Dict: 상세 정보 (상품명, 가격, 설명, 판매자 정보 등)
    """
    # URL 검증
    if not product_url.startswith("https://web.joongna.com/product/"):
        return {
            "error": "유효하지 않은 중고나라 상품 URL입니다."
        }
    
    print(f"상품 상세 정보 크롤링: {product_url}")
    
    # 크롤러 스크립트 생성
    script_path = create_crawler_script()
    
    try:
        # 별도 프로세스로 크롤러 실행
        cmd = ['python', script_path, '--type', 'detail', '--url', product_url]
        process = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # 결과 파싱
        result = json.loads(process.stdout)
        return result
    
    except subprocess.CalledProcessError as e:
        print(f"크롤링 프로세스 오류: {e}")
        print(f"표준 오류: {e.stderr}")
        return {
            "error": f"상품 정보를 가져오는 중 오류가 발생했습니다: {e.stderr}"
        }
    except Exception as e:
        print(f"처리 중 오류 발생: {e}")
        return {
            "error": f"상품 정보를 가져오는 중 오류가 발생했습니다: {str(e)}"
        }
    finally:
        # 임시 파일 삭제
        try:
            os.remove(script_path)
        except:
            pass

@mcp.tool()
def clear_cache(keyword: str = None) -> Dict:
    """캐시를 비웁니다. 키워드를 지정하면 해당 키워드의 캐시만 삭제합니다.
    
    Args:
        keyword (str, optional): 삭제할 캐시의 키워드. 없으면 전체 캐시 삭제.
        
    Returns:
        Dict: 삭제 결과
    """
    if keyword:
        if keyword in cache_manager.cache:
            del cache_manager.cache[keyword]
            if keyword in cache_manager.last_accessed:
                del cache_manager.last_accessed[keyword]
            return {"success": True, "message": f"'{keyword}' 키워드의 캐시가 삭제되었습니다."}
        else:
            return {"success": False, "message": f"'{keyword}' 키워드의 캐시가 존재하지 않습니다."}
    else:
        cache_manager.cache.clear()
        cache_manager.last_accessed.clear()
        return {"success": True, "message": "모든 캐시가 삭제되었습니다."}

# ----------------------------------------------------------------

class Location(BaseModel):
    name: str
    latitude: float
    longitude: float

class POI(BaseModel):
    name: str
    address: str
    category: str
    latitude: float
    longitude: float
    distance: float  # 중간 지점으로부터의 거리 (미터)

class SearchResponse(BaseModel):
    midpoint: Location
    facilities: List[POI]

class MidGeoRequest(BaseModel):
    geo1: str
    geo2: str

class MidGeoResponse(BaseModel):
    first: str
    second: str

class MidpointFacilitiesRequest(BaseModel):
    geo1: str
    geo2: str
    facility_types: Optional[List[str]] = None
    radius: int = 1000
    


def get_coordinates(location_name: str) -> tuple:
    """
    장소 이름으로부터 좌표(위도, 경도)를 얻습니다.
    """
    if KAKAO_API_KEY:
        # Kakao Maps API 사용
        url = "https://dapi.kakao.com/v2/local/search/address.json"
        headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
        params = {"query": location_name}
        
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        
        if response.status_code != 200 or not data.get("documents"):
            # 주소 검색 실패 시 키워드 검색으로 시도
            url = "https://dapi.kakao.com/v2/local/search/keyword.json"
            response = requests.get(url, headers=headers, params=params)
            data = response.json()
            
            if response.status_code != 200 or not data.get("documents"):
                raise ValueError(f"위치를 찾을 수 없습니다: {location_name}")
        
        # 첫 번째 결과 사용
        first_result = data["documents"][0]
        latitude = float(first_result.get("y"))
        longitude = float(first_result.get("x"))
        
        # 첫 번째 결과 사용
        location = data["results"][0]["geometry"]["location"]
        latitude = location["lat"]
        longitude = location["lng"]
    else:
        raise ValueError("API 키가 설정되지 않았습니다. Kakao 또는 Google Maps API 키를 설정해주세요.")
    
    return latitude, longitude

def calculate_midpoint(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple:
    """
    두 좌표 간의 중간 지점을 계산합니다.
    """
    # 라디안으로 변환
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # 중간 지점 계산
    bx = math.cos(lat2_rad) * math.cos(lon2_rad - lon1_rad)
    by = math.cos(lat2_rad) * math.sin(lon2_rad - lon1_rad)
    
    lat3_rad = math.atan2(
        math.sin(lat1_rad) + math.sin(lat2_rad),
        math.sqrt((math.cos(lat1_rad) + bx) ** 2 + by ** 2)
    )
    lon3_rad = lon1_rad + math.atan2(by, math.cos(lat1_rad) + bx)
    
    # 도(degree)로 변환
    lat3 = math.degrees(lat3_rad)
    lon3 = math.degrees(lon3_rad)
    
    return lat3, lon3

def find_nearby_facilities(latitude: float, longitude: float, facility_type: str, radius: int = 1000) -> List[POI]:
    """
    지정된 좌표 주변의 시설을 검색합니다.
    """
    results = []
    
    if KAKAO_API_KEY:
        # Kakao Local API 사용
        url = "https://dapi.kakao.com/v2/local/search/category.json"
        headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
        
        # 카테고리 코드 매핑
        category_codes = {
            "police": "PO3",    # 경찰서
            "subway": "SW8",    # 지하철역
           # "cafe": "FD6",      # 카페
           # "restaurant": "FD6", # 음식점
           # "hospital": "HP8",  # 병원
           # "bank": "BK9",      # 은행
           # "store": "MT1",     # 마트
        }
        
        # 카테고리 코드가 없으면 키워드 검색 사용
        if facility_type in category_codes:
            params = {
                "category_group_code": category_codes[facility_type],
                "x": longitude,
                "y": latitude,
                "radius": radius,
                "sort": "distance"
            }
            endpoint = "category.json"
        else:
            params = {
                "query": facility_type,
                "x": longitude,
                "y": latitude,
                "radius": radius,
                "sort": "distance"
            }
            endpoint = "keyword.json"
            
        url = f"https://dapi.kakao.com/v2/local/search/{endpoint}"
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            for place in data.get("documents", []):
                results.append(POI(
                    name=place.get("place_name"),
                    address=place.get("address_name"),
                    category=place.get("category_name"),
                    latitude=float(place.get("y")),
                    longitude=float(place.get("x")),
                    distance=float(place.get("distance"))
                ))
    
    return results

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    두 좌표 사이의 거리를 계산합니다 (Haversine 공식 사용).
    """
    # 지구 반경 (미터)
    R = 6371000
    
    # 라디안으로 변환
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # 위도, 경도 차이
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine 공식
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance

def get_address_from_coordinates(longitude, latitude):
    """
    카카오 지도 API를 사용해 위도와 경도로부터 주소 정보를 가져옵니다.
    
    Args:
        longitude (float): 경도 값
        latitude (float): 위도 값
        api_key (str): 카카오 API 키
        
    Returns:
        dict: 주소 정보가 담긴 딕셔너리
    """
    api_key = KAKAO_API_KEY
    url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {
        "x": longitude,  # 경도(x)
        "y": latitude,   # 위도(y)
    }
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if data.get("documents"):
            # 첫 번째 결과 반환
            return data["documents"][0]["address"]
        else:
            return {"error": "해당 좌표에 대한 주소 정보가 없습니다."}
    else:
        return {"error": f"API 요청 실패: {response.status_code}"}

# 시설 검색을 더 강력하게 하는 함수
def find_nearby_facilities_robust(latitude: float, longitude: float, facility_type: str, radius: int = 1000) -> List[POI]:
    """
    지정된 좌표 주변의 시설을 검색합니다.
    여러 방법을 시도하여 시설을 찾지 못하는 경우를 최소화합니다.
    """
    results = []
    
    if not KAKAO_API_KEY:
        return results
    
    # 카테고리 코드 매핑
    category_codes = {
        "경찰서": "PO3",      # 경찰서
        "지하철역": "SW8",    # 지하철역
        "공공기관": "PO3",    # 공공기관 (경찰서 코드 사용)
        "police": "PO3",
        "subway": "SW8",
        "public": "PO3",
    }
    
    # 시도 1: 카테고리 검색
    try:
        category_code = category_codes.get(facility_type)
        if category_code:
            url = "https://dapi.kakao.com/v2/local/search/category.json"
            headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
            params = {
                "category_group_code": category_code,
                "x": longitude,
                "y": latitude,
                "radius": radius,
                "sort": "distance"
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                for place in data.get("documents", []):
                    results.append(POI(
                        name=place.get("place_name"),
                        address=place.get("address_name"),
                        category=place.get("category_name"),
                        latitude=float(place.get("y")),
                        longitude=float(place.get("x")),
                        distance=float(place.get("distance"))
                    ))
    except Exception as e:
        print(f"카테고리 검색 실패: {str(e)}")
    
    # 시도 2: 키워드 검색 (카테고리 검색이 실패했거나 결과가 없는 경우)
    if not results:
        try:
            url = "https://dapi.kakao.com/v2/local/search/keyword.json"
            headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
            params = {
                "query": facility_type,
                "x": longitude,
                "y": latitude,
                "radius": radius,
                "sort": "distance"
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                for place in data.get("documents", []):
                    results.append(POI(
                        name=place.get("place_name"),
                        address=place.get("address_name"),
                        category=place.get("category_name"),
                        latitude=float(place.get("y")),
                        longitude=float(place.get("x")),
                        distance=float(place.get("distance"))
                    ))
        except Exception as e:
            print(f"키워드 검색 실패: {str(e)}")
    
    return results

# 좌표 획득을 더 강력하게 하는 함수
def get_coordinates_robust(location_name: str) -> tuple:
    """
    장소 이름으로부터 좌표(위도, 경도)를 얻습니다.
    여러 방법을 시도하여 좌표를 얻지 못하는 경우를 최소화합니다.
    """
    if not KAKAO_API_KEY:
        raise ValueError("API 키가 설정되지 않았습니다.")
    
    # 시도 1: 주소 검색
    try:
        url = "https://dapi.kakao.com/v2/local/search/address.json"
        headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
        params = {"query": location_name}
        
        response = requests.get(url, headers=headers, params=params, timeout=5)
        data = response.json()
        
        if response.status_code == 200 and data.get("documents"):
            first_result = data["documents"][0]
            latitude = float(first_result.get("y"))
            longitude = float(first_result.get("x"))
            return latitude, longitude
    except Exception as e:
        print(f"주소 검색 실패: {str(e)}")
    
    # 시도 2: 키워드 검색
    try:
        url = "https://dapi.kakao.com/v2/local/search/keyword.json"
        headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
        params = {"query": location_name}
        
        response = requests.get(url, headers=headers, params=params, timeout=5)
        data = response.json()
        
        if response.status_code == 200 and data.get("documents"):
            first_result = data["documents"][0]
            latitude = float(first_result.get("y"))
            longitude = float(first_result.get("x"))
            return latitude, longitude
    except Exception as e:
        print(f"키워드 검색 실패: {str(e)}")
    
    # 시도 3: 지역명으로 분할 검색
    try:
        # 공백으로 지역명 분할 (예: "서울시 강남구" -> "서울시", "강남구")
        parts = location_name.split()
        if len(parts) > 1:
            # 첫 번째 부분만 사용
            partial_name = parts[0]
            
            url = "https://dapi.kakao.com/v2/local/search/keyword.json"
            headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
            params = {"query": partial_name}
            
            response = requests.get(url, headers=headers, params=params, timeout=5)
            data = response.json()
            
            if response.status_code == 200 and data.get("documents"):
                first_result = data["documents"][0]
                latitude = float(first_result.get("y"))
                longitude = float(first_result.get("x"))
                return latitude, longitude
    except Exception as e:
        print(f"부분 지역명 검색 실패: {str(e)}")
    
    # 모든 검색 실패 시
    print(f"위치를 찾을 수 없습니다: {location_name}")
    return None, None

@mcp.tool()
def get_midpoint_facilities(geo1: str, geo2: str, facility_types: List[str] = None, radius: int = 1000) -> Dict:
    """
    두 지역 사이의 중간 지점을 찾고, 주변의 공공 시설(지하철역, 경찰서, 공공기관)을 검색합니다.
    시설이 충분히 나오지 않을 경우 반경을 점진적으로 넓힙니다.
    
    Args:
        geo1 (str): 첫 번째 지역 (예: '서울시 강남구')
        geo2 (str): 두 번째 지역 (예: '서울시 송파구')
        facility_types (List[str], 선택): 검색할 시설 유형 목록 (기본값: ['subway', 'police', 'public'])
        radius (int, 선택): 초기 검색 반경 (미터), 기본값은 1000미터
    
    Returns:
        Dict: 중간 지점 정보와 주변 시설 목록
    """
    try:
        # 시작 시간 기록
        start_time = time.time()
        
        # API 키 확인
        if not KAKAO_API_KEY:
            return {"error": "KAKAO_API_KEY가 설정되지 않았습니다."}
        
        # 기본 시설 유형 설정 - 공공 시설로 한정
        if facility_types is None or len(facility_types) == 0:
            facility_types = ["subway", "police", "public"]
        
        # 시설 타입 필터링 - 공공 시설만 유지
        allowed_types = ["subway", "police", "public", "government", "institution"]
        facility_types = [ft for ft in facility_types if ft.lower() in allowed_types]
        
        if not facility_types:
            facility_types = ["subway", "police", "public"]
        
        # 위치 좌표 얻기 - 오류 처리 강화
        try:
            lat1, lon1 = get_coordinates_robust(geo1)
            lat2, lon2 = get_coordinates_robust(geo2)
            
            if not (lat1 and lon1 and lat2 and lon2):
                # 좌표를 얻지 못했을 경우 기본값 사용 (서울시 중심)
                if not (lat1 and lon1):
                    print(f"Warning: {geo1}의 좌표를 얻지 못했습니다. 기본값을 사용합니다.")
                    lat1, lon1 = 37.5665, 126.9780  # 서울시 중심
                if not (lat2 and lon2):
                    print(f"Warning: {geo2}의 좌표를 얻지 못했습니다. 기본값을 사용합니다.")
                    lat2, lon2 = 37.5665, 126.9780  # 서울시 중심
                
        except Exception as e:
            # 에러 발생 시 대략적인 좌표 사용 (서울시 내 위치)
            print(f"좌표 획득 중 오류: {str(e)}. 대략적인 좌표를 사용합니다.")
            lat1, lon1 = 37.5665, 126.9780  # 서울시 중심
            lat2, lon2 = 37.5665, 126.9780  # 서울시 중심
        
        # 중간 지점 계산
        mid_lat, mid_lon = calculate_midpoint(lat1, lon1, lat2, lon2)
        
        # 주소 정보 가져오기 (실패 시 대체 텍스트 사용)
        try:
            address_info = get_address_from_coordinates(mid_lon, mid_lat)
            if isinstance(address_info, dict) and "error" not in address_info:
                address_string = f"행정구역: {address_info.get('region_1depth_name', '')} {address_info.get('region_2depth_name', '')} {address_info.get('region_3depth_name', '')}"
            else:
                # 두 지역명을 조합한 설명 사용
                address_string = f"{geo1}와 {geo2} 사이의 중간 지점"
        except Exception:
            address_string = f"{geo1}와 {geo2} 사이의 중간 지점"
        
        # 시설 검색 - 반경을 점진적으로 늘리며 검색
        all_facilities = []
        max_execution_time = 15  # 최대 실행 시간 (초)
        min_facilities_total = 3  # 최소 필요 시설 총 개수
        max_radius = 10000  # 최대 검색 반경 (미터)
        current_radius = radius  # 시작 반경
        radius_increment = 1000  # 반경 증가량 (미터)
        
        # 시설 검색 전략 변경: 총 개수로 판단
        while current_radius <= max_radius:
            # 시간 초과 확인
            if time.time() - start_time > max_execution_time:
                break
            
            # 충분한 시설을 찾았는지 확인
            if len(all_facilities) >= min_facilities_total:
                break
            
            for facility_type in facility_types:
                # 시간 초과 확인
                if time.time() - start_time > max_execution_time:
                    break
                    
                try:
                    # 공공 시설에 맞는 키워드 사용
                    search_keyword = facility_type
                    if facility_type == "public":
                        search_keyword = "공공기관"
                    elif facility_type == "police":
                        search_keyword = "경찰서"
                    elif facility_type == "subway":
                        search_keyword = "지하철역"
                    
                    new_facilities = find_nearby_facilities_robust(mid_lat, mid_lon, search_keyword, current_radius)
                    
                    # 이미 찾은 시설과 중복 제거
                    for facility in new_facilities:
                        is_duplicate = False
                        for existing in all_facilities:
                            if (existing.name == facility.name and 
                                abs(existing.latitude - facility.latitude) < 0.0001 and 
                                abs(existing.longitude - facility.longitude) < 0.0001):
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            all_facilities.append(facility)
                            
                except Exception as e:
                    print(f"시설 '{facility_type}' 검색 중 오류: {e}")
            
            # 반경 증가
            current_radius += radius_increment
        
        # 거리순으로 정렬
        all_facilities.sort(key=lambda x: x.distance)
        
        # 결과를 딕셔너리로 변환
        midpoint = {
            "name": address_string,
            "latitude": mid_lat,
            "longitude": mid_lon
        }
        
        facilities_list = []
        for facility in all_facilities:
            facilities_list.append({
                "name": facility.name,
                "address": facility.address,
                "category": facility.category,
                "latitude": facility.latitude,
                "longitude": facility.longitude,
                "distance": facility.distance
            })
        
        # 응답 구성
        response_data = {
            "midpoint": midpoint,
            "facilities": facilities_list,
            "initial_radius": radius,
            "final_radius_used": current_radius - radius_increment if current_radius > radius else radius
        }
        
        # 시설을 찾지 못한 경우에 대한 메시지 추가
        if not facilities_list:
            response_data["message"] = "검색 범위 내에서 시설을 찾지 못했습니다."
        
        return response_data
    
    except Exception as e:
        return {"error": f"처리 중 오류가 발생했습니다: {str(e)}"}


# 서버 시작 시 메시지 출력
print("중고나라 크롤링 서버가 시작되었습니다.")

if __name__ == "__main__":
    print("중고나라 크롤링 MCP 서버 시작 중...")
    mcp.run()