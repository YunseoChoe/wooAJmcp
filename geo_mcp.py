from mcp.server.fastmcp import FastMCP
from typing import Dict, List, Optional
import requests
import math
import os
from dotenv import load_dotenv
from pydantic import BaseModel

# .env 파일에서 환경 변수 불러오기
load_dotenv()

# Kakao API 키 또는 Google Maps API 키를 환경 변수에서 불러오기
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# MCP 서버 생성
mcp = FastMCP("geo_server")

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
        
    elif GOOGLE_MAPS_API_KEY:
        # Google Maps Geocoding API 사용
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            "address": location_name,
            "key": GOOGLE_MAPS_API_KEY
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if response.status_code != 200 or data.get("status") != "OK":
            raise ValueError(f"위치를 찾을 수 없습니다: {location_name}")
        
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
            "cafe": "FD6",      # 카페
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
    
    elif GOOGLE_MAPS_API_KEY:
        # Google Places API 사용
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{latitude},{longitude}",
            "radius": radius,
            "type": facility_type,
            "key": GOOGLE_MAPS_API_KEY
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            for place in data.get("results", []):
                # 거리 계산
                place_lat = place["geometry"]["location"]["lat"]
                place_lng = place["geometry"]["location"]["lng"]
                distance = calculate_distance(latitude, longitude, place_lat, place_lng)
                
                results.append(POI(
                    name=place.get("name"),
                    address=place.get("vicinity", ""),
                    category=", ".join(place.get("types", [])),
                    latitude=place_lat,
                    longitude=place_lng,
                    distance=distance
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

@mcp.tool()
def get_mid_geo(geo1: str, geo2: str) -> Dict:
    """
    사용자의 응답으로부터 두 지역을 결과로 반환합니다.
    
    Args:
        geo1 (str): 첫 번째 지역
        geo2 (str): 두 번째 지역
    
    Returns:
        Dict: 결과 반환 (첫 번째 지역, 두 번째 지역)
    """
    
    response_data = {
        'first': geo1,
        'second': geo2
    }
    
    return response_data

@mcp.tool()
def get_midpoint_facilities(geo1: str, geo2: str, facility_types: List[str] = None, radius: int = 1000) -> Dict:
    """
    두 지역 사이의 중간 지점을 찾고, 주변 시설을 검색합니다.
    
    Args:
        geo1 (str): 첫 번째 지역 (예: '서울시 강남구')
        geo2 (str): 두 번째 지역 (예: '서울시 송파구')
        facility_types (List[str], 선택): 검색할 시설 유형 목록 (예: ['subway', 'police'])
        radius (int, 선택): 검색 반경 (미터), 기본값은 1000미터
    
    Returns:
        Dict: 중간 지점 정보와 주변 시설 목록
    """
    try:
        # 기본 시설 유형 설정
        if facility_types is None or len(facility_types) == 0:
            facility_types = ["subway", "police"]
        
        # 위치 좌표 얻기
        lat1, lon1 = get_coordinates(geo1)
        lat2, lon2 = get_coordinates(geo2)
        
        # 중간 지점 계산
        mid_lat, mid_lon = calculate_midpoint(lat1, lon1, lat2, lon2)
        
        # 모든 시설 유형에 대해 검색 결과 합치기
        all_facilities = []
        for facility_type in facility_types:
            facilities = find_nearby_facilities(mid_lat, mid_lon, facility_type, radius)
            all_facilities.extend(facilities)
        
        # 거리순으로 정렬
        all_facilities.sort(key=lambda x: x.distance)
        
        # 결과를 딕셔너리로 변환
        midpoint = {
            "name": "중간 지점",
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
            "facilities": facilities_list
        }
        
        return response_data
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("지리 정보 MCP 서버 시작 중...")
    mcp.run()