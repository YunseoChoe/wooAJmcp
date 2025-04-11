from mcp.server.fastmcp import FastMCP
import subprocess
import json
import tempfile
import os
import time
from typing import Dict, List, Optional
from collections import defaultdict

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
mcp = FastMCP("joongna_crawler_server")

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

# 서버 시작 시 메시지 출력
print("중고나라 크롤링 서버가 시작되었습니다.")

if __name__ == "__main__":
    print("중고나라 크롤링 MCP 서버 시작 중...")
    mcp.run()