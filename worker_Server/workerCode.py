import os
import requests
import logging
import re
import html
import json
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pymongo import MongoClient, errors
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv

# --- 1. 초기 설정: 환경 변수, 로깅, 상수 ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# .env 파일에서 설정값 로드
NAVER_CLIENT_ID = os.getenv('NAVER_CLIENT_ID')
NAVER_CLIENT_SECRET = os.getenv('NAVER_CLIENT_SECRET')
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB = os.getenv('MONGODB_DB', 'lineChatbotDB')
MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION', 'articles')
BASE_URL = "https://openapi.naver.com/v1/search/news.json"

# 파일 및 모델 경로 설정
MODEL_PATH = "./model_files/"
COMPANY_MAP_FILE = "company_map.json"
DEVICE = "cpu"

# --- 2. 헬퍼 함수 및 핵심 로직 클래스 ---

def cleanse_html(text: str) -> str:
    if not text: return ""
    return html.unescape(re.sub(r"<[^>]+>", "", text))

def load_company_map(filepath: str) -> Dict[str, str]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            company_map = json.load(f)
        logger.info(f"성공적으로 {len(company_map)}개의 회사명-별칭 매핑을 로드했습니다.")
        return company_map
    except Exception as e:
        logger.error(f"치명적 오류: {filepath} 파일 로드 실패 - {e}")
        return {}

class SentimentAnalyzer:
    def __init__(self, model_path: str, device: str):
        self.model, self.tokenizer = self._load_model(model_path, device)
        self.label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
        self.device = device

    def _load_model(self, model_path: str, device: str) -> Tuple:
        try:
            logger.info(f"'{model_path}'에서 감성 분석 모델을 로드합니다...")
            model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model.to(device)
            model.eval()
            logger.info("모델 로드에 성공했습니다.")
            return model, tokenizer
        except Exception as e:
            logger.error(f"모델 로드 중 심각한 오류 발생: {e}")
            return None, None

    def analyze(self, text: str) -> Dict[str, Any]:
        if not self.model:
            return {"prediction": "분석 불가", "probability": 0.0}
        try:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=128, padding="max_length", truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            idx = np.argmax(probs)
            return {"prediction": self.label_map.get(idx, "알 수 없음"), "probability": float(probs[idx])}
        except Exception as e:
            logger.error(f"감성 분석 중 오류 발생: {e}")
            return {"prediction": "분석 오류", "probability": 0.0}

def fetch_naver_news_all(query: str, total: int = 1000) -> List[Dict[str, Any]]:
    headers = {"X-Naver-Client-Id": NAVER_CLIENT_ID, "X-Naver-Client-Secret": NAVER_CLIENT_SECRET}
    all_articles = []
    start = 1
    while start <= total:
        params = {"query": query, "display": 100, "start": start, "sort": "date"}
        try:
            res = requests.get(BASE_URL, headers=headers, params=params)
            res.raise_for_status()
            items = res.json().get("items", [])
            if not items:
                break
            all_articles.extend(items)
            logger.info(f"Collected {start}~{start+len(items)-1} articles for query: {query}")
            start += len(items)
            time.sleep(0.3)
        except Exception as e:
            logger.error(f"Failed to fetch from {start} for query {query}: {e}")
            break
    return all_articles

class NewsToMongoDB:
    def __init__(self, connection_string: str, database_name: str, collection_name: str):
        self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        logger.info(f"Connected to MongoDB: {database_name}.{collection_name}")

    def disconnect(self):
        self.client.close()
        logger.info("MongoDB connection closed")

    def setup_indexes(self):
        try:
            self.collection.create_index("originalLink", unique=True, name="originalLink_unique")
            self.collection.create_index("keywords", name="keywords_multikey")
            logger.info("인덱스 설정 완료: originalLink, keywords")
        except errors.OperationFailure as e:
            logger.warning(f"인덱스 생성 오류 또는 이미 존재함: {e}")

    def delete_old_articles(self, days_to_keep: int = 2):
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            logger.info(f"{cutoff_date} 이전의 오래된 데이터를 삭제합니다...")
            query = {"createdAt": {"$lt": cutoff_date}}
            delete_result = self.collection.delete_many(query)
            if delete_result.deleted_count > 0:
                logger.info(f"성공적으로 {delete_result.deleted_count}개의 오래된 문서를 삭제했습니다.")
            else:
                logger.info("삭제할 오래된 문서가 없습니다.")
        except Exception as e:
            logger.error(f"오래된 데이터 삭제 중 오류 발생: {e}")

    def save_processed_articles(self, articles: List[Dict[str, Any]]) -> Dict[str, int]:
        result = {"inserted": 0, "skipped": 0, "failed": 0, "updated": 0}
        for article in articles:
            try:
                #API가 주는 두 링크를 먼저 변수에 저장
                api_link = article.get("link")
                api_originallink = article.get("originallink")
                naver_link = None
                press_link = None

                # (핵심 로직) 두 링크를 검사하여 어떤 것이 네이버 링크인지 판별
                if api_link and "n.news.naver.com" in api_link:
                    # case 1: 'link'가 네이버 링크인 가장 일반적인 경우
                    naver_link = api_link
                    press_link = api_originallink
                elif api_originallink and "n.news.naver.com" in api_originallink:
                    # case 2: 'originallink'가 네이버 링크인 예외적인 경우
                    naver_link = api_originallink
                    press_link = api_link
                else:
                    # case 3: 둘 다 네이버 링크가 아닌 경우 (블로그 등), 크롤링이 불가능
                    logger.warning(f"네이버 뉴스 링크를 찾을 수 없어 건너뜁니다: {article.get('title')}")
                    result["skipped"] += 1
                    continue

                # 'originalLink' 필드에는 스크래핑에 사용할 네이버 링크를 저장
                # 'link' 필드에는 '원문 보기' 버튼에 사용할 언론사 링크를 저장
                doc = {
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "originalLink": naver_link,
                    "link": press_link,
                    "pubDate": article.get("pubDate"),
                    "keywords": article.get("mentioned_companies", []),
                    "prediction": article.get("sentiment", {}).get("prediction"),
                    "probability": article.get("sentiment", {}).get("probability"),
                    "createdAt": datetime.now()
                }

                if not doc["originalLink"]:
                    result["skipped"] += 1
                    continue

                update_result = self.collection.update_one(
                    {"originalLink": doc["originalLink"]},
                    {"$setOnInsert": doc},
                    upsert=True
                )
                if update_result.upserted_id:
                    result["inserted"] += 1
                else:
                    result["updated"] += 1
            except Exception as e:
                logger.error(f"기사 저장 실패 (링크: {api_link}): {e}")
                result["failed"] += 1
        return result

# --- 3. 메인 실행 함수 ---
def main():
    logger.info("===== 뉴스 분석 및 저장 작업 시작 =====")
    start_time = time.time()

    company_map = load_company_map(COMPANY_MAP_FILE)
    analyzer = SentimentAnalyzer(MODEL_PATH, DEVICE)
    news_db = NewsToMongoDB(MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION)

    if not company_map or not analyzer.model:
        logger.critical("필수 파일 로드 실패. 작업 중단."); return

    news_db.setup_indexes()

    search_keywords = ["경제", "금융", "주식", "부동산", "환율", "기업", "산업", "투자", "경기"]
    articles_to_save = []

    try:
        for keyword in search_keywords:
            articles_raw = fetch_naver_news_all(keyword, total=1000)           
            for article in articles_raw:
                title = cleanse_html(article.get("title"))
                description = cleanse_html(article.get("description"))
                mentioned_companies = set()
                text_to_search = title + " " + description
                for alias, official_name in company_map.items():
                    if alias in text_to_search:
                        mentioned_companies.add(official_name)

                if mentioned_companies:
                    sentiment_result = analyzer.analyze(title)
                    articles_to_save.append({
                        "title": title,
                        "description": description,
                        "originallink": article.get("originallink"),
                        "link": article.get("link"),
                        "pubDate": article.get("pubDate"),
                        "mentioned_companies": list(mentioned_companies),
                        "sentiment": sentiment_result
                    })

            logger.info(f"키워드 '{keyword}' 처리 완료. 저장 대상 기사 수: {len(articles_to_save)}")

        if articles_to_save:
            logger.info(f"총 {len(articles_to_save)}개의 처리된 뉴스를 DB에 저장합니다...")
            result = news_db.save_processed_articles(articles_to_save)
            logger.info(f"DB 저장 결과: {result}")
        else:
            logger.info("새롭게 저장할 뉴스가 없습니다.")

    except Exception as e:
        logger.error(f"메인 파이프라인 실행 중 오류 발생: {e}", exc_info=True)
    finally:
        logger.info("작업 완료 후 데이터베이스 정리 작업을 시작합니다.")
        news_db.delete_old_articles(days_to_keep=2)
        news_db.disconnect()
        end_time = time.time()
        logger.info(f"===== 작업 종료 (총 소요 시간: {end_time - start_time:.2f}초) =====")

if __name__ == "__main__":
    main()
                                                                 
                                                                          