# --- 1. 라이브러리 임포트 ---
import os
import logging
import pytz
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request, HTTPException
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, Any
from urllib.parse import parse_qs

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    FollowEvent, PostbackEvent, FlexSendMessage, BubbleContainer, CarouselContainer,
    BoxComponent, TextComponent, ButtonComponent, URIAction, PostbackAction, FillerComponent
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# --- 2. 설정 로드 및 객체 초기화 ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# .env 파일에서 환경 변수 로드
CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')
MONGODB_URI = os.getenv('MONGODB_URI')
MONGODB_DB = os.getenv('MONGODB_DB')
CLOVA_STUDIO_API_KEY = os.getenv('CLOVA_STUDIO_API_KEY')

# 핵심 객체 초기화
app = FastAPI()
line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client[MONGODB_DB]
scheduler = AsyncIOScheduler()
KST = pytz.timezone('Asia/Seoul')

# --- 3. 헬퍼 함수: 스크래핑 및 Clova API 호출 ---
def scrape_article_text(url: str) -> str:
    # 사용자가 요청한 뉴스의 원문을 가져오기 위한 스크래핑 함수
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.select_one('#dic_area') or soup.select_one('#articeBody') or soup.select_one('#newsEndContents')
        return content.get_text(separator='\n', strip=True) if content else ""
    except Exception as e:
        logger.error(f"스크래핑 실패 (URL: {url}): {e}")
        return ""

def summarize_with_clova(text: str) -> str:
    # Clova 텍스트 요약 API를 호출하는 함수
    if not CLOVA_STUDIO_API_KEY:
        return "Clova API 설정이 필요합니다."
    
    headers = {
        'Content-Type': 'application/json; charset=utf-8',
        'Authorization': f"Bearer {CLOVA_STUDIO_API_KEY}"
    }
    data = {"texts": [text[:2000]]} # API 글자 수 제한
    api_endpoint = "https://clovastudio.stream.ntruss.com/v1/api-tools/summarization/v2"

    try:
        response = requests.post(api_endpoint, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        if result.get('status', {}).get('code') == '20000':
            return result.get('result', {}).get('text', '요약 결과가 없습니다.')
        else:
            logger.error(f"Clova 요약 API 에러: {result.get('status', {}).get('message')}")
            return "요약 중 오류가 발생했습니다."
    except Exception as e:
        logger.error(f"Clova 요약 API 호출 실패: {e}")
        return "요약 중 오류가 발생했습니다."

# --- 4. 핵심 로직: Flex Message 생성 함수 ---
def create_news_bubble(article: Dict[str, Any], company_name: str) -> BubbleContainer:
    prediction = article.get('prediction', 'neutral')
    probability = article.get('probability', 0)
    
    if prediction == 'positive':
        header_color, bar_color, bg_color = "#27ACB2", "#0D8186", "#9FD8E36E"
        sentiment_text = f"{probability:.0%} 긍정"
    elif prediction == 'negative':
        header_color, bar_color, bg_color = "#FF6B6E", "#DE5658", "#FAD2A76E"
        sentiment_text = f"{probability:.0%} 부정"
    else: # neutral or N/A
        header_color, bar_color, bg_color = "#A9A9A9", "#696969", "#D3D3D36E"
        sentiment_text = f"{probability:.0%} 중립"

    summary_button_action = PostbackAction(
        label='AI 요약하기',
        data=f"action=summarize&link={article.get('originalLink', '')}",
        displayText='선택한 뉴스를 AI로 요약하고 있습니다...'
    )

    return BubbleContainer(
        size="giga",
        header=BoxComponent(
            layout='vertical', background_color=header_color, padding_all='12px',
            contents=[
                TextComponent(text=company_name, color='#ffffff', size='lg', gravity='center'),
                TextComponent(text=f"AI 분석 - {sentiment_text}", color='#ffffff', size='xs', gravity='center', margin='lg'),
                BoxComponent(
                    layout='vertical', background_color=bg_color, height='6px', margin='sm',
                    contents=[BoxComponent(layout='vertical', background_color=bar_color, height='6px', width=f"{probability:.0%}", contents=[FillerComponent()])]
                )
            ]
        ),
        body=BoxComponent(
            layout='vertical', padding_all='12px', spacing='md',
            contents=[
                BoxComponent(layout='horizontal', flex=1, contents=[TextComponent(text=article.get('description', '내용 없음'), color='#8C8C8C', size='sm', wrap=True)]),
                ButtonComponent(height='sm', action=URIAction(label='원문 보기', uri=article.get('link', 'https://www.naver.com'))),
                ButtonComponent(height='sm', action=summary_button_action)
            ]
        )
    )

# --- 5. 스케줄링될 자동 알림 함수 ---
def send_scheduled_notifications():
    logger.info("[Scheduler] 알림 발송 작업 시작...")
    current_time_kst = datetime.now(KST)
    current_time_str = current_time_kst.strftime("%H:%M")
    
    logger.info(f"[Scheduler] 현재 한국 시간({current_time_str}) 기준으로 알림 대상을 찾습니다.")
    
    users_to_notify = list(db.users.find({"notificationTime": current_time_str}))
    
    if not users_to_notify:
        logger.info("[Scheduler] 현재 시간에 알림을 받을 사용자가 없습니다.")
        return

    logger.info(f"[Scheduler] 총 {len(users_to_notify)}명의 발송 대상자를 찾았습니다.")
    
    for user in users_to_notify:
        user_id = user['_id']
        interest_keywords = user.get('keywords', [])
        logger.info(f"[Scheduler] 발송 처리 중: {user_id}, 키워드: {interest_keywords}")
        
        if not interest_keywords: continue
        bubbles = []
        for company in interest_keywords:
            pipeline = [{"$match": {"keywords": company}}, {"$sort": {"prediction": 1, "probability": -1, "createdAt": -1}}, {"$limit": 1}]
            articles = list(db.articles.aggregate(pipeline))
            if articles:
                bubbles.append(create_news_bubble(articles[0], company))
        
        if bubbles:
            carousel_message = FlexSendMessage(alt_text="오늘의 맞춤 뉴스 도착!", contents=CarouselContainer(contents=bubbles))
            try:
                line_bot_api.push_message(user_id, messages=carousel_message)
                logger.info(f"[Scheduler] {user_id}에게 맞춤 뉴스 발송 성공.")
            except Exception as e:
                logger.error(f"[Scheduler] {user_id}에게 메시지 발송 실패: {e}")

# --- 6. FastAPI 앱 시작/종료 이벤트 ---
@app.on_event("startup")
async def startup_event():
    scheduler.configure(timezone='Asia/Seoul')
    scheduler.add_job(send_scheduled_notifications, 'interval', minutes=1)
    scheduler.start()
    logger.info("APScheduler가 한국 시간 기준으로 시작되었습니다.")

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()

# --- 7. FastAPI 엔드포인트 및 LINE 이벤트 핸들러 ---
@app.get("/")
def read_root(): return {"status": "ok", "server": "api-server"}

@app.post("/webhook")
async def webhook(request: Request):
    signature = request.headers['X-Line-Signature']
    body = await request.body()
    try: handler.handle(body.decode(), signature)
    except InvalidSignatureError: raise HTTPException(status_code=400, detail="Invalid signature")
    return 'OK'

@handler.add(PostbackEvent)
def handle_postback(event):
    user_id = event.source.user_id
    postback_data = parse_qs(event.postback.data)
    action = postback_data.get('action', [None])[0]

    if action == 'summarize':
        link = postback_data.get('link', [None])[0]
        if not link:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="오류: 요약할 기사 링크를 찾을 수 없습니다."))
            return

        today_str = datetime.now(KST).strftime("%Y-%m-%d")
        user_doc = db.users.find_one({"_id": user_id})
        usage = user_doc.get('summaryUsage', {})
        
        if usage.get('date') == today_str and usage.get('count', 0) >= 3:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="AI 요약 기능은 하루 3번까지만 사용할 수 있어요. 내일 다시 이용해주세요!"))
            return

        article_text = scrape_article_text(link)
        if not article_text:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="기사 내용을 가져오는 데 실패했습니다."))
            return
            
        summary_text = summarize_with_clova(article_text)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"🤖 AI 요약 결과입니다:\n\n{summary_text}"))
        
        db.users.update_one(
            {"_id": user_id},
            {"$inc": {"summaryUsage.count": 1}, "$set": {"summaryUsage.date": today_str}},
            upsert=True
        )

@handler.add(FollowEvent)
def handle_follow(event):
    user_id = event.source.user_id; profile = line_bot_api.get_profile(user_id)
    db.users.update_one({"_id": user_id}, {"$setOnInsert": {"displayName": profile.display_name, "createdAt": datetime.now()}}, upsert=True)
    reply_text = f"안녕하세요, {profile.display_name}님!\n'관심 주식 등록' 또는 '알림 시간 설정'이라고 입력하여 맞춤 뉴스 알림을 설정해보세요."
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id
    text = event.message.text.strip()
    state_doc = db.user_states.find_one({"_id": user_id})
    current_state = state_doc['state'] if state_doc else "normal"

    if current_state == "normal":
        if text in ["관심 주식 등록", "관심주식", "주식 등록"]:
            db.user_states.update_one({"_id": user_id}, {"$set": {"state": "awaiting_keywords"}}, upsert=True)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="알림받을 관심 주식을 쉼표(,)로 구분하여 입력해주세요.\n(예: 삼성전자,카카오,NAVER)"))
        elif text in ["알림 시간 설정", "시간 설정", "알림 시간"]:
            db.user_states.update_one({"_id": user_id}, {"$set": {"state": "awaiting_time"}}, upsert=True)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="알림받을 시간을 'HH:MM' 24시간 형식으로 입력해주세요.\n(예: 08:30 또는 15:00)"))
        else:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="안녕하세요! '관심 주식 등록' 또는 '알림 시간 설정'이라고 말씀해주세요."))
    
    elif current_state == "awaiting_keywords":
        keywords = [k.strip() for k in text.split(',')]
        db.users.update_one({"_id": user_id}, {"$set": {"keywords": keywords, "updatedAt": datetime.now()}}, upsert=True)
        db.user_states.update_one({"_id": user_id}, {"$set": {"state": "awaiting_time"}})
        reply_text = f"'{', '.join(keywords)}'이(가) 등록되었습니다.\n\n이제 알림받을 시간을 'HH:MM' 24시간 형식으로 바로 입력해주세요."
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

    elif current_state == "awaiting_time":
        try:
            # 사용자가 입력한 시간을 HH:MM 형식으로 변환하여 저장합니다.
            time_obj = datetime.strptime(text, "%H:%M")
            notification_time = time_obj.strftime("%H:%M")
            
            db.users.update_one({"_id": user_id}, {"$set": {"notificationTime": notification_time, "updatedAt": datetime.now()}}, upsert=True)
            db.user_states.update_one({"_id": user_id}, {"$set": {"state": "normal"}})
            reply_text = f"알림 시간이 {notification_time}으로 설정되었습니다. 이제 매일 해당 시간에 맞춤 뉴스를 보내드릴게요!"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
        except ValueError:
            # 사용자가 잘못된 형식으로 입력한 경우, 다시 입력을 요청합니다.
            reply_text = "시간 형식이 올바르지 않습니다. '08:30' 또는 '15:00' 과 같이 'HH:MM' 24시간 형식으로 다시 입력해주세요."
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))