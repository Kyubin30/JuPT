import json
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn.functional import softmax
import numpy as np

# 라벨 매핑 (학습 시 사용한 것과 동일하게)
label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}

def load_model(model_path):
    """학습된 모델과 토크나이저를 로드합니다."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    return model, tokenizer, device

def predict_sentiment(text, model, tokenizer, device, max_length=128):
    """텍스트의 감정을 예측합니다."""
    try:
        # 텍스트 토큰화
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True
        ).to(device)
        
        # 예측
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 모델 출력 형식에 따라 처리
        if isinstance(outputs, tuple):
            # 튜플인 경우 첫 번째 요소가 logits이라고 가정
            logits = outputs[0]
        elif hasattr(outputs, 'logits'):
            # logits 속성이 있는 경우
            logits = outputs.logits
        else:
            # 그 외의 경우 outputs 자체를 사용
            logits = outputs
        
        # 확률 계산
        probs = softmax(logits, dim=1).cpu().numpy()[0]
        predicted_label_idx = np.argmax(probs)
        
        return label_map[predicted_label_idx], probs
    except Exception as e:
        print(f"예측 중 오류 발생 (텍스트 길이: {len(text)}): {str(e)}")
        # 오류 발생 시 중립으로 처리
        return 'neutral', np.array([0.2, 0.2, 0.6])  # 중립에 가중치를 더 줌

def process_articles(input_file, output_file, model_path='./results/best_model'):
    """기사 JSON 파일을 처리하고 결과를 CSV로 저장합니다."""
    # 모델 로드
    print("모델을 로드하는 중입니다...")
    model, tokenizer, device = load_model(model_path)
    
    # JSON 파일 로드
    print(f"{input_file} 파일을 로드하는 중입니다...")
    with open(input_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    results = []
    total = len(articles)
    
    # 각 기사 처리
    print("기사 감정 분석을 시작합니다...")
    for i, article in enumerate(articles, 1):
        try:
            description = article.get('description', '').strip()
            if not description:
                continue
                
            # 감정 분석
            label, probs = predict_sentiment(description, model, tokenizer, device)
            
            # 결과 저장
            results.append({
                'description': description,
                'sentiment': label,
                'positive_prob': round(float(probs[0]), 4),
                'negative_prob': round(float(probs[1]), 4),
                'neutral_prob': round(float(probs[2]), 4)
            })
            
            if i % 10 == 0 or i == total:
                print(f"진행 상황: {i}/{total} ({(i/total)*100:.1f}%)")
                
        except Exception as e:
            print(f"기사 처리 중 오류 발생 (인덱스 {i}): {str(e)}")
    
    # 결과를 DataFrame으로 변환
    df = pd.DataFrame(results)
    
    # CSV로 저장
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n분석이 완료되었습니다. 결과가 {output_file}에 저장되었습니다.")
    print(f"총 {len(df)}개의 기사가 처리되었습니다.")
    
    # 감정 분포 출력
    if not df.empty:
        print("\n감정 분포:")
        print(df['sentiment'].value_counts())

if __name__ == "__main__":
    input_file = "article.json"  # 입력 JSON 파일 경로
    output_file = "article_sentiment_analysis.csv"  # 출력 CSV 파일 경로
    
    process_articles(input_file, output_file)
