import os
import torch
from data_processing import load_and_preprocess_data
from model_trainer import SentimentTrainer

def main():
    # 디렉토리 생성
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 장치: {device}")
    
    try:
        # 1. 데이터 로드 및 전처리
        print("\n데이터를 불러오고 전처리 중입니다...")
        train_df, val_df, label_map = load_and_preprocess_data('data.csv')
        
        # 2. 모델 초기화
        print("\n모델을 초기화 중입니다...")
        trainer = SentimentTrainer()
        
        # 3. 데이터셋 준비
        print("\n학습 데이터를 준비 중입니다...")
        train_dataset, val_dataset = trainer.prepare_data(train_df, val_df)
        
        # 4. 모델 학습
        print("\n모델 학습을 시작합니다. 이 과정은 시간이 소요될 수 있습니다...")
        trainer.train(train_dataset, val_dataset)
        
        print("\n모든 과정이 성공적으로 완료되었습니다!")
        print("학습된 모델은 'results/best_model' 디렉토리에 저장되었습니다.")
        
    except Exception as e:
        print(f"\n오류가 발생했습니다: {str(e)}")
        print("문제 해결을 위해 오류 메시지를 확인해주세요.")

if __name__ == "__main__":
    main()
