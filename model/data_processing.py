import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_and_preprocess_data(file_path='data.csv'):
    """
    데이터를 로드하고 전처리하는 함수
    
    Args:
        file_path (str): 데이터 파일 경로
        
    Returns:
        tuple: (train_df, val_df) 학습용과 검증용 데이터프레임
    """
    # 데이터 로드
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"'{file_path}' 파일을 찾을 수 없습니다. 데이터 파일 경로를 확인해주세요.")
    
    df = pd.read_csv(file_path)
    
    # 필요한 컬럼만 선택하고 결측치 제거
    df = df[['kor_sentence', 'labels']].dropna()
    
    # 레이블을 숫자로 변환
    label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
    df['labels'] = df['labels'].map(label_map)
    
    # 데이터셋 분리 (학습 80%, 검증 20%)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['labels'])
    
    print(f"\n전체 데이터 수: {len(df)}")
    print(f"학습 데이터 수: {len(train_df)}")
    print(f"검증 데이터 수: {len(val_df)}")
    
    return train_df, val_df, label_map

if __name__ == "__main__":
    try:
        train_df, val_df, label_map = load_and_preprocess_data()
        print("\n데이터 전처리가 완료되었습니다.")
        print("학습 데이터 샘플:")
        print(train_df.head())
    except Exception as e:
        print(f"오류 발생: {str(e)}")
