import pandas as pd

try:
    # 데이터 로드
    df = pd.read_csv('data.csv')
    
    # 데이터프레임 정보 출력
    print("\n=== 데이터프레임 정보 ===")
    print(f"행 수: {len(df)}")
    print(f"컬럼 목록: {df.columns.tolist()}")
    
    # 처음 3행 출력
    print("\n=== 데이터 샘플 ===")
    print(df.head(3))
    
    # 컬럼별 결측치 확인
    print("\n=== 결측치 개수 ===")
    print(df.isnull().sum())
    
except Exception as e:
    print(f"오류 발생: {str(e)}")
