import pandas as pd

# Excel 파일 읽기
df = pd.read_excel(r'C:\ABCD_C\2025.09.09\data.xlsx')

# 데이터 정리: 첫 번째 행을 컬럼명으로 설정
df.columns = df.iloc[0]  # 첫 번째 행을 컬럼명으로
df = df.drop(df.index[0])  # 첫 번째 행 삭제
df = df.drop(df.columns[0], axis=1)  # 첫 번째 열(빈 열) 삭제

# 인덱스 리셋
df = df.reset_index(drop=True)

# 데이터 확인
print("정리된 데이터:")
print(df)
print(f"\n데이터 크기: {df.shape[0]}행 {df.shape[1]}열")
print(f"\n컬럼명: {list(df.columns)}")