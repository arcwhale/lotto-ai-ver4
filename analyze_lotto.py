import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 🎯 CSV 파일 불러오기 (오류 방지 옵션 추가!)
try:
    df = pd.read_csv("lotto_data.csv", error_bad_lines=False, encoding="utf-8")
except Exception as e:
    print("❌ CSV 파일을 불러오는 중 오류 발생:", e)
    exit()

# 📌 숫자별 출현 빈도수 계산
lotto_flat = df.values.flatten()  # 2D 데이터를 1D 배열로 변환
numbers, counts = np.unique(lotto_flat, return_counts=True)

# 📊 히스토그램 시각화 (출현 빈도)
plt.figure(figsize=(12, 6))
sns.barplot(x=numbers, y=counts, palette="coolwarm")
plt.xlabel("번호")
plt.ylabel("출현 빈도")
plt.title("로또 번호 출현 빈도수")
plt.show()
