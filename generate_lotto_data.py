import pandas as pd
import requests
import os

# 🎯 로또 데이터 가져오기 (공식 API 사용)
url = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo="

# 📌 기존 데이터 로드 (파일이 존재하는 경우)
if os.path.exists("lotto_data.csv"):
    df = pd.read_csv("lotto_data.csv")
    
    # NaN 값 제거 (파일이 손상되었을 가능성 고려)
    df.dropna(inplace=True)

    # 마지막 저장된 회차 찾기
    last_round = len(df) if not df.empty else 0
    print(f"📌 기존 데이터 발견! 마지막 저장된 회차: {last_round}회")
else:
    df = pd.DataFrame(columns=["No1", "No2", "No3", "No4", "No5", "No6"])
    last_round = 0
    print("🚀 기존 데이터 없음! 새로 다운로드 시작")

# 🔍 최신 회차 확인
latest_round = None
for i in range(1100, 1200):  # 1100회 이후부터 최신 회차를 찾음
    res = requests.get(url + str(i))
    if res.status_code == 200:
        data = res.json()
        if data["returnValue"] == "success":
            latest_round = i
        else:
            break  # 최신 회차 이후 데이터 없음
    else:
        break  # API 요청 실패

if latest_round is None:
    print("❌ 최신 회차 정보를 가져오지 못했습니다.")
    exit()

print(f"🔄 최신 로또 회차: {latest_round}회")

# 🔹 새로운 회차 데이터 가져오기
lotto_data = []

for i in range(last_round + 1, latest_round + 1):  # 기존 회차 이후부터 최신 회차까지 가져오기
    res = requests.get(url + str(i))

    if res.status_code == 200:
        data = res.json()
        if data["returnValue"] == "success":
            numbers = [
                data["drwtNo1"], data["drwtNo2"], data["drwtNo3"], 
                data["drwtNo4"], data["drwtNo5"], data["drwtNo6"]
            ]
            lotto_data.append(numbers)
            print(f"✅ {i}회 데이터 추가됨: {numbers}")
    else:
        print(f"❌ 데이터 가져오기 실패 (회차: {i})")

# 📌 DataFrame으로 변환 & 기존 데이터에 추가 저장
if lotto_data:
    new_df = pd.DataFrame(lotto_data, columns=["No1", "No2", "No3", "No4", "No5", "No6"])
    df = pd.concat([df, new_df], ignore_index=True)  # 기존 데이터에 추가
    df.to_csv("lotto_data.csv", index=False)
    print(f"✅ 총 {len(lotto_data)}개 회차 추가 저장 완료! (최신 회차: {latest_round}회)")
else:
    print("🚀 새로운 회차 데이터 없음! 기존 데이터 유지")
