import pandas as pd
import requests
import os
import schedule
import time
from train_lstm import train_lstm_model  # LSTM 학습 모듈 가져오기

# 🎯 로또 데이터 API URL
API_URL = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo="

# ✅ 기존 데이터 로드
def load_existing_data():
    if os.path.exists("lotto_data.csv"):
        df = pd.read_csv("lotto_data.csv")
        df.dropna(inplace=True)  # NaN 값 제거
        last_round = df.shape[0] + 1161  # 현재 데이터가 몇 개 있는지 확인하여 마지막 회차 결정
        print(f"📌 기존 데이터 발견! 마지막 저장된 회차: {last_round}회")
    else:
        df = pd.DataFrame(columns=["No1", "No2", "No3", "No4", "No5", "No6"])
        last_round = 1161  # 1161회차까지는 유지, 1162회차부터 추가
        print("🚀 기존 데이터 없음! 1162회차부터 다운로드 시작")
    
    return df, last_round

# ✅ 최신 로또 회차 확인 (1162회차부터 적용)
def get_latest_round():
    for i in range(1162, 2000):  # 1162회부터 최신 회차를 확인
        res = requests.get(API_URL + str(i))
        if res.status_code == 200:
            data = res.json()
            if data["returnValue"] == "success":
                continue  # 최신 회차 계속 찾기
            else:
                return i - 1  # 이전 회차가 최신 회차
        else:
            break
    return None

# ✅ 새로운 데이터 가져오기
def update_lotto_data():
    df, last_round = load_existing_data()
    latest_round = get_latest_round()

    if latest_round is None:
        print("❌ 최신 회차 정보를 가져오지 못했습니다.")
        return
    
    print(f"🔄 최신 로또 회차: {latest_round}회")

    if last_round >= latest_round:
        print("🚀 최신 데이터와 동일합니다. 업데이트 필요 없음.")
        return

    lotto_data = []

    for i in range(last_round + 1, latest_round + 1):  # 새로운 회차만 가져오기
        res = requests.get(API_URL + str(i))

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

        # ✅ 데이터 업데이트 후 LSTM 모델 재학습
        print("🚀 새로운 데이터를 반영하여 LSTM 모델을 학습합니다...")
        train_lstm_model()
    else:
        print("🚀 새로운 회차 데이터 없음! 기존 데이터 유지")

# ✅ 매주 토요일 오후 9시(21:00)에 실행되도록 설정
schedule.every().saturday.at("21:00").do(update_lotto_data)

# ✅ 스케줄 실행
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)  # 1분마다 스케줄 확인

if __name__ == "__main__":
    update_lotto_data()  # 최초 실행 시 업데이트 수행
    run_scheduler()  # 주기적 실행
