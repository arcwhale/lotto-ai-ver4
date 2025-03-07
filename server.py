import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from flask import Flask, render_template, jsonify
import random
from sklearn.preprocessing import MinMaxScaler
from collections import Counter, deque
import os
import schedule
import time
import threading
from sklearn.linear_model import SGDRegressor

app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return "", 200

MODEL_PATH = "lstm_model.h5"

# ✅ 히스토리 저장 (최대 100개)
history = deque(maxlen=100)

# ✅ 로또 데이터 로드
def load_lotto_data():
    try:
        print("🔄 [DEBUG] 로또 데이터 로딩 중...")
        lotto_data = np.loadtxt('lotto_data.csv', delimiter=',', dtype=int, skiprows=1)
        print(f"✅ [DEBUG] 데이터 로드 완료, 크기: {lotto_data.shape}")
        return lotto_data
    except Exception as e:
        print(f"❌ Error loading lotto data: {e}")
        raise e

# ✅ 데이터 전처리 함수 (20회차씩 분석)
def preprocess_data(lotto_data, sequence_length=20):
    X, y = [], []
    for i in range(len(lotto_data) - sequence_length):
        X.append(lotto_data[i:i+sequence_length])
        y.append(lotto_data[i+sequence_length])
    return np.array(X), np.array(y)

# ✅ LSTM 모델 생성
def create_lstm_model():
    model = Sequential([
        LSTM(128, activation='relu', return_sequences=True, input_shape=(20, 6)),
        Dropout(0.4),  # ✅ 과적합 방지
        LSTM(64, activation='relu', return_sequences=True),  
        Dropout(0.4),  # ✅ 과적합 방지
        LSTM(32, activation='relu', return_sequences=False),  
        Dense(6, activation='linear')  # ✅ 숫자 균형 유지
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss=tf.keras.losses.MeanSquaredError())
    return model

# ✅ LSTM 예측 함수 (50세트)
import numpy as np

def generate_lstm_numbers(X, num_predictions=50):
    try:
        if len(X) == 0:
            return [sorted(random.sample(range(1, 46), 6)) for _ in range(num_predictions)]

        if os.path.exists(MODEL_PATH):
            lstm_model = load_model(MODEL_PATH, compile=False)
            lstm_model.compile(optimizer=Adam(learning_rate=0.0005), loss=tf.keras.losses.MeanSquaredError())
        else:
            lstm_model = create_lstm_model()

        lstm_predictions = lstm_model.predict(np.tile(X[-1].reshape(1, 20, 6), (num_predictions, 1, 1)))

        # 🎯 예측값 조정: 랜덤 노이즈 + 확률적 샘플링 적용
        lstm_numbers = []
        for pred in lstm_predictions:
            pred = pred + np.random.normal(0, 3.0, size=6)  
    
            # 🎯 확률적 샘플링 (최빈 숫자 과다 출현 방지)
            valid_numbers = sorted(set(np.clip(np.round(pred * 45).astype(int), 1, 45)))

            # 🎯 중복 숫자 보정
            while len(valid_numbers) < 6:
                valid_numbers.append(random.randint(1, 45))

            # 🎯 숫자 균형 유지 (랜덤 재배치)
            random.shuffle(valid_numbers)
            lstm_numbers.append(valid_numbers[:6])

        return lstm_numbers

    except Exception as e:
        return [sorted(random.sample(range(1, 46), 6)) for _ in range(num_predictions)]

# ✅ 보상 계산 함수 (로또 예측 정확도에 따라 가중치 부여)
def calculate_rewards(predicted_sets, actual_data):
    rewards = []
    for pred in predicted_sets:
        best_score = 0
        for actual in actual_data:
            matches = len(set(pred) & set(actual))
            if matches >= 3:
                score = matches ** 2
            else:
                score = matches
            best_score = max(best_score, score)
        rewards.append(best_score)
    return rewards

# ✅ RL 학습 (SGD 기반 보상 강화)
def train_rl_model(lstm_numbers, actual_data):
    rewards = calculate_rewards(lstm_numbers, actual_data)
    X_train = np.array(lstm_numbers)
    y_train = np.array(rewards)

    if len(X_train.shape) == 1 or X_train.shape[1] != 6:
        return None  

    model = SGDRegressor()
    try:
        model.fit(X_train, y_train)
    except Exception:
        model.partial_fit(X_train, y_train)

    return model

# ✅ Monte Carlo 시뮬레이션 최적화 (중복 방지 추가)
def monte_carlo_simulation(rl_model, num_simulations=10000):
    print("🔄 [DEBUG] Monte Carlo 시뮬레이션 시작...")
    simulated_results = []

    for _ in range(num_simulations):
        sample_numbers = sorted(random.sample(range(1, 46), 6))

        # 🎯 Mutation 적용 (20% 확률로 숫자 1개 변경)
        if random.random() < 0.2:
            idx = random.randint(0, 5)
            sample_numbers[idx] = random.randint(1, 45)

        score = float(rl_model.predict([sample_numbers])[0])
        simulated_results.append((sample_numbers, score))

    simulated_results.sort(key=lambda x: x[1], reverse=True)
    best_samples = [x[0] for x in simulated_results[:10]]

    # ✅ 숫자 균형 필터 적용 (중복 방지)
    number_counts = Counter(num for group in best_samples for num in group)
    unique_best_samples = []

    for numbers in best_samples:
        adjusted_numbers = sorted(numbers, key=lambda n: number_counts[n])
        unique_best_samples.append(adjusted_numbers[:6])

    # 🎯 랜덤 섞기 (너무 일정한 패턴 방지)
    for sample in unique_best_samples:
        random.shuffle(sample)

    print(f"✅ [DEBUG] Monte Carlo 최적화 완료: {unique_best_samples[:5]}")
    return unique_best_samples[:5]

# ✅ JSON 변환 오류 방지: numpy.int64 → int 변환 + 리스트 확인
def convert_to_int(obj):
    """ JSON 직렬화 오류 방지: numpy.int64 → int 변환 + 중첩 리스트 처리 """
    if isinstance(obj, list):  
        return [convert_to_int(item) for item in obj]  
    elif isinstance(obj, np.ndarray):  
        return [convert_to_int(item) for item in obj.tolist()]  
    elif isinstance(obj, (np.int64, np.float64)):  
        return int(obj)  
    else:
        return obj  

@app.route('/predict', methods=['GET'])
def predict():
    try:
        lotto_data = load_lotto_data()
        X, y = preprocess_data(lotto_data)
        lstm_numbers = generate_lstm_numbers(X)
        trained_rl_model = train_rl_model(lstm_numbers, lotto_data)
        final_games = monte_carlo_simulation(trained_rl_model, num_simulations=10000)
        history.appendleft(final_games)

        return jsonify({
            "Optimal Games": convert_to_int(final_games),
            "History": convert_to_int(list(history)),
            "LSTM Predictions": convert_to_int(lstm_numbers)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ✅ LSTM 학습 함수 (Epoch 증가 300~500)
def train_lstm_model():
    try:
        print("🔄 최신 로또 데이터 로드 중...")
        lotto_data = load_lotto_data()
        X, y = preprocess_data(lotto_data)

        print(f"🚀 LSTM 모델 학습 시작 (데이터 크기: {X.shape})...")
        model = create_lstm_model()

        # ✅ 학습 횟수 증가 (300~500 Epoch)
        model.fit(X, y, epochs=300, batch_size=16, verbose=1)

        model.save(MODEL_PATH)
        print("✅ 학습 완료! 모델이 저장되었습니다.")

    except Exception as e:
        print(f"❌ Training Error: {e}")

# ✅ 모델 학습 스케줄링 (매주 토요일 21:00)
schedule.every().saturday.at("21:00").do(train_lstm_model)

# ✅ 스케줄 실행을 위한 스레드
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == '__main__':
    threading.Thread(target=run_scheduler, daemon=True).start()
    app.run(debug=True, port=5001)
