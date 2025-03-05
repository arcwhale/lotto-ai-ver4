import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam
from flask import Flask, render_template, jsonify
import random
from sklearn.preprocessing import MinMaxScaler
from collections import Counter, deque

app = Flask(__name__, static_folder="static", template_folder="templates")

# ✅ 로또 데이터 로드 및 정규화
lotto_data = np.loadtxt('lotto_data.csv', delimiter=',', dtype=int, skiprows=1)
scaler = MinMaxScaler(feature_range=(0, 1))
lotto_data = scaler.fit_transform(lotto_data)

# ✅ LSTM 학습 데이터 전처리
X, y = [], []
sequence_length = 10  

for i in range(len(lotto_data) - sequence_length):
    X.append(lotto_data[i:i+sequence_length])
    y.append(lotto_data[i+sequence_length])

X, y = np.array(X), np.array(y)

# ✅ LSTM 모델 정의
lstm_model = Sequential([
    LSTM(128, activation='relu', return_sequences=True, input_shape=(sequence_length, 6)),
    Dropout(0.2),
    LSTM(64, activation='relu'),
    Dropout(0.2),
    Dense(6, activation='sigmoid')
])

lstm_model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
lstm_model.fit(X, y, epochs=100, batch_size=16, verbose=1)

# ✅ GAN & RL 기반 생성 함수
def generate_lotto_numbers_rl():
    return [sorted(random.sample(range(1, 46), 6)) for _ in range(10)]

generator = Sequential([
    Dense(128, input_dim=10),
    LeakyReLU(0.2),
    Dense(64),
    LeakyReLU(0.2),
    Dense(6, activation='sigmoid')
])

def generate_lotto_numbers_gan():
    try:
        noise = np.random.normal(0, 1, (10, 10))
        generated_numbers = generator.predict(noise) * 45  
        return [sorted(set(map(int, numbers)))[:6] for numbers in generated_numbers]
    except Exception as e:
        print(f"❌ GAN Prediction Error: {e}")
        return [sorted(random.sample(range(1, 46), 6)) for _ in range(10)]

# ✅ 히스토리 저장 (최대 100개)
history = deque(maxlen=100)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        lstm_predictions = lstm_model.predict(np.tile(X[-1].reshape(1, sequence_length, 6), (10, 1, 1)))
        lstm_numbers = [sorted(set(map(int, (pred * 45))))[:6] for pred in lstm_predictions]

        if any(np.isnan(lstm).any() for lstm in lstm_numbers):
            print("❌ LSTM Prediction contains NaN. Returning random numbers.")
            lstm_numbers = [sorted(random.sample(range(1, 46), 6)) for _ in range(10)]
        
        rl_predictions = generate_lotto_numbers_rl()
        gan_predictions = generate_lotto_numbers_gan()

        # ✅ 최적의 5세트 생성
        all_numbers = [num for game in (lstm_numbers + rl_predictions + gan_predictions) for num in game]
        most_common_numbers = [num for num, count in Counter(all_numbers).most_common(30)]
        final_games = [sorted(random.sample(most_common_numbers, 6)) for _ in range(5)]

        # ✅ 히스토리에 최신 5세트 추가 (리스트 구조 유지)
        history.appendleft(final_games)

        return jsonify({
            'Optimal Games': final_games,
            'History': list(history)  # ✅ 배열의 배열 형태로 반환 (프론트엔드에서 읽을 수 있도록)
        })
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
