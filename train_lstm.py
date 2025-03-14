import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import random
import os

MODEL_PATH = "lstm_model.h5"

# ✅ 로또 데이터 로드
def load_lotto_data():
    lotto_data = np.loadtxt('lotto_data.csv', delimiter=',', dtype=int, skiprows=1)
    return lotto_data

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
        Dropout(0.4),
        LSTM(64, activation='relu', return_sequences=True),
        Dropout(0.4),
        LSTM(32, activation='relu'),
        Dense(6, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    return model

# ✅ 모델 학습 후 저장
def train_lstm_model():
    lotto_data = load_lotto_data()
    X, y = preprocess_data(lotto_data)

    model = create_lstm_model()

    model.fit(X, y, epochs=300, batch_size=16, verbose=1)
    model.save("lstm_model.h5")

# ✅ LSTM 번호 예측 함수 (정확한 코드)
from tensorflow.keras.models import load_model
import random
MODEL_PATH = "lstm_model.h5"

def generate_lstm_numbers(X, num_predictions=50):
    try:
        lstm_model = load_model(MODEL_PATH, compile=False)
        predictions = lstm_model.predict(np.tile(X[-1].reshape(1, 20, 6), (num_predictions, 1, 1)))

        lstm_numbers = []
        for pred in predictions:
            pred_numbers = sorted(set(np.clip(np.round(pred).astype(int), 1, 45)))
            while len(pred_numbers) < 6:
                new_num = random.randint(1, 45)
                if new_num not in pred_numbers:
                    pred_numbers.append(new_num)
            random.shuffle(pred_numbers)
            lstm_numbers.append(pred_numbers[:6])

        return lstm_numbers

    except Exception as e:
        print(f"❌ generate_lstm_numbers error: {e}")
        return [sorted(random.sample(range(1, 46), 6)) for _ in range(num_predictions)]

# ✅ LSTM 모델 생성
def create_lstm_model():
    model = Sequential([
        LSTM(128, activation='relu', return_sequences=True, input_shape=(20, 6)),
        Dropout(0.4),
        LSTM(64, activation='relu', return_sequences=True),
        Dropout(0.4),
        LSTM(32, activation='relu'),
        Dense(6, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    return model

if __name__ == "__main__":
    train_lstm_model()
