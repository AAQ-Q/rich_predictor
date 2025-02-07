from pywebio.input import *
from pywebio.output import *
from pywebio import start_server
from sklearn.ensemble import RandomForestClassifier
import numpy as np

model = RandomForestClassifier()
game_results = []

def encode_data(data):
    return [1 if x == "B" else 0 for x in data]

def decode_data(data):
    return "B" if data == 1 else "P"

def train_model():
    if len(game_results) < 5:
        return False
    X = []
    y = []
    for i in range(3, len(game_results)):
        X.append(game_results[i - 3:i])
        y.append(game_results[i])
    X = np.array([encode_data(x) for x in X])
    y = np.array(encode_data(y))
    model.fit(X, y)
    return True

def predict_next():
    if len(game_results) < 3:
        return "未知"
    last_sequence = game_results[-3:]
    X = np.array([encode_data(last_sequence)])
    prediction = model.predict(X)[0]
    return decode_data(prediction)

def main():
    put_text("富豪預測系統")
    while True:
        result = radio("請選擇最近的遊戲結果：", ["莊 (B)", "閒 (P)"])
        game_results.append("B" if result == "莊 (B)" else "P")
        train_model()
        put_text("歷史記錄：" + " ".join(game_results))
        put_text(f"下一次可能是：{predict_next()}")

start_server(main, port=8080, debug=True)
