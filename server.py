import torch
import torch.nn as nn
import sqlite3
import numpy as np
import time
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS  # 允许 CORS 访问

# 初始化 Flask
app = Flask(__name__)
CORS(app)  # 允许所有跨域请求

# 初始化 SQLite 数据库
DATABASE = 'dash_metrics.db'

def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS dash_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            bitrate INTEGER,
            buffer REAL,
            throughput INTEGER,
            latency REAL,
            switch_count INTEGER
        )
    ''')
    conn.commit()
    conn.close()

# 定义 Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 码率选项（单位：kbps）
bitrate_options = [500, 1000, 2500, 5000, 8000]

# 加载模型
STATE_SIZE = 5  # (bitrate, buffer, throughput, latency, switch_count)
ACTION_SIZE = len(bitrate_options)
q_network = QNetwork(STATE_SIZE, ACTION_SIZE)

try:
    q_network.load_state_dict(torch.load("abr_q_network.pth"))
    q_network.eval()
    print("DRL 模型加载成功")
except Exception as e:
    print(f"DRL 模型加载失败: {e}")

# 归一化函数
def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

@app.route('/log', methods=['POST'])
def log_data():
    try:
        data = request.json
        timestamp = time.time()
        bitrate = data.get("bitrate", 0)
        buffer = data.get("buffer", 0)
        throughput = data.get("throughput", 0)
        latency = data.get("latency", 0)
        switch_count = data.get("switch_count", 0)

        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute("INSERT INTO dash_metrics (timestamp, bitrate, buffer, throughput, latency, switch_count) VALUES (?, ?, ?, ?, ?, ?)", 
                  (timestamp, bitrate, buffer, throughput, latency, switch_count))
        conn.commit()
        conn.close()

        return jsonify({"status": "success", "data": data})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute("SELECT * FROM dash_metrics ORDER BY timestamp DESC LIMIT 100")
        rows = c.fetchall()
        conn.close()

        data = [
            {"id": row[0], "timestamp": row[1], "bitrate": row[2], "buffer": row[3], 
             "throughput": row[4], "latency": row[5], "switch_count": row[6]}
            for row in rows
        ]

        return jsonify({"status": "success", "data": data})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/predict_bitrate', methods=['POST'])
def predict_bitrate():
    try:
        data = request.json
        state = np.array([
            data["bitrate"],
            data["buffer"],
            data["throughput"],
            data["latency"],
            data["switch_count"]
        ], dtype=np.float32)

        # 归一化数据
        state[1] = normalize(state[1], 0, 10)  # 假设 buffer 最大值为 10 秒
        state[2] = normalize(state[2], 0, 10000)  # 假设带宽最大值 10 Mbps
        state[3] = normalize(state[3], 0, 5)  # 假设时延最大值 5s
        state[4] = normalize(state[4], 0, 10)  # 码率切换最大值 10

        # 转换为 Tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # 预测码率
        with torch.no_grad():
            action_idx = torch.argmax(q_network(state_tensor)).item()

        selected_bitrate = bitrate_options[action_idx]
        return jsonify({"bitrate": selected_bitrate})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000)
