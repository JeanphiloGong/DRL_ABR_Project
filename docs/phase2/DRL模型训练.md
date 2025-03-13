### **🚀 训练 DRL 模型步骤**
你已经成功收集了数据，接下来可以使用 **深度强化学习（DRL）** 来训练一个 **智能码率控制（ABR）策略**。这里我们使用 **PyTorch** 结合 **DQN（Deep Q-Network）** 进行训练。

---

## **✅ 1. 定义 DRL 任务**
### **🎯 状态空间（State Space）**
你的数据集包含：
- **网络指标**
  - `throughput`（网络带宽估计）
  - `latency`（网络时延）
- **客户端指标**
  - `buffer`（缓冲区大小）
  - `bitrate`（当前码率）
  - `switch_count`（码率切换次数）

📌 **定义状态向量**：
\[
S_t = (bitrate, buffer, throughput, latency, switch\_count)
\]
---

### **🎯 动作空间（Action Space）**
- 选择不同的码率：
  - `240p (500kbps)`
  - `360p (1000kbps)`
  - `720p (2500kbps)`
  - `1080p (5000kbps)`
  - `4K (8000kbps)`

📌 **定义动作集合**：
\[
A = \{500, 1000, 2500, 5000, 8000\}
\]

---

### **🎯 奖励函数（Reward Function）**
一般采用：
\[
R = w_1 \times bitrate - w_2 \times rebuffering - w_3 \times quality\_switches
\]
- **清晰度奖励**：更高的 `bitrate` 得到更高的奖励（`w_1`）
- **卡顿惩罚**：如果 `buffer` 太低，惩罚 `rebuffering`（`w_2`）
- **码率切换惩罚**：避免频繁切换（`w_3`）

📌 **具体实现**
```python
def compute_reward(bitrate, rebuffering, switch_count, w1=1.0, w2=3.0, w3=1.5):
    return w1 * bitrate - w2 * rebuffering - w3 * switch_count
```

---

## **✅ 2. 预处理数据**
先读取数据并转换成 DRL 训练格式。

### **📌 读取 SQLite 数据**
```python
import sqlite3
import pandas as pd

DATABASE = 'dash_metrics.db'

# 读取数据
conn = sqlite3.connect(DATABASE)
df = pd.read_sql_query("SELECT * FROM dash_metrics ORDER BY timestamp", conn)
conn.close()

print(df.head())
```

### **📌 预处理数据**
```python
import numpy as np

# 归一化数据
def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

df['buffer'] = normalize(df['buffer'], 0, df['buffer'].max())
df['throughput'] = normalize(df['throughput'], 0, df['throughput'].max())
df['latency'] = normalize(df['latency'], 0, df['latency'].max())

# 计算奖励
df['reward'] = df.apply(lambda row: compute_reward(row['bitrate'], 1/row['buffer'] if row['buffer'] > 0 else 10, row['switch_count']), axis=1)

# 状态-动作对
states = df[['bitrate', 'buffer', 'throughput', 'latency', 'switch_count']].values
rewards = df['reward'].values
```

---

## **✅ 3. 训练 DRL 模型（DQN）**
我们使用 **Deep Q-Network（DQN）** 来训练 ABR 策略。

### **📌 1. 定义 Q 网络**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Q-Network 结构
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # 输出 Q 值
```

---

### **📌 2. 训练 DQN**
```python
# 定义参数
STATE_SIZE = states.shape[1]
ACTION_SIZE = 5  # 5 种码率选择
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON = 1.0  # 初始探索率
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
BATCH_SIZE = 32
MEMORY_SIZE = 1000

# 初始化网络
q_network = QNetwork(STATE_SIZE, ACTION_SIZE)
optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# 经验回放
memory = []

# 训练循环
for episode in range(500):  # 训练 500 轮
    state = states[random.randint(0, len(states) - 1)]  # 随机初始化状态
    total_reward = 0
    
    for step in range(100):  # 每轮最多 100 步
        # 选择动作 (Epsilon-Greedy)
        if random.random() < EPSILON:
            action = random.randint(0, ACTION_SIZE - 1)  # 随机动作
        else:
            with torch.no_grad():
                action = torch.argmax(q_network(torch.tensor(state, dtype=torch.float32))).item()

        # 计算奖励
        reward = rewards[action]
        next_state = states[random.randint(0, len(states) - 1)]  # 随机选择下一状态

        # 存入经验回放池
        memory.append((state, action, reward, next_state))
        if len(memory) > MEMORY_SIZE:
            memory.pop(0)

        # 经验回放训练
        if len(memory) > BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            state_batch = torch.tensor([b[0] for b in batch], dtype=torch.float32)
            action_batch = torch.tensor([b[1] for b in batch], dtype=torch.int64)
            reward_batch = torch.tensor([b[2] for b in batch], dtype=torch.float32)
            next_state_batch = torch.tensor([b[3] for b in batch], dtype=torch.float32)

            # 计算 Q 值
            q_values = q_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                target_q_values = reward_batch + GAMMA * q_network(next_state_batch).max(dim=1)[0]

            loss = criterion(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_reward += reward
        state = next_state

    EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)
    print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {EPSILON:.3f}")
```

---

## **✅ 4. 训练完成后，保存模型**
```python
torch.save(q_network.state_dict(), "abr_q_network.pth")
```

---

## **🚀 总结**
| **步骤** | **任务** |
|----------|---------|
| **1. 数据预处理** | 读取 SQLite 数据，归一化 |
| **2. 定义状态空间** | `bitrate, buffer, throughput, latency, switch_count` |
| **3. 定义奖励函数** | `bitrate - rebuffering - switch_count` |
| **4. 训练 DQN** | 经验回放 + Q 网络更新 |
| **5. 训练完成，保存模型** | `torch.save()` |

---

## **🚀 下一步**
- 用 **DRL 训练出的模型** 进行 **在线推理**，让 DASH.js **使用 Q-Network 选择码率**
- **调试和优化超参数**（学习率、折扣因子等）
