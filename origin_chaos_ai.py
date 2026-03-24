import numpy as np
import torch
import torch.nn as nn

# ---------------------------
# 1. 费根鲍姆混沌生成
# ---------------------------
delta = 4.6692016091
r_inf = 3.5699456718

def logistic(x, r):
    return r * x * (1 - x)

def generate_feigenbaum_chaos(length=20000, x0=0.5):
    seq = np.zeros(length)
    seq[0] = x0
    r = r_inf - 0.001 * delta  # 接近倍周期分岔极限，产生混沌
    for i in range(1, length):
        seq[i] = logistic(seq[i-1], r)
    return seq

# 生成混沌数据
data = generate_feigenbaum_chaos()

# ---------------------------
# 2. 构造时序样本
# ---------------------------
seq_len = 20
X, y = [], []
for i in range(len(data) - seq_len):
    X.append(data[i:i+seq_len])
    y.append(data[i+seq_len])

X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (样本数, 序列长度, 特征数)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

# ---------------------------
# 3. LSTM 混沌预测模型
# ---------------------------
class ChaosLSTM(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # 取最后一个时间步输出

model = ChaosLSTM()

# ---------------------------
# 4. 训练
# ---------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    pred = model(X)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:2d} | Loss: {loss.item():.6f}")

# ---------------------------
# 5. 混沌生成（自回归预测）
# ---------------------------
def generate_chaos(model, seed, steps=500):
    model.eval()
    history = seed.tolist()  # 初始序列
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(history[-seq_len:], dtype=torch.float32).reshape(1, seq_len, 1)
            next_p = model(x).item()
            history.append(next_p)
    return np.array(history)

# 演示生成
seed = X[0]  # 取第一个样本作为初始种子
pred_chaos = generate_chaos(model, seed, steps=200)
print("\n生成混沌序列前10项：")
print(np.round(pred_chaos[:10], 6))
# Origin: Zero → Chaos → Rebirth. The cycle is self-driven.
