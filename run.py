import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler

from load import n_items, train_df, test_df, train_lstm, train_rbm_knn, train_gnn, train_gat, n_users


# User statistics

def build_user_stats(df):
    g = df.groupby('user_id')['rating']
    mean_r = g.mean()
    var_r = g.var().fillna(0.0)
    cnt = g.count()
    skew_r = g.apply(lambda x: skew(x) if len(x) >= 3 else 0.0)
    us = pd.DataFrame({
        'user_id': mean_r.index,
        'u_mean': mean_r.values,
        'u_var': var_r.values,
        'u_count': cnt.values,
        'u_skew': skew_r.values
    }).set_index('user_id')
    scaler = StandardScaler()
    us[['u_mean','u_var','u_count','u_skew']] = scaler.fit_transform(us[['u_mean','u_var','u_count','u_skew']])
    return us


# Bahdanau Attention

class BahdanauAttention(nn.Module):
    def __init__(self, emb_dim, stat_dim, att_dim=32):
        super().__init__()
        self.W1 = nn.Linear(emb_dim, att_dim)
        self.W2 = nn.Linear(stat_dim, att_dim)
        self.V  = nn.Linear(att_dim, 1)
        self.layernorm = nn.LayerNorm(att_dim)

    def forward(self, emb, stats):
        score = torch.tanh(torch.clamp(self.W1(emb) + self.W2(stats), min=-10, max=10))
        score = self.layernorm(score)
        att_w = torch.sigmoid(self.V(score))
        output = att_w * emb
        return output


# HybridNet

class HybridNet(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64, stat_dim=4):
        super().__init__()
        self.u_emb = nn.Embedding(n_users, emb_dim)
        self.i_emb = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.u_emb.weight)
        nn.init.xavier_uniform_(self.i_emb.weight)

        self.stat_fc = nn.Sequential(
            nn.Linear(stat_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.attention = BahdanauAttention(emb_dim*2, 16, att_dim=32)
        self.final = nn.Sequential(
            nn.Linear(emb_dim*2 + 16, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        self.emb_norm = nn.LayerNorm(emb_dim)

    def forward(self, u, i, s):
        ue = self.emb_norm(self.u_emb(u))
        ie = self.emb_norm(self.i_emb(i))
        se = self.stat_fc(s)
        se = torch.clamp(se, min=-10, max=10)
        emb_cat = torch.cat([ue, ie], dim=1)
        emb_att = self.attention(emb_cat, se)
        x = torch.cat([emb_cat, se], dim=1)
        out = self.final(x)
        out = torch.clamp(out, 1.0, 5.0)
        out[out != out] = 0.0  # replace NaNs
        return out.squeeze(1)


# Training Proposed

def train_proposed(train_df, test_df, n_users, n_items, epochs=100, batch_size=2048, lr=1e-4):
    user_stats = build_user_stats(train_df)

    def stat_array(df_):
        return np.vstack([user_stats.loc[int(r['user_id'])].values for _, r in df_.iterrows()])

    tr_stat = stat_array(train_df)
    te_stat = stat_array(test_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridNet(n_users, n_items).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    loss_fn = nn.MSELoss()

    tr_u = torch.tensor(train_df['user_id'].values, dtype=torch.long)
    tr_i = torch.tensor(train_df['item_id'].values, dtype=torch.long)
    tr_s = torch.tensor(tr_stat, dtype=torch.float32)
    tr_y = torch.tensor(train_df['rating'].values, dtype=torch.float32)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tr_u, tr_i, tr_s, tr_y),
        batch_size=batch_size, shuffle=True
    )

    epoch_losses = []
    for ep in range(epochs):
        model.train()
        total = 0.0
        for bu, bi, bs, by in loader:
            bu, bi, bs, by = bu.to(device), bi.to(device), bs.to(device), by.to(device)
            opt.zero_grad()
            preds = model(bu, bi, bs)
            loss = loss_fn(preds, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(by)
        avg_loss = total / len(train_df)
        epoch_losses.append(avg_loss)
        scheduler.step(avg_loss)
        print(f"[Proposed] Epoch {ep+1}/{epochs} | Loss={avg_loss:.4f}")

    # Loss plot
    plt.figure(figsize=(6,4))
    plt.plot(epoch_losses, marker='o', color='darkorange')
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title("Proposed Hybrid Model - Training Loss")
    plt.grid(True); #plt.show()
    plt.close()

    # Evaluation
    model.eval()
    with torch.no_grad():
        te_u = torch.tensor(test_df['user_id'].values, dtype=torch.long).to(device)
        te_i = torch.tensor(test_df['item_id'].values, dtype=torch.long).to(device)
        te_s = torch.tensor(te_stat, dtype=torch.float32).to(device)
        preds = model(te_u, te_i, te_s).cpu().numpy()

    return preds


# Metrics

def evaluate_all(y_true, y_pred):
    return {
        'RMSE': math.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'ExplainedVariance': explained_variance_score(y_true, y_pred),
        'MaxError': max_error(y_true, y_pred),
        'PearsonCorr': np.corrcoef(y_true, y_pred)[0, 1],
        'MAPE': np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-3, None))) * 100,
        'Recall': recall_score((y_true>=4).astype(int), (y_pred>=4).astype(int)),
        'F1': f1_score((y_true>=4).astype(int), (y_pred>=4).astype(int))
    }


# Train & compare all models

print("\n=== Training and Evaluating All Models ===")
pred_lstm = train_lstm(train_df, test_df)
pred_rbm  = train_rbm_knn(train_df, test_df)
pred_gnn  = train_gnn(train_df, test_df)
pred_gat  = train_gat(train_df, test_df)
pred_prop = train_proposed(train_df, test_df, n_users, n_items)

y_true = test_df['rating'].values
models = ['LSTM', 'RBM+kNN', 'GNN', 'GAT', 'Proposed']
preds  = [pred_lstm, pred_rbm, pred_gnn, pred_gat, pred_prop]

metrics_list = []
for name, pred in zip(models, preds):
    metrics = evaluate_all(y_true, pred)
    metrics['Model'] = name
    metrics_list.append(metrics)

results = pd.DataFrame(metrics_list)
results = results[['Model','RMSE','MAE','R2','MSE','ExplainedVariance','MaxError','PearsonCorr','MAPE','Recall','F1']]

print("\n=== Final Comparison Table ===")
print(results.round(4))
# results.to_csv("evaluation_results.csv", index=False)
print("Saved results to evaluation_results.csv")
