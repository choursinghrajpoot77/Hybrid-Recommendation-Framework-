
from transformers import RobertaTokenizer
import os
import sys
import time
import math
import warnings
warnings.simplefilter("ignore")
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, r2_score
from sklearn.metrics import precision_score, recall_score
from sklearn.neural_network import BernoulliRBM
import pandas as pd
import torch
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

def _Pre_():
    # 1. Load Data
    df = pd.read_csv('Dataset//reviews_dataset.csv')
    print('Loaded Data')
    print(df.head())
    print('Preprocessing..........')
    # 2. Data Cleaning
    # Drop rows with null ratings or user/item IDs
    df = df.dropna(subset=['reviewerID', 'asin', 'overall'])
    df = df.drop_duplicates(subset=['reviewerID', 'asin', 'reviewTime'])
    print('Data Cleaning..........')
    print('Cleaned Data')
    print(df.head())
    # 3. Text Normalization
    # Initialize RoBERTa tokenizer
    print('Text Normalization..........')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # text preprocessing function
    def preprocess_text(text):
        text = str(text).lower()  # lowercase
        text = text.replace('\n', ' ')  # remove line breaks
        return text

    # Combine title (summary) + reviewText for tokenization
    df['combined_text'] = df['summary'].fillna('') + " " + df['reviewText'].fillna('')
    df['combined_text'] = df['combined_text'].apply(preprocess_text)

    # Tokenize
    df['tokenized'] = df['combined_text'].apply(lambda x: tokenizer.encode(x, truncation=True, max_length=128))

    # 4. Interaction Matrix
    # Create a user-item interaction matrix
    users = df['reviewerID'].unique()
    items = df['asin'].unique()
    user_encoder = LabelEncoder().fit(users)
    item_encoder = LabelEncoder().fit(items)

    df['user_idx'] = user_encoder.transform(df['reviewerID'])
    df['item_idx'] = item_encoder.transform(df['asin'])

    # Ratings as interaction matrix
    interaction_matrix = np.zeros((len(users), len(items)))
    for _, row in df.iterrows():
        interaction_matrix[row['user_idx'], row['item_idx']] = row['overall']

    # 5. Negative Sampling
    def negative_sampling(inter_matrix, k=3):
        neg_samples = []
        n_users, n_items = inter_matrix.shape
        for u in range(n_users):
            pos_items = np.where(inter_matrix[u] > 0)[0]
            all_items = set(range(n_items))
            neg_items = list(all_items - set(pos_items))
            if len(neg_items) > 0:
                sampled = np.random.choice(neg_items, size=min(k, len(neg_items)), replace=False)
                for i in sampled:
                    neg_samples.append((u, i, 0))  # 0 for negative
        return neg_samples

    neg_samples = negative_sampling(interaction_matrix, k=3)

    # 6. Feature Scaling
    cont_features = ['exclamationcount', 'questioncount', 'charcount', 'wordcount', 'capcount', 'avgrating', 'diffrating']
    scaler = StandardScaler()
    df[cont_features] = scaler.fit_transform(df[cont_features])

    # Categorical encoding
    df['user_id_enc'] = df['user_idx']
    df['item_id_enc'] = df['item_idx']

    print("Preprocessing complete.")
    print(df.head())
    print("Sample negative samples:", neg_samples[:5])
    final_csv = "Pre_evaluated//_preprocessed.csv"
    df .to_csv(final_csv, index=False, encoding='utf-8')
    print(f"Preprocessed data saved to {final_csv}")

# Assume 'user_stats' has your statistical features
features = ['mean_rating', 'rating_var', 'interaction_count', 'rating_skew']
df = pd.read_csv('Pre_evaluated//_preprocessed.csv')
print(len(df))
print(df.columns)
# Group by user
user_group = df.groupby('reviewerID')['overall']

# Mean rating
mean_rating = user_group.mean()

# Rating variance
rating_var = user_group.var()

# Interaction frequency
interaction_count = user_group.count()

# Skewness
rating_skew = user_group.apply(lambda x: skew(x))

# Combine into a DataFrame
user_stats = pd.DataFrame({
    'mean_rating': mean_rating.fillna(0),   # var can be NaN if only 1 rating,
    'rating_var': rating_var.fillna(0),
    'interaction_count': interaction_count.fillna(0),
    'rating_skew': rating_skew.fillna(0)
}).reset_index()

print(user_stats.head())
# Standardize features (optional, BatchNorm also handles it)
scaler = StandardScaler()
X = scaler.fit_transform(user_stats[features])

# Convert to PyTorch tensor
X_tensor = torch.tensor(X, dtype=torch.float32)

# DNN layer

class StatDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)  # BatchNorm applied after linear
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.out = nn.Linear(32, 1)    # Output: could be rating prediction

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.out(x)
        return x

# Instantiate model
dnn_ = StatDNN(input_dim=X_tensor.shape[1])
output_features = dnn_(X_tensor)
print(output_features.detach().numpy())
print(len(output_features))
# train_eval_noMF.py
"""
 

Models:
 - Stacked LSTM
 - RBM + kNN hybrid
 - GNN (PyTorch Geometric required)
 - GAT (PyTorch Geometric required)
 - Proposed Hybrid (stat features + embeddings)

Outputs:
 - evaluation_results .csv
"""

# --- PyG: required for GNN/GAT ---
USE_PYG = True
try:
    from torch_geometric.data import Data as GeomData
    from torch_geometric.nn import GCNConv, GATConv
except Exception as e:
    USE_PYG = False

# -------------------------
# Helpers: metrics & utils
# -------------------------
def rmse(y_true, y_pred): return math.sqrt(mean_squared_error(y_true, y_pred))
def mae(y_true, y_pred): return mean_absolute_error(y_true, y_pred)

def precision_at_k(ranked_items, relevant_set, k):
    topk = ranked_items[:k]
    return sum(1 for i in topk if i in relevant_set) / k

def recall_at_k(ranked_items, relevant_set, k):
    topk = ranked_items[:k]
    if len(relevant_set) == 0:
        return 0.0
    return sum(1 for i in topk if i in relevant_set) / len(relevant_set)

def dcg_at_k(scores, k):
    scores = np.asfarray(scores)[:k]
    if scores.size:
        return np.sum((2**scores - 1) / np.log2(np.arange(2, scores.size + 2)))
    return 0.

def ndcg_binary_at_k(ranked_items, relevant_set, k):
    # build binary relevances of ranked_items
    relevances = [1 if it in relevant_set else 0 for it in ranked_items[:k]]
    dcg = 0.0
    for i, rel in enumerate(relevances):
        dcg += (2**rel - 1) / math.log2(i + 2)
    # ideal DCG
    ideal = sorted(relevances, reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal):
        idcg += (2**rel - 1) / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0

# ranking evaluation for test set
def evaluate_ranking_on_test(test_df, score_col='pred_score', k=10, pos_threshold=4.0):
    users = test_df['user_id'].unique()
    p_list, r_list, ndcg_list, hit_list = [], [], [], []
    for u in users:
        user_rows = test_df[test_df['user_id'] == u]
        # rank items by predicted score descending
        ranked = user_rows.sort_values(score_col, ascending=False)
        ranked_items = ranked['item_id'].tolist()
        relevant = set(ranked[ranked['true_rating'] >= pos_threshold]['item_id'].tolist())
        if len(ranked_items) == 0:
            continue
        p_list.append(precision_at_k(ranked_items, relevant, k))
        r_list.append(recall_at_k(ranked_items, relevant, k))
        ndcg_list.append(ndcg_binary_at_k(ranked_items, relevant, k))
        hit_list.append(1.0 if len(set(ranked_items[:k]).intersection(relevant)) > 0 else 0.0)
    return {
        f'Precision': float(np.mean(p_list)) if p_list else 0.0,
        f'Recall': float(np.mean(r_list)) if r_list else 0.0,
        f'NDCG': float(np.mean(ndcg_list)) if ndcg_list else 0.0,
        f'HitRate': float(np.mean(hit_list)) if hit_list else 0.0
    }


DATA_PATH = "Pre_evaluated/_preprocessed.csv"
if not os.path.exists(DATA_PATH):
    print("Dataset not found at", DATA_PATH)
    sys.exit(1)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)[:1000]
print("Rows:", len(df))
df.columns = df.columns.str.strip()

required_cols = ['reviewerID', 'asin', 'overall']
for c in required_cols:
    if c not in df.columns:
        raise RuntimeError(f"Required column '{c}' missing in CSV")

# compute helpful ratio from helpful_num/helpful_den if present, else try 'helpful' text
if 'helpful_num' in df.columns and 'helpful_den' in df.columns:
    df['helpful_ratio'] = df.apply(lambda r: r['helpful_num']/r['helpful_den'] if r['helpful_den']>0 else 0.0, axis=1)
elif 'helpful' in df.columns:
    def _safe_helpful(x):
        try:
            t = eval(x)
            return t[0]/t[1] if t[1]>0 else 0.0
        except Exception:
            return 0.0
    df['helpful_ratio'] = df['helpful'].apply(_safe_helpful)
else:
    df['helpful_ratio'] = 0.0

# enforce numeric rating column
df['rating'] = df['overall'].astype(float)

# encode user/item ids
user_enc = LabelEncoder()
item_enc = LabelEncoder()
df['user_id'] = user_enc.fit_transform(df['reviewerID'])
df['item_id'] = item_enc.fit_transform(df['asin'])
n_users = df['user_id'].nunique()
n_items = df['item_id'].nunique()
print("n_users:", n_users, "n_items:", n_items)

# split per-user: leave-out method (keeps at least one test per user when possible)
def train_test_by_user(df, test_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for uid, g in df.groupby('user_id'):
        idxs = g.index.tolist()
        rng.shuffle(idxs)
        ntest = max(1, int(math.ceil(len(idxs) * test_frac))) if len(idxs) > 1 else 0
        if ntest == 0:
            train_idx.extend(idxs)
        else:
            test_idx.extend(idxs[:ntest])
            train_idx.extend(idxs[ntest:])
    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)
    return train_df, test_df

train_df, test_df = train_test_by_user(df, test_frac=0.2)
print("Train interactions:", len(train_df), "Test interactions:", len(test_df))

# -------------------------
# Model 1: Stacked LSTM
# -------------------------
class LSTMRec(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64, hidden_dim=64):
        super().__init__()
        self.u_emb = nn.Embedding(n_users, emb_dim)
        self.i_emb = nn.Embedding(n_items, emb_dim)
        self.lstm = nn.LSTM(emb_dim * 2, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.out = nn.Linear(hidden_dim, 1)
    def forward(self, u_idx, i_idx):
        u = self.u_emb(u_idx)
        i = self.i_emb(i_idx)
        x = torch.cat([u, i], dim=1).unsqueeze(1)  # batch x seq_len(1) x feat
        out, _ = self.lstm(x)
        return self.out(out[:, -1, :]).squeeze(1)

def train_lstm(train_df, test_df, epochs=6, batch_size=2048, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRec(n_users, n_items, emb_dim=64, hidden_dim=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_u = torch.tensor(train_df['user_id'].values, dtype=torch.long)
    train_i = torch.tensor(train_df['item_id'].values, dtype=torch.long)
    train_y = torch.tensor(train_df['rating'].values, dtype=torch.float32)
    ds = TensorDataset(train_u, train_i, train_y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    for ep in range(epochs):
        model.train()
        total = 0.0
        for bu, bi, by in loader:
            bu, bi, by = bu.to(device), bi.to(device), by.to(device)
            opt.zero_grad()
            preds = model(bu, bi)
            loss = loss_fn(preds, by)
            loss.backward()
            opt.step()
            total += loss.item() * len(by)
        print(f"[LSTM] Epoch {ep+1}/{epochs} train_loss={total/len(train_df):.4f}")

    # predict on test
    model.eval()
    with torch.no_grad():
        tu = torch.tensor(test_df['user_id'].values, dtype=torch.long).to(device)
        ti = torch.tensor(test_df['item_id'].values, dtype=torch.long).to(device)
        preds = model(tu, ti).cpu().numpy()
    return preds

# -------------------------
# Model 2: RBM + kNN hybrid
# -------------------------
def train_rbm_knn(train_df, test_df, latent_dim=50, k=5):
    # Build user-item rating matrix from train
    mat = np.zeros((n_users, n_items), dtype=float)
    for _, r in train_df.iterrows():
        mat[int(r['user_id']), int(r['item_id'])] = r['rating']
    # create binarized presence matrix for RBM
    mat_bin = (mat > 0).astype(float)
    try:
        rbm = BernoulliRBM(n_components=latent_dim, learning_rate=0.01, batch_size=64, n_iter=10, random_state=42)
        rbm.fit(mat_bin)
        user_latent = rbm.transform(mat_bin)  # shape n_users x latent_dim
    except Exception:
        pca = PCA(n_components=latent_dim, random_state=42)
        user_latent = pca.fit_transform(mat)

    # per-item kNN regressors using user_latent -> rating
    preds = []
    # pre-build mapping for each item to train user latent & ratings
    item_user_map = {}
    for i in np.unique(train_df['item_id'].values):
        sub = train_df[train_df['item_id'] == i]
        us = sub['user_id'].values.astype(int)
        ys = sub['rating'].values.astype(float)
        if len(us) > 0:
            item_user_map[int(i)] = (user_latent[us], ys)
    for idx, r in test_df.iterrows():
        u = int(r['user_id']); i = int(r['item_id'])
        if i not in item_user_map:
            # cold item -> global item mean fallback using train mean
            # compute train item mean
            mean_val = train_df[train_df['item_id'] == i]['rating'].mean()
            if np.isnan(mean_val):
                mean_val = train_df['rating'].mean()
            preds.append(mean_val)
            continue
        X_item, y_item = item_user_map[i]
        neigh = KNeighborsRegressor(n_neighbors=min(k, max(1, len(y_item))))
        neigh.fit(X_item, y_item)
        pred = neigh.predict(user_latent[u].reshape(1, -1))[0]
        preds.append(pred)
    return np.array(preds)

# -------------------------
# Model 3: GNN using PyG
# -------------------------
def train_gnn(train_df, test_df, epochs=20):
    if not USE_PYG:
        print("PyTorch Geometric (PyG) not installed. Install PyG to run GNN/GAT.")
        print("See https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html")
        sys.exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # build bipartite graph edges (user -> item and item -> user)
    edge_u = []
    edge_v = []
    for _, r in train_df.iterrows():
        u = int(r['user_id']); v = int(r['item_id']) + n_users
        edge_u.extend([u, v]); edge_v.extend([v, u])
    edge_index = torch.tensor([edge_u, edge_v], dtype=torch.long)
    num_nodes = n_users + n_items
    # node features: small random features (or identity if small)
    if num_nodes <= 2000:
        x = torch.eye(num_nodes, dtype=torch.float)
    else:
        x = torch.randn(num_nodes, 64, dtype=torch.float)
    data = GeomData(x=x, edge_index=edge_index).to(device)

    class GCNRec(nn.Module):
        def __init__(self, in_dim, hid=64):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hid)
            self.conv2 = GCNConv(hid, hid)
        def forward(self, data):
            x = self.conv1(data.x, data.edge_index).relu()
            x = self.conv2(x, data.edge_index).relu()
            return x

    model = GCNRec(data.num_node_features, hid=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        node_emb = model(data)
        # sample a mini-batch of interactions to compute MSE loss
        sample_idx = np.random.choice(len(train_df), size=min(4096, len(train_df)), replace=False)
        us = torch.tensor(train_df.iloc[sample_idx]['user_id'].values, dtype=torch.long).to(device)
        is_ = torch.tensor(train_df.iloc[sample_idx]['item_id'].values + n_users, dtype=torch.long).to(device)
        true_r = torch.tensor(train_df.iloc[sample_idx]['rating'].values, dtype=torch.float32).to(device)
        preds = (node_emb[us] * node_emb[is_]).sum(dim=1)
        loss = nn.MSELoss()(preds, true_r)
        loss.backward(); opt.step()
        if (ep+1) % 5 == 0:
            print(f"[GNN] Epoch {ep+1}/{epochs} loss={loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        node_emb = model(data)
        us = torch.tensor(test_df['user_id'].values, dtype=torch.long).to(device)
        is_ = torch.tensor(test_df['item_id'].values + n_users, dtype=torch.long).to(device)
        preds = (node_emb[us] * node_emb[is_]).sum(dim=1).cpu().numpy()
    return preds

# -------------------------
# Model 4: GAT (PyG)
# -------------------------
def train_gat(train_df, test_df, epochs=20):
    if not USE_PYG:
        print("PyTorch Geometric (PyG) not installed. Install PyG to run GNN/GAT.")
        sys.exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_u = []; edge_v = []
    for _, r in train_df.iterrows():
        u = int(r['user_id']); v = int(r['item_id']) + n_users
        edge_u.extend([u, v]); edge_v.extend([v, u])
    edge_index = torch.tensor([edge_u, edge_v], dtype=torch.long)
    num_nodes = n_users + n_items
    if num_nodes <= 2000:
        x = torch.eye(num_nodes, dtype=torch.float)
    else:
        x = torch.randn(num_nodes, 64, dtype=torch.float)
    data = GeomData(x=x, edge_index=edge_index).to(device)

    class GATRec(nn.Module):
        def __init__(self, num_features, hidden_dim=64, out_dim=32):
            super(GATRec, self).__init__()
            self.gat1 = GATConv(num_features, hidden_dim, heads=4)
            self.gat2 = GATConv(hidden_dim * 4, out_dim, heads=1)

        def forward(self, data):
            x = F.elu(self.gat1(data.x, data.edge_index))
            x = self.gat2(x, data.edge_index)
            return x

    model = GATRec(data.num_node_features, hidden_dim=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        node_emb = model(data)
        sample_idx = np.random.choice(len(train_df), size=min(4096, len(train_df)), replace=False)
        us = torch.tensor(train_df.iloc[sample_idx]['user_id'].values, dtype=torch.long).to(device)
        is_ = torch.tensor(train_df.iloc[sample_idx]['item_id'].values + n_users, dtype=torch.long).to(device)
        true_r = torch.tensor(train_df.iloc[sample_idx]['rating'].values, dtype=torch.float32).to(device)
        preds = (node_emb[us] * node_emb[is_]).sum(dim=1)
        loss = nn.MSELoss()(preds, true_r)
        loss.backward(); opt.step()
        if (ep+1) % 5 == 0:
            print(f"[GAT] Epoch {ep+1}/{epochs} loss={loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        node_emb = model(data)
        us = torch.tensor(test_df['user_id'].values, dtype=torch.long).to(device)
        is_ = torch.tensor(test_df['item_id'].values + n_users, dtype=torch.long).to(device)
        preds = (node_emb[us] * node_emb[is_]).sum(dim=1).cpu().numpy()
    return preds

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Bahdanau Attention (Additive Attention) ---
class BahdanauAttention(nn.Module):
    def __init__(self, query_dim, key_dim, attn_dim):
        super().__init__()
        self.Wq = nn.Linear(query_dim, attn_dim)
        self.Wk = nn.Linear(key_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, query, keys, values):
        """
        query: [B, query_dim] — context (e.g., user embedding)
        keys:  [B, seq_len, key_dim] — item + stat embeddings
        values:[B, seq_len, value_dim]
        """
        # Expand query for all keys
        q = self.Wq(query).unsqueeze(1)               # [B,1,attn_dim]
        k = self.Wk(keys)                             # [B,seq_len,attn_dim]

        # Compute attention scores
        energy = torch.tanh(q + k)                    # [B,seq_len,attn_dim]
        scores = self.v(energy).squeeze(-1)           # [B,seq_len]

        # Attention weights
        attn_weights = F.softmax(scores, dim=1)       # [B,seq_len]

        # Weighted sum of values
        context = torch.bmm(attn_weights.unsqueeze(1), values).squeeze(1)  # [B,value_dim]
        return context, attn_weights


# --- Hybrid Recommendation Model with Bahdanau Attention ---
class HybridNet(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64, stat_dim=4, attn_dim=32):
        super().__init__()
        # User and Item embeddings
        self.u_emb = nn.Embedding(n_users, emb_dim)
        self.i_emb = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.u_emb.weight)
        nn.init.xavier_uniform_(self.i_emb.weight)

        # Statistical feature branch
        self.stat_fc = nn.Sequential(
            nn.Linear(stat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, emb_dim),
            nn.ReLU()
        )

        # Bahdanau attention
        self.attn = BahdanauAttention(query_dim=emb_dim, key_dim=emb_dim, attn_dim=attn_dim)

        # Final prediction network
        self.final = nn.Sequential(
            nn.Linear(emb_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, u, i, s):
        # Embeddings
        ue = self.u_emb(u)
        ie = self.i_emb(i)
        se = self.stat_fc(s)

        # Normalize embeddings
        ue = F.normalize(ue, dim=-1)
        ie = F.normalize(ie, dim=-1)
        se = F.normalize(se, dim=-1)

        # Stack item + stat features for attention
        keys = torch.stack([ie, se], dim=1)  # [B,2,emb_dim]
        values = keys.clone()

        # Apply Bahdanau attention with user embedding as query
        context, attn_w = self.attn(ue, keys, values)

        # Concatenate user embedding and attended context
        x = torch.cat([ue, context], dim=1)

        # Predict rating
        out = self.final(x)

        # Clamp for stability (typical rating 1–5)
        out = torch.clamp(out, 1.0, 5.0)
        out[out != out] = 0.0
        return out.squeeze(1), attn_w


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

import matplotlib.pyplot as plt

def train_proposed(train_df, test_df, epochs=8, batch_size=2048, lr=5e-5):
    user_stats = build_user_stats(train_df)

    def stat_array(df_):
        arr = []
        for _, r in df_.iterrows():
            uid = int(r['user_id'])
            arr.append(user_stats.loc[uid].values)
        return np.vstack(arr)

    tr_stat = stat_array(train_df)
    te_stat = stat_array(test_df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridNet(n_users, n_items, emb_dim=64, stat_dim=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    tr_u = torch.tensor(train_df['user_id'].values, dtype=torch.long)
    tr_i = torch.tensor(train_df['item_id'].values, dtype=torch.long)
    tr_s = torch.tensor(tr_stat, dtype=torch.float32)
    tr_y = torch.tensor(train_df['rating'].values, dtype=torch.float32)
    ds = TensorDataset(tr_u, tr_i, tr_s, tr_y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    epoch_losses = []  # <-- store loss per epoch

    for ep in range(epochs):
        model.train()
        total = 0.0
        for bu, bi, bs, by in loader:
            bu, bi, bs, by = bu.to(device), bi.to(device), bs.to(device), by.to(device)
            opt.zero_grad()
            preds, _ = model(bu, bi, bs)  # <-- UNPACK here
            loss = loss_fn(preds, by)
            loss.backward()
            opt.step()
            total += loss.item() * len(by)

        avg_loss = total / len(train_df)
        epoch_losses.append(avg_loss)
        print(f"[Hybrid] Epoch {ep + 1}/{epochs} loss={avg_loss:.4f}")

    # --- Plot Epoch vs Loss ---
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o', color='blue')
    plt.title('Epoch vs Training Loss (Proposed Hybrid)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig('loss.png', dpi = 300)
    # plt.show()
    plt.close()

    # predict
    model.eval()
    with torch.no_grad():
        tu = torch.tensor(test_df['user_id'].values, dtype=torch.long).to(device)
        ti = torch.tensor(test_df['item_id'].values, dtype=torch.long).to(device)
        ts = torch.tensor(te_stat, dtype=torch.float32).to(device)
        preds, attn_w = model(tu, ti, ts)  # <-- UNPACK here
        preds = preds.cpu().numpy()
    return preds


# # -------------------------
# # Run all model trainings & evaluations
# # -------------------------
# runners = [
#     ("Stacked LSTM", train_lstm),
#     ("RBM + kNN", train_rbm_knn),
#     ("GNN ", train_gnn),
#     ("GAT", train_gat),
#     ("Proposed Hybrid", train_proposed)
# ]
#
# results = []
# start_all = time.time()
# for name, fn in runners:
#     print("\n" + "="*40)
#     print("Model:", name)
#     t0 = time.time()
#     preds = fn(train_df, test_df)
#     t1 = time.time()
#     y_true = test_df['rating'].values
#     y_pred = np.array(preds)
#     print("NaNs in y_true:", np.isnan(y_true).sum())
#     print("NaNs in y_pred:", np.isnan(y_pred).sum())
#     print("Inf in y_pred:", np.isinf(y_pred).sum())
#     mse = mean_squared_error(y_true, y_pred)
#     rmse_ = math.sqrt(mse)
#     mae_ = mean_absolute_error(y_true, y_pred)
#     r2_ = r2_score(y_true, y_pred)
#     basic = {'MSE': mse, 'RMSE': rmse(y_true, y_pred), 'MAE': mae(y_true, y_pred),'r2':r2_}
#     # prepare pred_df for ranking eval
#     pred_df = pd.DataFrame({
#         'user_id': test_df['user_id'].values,
#         'item_id': test_df['item_id'].values,
#         'pred_score': y_pred,
#         'true_rating': y_true
#     })
#     rank5 = evaluate_ranking_on_test(pred_df, score_col='pred_score', k=5)
#     rank10 = evaluate_ranking_on_test(pred_df, score_col='pred_score', k=10)
#     # optional AUC if ishelpful exists
#     auc_val = None
#     if 'ishelpful' in test_df.columns:
#         try:
#             # clamp preds to [0,1] by min-max across train predicted or sigmoid scaling; we simply min-max normalize for AUC
#             pmin, pmax = y_pred.min(), y_pred.max()
#             pnorm = (y_pred - pmin) / (pmax - pmin + 1e-9)
#             auc_val = float(roc_auc_score(test_df['ishelpful'].astype(int).values, pnorm))
#         except Exception:
#             auc_val = None
#     res = {
#         'Model': name,
#         'TrainTime_s': round(t1 - t0, 2),
#         'RMSE': basic['RMSE'],
#         'MAE': basic['MAE'],
#         'MSE': basic['MSE'],
#         'R2': basic['r2'],
#         'AUC': auc_val
#     }
#     # res.update(rank5); res.update(rank10)
#     results.append(res)
#     print(f"{name} done: RMSE={basic['RMSE']:.4f}, MAE={basic['MAE']:.4f}, time={t1-t0:.1f}s")
#
# # Save aggregated results
# results_df = pd.DataFrame(results)
# out_csv = "evaluation_results.csv"
# results_df.to_csv(out_csv, index=False)
# print("\nAll models finished. Results saved to:", out_csv)
# print(results_df)
# print("Total run time (s):", time.time() - start_all)


def _ablation_():
    # --- Prepare training tensors for ablation ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build user stats
    user_stats = build_user_stats(train_df)
    tr_stat = np.vstack([user_stats.loc[int(r['user_id'])].values for _, r in train_df.iterrows()])
    te_stat = np.vstack([user_stats.loc[int(r['user_id'])].values for _, r in test_df.iterrows()])

    tr_u = torch.tensor(train_df['user_id'].values, dtype=torch.long)
    tr_i = torch.tensor(train_df['item_id'].values, dtype=torch.long)
    tr_s = torch.tensor(tr_stat, dtype=torch.float32)
    tr_y = torch.tensor(train_df['rating'].values, dtype=torch.float32)
    te_u = torch.tensor(test_df['user_id'].values, dtype=torch.long)
    te_i = torch.tensor(test_df['item_id'].values, dtype=torch.long)
    te_s = torch.tensor(te_stat, dtype=torch.float32)

    # ---------------------------
    # Ablation variants (modified)
    # ---------------------------
    def ablation_variants_fixed(train_tensors, test_tensors):
        tr_u, tr_i, tr_s, tr_y = train_tensors
        te_u, te_i, te_s = test_tensors
        variants = {}

        # Full Proposed model
        preds_full = train_proposed(train_df, test_df, epochs=8)
        variants['Proposed (Full)'] = preds_full

        # No Statistical features
        class HybridNoStat(HybridNet):
            def forward(self, u, i, s):
                ue = self.u_emb(u)
                ie = self.i_emb(i)
                ue, ie = F.normalize(ue, dim=-1), F.normalize(ie, dim=-1)
                x = torch.cat([ue, ie], dim=1)
                out = self.final(x)
                return torch.clamp(out, 1.0, 5.0).squeeze(1), None

        model_ns = HybridNoStat(n_users, n_items).to(device)
        model_ns.eval()
        with torch.no_grad():
            preds_ns, _ = model_ns(tr_u.to(device), tr_i.to(device), tr_s.to(device))
            variants['No Statistical features'] = preds_ns.cpu().numpy()

        # Attention replaced with concat
        class HybridConcat(HybridNet):
            def forward(self, u, i, s):
                ue = self.u_emb(u)
                ie = self.i_emb(i)
                se = self.stat_fc(s)
                ue, ie, se = F.normalize(ue, dim=-1), F.normalize(ie, dim=-1), F.normalize(se, dim=-1)
                context = torch.cat([ie, se], dim=1)
                x = torch.cat([ue, context], dim=1)
                out = self.final(x)
                return torch.clamp(out, 1.0, 5.0).squeeze(1), None

        model_concat = HybridConcat(n_users, n_items).to(device)
        model_concat.eval()
        with torch.no_grad():
            preds_concat, _ = model_concat(tr_u.to(device), tr_i.to(device), tr_s.to(device))
            variants['Attention replaced with concat'] = preds_concat.cpu().numpy()

        return variants

    # --- Run ablation ---
    train_tensors = (tr_u, tr_i, tr_s, tr_y)
    test_tensors = (te_u, te_i, te_s)
    variants = ablation_variants_fixed(train_tensors, test_tensors)
    def compute_ablation_metrics(variants, test_df, k=10):
        results = []
        for name, preds in variants.items():
            test_df_copy = test_df.copy()
            test_df_copy['pred_score'] = preds
            metrics = evaluate_ranking_on_test(test_df_copy, score_col='pred_score', k=k, pos_threshold=4.0)
            results.append({
                'Model Variant': name,
                'Recall': metrics['Recall'],
                'F1': metrics['F1']
            })
        return pd.DataFrame(results)

    df_ablation = compute_ablation_metrics(variants, test_df, k=10)
    df_ablation.to_csv("ablation_.csv", index=False)
    print("Saved ablation results to ablation_.csv")
    print(df_ablation)
