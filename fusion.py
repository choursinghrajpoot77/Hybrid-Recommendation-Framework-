import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaModel
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew
import torch.nn as nn
import numpy as np


# 1. Load preprocessed data

df = pd.read_csv('Pre_evaluated//_preprocessed.csv')
print("Loaded data:", len(df))


# 2. Statistical features

user_group = df.groupby('reviewerID')['overall']

user_stats = pd.DataFrame({
    'reviewerID': user_group.groups.keys(),
    'mean_rating': user_group.mean().fillna(0),
    'rating_var': user_group.var().fillna(0),
    'interaction_count': user_group.count().fillna(0),
    'rating_skew': user_group.apply(lambda x: skew(x)).fillna(0)
})

scaler = StandardScaler()
X_stats = scaler.fit_transform(user_stats[['mean_rating', 'rating_var', 'interaction_count', 'rating_skew']])
X_stats_tensor = torch.tensor(X_stats, dtype=torch.float32)



# 3. DNN for statistical features

class StatDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.out = nn.Linear(32, 32)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.out(x)
        return x


dnn_ = StatDNN(input_dim=X_stats_tensor.shape[1])
dnn_features = dnn_(X_stats_tensor)


# 4. RoBERTa embeddings (batch-wise)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
model.eval()

batch_size = 128  # adjust according to your RAM/GPU
all_embeds = []


def tokenize_batch(texts):
    return tokenizer(texts, padding=True, truncation=True,
                     max_length=128, return_tensors='pt')


for start in range(0, len(df), batch_size):
    end = min(start + batch_size, len(df))
    batch_texts = df['combined_text'].iloc[start:end].fillna('').tolist()
    tokens = tokenize_batch(batch_texts)

    with torch.no_grad():
        outputs = model(**tokens)
        batch_embeds = outputs.pooler_output  # [batch_size, 768]
        all_embeds.append(batch_embeds)

# Concatenate all batches
roberta_emb = torch.cat(all_embeds, dim=0)
print("Fused RoBERTa embedding shape:", roberta_emb.shape)


# 5. Fuse embeddings

fused_emb = torch.cat([roberta_emb, dnn_features.repeat(int(np.ceil(len(df) / len(user_stats))), 1)[:len(df)]], dim=1)


# 6. Save fused embeddings

fused_df = pd.DataFrame(fused_emb.detach().numpy(),
                        columns=[f'feat_{i}' for i in range(fused_emb.shape[1])])
fused_df['reviewerID'] = df['reviewerID'].values
fused_df['asin'] = df['asin'].values

fused_df.to_csv('Pre_evaluated//_fused_embeddings_full_batch.csv', index=False)
print("Saved fused embeddings CSV with batching:", fused_df.shape)
