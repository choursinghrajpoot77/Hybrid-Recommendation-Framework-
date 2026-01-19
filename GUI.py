
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from datetime import datetime
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import RobertaTokenizer

def _fsn():
    import pandas as pd
    import torch
    from transformers import RobertaTokenizer, RobertaModel
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import skew
    import torch.nn as nn
    import numpy as np

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
    cont_features = ['exclamationcount', 'questioncount', 'charcount', 'wordcount', 'capcount', 'avgrating',
                     'diffrating']
    scaler = StandardScaler()
    df[cont_features] = scaler.fit_transform(df[cont_features])

    # Categorical encoding
    df['user_id_enc'] = df['user_idx']
    df['item_id_enc'] = df['item_idx']

    print("Preprocessing complete.")
    print(df.head())
    print("Sample negative samples:", neg_samples[:5])
    final_csv = "Pre_evaluated//_preprocessed.csv"
    df.to_csv(final_csv, index=False, encoding='utf-8')
    print(f"Preprocessed data saved to {final_csv}")

    import pandas as pd
    import torch
    from scipy.stats import skew
    from sklearn.preprocessing import StandardScaler
    import torch.nn as nn

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
        'mean_rating': mean_rating.fillna(0),  # var can be NaN if only 1 rating,
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
            self.out = nn.Linear(32, 1)  # Output: could be rating prediction

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

    df = pd.read_csv('Pre_evaluated//_preprocessed.csv')
    print("Loaded data:", len(df))

    # -----------------------------
    # 2. Statistical features
    # -----------------------------
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

    # -----------------------------
    # 3. DNN for statistical features
    # -----------------------------
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

    # -----------------------------
    # 4. RoBERTa embeddings (batch-wise)
    # -----------------------------
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

    # -----------------------------
    # 5. Fuse embeddings
    # -----------------------------
    fused_emb = torch.cat([roberta_emb, dnn_features.repeat(int(np.ceil(len(df) / len(user_stats))), 1)[:len(df)]],
                          dim=1)

    # -----------------------------
    # 6. Save fused embeddings
    # -----------------------------
    fused_df = pd.DataFrame(fused_emb.detach().numpy(),
                            columns=[f'feat_{i}' for i in range(fused_emb.shape[1])])
    fused_df['reviewerID'] = df['reviewerID'].values
    fused_df['asin'] = df['asin'].values

    # fused_df.to_csv('Pre_evaluated//_fused_embeddings_full_batch.csv', index=False)
    print("Saved fused embeddings CSV with batching:", fused_df.shape)


# === Load dataset ===
try:
    df = pd.read_csv('Pre_evaluated/_preprocessed.csv')
    print("Loaded dataset:", len(df))
    print(df.columns)
except FileNotFoundError:
    messagebox.showerror("Error", "Dataset not found. Please check the file path.")
    exit()

# === Clean and Prepare ===
df['helpful_ratio'] = df['helpful'].apply(
    lambda x: eval(x)[0]/eval(x)[1] if isinstance(x, str) and eval(x)[1] > 0 else 0
)

# Convert review time if exists
if 'reviewTime' in df.columns:
    df['reviewTime'] = pd.to_datetime(df['reviewTime'], errors='coerce')
    df['recency_weight'] = (df['reviewTime'] - df['reviewTime'].min()).dt.days
    df['recency_weight'] /= df['recency_weight'].max()
else:
    df['recency_weight'] = 0.5  # neutral default

df['ishelpful'] = df['ishelpful'].astype(str)

# === Encode users and items ===
users = df['reviewerID'].unique()
items = df['asin'].unique()
user_encoder = {u: i for i, u in enumerate(users)}
item_encoder = {i: j for j, i in enumerate(items)}

# === Build rating matrix ===
interaction_matrix = np.zeros((len(users), len(items)))
for _, row in df.iterrows():
    u_idx = user_encoder[row['reviewerID']]
    i_idx = item_encoder[row['asin']]
    interaction_matrix[u_idx, i_idx] = row['overall']

# === Compute item meta info ===
item_meta = df.groupby('asin').agg({
    'overall': 'mean',
    'helpful_ratio': 'mean',
    'recency_weight': 'mean',
    'summary': lambda x: x.mode()[0] if not x.mode().empty else "",
    'reviewText': lambda x: x.iloc[0] if isinstance(x.iloc[0], str) else "",
    'ishelpful': lambda x: "Yes" if (x == "True").sum() > 0 else "No"
}).reset_index()

item_meta['score'] = (
    0.5 * item_meta['overall'] +
    0.3 * item_meta['helpful_ratio'] +
    0.2 * item_meta['recency_weight']
)
item_meta = item_meta.set_index('asin').to_dict('index')

# === Recommendation Function ===
def recommend_for_user(user_id, top_k=5):
    if user_id not in user_encoder:
        return []
    u_idx = user_encoder[user_id]
    user_ratings = interaction_matrix[u_idx]
    unrated_indices = np.where(user_ratings == 0)[0]
    if len(unrated_indices) == 0:
        return []
    ranked = sorted(unrated_indices, key=lambda x: item_meta.get(items[x], {}).get('score', 0), reverse=True)
    return [(items[i], item_meta.get(items[i], {})) for i in ranked[:top_k]]


# === Hover Animation ===
def on_enter(event):
    event.widget.configure(bg="#b3e5fc")

def on_leave(event):
    event.widget.configure(bg="#ffffff")


# === GUI ===
root = tk.Tk()
root.title("Product Recommender")
root.geometry("950x700")
root.configure(bg="#e6f0ff")

# Header Gradient Canvas
header_canvas = tk.Canvas(root, height=70, width=950, bg="#1a237e", highlightthickness=0)
header_canvas.pack(fill="x")
for i in range(950):
    color = f'#{int(26 + (i/950)*(100)):02x}{int(35 + (i/950)*(150)):02x}{int(126 + (i/950)*50):02x}'
    header_canvas.create_line(i, 0, i, 70, fill=color)
header_text = header_canvas.create_text(480, 35, text="Product Recommendation System",
                                        font=("Segoe UI", 20, "bold"), fill="white")

# Input Section
frame_input = tk.Frame(root, bg="#e6f0ff")
frame_input.pack(pady=15)
tk.Label(frame_input, text="Enter User ID:", font=("Segoe UI", 12, "bold"), bg="#e6f0ff", fg="#0d47a1").grid(row=0, column=0, padx=10)
entry_user = tk.Entry(frame_input, width=35, font=("Segoe UI", 12), relief="solid", bd=1)
entry_user.grid(row=0, column=1, padx=10)
btn = tk.Button(frame_input, text=" Get Recommendations", font=("Segoe UI", 12, "bold"),
                bg="#00bfa5", fg="white", activebackground="#00796b",
                command=lambda: show_recommendations())
btn.grid(row=0, column=2, padx=10)

# Scrollable Frame for Results
frame_canvas = tk.Frame(root, bg="#e6f0ff")
frame_canvas.pack(fill="both", expand=True, padx=10, pady=10)

canvas = tk.Canvas(frame_canvas, bg="#f0faff", highlightthickness=0)
scrollbar = ttk.Scrollbar(frame_canvas, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg="#f0faff")

scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

frame_results = scrollable_frame


# === Recommendation Display ===
def show_recommendations():
    user_id = entry_user.get().strip()
    if not user_id:
        messagebox.showwarning("⚠️ Input Error", "Please enter a valid User ID")
        return

    recs = recommend_for_user(user_id, top_k=10)
    for widget in frame_results.winfo_children():
        widget.destroy()

    if not recs:
        tk.Label(frame_results, text=f"No recommendations found for user '{user_id}'",
                 font=("Segoe UI", 13, "italic"), bg="#f0faff", fg="gray").pack(pady=20)
        return

    colors = ["#ffffff", "#e3f2fd", "#e8f5e9", "#fff3e0"]

    for rank, (item_id, meta) in enumerate(recs, 1):
        card = tk.Frame(frame_results, bg=random.choice(colors), relief="raised", bd=3, padx=10, pady=8)
        card.pack(padx=15, pady=10, fill="x")

        card.bind("<Enter>", on_enter)
        card.bind("<Leave>", on_leave)

        title = f"Rank {rank}  {meta.get('summary', 'N/A')}"
        tk.Label(card, text=title, font=("Segoe UI", 14, "bold"), bg=card["bg"], fg="#0d47a1", wraplength=850).pack(anchor="w")

        tk.Label(card, text=f"🆔 Item ID: {item_id}", font=("Segoe UI", 11), bg=card["bg"], fg="#004d40").pack(anchor="w", pady=2)
        tk.Label(card, text=f"⭐ Avg Rating: {meta.get('overall', 0):.2f}   👍 Helpful: {meta.get('helpful_ratio', 0):.2f}",
                 font=("Segoe UI", 11), bg=card["bg"], fg="#1b5e20").pack(anchor="w", pady=2)
        tk.Label(card, text=f"💬 Description: {meta.get('reviewText', 'N/A')[:250]}...",
                 font=("Segoe UI", 10), bg=card["bg"], wraplength=850, fg="#37474f").pack(anchor="w", pady=3)
        # tk.Label(card, text=f" Helpful Review: {meta.get('ishelpful', 'N/A')}",
        #          font=("Segoe UI", 10, "italic"), bg=card["bg"], fg="#6a1b9a").pack(anchor="w")


root.mainloop()
