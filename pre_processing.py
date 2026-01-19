import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import RobertaTokenizer


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