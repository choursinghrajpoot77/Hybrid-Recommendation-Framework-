import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load preprocessed data
df = pd.read_csv('Pre_evaluated//_preprocessed.csv')

# Folder to save plots
plot_folder = "EDA_Plots"
os.makedirs(plot_folder, exist_ok=True)

# 1. Basic Statistics
num_users = df['reviewerID'].nunique()
num_items = df['asin'].nunique()
num_reviews = df.shape[0]
interaction_density = num_reviews / (num_users * num_items)
print(f"Number of users: {num_users}")
print(f"Number of items: {num_items}")
print(f"Number of reviews: {num_reviews}")
print(f"Interaction density: {interaction_density:.4f}")

# 2. Rating Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='overall', data=df, color='green')
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, 'rating_distribution.png'))
plt.close()

# 3. Textual Features
df['wordcount'] = df['combined_text'].apply(lambda x: len(str(x).split()))
df['charcount'] = df['combined_text'].apply(len)
df['capcount'] = df['combined_text'].apply(lambda x: sum(1 for c in x if c.isupper()))
print(df[['wordcount','charcount','capcount']].describe())

# 4. Positive Interaction Ratio
positive_ratio = (df['overall'] >= 4).mean()
print(f"Proportion of positive interactions (rating >=4): {positive_ratio:.2f}")

# 5. Continuous Feature Distributions
cont_features = ['exclamationcount', 'questioncount', 'charcount', 'wordcount', 'capcount', 'avgrating', 'diffrating']
for col in cont_features:
    plt.figure(figsize=(5,3))
    sns.histplot(df[col], bins=30, kde=True, color='green')
    plt.title(f'{col} Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f'{col}_distribution.png'))
    plt.close()

# 6. Top Users and Items
top_users = df['reviewerID'].value_counts().head(10)
top_items = df['asin'].value_counts().head(10)
print("Top 10 users by number of reviews:\n", top_users)
print("Top 10 items by number of reviews:\n", top_items)
