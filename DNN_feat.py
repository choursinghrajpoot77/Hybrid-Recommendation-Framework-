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