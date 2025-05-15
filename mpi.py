from mpi4py import MPI
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time
import pandas as pd
from imblearn.over_sampling import SMOTE


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

df = pd.read_csv('pdc_dataset_with_target.csv')
# Removed Nan value rows because they were giving better results then filling them with mean
nan_rows = df.isnull().any(axis=1).sum()
total_rows = len(df)
print(f"Rows with NaN: {nan_rows} out of {total_rows} ({nan_rows/total_rows:.2%})")
df = df.dropna()
df = pd.get_dummies(df, drop_first=True)
features = df.drop('target', axis=1)
scaler = StandardScaler()
features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
df_scaled = features_scaled.copy()
df_scaled['target'] = df['target'].values
# balancing the dataset because it was a bit imbalanced
smote = SMOTE(random_state=42)
X = df_scaled.drop('target', axis=1)
y = df_scaled['target']
X_resampled, y_resampled = smote.fit_resample(X, y)
X_resampled.shape, y_resampled.shape
y_resampled.value_counts(normalize=True)
df_scaled = pd.concat([X_resampled, y_resampled], axis=1)
df_scaled['target'].value_counts(normalize=True)
start = time.time()
from sklearn.model_selection import train_test_split

# Separate features and target
X = df_scaled.drop('target', axis=1)
y = df_scaled['target']

# Step 1: Take 15% of the data as initial set
X_remaining, x_init, y_remaining, y_init = train_test_split(
    X, y, test_size=0.15, random_state=42)

# Step 2: Split the remaining 85% into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_remaining, y_remaining, test_size=0.2, random_state=42)

# Each rank trains a model (can use rank as seed)
model = RandomForestClassifier(n_estimators=100, random_state=rank, n_jobs=1)
model.fit(x_init, y_init)
acc = accuracy_score(y_init, model.predict(x_init))

print(f"[Process {rank}] Accuracy: {acc:.4f}")

end = time.time()
if rank == 0:
    print(f"Total Time taken: {end - start:.2f} seconds")
