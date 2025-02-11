import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

# Load dataset
file_path = "sample_test.csv"  # path

df = pd.read_csv(file_path)

# Convert timestamp to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Encode categorical labels
label_encoder = LabelEncoder()
df['activity_encoded'] = label_encoder.fit_transform(df['activity'])

# Select features and targets
features = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
X = df[features]
y_heart_rate = df['heart_rate']  # Target for Linear Regression
y_activity = df['activity_encoded']  # Target for Decision Tree

# Standardize feature values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split for supervised models
X_train, X_test, y_train_hr, y_test_hr = train_test_split(X_scaled, y_heart_rate, test_size=0.2, random_state=42)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_scaled, y_activity, test_size=0.2, random_state=42)

# Train Linear Regression (Predict Heart Rate)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train_hr)
joblib.dump(lr_model, "linear_regression_model.pkl")

# Train Decision Tree (Classify Activity)
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train_cls, y_train_cls)
joblib.dump(dt_model, "decision_tree_model.pkl")

# Train K-Means Clustering (Find Movement Patterns)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
joblib.dump(kmeans, "kmeans_model.pkl")

print("Models trained and saved successfully.")

# Function to load and use the models
def predict_new_data(new_data):
    new_data_scaled = scaler.transform([new_data])
    lr_model = joblib.load("linear_regression_model.pkl")
    dt_model = joblib.load("decision_tree_model.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    
    heart_rate_pred = lr_model.predict(new_data_scaled)
    activity_pred = dt_model.predict(new_data_scaled)
    cluster_pred = kmeans.predict(new_data_scaled)
    
    return {
        "Predicted Heart Rate": heart_rate_pred[0],
        "Predicted Activity": label_encoder.inverse_transform([activity_pred[0]])[0],
        "Predicted Cluster": cluster_pred[0]
    }

# Example usage with test data
example_input = X.iloc[0].tolist()
print(predict_new_data(example_input))
