import pandas as pd
from sklearn.ensemble import IsolationForest
import os

# Get absolute path of project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build correct paths
DATA_PATH = os.path.join(BASE_DIR, "data", "user_logs.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "output", "anomaly_report.csv")

# Load dataset
data = pd.read_csv(DATA_PATH)

# Select features
features = data[['login_hour', 'login_count', 'session_duration']]

# Initialize model
model = IsolationForest(
    n_estimators=100,
    contamination=0.25,
    random_state=42
)

# Train and predict
data['anomaly'] = model.fit_predict(features)

# Extract anomalies
anomalies = data[data['anomaly'] == -1]

# Save report
anomalies.to_csv(OUTPUT_PATH, index=False)

print("✅ Anomaly detection completed successfully")
print(f"⚠️ Report saved at: {OUTPUT_PATH}")