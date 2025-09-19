import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Step 1: Load the dataset
df = pd.read_csv("dataset_cloud.csv")

# Step 2: Drop non-useful columns
df = df.drop(columns=["Task_ID"])

# Step 3: Encode categorical features using one-hot encoding
categorical_cols = ["Task_Type", "User_Type", "Resource_Class"]
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Step 4: Encode the target column ("Priority")
label_encoder = LabelEncoder()
df_encoded["Priority"] = label_encoder.fit_transform(df_encoded["Priority"])

# Step 5: Save the label encoder
joblib.dump(label_encoder, "label_encoder.pkl")

# Step 6: Split into features and target
X = df_encoded.drop("Priority", axis=1)
y = df_encoded["Priority"]

# Step 7: Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# Step 8: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 9: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 10: Save the model and feature names
joblib.dump(model, "priority_model.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")  # Save feature names for testing alignment

print("✅ Model, scaler, label encoder, and feature names saved successfully.")
