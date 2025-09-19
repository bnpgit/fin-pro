import pandas as pd
import joblib

# Load model and preprocessing tools
model = joblib.load("priority_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")

# Load dataset
df = pd.read_csv("dataset_cloud.csv")

# Drop Task_ID and Priority column
X = df.drop(columns=["Task_ID", "Priority"])

# One-hot encode
X_encoded = pd.get_dummies(X, columns=["Task_Type", "User_Type", "Resource_Class"])

# Align with training features
X_encoded = X_encoded.reindex(columns=feature_names, fill_value=0)

# Randomly pick 3 samples
X_sample_raw = X.sample(n=3, random_state=None)
X_sample_encoded = X_encoded.loc[X_sample_raw.index]

# Scale the encoded input
X_scaled = scaler.transform(X_sample_encoded)

# Predict class and probability
predictions = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)
predicted_labels = label_encoder.inverse_transform(predictions)

# Show input + predictions
print("\n🎯 Predicted Priorities with Inputs & Probabilities:\n")

task_list = []
for i in range(3):
    print(f"🔹 Task {i+1} Input Parameters:")
    print(X_sample_raw.iloc[i])
    print(f"→ Predicted Priority: {predicted_labels[i]}")
    
    # Show class probabilities
    class_prob = dict(zip(label_encoder.classes_, probabilities[i]))
    for cls in sorted(class_prob, key=lambda x: -class_prob[x]):
        print(f"   - {cls}: {class_prob[cls]*100:.2f}%")
    
    print("-" * 50)

    # Store for final decision
    task_list.append({
        "index": i + 1,
        "priority": predicted_labels[i],
        "confidence": class_prob[predicted_labels[i]],
        "raw_data": X_sample_raw.iloc[i]
    })

# Sort task list by priority level and confidence
priority_order = {"High": 3, "Medium": 2, "Low": 1}
sorted_tasks = sorted(task_list, key=lambda x: (-priority_order[x["priority"]], -x["confidence"]))

print("\n📌 Suggested Task Execution Order (High Priority & High Confidence First):")
for rank, task in enumerate(sorted_tasks, 1):
    print(f"{rank}. Task {task['index']} → {task['priority']} ({task['confidence']*100:.2f}% confidence)")
