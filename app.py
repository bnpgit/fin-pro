from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import joblib
import secrets
from werkzeug.security import generate_password_hash, check_password_hash
import matplotlib.pyplot as plt
import io
import base64
import random
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Mock user database (in production, use a real database)
users = {
    "admin": {
        "password": generate_password_hash("admin123"),
        "name": "System Administrator",
        "email": "admin@cloudsys.com"
    }
}

# Load ML model and preprocessing tools
try:
    model = joblib.load("priority_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    feature_names = joblib.load("feature_names.pkl")
except Exception as e:
    print(f"Error loading model files: {e}")
    exit(1)

# Load dataset
df = pd.read_csv("dataset_cloud.csv")

# Preprocess the data
X = df.drop(columns=["Task_ID", "Priority"])
X_encoded = pd.get_dummies(X, columns=["Task_Type", "User_Type", "Resource_Class"])
X_encoded = X_encoded.reindex(columns=feature_names, fill_value=0)

# Color scheme
COLORS = {
    "High": "#ff6b6b",
    "Medium": "#ffd166",
    "Low": "#06d6a0",
    "background": "#f8f9fa",
    "card": "#ffffff",
    "text": "#343a40",
    "primary": "#4e73df",
    "secondary": "#858796"
}
# Add this after the COLORS dictionary definition in app.py

@app.context_processor
def inject_colors():
    return dict(colors=COLORS)
# Helper functions
def generate_task_priority_chart(probabilities, classes):
    plt.figure(figsize=(6, 4))
    plt.bar(classes, probabilities, color=[COLORS[cls] for cls in classes])
    plt.title('Task Priority Probabilities', fontsize=12)
    plt.ylabel('Probability', fontsize=10)
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=80)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode('utf-8')

def generate_system_load_chart():
    # Mock system load data
    hours = [f"{h}:00" for h in range(24)]
    load = [random.randint(30, 90) for _ in range(24)]
    
    plt.figure(figsize=(8, 3))
    plt.plot(hours, load, color=COLORS['primary'], marker='o', markersize=4)
    plt.fill_between(hours, load, color=COLORS['primary'], alpha=0.1)
    plt.title('System Load (Last 24 Hours)', fontsize=12)
    plt.ylabel('Load %', fontsize=10)
    plt.xticks([hours[i] for i in range(0, 24, 4)], rotation=45, fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=80)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode('utf-8')

def generate_priority_distribution():
    priorities = ['High', 'Medium', 'Low']
    counts = [random.randint(5, 20) for _ in range(3)]
    
    plt.figure(figsize=(4, 4))
    plt.pie(counts, labels=priorities, colors=[COLORS[p] for p in priorities],
            autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1})
    plt.title('Priority Distribution', fontsize=12)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=80)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode('utf-8')

def get_random_tasks(n=3):
    X_sample_raw = X.sample(n=n, random_state=None)
    X_sample_encoded = X_encoded.loc[X_sample_raw.index]
    X_scaled = scaler.transform(X_sample_encoded)
    
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    predicted_labels = label_encoder.inverse_transform(predictions)
    
    tasks = []
    for i in range(n):
        class_prob = dict(zip(label_encoder.classes_, probabilities[i]))
        tasks.append({
            "id": X_sample_raw.index[i],
            "priority": predicted_labels[i],
            "confidence": class_prob[predicted_labels[i]],
            "raw_data": X_sample_raw.iloc[i].to_dict(),
            "probabilities": class_prob,
            "chart": generate_task_priority_chart(
                [class_prob[cls] for cls in label_encoder.classes_],
                label_encoder.classes_
            )
        })
    
    # Sort by priority and confidence
    priority_order = {"High": 3, "Medium": 2, "Low": 1}
    return sorted(tasks, key=lambda x: (-priority_order[x["priority"]], -x["confidence"]))

# Routes
@app.route('/')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users and check_password_hash(users[username]['password'], password):
            session['username'] = username
            session['name'] = users[username]['name']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        name = request.form['name']
        email = request.form['email']
        
        if username in users:
            flash('Username already exists', 'danger')
        else:
            users[username] = {
                "password": generate_password_hash(password),
                "name": name,
                "email": email
            }
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    try:
        # Get random tasks (reduce if manual tasks exist)
        num_random_tasks = max(0, 3 - len(session.get('manual_tasks', [])))
        tasks = get_random_tasks(num_random_tasks)
        
        system_load_chart = generate_system_load_chart()
        priority_dist_chart = generate_priority_distribution()
        
        return render_template('dashboard.html',
                            name=session['name'],
                            tasks=tasks,
                            colors=COLORS,
                            system_load_chart=system_load_chart,
                            priority_dist_chart=priority_dist_chart,
                            current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as e:
        app.logger.error(f"Error generating dashboard: {str(e)}")
        flash("Error generating dashboard. Please try again.", "danger")
        return redirect(url_for('login'))
    
@app.route('/refresh')
def refresh():
    if 'username' not in session:
        return redirect(url_for('login'))
    flash('Dashboard refreshed with new tasks', 'info')
    return redirect(url_for('dashboard'))

@app.route('/add_manual_task', methods=['POST'])
def add_manual_task():
    if 'username' not in session:
        return redirect(url_for('login'))
    try:
        manual_tasks = []
        
        # Process each task (1-3)
        for i in range(1, 4):
            task_key = f'task{i}'
            task_data = {
                'Task_Type': request.form[f'{task_key}_task_type'],
                'User_Type': request.form[f'{task_key}_user_type'],
                'Resource_Class': request.form[f'{task_key}_resource_class'],
                'CPU_GHZ': float(request.form[f'{task_key}_cpu_ghz']),
                'Memory_MB': int(request.form[f'{task_key}_memory_mb']),
                'Bandwidth_Mbps': int(request.form[f'{task_key}_bandwidth_mbps']),
                'Disk_IO_MBps': int(request.form[f'{task_key}_disk_io']),
                'Runtime_Estimate_sec': int(request.form[f'{task_key}_runtime_sec']),
                'Submission_Hour': int(request.form[f'{task_key}_submission_hour']),
                'Deadline_sec': int(request.form[f'{task_key}_deadline_sec']),
                'Retry_Count': int(request.form[f'{task_key}_retry_count']),
                'Task_Size_MB': int(request.form[f'{task_key}_task_size_mb']),
                'Network_Latency_ms': int(request.form[f'{task_key}_network_latency'])
            }
            
            # Create DataFrame and encode
            manual_task = pd.DataFrame([task_data])
            manual_task_encoded = pd.get_dummies(manual_task, columns=["Task_Type", "User_Type", "Resource_Class"])
            
            # Ensure all expected columns are present
            for col in feature_names:
                if col not in manual_task_encoded.columns:
                    manual_task_encoded[col] = 0
                    
            manual_task_encoded = manual_task_encoded[feature_names]
            
            # Scale and predict
            X_scaled = scaler.transform(manual_task_encoded)
            prediction = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)
            predicted_label = label_encoder.inverse_transform(prediction)[0]
            
            # Create task result
            class_prob = dict(zip(label_encoder.classes_, probabilities[0]))
            manual_task_result = {
                "id": f"M{i}-{datetime.now().strftime('%H%M%S')}",
                "priority": predicted_label,
                "confidence": class_prob[predicted_label],
                "raw_data": task_data,
                "probabilities": class_prob,
                "chart": generate_task_priority_chart(
                    [class_prob[cls] for cls in label_encoder.classes_],
                    label_encoder.classes_
                )
            }
            manual_tasks.append(manual_task_result)
        
        flash('Task priorities predicted successfully!', 'success')
        tasks = manual_tasks
            
        system_load_chart = generate_system_load_chart()
        priority_dist_chart = generate_priority_distribution()
        
        return render_template('dashboard.html',
                            name=session['name'],
                            tasks=tasks,
                            colors=COLORS,
                            system_load_chart=system_load_chart,
                            priority_dist_chart=priority_dist_chart,
                            current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as e:
        print(f"Error generating dashboard: {str(e)}")
        flash("Error generating dashboard. Please try again.", "danger")
        return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
