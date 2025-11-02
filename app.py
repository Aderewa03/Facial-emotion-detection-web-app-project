# ===================================================================
# FACIAL EMOTION RECOGNITION - FLASK WEB APPLICATION
# ===================================================================
# This Flask app provides a web interface for users to:
# 1. Enter their personal information
# 2. Upload a face image
# 3. Get emotion prediction from the trained model
# 4. Save data to SQLite database
# ===================================================================

import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow import keras
import cv2
from werkzeug.utils import secure_filename
import base64

# ===================================================================
# FLASK APP CONFIGURATION
# ===================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_IMAGES_FOLDER'] = 'static/images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Emotion labels (must match the order from training)
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Emotion response messages
EMOTION_RESPONSES = {
    'angry': "You are frowning. Why are you angry? Take a deep breath! 😠",
    'disgust': "You look disgusted. Is something bothering you? 🤢",
    'fear': "You seem scared. Don't worry, everything will be okay! 😨",
    'happy': "You are smiling! Keep up the positive energy! 😊",
    'sad': "You are frowning. Why are you sad? Cheer up! 😢",
    'surprise': "You look surprised! What caught your attention? 😲",
    'neutral': "You have a neutral expression. Calm and composed! 😐"
}

print("="*70)
print("🚀 FACIAL EMOTION RECOGNITION - FLASK WEB APP")
print("="*70)

# ===================================================================
# LOAD TRAINED MODEL
# ===================================================================

print("\n📦 Loading trained emotion detection model...")

MODEL_PATH = 'face_emotionModel.h5'

try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from '{MODEL_PATH}'")
    print(f"   Model input shape: {model.input_shape}")
    print(f"   Model output shape: {model.output_shape}")
except Exception as e:
    print(f"❌ ERROR: Could not load model: {e}")
    print("   Make sure 'face_emotionModel.h5' exists in the project folder!")
    model = None

# ===================================================================
# DATABASE SETUP
# ===================================================================

DATABASE_NAME = 'database.db'

def init_database():
    """Initialize SQLite database with users table"""
    print("\n🗄️ Setting up database...")
    
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            image_path TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    
    print(f"✅ Database '{DATABASE_NAME}' initialized successfully!")
    print("   Table: users (id, name, email, age, gender, emotion, confidence, image_path, timestamp)")

# Initialize database on startup
init_database()

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """
    Preprocess uploaded image for model prediction
    
    Steps:
    1. Read image in grayscale
    2. Resize to 48x48 pixels
    3. Normalize pixel values to 0-1
    4. Reshape to match model input shape
    """
    try:
        # Read image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError("Could not read image file")
        
        # Resize to 48x48
        img = cv2.resize(img, (48, 48))
        
        # Normalize to 0-1
        img = img / 255.0
        
        # Reshape to (1, 48, 48, 1) for model input
        img = img.reshape(1, 48, 48, 1)
        
        return img
    
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None

def predict_emotion(image_path):
    """
    Predict emotion from image using trained model
    
    Returns:
        tuple: (emotion_label, confidence_score)
    """
    if model is None:
        return None, 0.0
    
    # Preprocess image
    processed_img = preprocess_image(image_path)
    
    if processed_img is None:
        return None, 0.0
    
    # Make prediction
    predictions = model.predict(processed_img, verbose=0)
    
    # Get emotion with highest probability
    emotion_index = np.argmax(predictions[0])
    confidence = float(predictions[0][emotion_index])
    emotion = EMOTION_LABELS[emotion_index]
    
    return emotion, confidence

def save_to_database(name, email, age, gender, emotion, confidence, image_path):
    """Save user data and emotion prediction to database"""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO users (name, email, age, gender, emotion, confidence, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (name, email, age, gender, emotion, confidence, image_path))
        
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        
        print(f"✅ User data saved to database (ID: {user_id})")
        return user_id
    
    except Exception as e:
        print(f"❌ Error saving to database: {e}")
        return None

# ===================================================================
# FLASK ROUTES
# ===================================================================

@app.route('/')
def index():
    """Home page - display the upload form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle form submission:
    1. Validate form data
    2. Save uploaded image
    3. Predict emotion
    4. Save to database
    5. Return results
    """
    try:
        # ===== STEP 1: VALIDATE FORM DATA =====
        print("\n" + "="*70)
        print("📥 NEW PREDICTION REQUEST")
        print("="*70)
        
        # Get form data
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        age = request.form.get('age', '').strip()
        gender = request.form.get('gender', '').strip()
        
        # Validate required fields
        if not name or not email:
            return jsonify({
                'success': False,
                'error': 'Name and email are required!'
            }), 400
        
        # Convert age to integer
        try:
            age = int(age) if age else None
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Age must be a valid number!'
            }), 400
        
        print(f"👤 User Info:")
        print(f"   Name: {name}")
        print(f"   Email: {email}")
        print(f"   Age: {age}")
        print(f"   Gender: {gender}")
        
        # ===== STEP 2: VALIDATE AND SAVE IMAGE =====
        
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image uploaded!'
            }), 400
        
        file = request.files['image']
        
        # Check if filename is empty
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No image selected!'
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type! Please upload PNG, JPG, or GIF.'
            }), 400
        
        # Create secure filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(file.filename)
        filename = f"{timestamp}_{filename}"
        
        # Save to uploads folder (temporary)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        print(f"📁 Image saved to: {upload_path}")
        
        # ===== STEP 3: PREDICT EMOTION =====
        
        print("🧠 Running emotion prediction...")
        emotion, confidence = predict_emotion(upload_path)
        
        if emotion is None:
            return jsonify({
                'success': False,
                'error': 'Could not process image. Please try another photo.'
            }), 500
        
        confidence_percent = confidence * 100
        
        print(f"✅ Prediction Complete:")
        print(f"   Emotion: {emotion}")
        print(f"   Confidence: {confidence_percent:.2f}%")
        
        # ===== STEP 4: SAVE TO PERMANENT STORAGE =====
        
        # Copy image to static/images for permanent storage
        static_image_path = os.path.join(app.config['STATIC_IMAGES_FOLDER'], filename)
        
        # Read and save image
        import shutil
        shutil.copy2(upload_path, static_image_path)
        
        # Store relative path for database
        db_image_path = f"static/images/{filename}"
        
        print(f"💾 Image copied to permanent storage: {db_image_path}")
        
        # ===== STEP 5: SAVE TO DATABASE =====
        
        user_id = save_to_database(
            name=name,
            email=email,
            age=age,
            gender=gender,
            emotion=emotion,
            confidence=confidence,
            image_path=db_image_path
        )
        
        if user_id is None:
            return jsonify({
                'success': False,
                'error': 'Could not save data to database.'
            }), 500
        
        # ===== STEP 6: RETURN RESULTS =====
        
        response_message = EMOTION_RESPONSES.get(emotion, "Emotion detected!")
        
        print("="*70)
        print("✅ REQUEST COMPLETED SUCCESSFULLY")
        print("="*70)
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'emotion': emotion,
            'confidence': round(confidence_percent, 2),
            'message': response_message,
            'image_path': db_image_path
        })
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred. Please try again.'
        }), 500

@app.route('/stats')
def stats():
    """Display database statistics (optional feature)"""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        
        # Get total users
        cursor.execute('SELECT COUNT(*) FROM users')
        total_users = cursor.fetchone()[0]
        
        # Get emotion distribution
        cursor.execute('''
            SELECT emotion, COUNT(*) as count 
            FROM users 
            GROUP BY emotion 
            ORDER BY count DESC
        ''')
        emotion_stats = cursor.fetchall()
        
        conn.close()
        
        stats_html = f"""
        <html>
        <head>
            <title>Database Statistics</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 50%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>📊 Database Statistics</h1>
            <p><strong>Total Users:</strong> {total_users}</p>
            
            <h2>Emotion Distribution</h2>
            <table>
                <tr>
                    <th>Emotion</th>
                    <th>Count</th>
                </tr>
        """
        
        for emotion, count in emotion_stats:
            stats_html += f"<tr><td>{emotion.capitalize()}</td><td>{count}</td></tr>"
        
        stats_html += """
            </table>
            <br>
            <a href="/">← Back to Home</a>
        </body>
        </html>
        """
        
        return stats_html
    
    except Exception as e:
        return f"<h1>Error loading statistics</h1><p>{e}</p>"

# ===================================================================
# RUN FLASK APP
# ===================================================================

if __name__ == '__main__':
    # Create necessary folders if they don't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['STATIC_IMAGES_FOLDER'], exist_ok=True)
    
    print("\n" + "="*70)
    print("🌐 STARTING FLASK WEB SERVER")
    print("="*70)
    print("\n📍 Server will run at: http://127.0.0.1:5000")
    print("📍 Stats page at: http://127.0.0.1:5000/stats")
    print("\n⚠️  Press CTRL+C to stop the server")
    print("="*70 + "\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)