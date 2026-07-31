# ============================================================
# IMPORTS
# ============================================================
import os
import gdown
import uuid
import re
import json
import time
import random
import base64
import pytz
from collections import defaultdict, Counter
from functools import wraps
from datetime import datetime, timezone, timedelta, date
import pymysql
pymysql.install_as_MySQLdb()
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify
)
from flask_mysqldb import MySQL
from flask_babel import Babel, gettext as _
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer
from deep_translator import GoogleTranslator

print("TensorFlow version:", tf.__version__)

# ============================================================
# APP INITIALIZATION & CONFIGURATION
# ============================================================
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'fallback-dev-key')

# ---- Mail ----
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'agappalay@gmail.com'
app.config['MAIL_PASSWORD'] = 'fvyb rytt hqxf juhx'
app.config['MAIL_DEFAULT_SENDER'] = 'agappalay@gmail.com'
mail = Mail(app)
serializer = URLSafeTimedSerializer(app.secret_key)

# ---- Session ----
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = False

# ---- Template ----
app.jinja_env.cache = None
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['TIMEZONE'] = 'UTC'

@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# ============================================================
# TIMEZONE HELPERS
# ============================================================
LOCAL_TZ = pytz.timezone('Asia/Manila')
UTC_TZ = pytz.UTC

def utc_to_local(utc_dt):
    if utc_dt is None:
        return None
    if utc_dt.tzinfo is None:
        utc_dt = UTC_TZ.localize(utc_dt)
    return utc_dt.astimezone(LOCAL_TZ)

def get_local_today():
    return datetime.now(LOCAL_TZ).date()

# ============================================================
# BABEL LOCALIZATION
# ============================================================
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
app.config['BABEL_DEFAULT_TIMEZONE'] = 'Asia/Manila'
app.config['LANGUAGES'] = {'en': 'English', 'tl': 'Tagalog'}
babel = Babel(app)

def get_locale():
    return session.get('language', request.accept_languages.best_match(app.config['LANGUAGES'].keys()))
babel.locale_selector_func = get_locale

@app.before_request
def set_language():
    lang = request.args.get('lang')
    if lang in ['en', 'tl']:
        session['language'] = lang
        session.permanent = True
        return
    if 'language' not in session:
        session['language'] = 'en'
        session.permanent = True

# ============================================================
# API KEYS & CONSTANTS
# ============================================================
OPENWEATHER_API_KEY = os.environ.get('WEATHER_API_KEY', '')

BARANGAYS = [
    'Alos', 'Amandiego', 'Amangbangan', 'Balangobong', 'Balayang',
    'Baleyadaan', 'Bisocol', 'Bolaney', 'Bued', 'Cabatuan',
    'Cayucay', 'Dulacac', 'Inerangan', 'Landoc', 'Linmansangan',
    'Lucap', 'Maawi', 'Macatiw', 'Magsaysay', 'Mona', 'Palamis', 'Pandan',
    'Pangapisan', 'Poblacion', 'Pocal-pocal', 'Pogo', 'Polo',
    'Quibuar', 'Sabangan', 'San Antonio (R. Magsaysay)', 'San Jose', 'San Roque',
    'San Vicente', 'Santa Maria', 'Tanaytay', 'Tangcarang', 'Tawintawin',
    'Telbang', 'Victoria'
]

# ============================================================
# CUSTOM MODEL LOADING
# ============================================================
class CustomDense(Dense):
    def __init__(self, units, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 quantization_config=None, **kwargs):
        super().__init__(units=units, activation=activation, use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         **kwargs)

class CompatibleInputLayer(InputLayer):
    @classmethod
    def from_config(cls, config):
        if 'batch_shape' in config:
            config['batch_input_shape'] = config.pop('batch_shape')
        return super().from_config(config)

custom_objects = {
    'InputLayer': CompatibleInputLayer,
    'Dense': CustomDense,
}
FILE_ID = "1B1X0YMYaKXSXIIahlA-tFSWjXut1qsql"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"
MODEL_PATH = 'model/rice_disease_models.h5'
model = None

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model... (this may take a few minutes)")
        os.makedirs("model", exist_ok=True)
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("✅ Model downloaded successfully.")
        download_model()
try:
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
    print("✅ Custom model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

IMG_SIZE = (224, 224)

class_names_raw = [
    'bacterial_leaf_blight',
    'brown_spot',
    'healthy_rice__leaf',
    'leaf_scald',
    'narrow_brown_leaf_spot',
    'not_rice_leaf',
    'pest_damage',
    'rice_blast',
    'sheath_blight',
    'tungro_virus'
]

DISEASE_NAME_MAP = {
    'bacterial_leaf_blight': 'Bacterial Leaf Blight',
    'brown_spot': 'Brown Spot',
    'healthy_rice__leaf': 'Healthy Rice Leaf',
    'leaf_scald': 'Leaf Scald',
    'narrow_brown_leaf_spot': 'Narrow Brown Leaf Spot',
    'not_rice_leaf': 'Not Rice Leaf',
    'pest_damage': 'Insect Damage',
    'rice_blast': 'Rice Blast',
    'sheath_blight': 'Sheath Blight',
    'tungro_virus': 'Tungro Virus'
}

# ============================================================
# MYSQL CONFIGURATION
# ============================================================
app.config['MYSQL_HOST'] = os.environ.get('MYSQL_HOST', 'localhost')
app.config['MYSQL_USER'] = os.environ.get('MYSQL_USER', 'root')
app.config['MYSQL_PASSWORD'] = os.environ.get('MYSQL_PASSWORD', '')
app.config['MYSQL_DB'] = os.environ.get('MYSQL_DB', 'sakascan_db')
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)

# ============================================================
# UPLOAD SETTINGS
# ============================================================
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_image_quality(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        arr = np.array(img)
        gray = np.mean(arr, axis=2)
        mean_bright = np.mean(gray)
        if mean_bright < 30:
            return False, 'Image is too dark. Please take a photo with better lighting.'
        if mean_bright > 240:
            return False, 'Image is overexposed. Please avoid direct sunlight.'
        blur_score = np.std(gray)
        if blur_score < 15:
            return False, 'Image is blurry. Please ensure the leaf is in focus.'
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        green_mask = (g > r) & (g > b)
        green_ratio = np.mean(green_mask)
        if green_ratio < 0.08:
            return False, 'No rice leaf detected. Please capture a rice leaf image.'
        return True, 'OK'
    except Exception as e:
        print(f'Quality check error: {e}')
        return True, 'OK'

def predict_disease(image_path):
    quality_ok, reason = check_image_quality(image_path)
    if not quality_ok:
        return None, 0.0, reason

    if model is None:
        return None, 0.0, "Model not loaded. Please try again later."

    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array, verbose=0)[0]
        sorted_indices = np.argsort(predictions)[::-1]
        top_idx = sorted_indices[0]
        top_name = class_names_raw[top_idx]
        top_conf = predictions[top_idx]

        if top_name in ['not_rice_leaf', 'pest_damage'] and top_conf > 0.5:
            return None, float(top_conf), 'This does not appear to be a rice leaf. Please capture a clear image of a rice leaf.'

        for idx in sorted_indices:
            raw_name = class_names_raw[idx]
            if raw_name not in ['not_rice_leaf', 'pest_damage']:
                display_name = DISEASE_NAME_MAP.get(raw_name, raw_name)
                disease_conf = float(predictions[idx])
                if disease_conf < 0.70:
                    return None, disease_conf, f'Low confidence ({disease_conf:.0%}). Please take a clearer photo.'
                return display_name, disease_conf, None

        display_name = DISEASE_NAME_MAP.get(top_name, top_name)
        return display_name, float(top_conf), None

    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0.0, "An error occurred during prediction. Please try again."

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'admin':
            flash('Admin access required.')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def get_disease_images(disease_id, limit=None):
    try:
        cur = mysql.connection.cursor()
        query = """
            SELECT id, image_path, is_primary, display_order
            FROM disease_images
            WHERE disease_id = %s
            ORDER BY display_order ASC, is_primary DESC
        """
        if limit:
            query += f" LIMIT {limit}"
        cur.execute(query, (disease_id,))
        images = cur.fetchall()
        cur.close()
        return images
    except Exception as e:
        print(f"Error fetching disease images: {e}")
        return []

# ---- Full name helper ----
def get_full_name(user):
    if not user:
        return ''
    first = (user.get('first_name') or '').strip()
    middle = (user.get('middle_name') or '').strip()
    last = (user.get('last_name') or '').strip()
    suffix = (user.get('suffix') or '').strip()
    parts = [p for p in [first, middle, last, suffix] if p]
    return ' '.join(parts)
# ---- Barangay risk functions ----
def get_barangay_disease_risk(barangay):
    try:
        cur = mysql.connection.cursor()
        query = """
            SELECT 
                s.predicted_name AS disease,
                COUNT(DISTINCT CONCAT(s.user_id, DATE(s.created_at))) AS report_count,
                COUNT(DISTINCT s.user_id) AS unique_farmers
            FROM scans s
            WHERE s.location = %s
              AND s.created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
              AND s.confidence >= 0.85
              AND s.predicted_name NOT IN ('Healthy', 'Not Rice Leaf', 'Insect Damage')
              AND s.disease_id IS NOT NULL
            GROUP BY s.predicted_name
            ORDER BY report_count DESC
        """
        cur.execute(query, (barangay,))
        results = cur.fetchall()
        cur.close()

        risk_data = []
        for row in results:
            disease = row['disease']
            count = row['report_count']
            farmers = row['unique_farmers']

            if count >= 10 and farmers >= 5:
                risk_level = 'Very High'
                color = 'danger'
                icon = 'exclamation-triangle'
                recommendation = (
                    f"A high number of {disease} cases have been reported in your barangay "
                    f"within the past seven days. This indicates a **Very High Disease Risk**. "
                    f"Farmers are strongly encouraged to inspect their fields immediately and "
                    f"seek guidance from their local agricultural office if symptoms continue to spread."
                )
            elif count >= 6 and farmers >= 3:
                risk_level = 'High'
                color = 'high'
                icon = 'exclamation-triangle'
                recommendation = (
                    f"Multiple cases of {disease} have been detected in your barangay during "
                    f"the past seven days. There is an increased risk of disease spread. "
                    f"Farmers should inspect their rice fields immediately and apply appropriate "
                    f"disease management practices when necessary."
                )
            elif count >= 3 and farmers >= 2:
                risk_level = 'Moderate'
                color = 'moderate'
                icon = 'info-circle'
                recommendation = (
                    f"Several cases of {disease} have recently been reported in your barangay. "
                    f"Farmers are advised to monitor their rice fields regularly and inspect plants "
                    f"for early signs of disease."
                )
            else:
                risk_level = 'Low'
                color = 'success'
                icon = 'check-circle'
                recommendation = (
                    f"Current reports indicate a **Low Risk** of disease occurrence in your barangay. "
                    f"Continue routine field monitoring and follow recommended crop management practices."
                )

            risk_data.append({
                'disease': disease,
                'count': count,
                'farmers': farmers,
                'risk_level': risk_level,
                'color': color,
                'icon': icon,
                'recommendation': recommendation
            })
        return risk_data
    except Exception as e:
        print(f"Error in get_barangay_disease_risk: {e}")
        return []

def get_barangay_overall_risk(barangay):
    risks = get_barangay_disease_risk(barangay)
    if not risks:
        return None

    risk_order = {'Low': 0, 'Moderate': 1, 'High': 2, 'Very High': 3}
    max_level = max(risks, key=lambda r: risk_order[r['risk_level']])['risk_level']
    top_risks = [r for r in risks if r['risk_level'] == max_level]

    if len(top_risks) == 1:
        display_disease = top_risks[0]['disease']
        display_risk = top_risks[0]
        multiple = False
    else:
        display_disease = f"{len(top_risks)} diseases"
        display_risk = top_risks[0]
        multiple = True

    return {
        'disease': display_disease,
        'count': display_risk['count'],
        'farmers': display_risk['farmers'],
        'risk_level': display_risk['risk_level'],
        'color': display_risk['color'],
        'icon': display_risk['icon'],
        'recommendation': display_risk['recommendation'],
        'all_risks': risks,
        'multiple': multiple,
        'top_diseases': [r['disease'] for r in top_risks]
    }

def get_risk_level_from_counts(reports, farmers):
    if reports >= 10 and farmers >= 5:
        return {'level': 'Very High', 'color': 'danger'}
    elif reports >= 6 and farmers >= 3:
        return {'level': 'High', 'color': 'high'}
    elif reports >= 3 and farmers >= 2:
        return {'level': 'Moderate', 'color': 'moderate'}
    else:
        return {'level': 'Low', 'color': 'success'}

# ============================================================
# TRANSLATION FUNCTIONS
# ============================================================
_translator = GoogleTranslator(source='en', target='tl')
_in_memory_cache = {}
_MAX_CACHE_SIZE = 500

def _cache_translation(source_text, target_lang, translated_text):
    key = (source_text, target_lang)
    if len(_in_memory_cache) >= _MAX_CACHE_SIZE:
        _in_memory_cache.pop(next(iter(_in_memory_cache)))
    _in_memory_cache[key] = translated_text

def translate_with_deep(text, target_lang='tl', source_lang='en'):
    if not text or target_lang == source_lang:
        return text
    try:
        if target_lang == 'tl' and source_lang == 'en':
            translator = _translator
        else:
            translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated = translator.translate(text)
        if translated and translated != text:
            return translated
        return text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def translate_with_cache(text, target_lang='tl'):
    if not text or target_lang == 'en':
        return text

    key = (text, target_lang)
    if key in _in_memory_cache:
        return _in_memory_cache[key]

    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT translated_text FROM translations WHERE source_text = %s AND target_lang = %s",
                    (text, target_lang))
        cached = cur.fetchone()
        if cached:
            translated = cached['translated_text']
            cur.close()
            _cache_translation(text, target_lang, translated)
            return translated
        cur.close()
    except Exception as e:
        print(f"Translation cache read error: {e}")

    translated = translate_with_deep(text, target_lang)
    if translated != text and len(text) < 5000:
        try:
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO translations (source_text, target_lang, translated_text) VALUES (%s, %s, %s)",
                        (text, target_lang, translated))
            mysql.connection.commit()
            cur.close()
        except Exception as e:
            print(f"Cache save error: {e}")
    _cache_translation(text, target_lang, translated)
    return translated

def translate_text(text):
    if not text:
        return text
    if session.get('language') == 'tl':
        return translate_with_cache(text)
    return text

@app.context_processor
def utility_processor():
    return dict(translate=translate_text)

app.jinja_env.filters['translate'] = translate_text

def prewarm_translation_cache():
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT id, name_en, description_en, symptoms_en, causes_en, prevention_en, treatment_en "
                    "FROM diseases WHERE status='published'")
        diseases = cur.fetchall()
        cur.close()
        fields = ['name_en', 'description_en', 'symptoms_en', 'causes_en', 'prevention_en', 'treatment_en']
        for disease in diseases:
            for field in fields:
                text = disease.get(field)
                if text:
                    translate_with_cache(text, 'tl')
        print(f"✅ Pre‑warmed translation cache with {len(diseases)} diseases.")
    except Exception as e:
        print(f"⚠️ Could not pre‑warm translation cache: {e}")

# ============================================================
# ADVISORY GENERATION
# ============================================================
def generate_weather_advisory(temp, humidity, rainfall, condition, user_id=None):
    try:
        cur = mysql.connection.cursor()
        if user_id:
            cur.execute("""
                SELECT id, created_at FROM advisories
                WHERE (user_id = %s OR (user_id IS NULL AND %s IS NULL))
                ORDER BY created_at DESC LIMIT 1
            """, (user_id, user_id))
        else:
            cur.execute("""
                SELECT id, created_at FROM advisories
                WHERE user_id IS NULL
                ORDER BY created_at DESC LIMIT 1
            """)
        recent = cur.fetchone()
        cur.close()
        if recent:
            time_diff = datetime.now(UTC_TZ) - recent['created_at']
            if time_diff.total_seconds() < 21600:
                return False

        messages = []
        severity = 'info'

        if temp > 30:
            messages.append("High temperature (>30°C) detected. Hot weather may stress rice plants and increase the risk of pest infestations. Regularly inspect your field and maintain proper water levels.")
            severity = 'warning'
        elif temp < 20:
            messages.append("Low temperature (<20°C) detected. Cool weather may slow rice growth and increase the risk of certain diseases. Continue monitoring your rice plants.")
            severity = 'warning'

        if humidity > 80:
            messages.append("High humidity (>80%) detected. These conditions favor diseases such as Rice Blast and Brown Spot. Inspect rice leaves regularly and remove infected leaves if necessary.")
            severity = 'danger'
        elif humidity > 65:
            messages.append("Moderate humidity detected. Some leaf diseases may develop under these conditions. Continue checking your rice plants regularly.")
            if severity == 'info':
                severity = 'warning'

        if rainfall > 5:
            messages.append("Heavy rainfall detected. Excess water may increase the spread of diseases such as Bacterial Leaf Blight. Ensure proper field drainage and inspect your crop after the rain.")
            severity = 'danger'
        elif rainfall > 1:
            messages.append("Light rainfall detected. Continue monitoring your rice plants and make sure the field has proper drainage.")

        if not messages:
            return False

        title = "Weather Advisory"
        full_message = " ".join(messages)

        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO advisories (user_id, title, message, severity)
            VALUES (%s, %s, %s, %s)
        """, (user_id, title, full_message, severity))
        mysql.connection.commit()
        cur.close()
        return True
    except Exception as e:
        print(f"Error generating advisory: {e}")
        return False

# ============================================================
# EMAIL VERIFICATION FUNCTIONS
# ============================================================
def generate_verification_token(email):
    return serializer.dumps(email, salt='email-verify')

def verify_token(token, expiration=3600):
    try:
        email = serializer.loads(token, salt='email-verify', max_age=expiration)
        return email
    except Exception:
        return None

def send_verification_email(email, link):
    try:
        msg = Message(
            subject='Verify Your Email - AGAP-PALAY',
            sender=app.config['MAIL_DEFAULT_SENDER'],
            recipients=[email],
            html=f'''
            <html>
            <body style="font-family: Arial, sans-serif; padding: 20px; background-color: #f4f4f4;">
                <div style="max-width: 500px; margin: 0 auto; background-color: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                    <h2 style="color: #2e7d32; text-align: center;">🌾 AGAP-PALAY</h2>
                    <h3 style="color: #333;">Verify Your Email</h3>
                    <p style="color: #555;">Thank you for registering! Please click the button below to verify your email address.</p>
                    <div style="text-align: center; margin: 25px 0;">
                        <a href="{link}" style="background-color: #2e7d32; color: #ffffff; padding: 12px 30px; text-decoration: none; border-radius: 5px; font-weight: bold;">
                            Verify Email
                        </a>
                    </div>
                    <p style="color: #888; font-size: 12px; text-align: center;">This link will expire in 1 hour.</p>
                    <hr style="border: none; border-top: 1px solid #eee;">
                    <p style="color: #aaa; font-size: 11px; text-align: center;">If you didn't create an account, you can safely ignore this email.</p>
                </div>
            </body>
            </html>
            '''
        )
        mail.send(msg)
        print(f"✅ Verification email sent to {email}")
        return True
    except Exception as e:
        print(f"❌ Email error: {e}")
        return False

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT 
                d.id, 
                d.name_en as name, 
                d.description_en as description,
                (SELECT image_path 
                 FROM disease_images 
                 WHERE disease_id = d.id AND is_primary = TRUE 
                 LIMIT 1) as image_path
            FROM diseases d
            WHERE d.status = 'published'
            ORDER BY d.created_at DESC
            LIMIT 6
        """)
        recent_diseases = cur.fetchall()
        cur.close()

        if session.get('language') == 'tl':
            for d in recent_diseases:
                d['name'] = translate_with_cache(d['name'] or '')
                d['description'] = translate_with_cache(d['description'] or '')

        return render_template('index.html', recent_diseases=recent_diseases)
    except Exception as e:
        print(f"Index error: {e}")
        flash('Unable to load homepage. Please try again later.', 'danger')
        return render_template('index.html', recent_diseases=[])

# ---- Registration (now with name parts) ----
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form.get('first_name', '').strip()
        middle_name = request.form.get('middle_name', '').strip() or None
        last_name = request.form.get('last_name', '').strip()
        suffix = request.form.get('suffix', '').strip() or None
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')
        language = request.form.get('language_preference', 'en')
        barangay = request.form.get('barangay')

        if not first_name or not last_name or not email or not password:
            flash(_('First name, last name, email, and password are required.'))
            return redirect(url_for('register'))

        if password != confirm:
            flash(_('Passwords do not match.'))
            return redirect(url_for('register'))

        try:
            cur = mysql.connection.cursor()
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            if cur.fetchone():
                flash(_('Email already registered. Please login.'))
                cur.close()
                return redirect(url_for('login'))

            hashed_password = generate_password_hash(password)
            cur.execute("""
                INSERT INTO users (first_name, middle_name, last_name, suffix, email, password, language_preference, barangay, is_verified)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, FALSE)
            """, (first_name, middle_name, last_name, suffix, email, hashed_password, language, barangay))
            mysql.connection.commit()
            cur.close()

            token = generate_verification_token(email)
            link = url_for('verify_email', token=token, _external=True)
            send_verification_email(email, link)

            flash(_('Registration successful! Please check your email to verify your account.'))
            return redirect(url_for('verification_pending'))
        except Exception as e:
            print(f"Registration error: {e}")
            flash(_('An error occurred during registration. Please try again.'))
            return redirect(url_for('register'))

    return render_template('register.html', barangays=BARANGAYS)

# ---- Verification routes ----
@app.route('/verification-pending')
def verification_pending():
    return render_template('verification_pending.html')

@app.route('/verify/<token>')
def verify_email(token):
    email = verify_token(token)
    if not email:
        return render_template('verification_error.html'), 400

    try:
        cur = mysql.connection.cursor()
        cur.execute("UPDATE users SET is_verified = TRUE WHERE email = %s", (email,))
        mysql.connection.commit()
        cur.close()
        return render_template('verification_success.html', email=email)
    except Exception as e:
        print(f"Verification error: {e}")
        flash('An error occurred during verification. Please try again.')
        return redirect(url_for('login'))

@app.route('/resend-verification')
def resend_verification():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT email, is_verified FROM users WHERE id = %s", (session['user_id'],))
        user = cur.fetchone()
        cur.close()

        if user and not user['is_verified']:
            token = generate_verification_token(user['email'])
            link = url_for('verify_email', token=token, _external=True)
            send_verification_email(user['email'], link)
            flash(_('A new verification email has been sent.'))
        else:
            flash(_('Your account is already verified.'))
    except Exception as e:
        print(f"Resend verification error: {e}")
        flash('An error occurred. Please try again.')

    return redirect(url_for('verification_pending'))

# ---- Login ----
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if not email or not password:
            flash(_('Email and password are required.'))
            return redirect(url_for('login'))

        try:
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cur.fetchone()
            cur.close()

            if user and check_password_hash(user['password'], password):
                if not user.get('is_verified', False):
                    flash(_('Please verify your email before logging in. Check your inbox.'))
                    return redirect(url_for('login'))

                session['user_id'] = user['id']
                session['first_name'] = user['first_name']
                session['last_name'] = user['last_name']
                session['role'] = user['role']
                session['language'] = user.get('language_preference', 'en')
                session['barangay'] = user.get('barangay')
                session.permanent = True
                session.modified = True

                full_name = get_full_name(user)
                flash(_('Welcome back, %(name)s!', name=full_name))
                if user['role'] == 'admin':
                    return redirect(url_for('admin_dashboard'))
                return redirect(url_for('dashboard'))
            else:
                flash(_('Invalid email or password.'))
                return redirect(url_for('login'))
        except Exception as e:
            print(f"Login error: {e}")
            flash('An error occurred during login. Please try again.')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.')
    return redirect(url_for('index'))

@app.route('/language/<lang>')
def set_language(lang):
    if lang in ['en', 'tl']:
        session['language'] = lang
        session.permanent = True
        session.modified = True
        if 'user_id' in session:
            try:
                cur = mysql.connection.cursor()
                cur.execute("UPDATE users SET language_preference=%s WHERE id=%s", (lang, session['user_id']))
                mysql.connection.commit()
                cur.close()
            except Exception as e:
                print(f"Language update error: {e}")
        flash('Language changed successfully.' if lang == 'en' else 'Matagumpay na napalitan ang wika.')
    return redirect(request.referrer or url_for('index'))

# ---- Dashboard ----
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    weather = None
    severe_advisory = None
    advisories = []
    overall_risk = None
    recent_scans = []
    labels = []
    counts = []
    scans_count = 0

    try:
        cur = mysql.connection.cursor()

        # ---- 1. Get user details (for name display) ----
        cur.execute("SELECT first_name, middle_name, last_name, suffix, barangay FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()
        user_barangay = user['barangay'] if user else None
        user_full_name = get_full_name(user) if user else "User"

        # ---- 2. Barangay Disease Risk ----
        if user_barangay:
            overall_risk = get_barangay_overall_risk(user_barangay)

        # ---- 3. Fetch weather ----
        lat, lon = 16.1558, 119.9808
        if user_barangay:
            cur.execute("SELECT latitude, longitude FROM barangay_coordinates WHERE name = %s", (user_barangay,))
            coords = cur.fetchone()
            if coords:
                lat, lon = coords['latitude'], coords['longitude']

        weather_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {'lat': lat, 'lon': lon, 'appid': OPENWEATHER_API_KEY, 'units': 'metric'}
        try:
            resp = requests.get(weather_url, params=params, timeout=5)
            data = resp.json()
            if data.get('cod') == 200:
                temp = data['main']['temp']
                humidity = data['main']['humidity']
                rainfall = data.get('rain', {}).get('1h', 0)
                condition = data['weather'][0]['description']
                weather = {
                    'temperature': temp,
                    'humidity': humidity,
                    'weather_condition': condition,
                    'rainfall': rainfall
                }
                generate_weather_advisory(temp, humidity, rainfall, condition, user_id=user_id)
            else:
                print(f"Weather API error: {data.get('message', 'Unknown error')}")
        except Exception as e:
            print(f"Weather fetch exception: {e}")

        # ---- Admin override ----
        view_barangay = request.args.get('barangay')
        if session.get('role') == 'admin' and view_barangay:
            user_barangay = view_barangay
        else:
            cur.execute("SELECT barangay FROM users WHERE id = %s", (user_id,))
            user = cur.fetchone()
            user_barangay = user['barangay'] if user else None

        if user_barangay:
            overall_risk = get_barangay_overall_risk(user_barangay)

        # ---- 4. Statistics ----
        cur.execute("SELECT COUNT(*) as total_scans FROM scans WHERE user_id = %s", (user_id,))
        scans_count = cur.fetchone()['total_scans']

        # ---- Recent scans ----
        cur.execute("""
            SELECT s.*, d.name_en as disease_name
            FROM scans s
            LEFT JOIN diseases d ON s.disease_id = d.id
            WHERE s.user_id = %s
              AND s.predicted_name NOT IN ('Not Rice Leaf', 'Insect Damage', 'Healthy Rice Leaf', 'Healthy')
            ORDER BY s.created_at DESC
            LIMIT 5
        """, (user_id,))
        recent_scans = cur.fetchall()
        for scan in recent_scans:
            if scan.get('created_at'):
                scan['created_at'] = utc_to_local(scan['created_at'])

        # ---- Chart data ----
        cur.execute("""
            SELECT d.name_en, COUNT(s.id) as count
            FROM diseases d
            LEFT JOIN scans s ON d.id = s.disease_id AND s.user_id = %s
            WHERE d.name_en NOT IN ('Not Rice Leaf', 'Insect Damage', 'Healthy Rice Leaf', 'Healthy')
            GROUP BY d.id
            HAVING count > 0
            ORDER BY count DESC
        """, (user_id,))
        chart_data = cur.fetchall()
        labels = [row['name_en'] for row in chart_data]
        counts = [row['count'] for row in chart_data]

        # ---- Advisories ----
        cur.execute("""
            SELECT * FROM advisories
            WHERE user_id IS NULL OR user_id = %s
            ORDER BY created_at DESC
            LIMIT 5
        """, (user_id,))
        advisories = cur.fetchall()
        for adv in advisories:
            if adv.get('created_at'):
                adv['created_at'] = utc_to_local(adv['created_at'])

        cur.execute("""
            SELECT * FROM advisories
            WHERE (user_id IS NULL OR user_id = %s)
              AND severity IN ('danger', 'warning')
            ORDER BY created_at DESC
            LIMIT 1
        """, (user_id,))
        severe_advisory = cur.fetchone()
        if severe_advisory and severe_advisory.get('created_at'):
            severe_advisory['created_at'] = utc_to_local(severe_advisory['created_at'])

        cur.close()

        # ---- Translate ----
        if session.get('language') == 'tl':
            for scan in recent_scans:
                scan['disease_name'] = translate_with_cache(scan['disease_name'] or '')

        return render_template('dashboard.html',
                               user_full_name=user_full_name,
                               scans_count=scans_count,
                               recent_scans=recent_scans,
                               labels=labels,
                               counts=counts,
                               advisories=advisories,
                               severe_advisory=severe_advisory,
                               weather=weather,
                               overall_risk=overall_risk)
    except Exception as e:
        print(f"Dashboard error: {e}")
        flash('Unable to load dashboard. Please try again.', 'danger')
        return render_template('dashboard.html',
                               user_full_name="User",
                               scans_count=0,
                               recent_scans=[],
                               labels=[],
                               counts=[],
                               advisories=[],
                               severe_advisory=None,
                               weather=None,
                               overall_risk=None)

# ---- Scan ----
@app.route('/scan', methods=['GET', 'POST'])
def scan():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT name FROM barangay_coordinates ORDER BY name")
        barangays = [row['name'] for row in cur.fetchall()]
        cur.close()
    except Exception as e:
        print(f"Barangay fetch error: {e}")
        barangays = BARANGAYS

    error = None
    error_type = None
    guidance = None
    image_preview = None
    user_barangay = session.get('barangay')

    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                flash('No file part.')
                return redirect(request.url)

            file = request.files['file']
            if file.filename == '':
                flash('No selected file.')
                return redirect(request.url)

            if not allowed_file(file.filename):
                flash('Unsupported file format. Please upload a PNG, JPG, JPEG, or GIF.')
                return redirect(request.url)

            filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            disease_name, confidence, error_msg = predict_disease(filepath)

            if error_msg or confidence < 0.70:
                if error_msg:
                    if 'not a rice leaf' in error_msg.lower():
                        error_type = 'not_leaf'
                        error_message = error_msg
                        guidance = 'Make sure the leaf fills most of the frame and is well‑lit. Avoid background objects.'
                    elif 'dark' in error_msg.lower():
                        error_type = 'quality'
                        error_message = error_msg
                        guidance = 'Move to a brighter area or use flash.'
                    elif 'blurry' in error_msg.lower():
                        error_type = 'quality'
                        error_message = error_msg
                        guidance = 'Hold the camera steady and tap the screen to focus.'
                    elif 'leaf' in error_msg.lower():
                        error_type = 'quality'
                        error_message = error_msg
                        guidance = 'Frame only the leaf, avoid background objects.'
                    else:
                        error_type = 'low_confidence'
                        error_message = error_msg
                        guidance = 'Try to take a closer, well‑lit photo.'
                else:
                    error_type = 'low_confidence'
                    error_message = f'I\'m not sure about this image (confidence: {confidence:.0%}).'
                    guidance = 'Try to take a closer, well‑lit photo.'

                with open(filepath, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                image_preview = f"data:image/jpeg;base64,{image_data}"
                os.remove(filepath)

                return render_template(
                    'scan.html',
                    barangays=barangays,
                    user_barangay=user_barangay,
                    error=error_message,
                    error_type=error_type,
                    guidance=guidance,
                    image_preview=image_preview
                )

            # ---- Store scan ----
            cur = mysql.connection.cursor()
            cur.execute("SELECT id FROM diseases WHERE LOWER(TRIM(name_en)) = LOWER(TRIM(%s))", (disease_name,))
            disease = cur.fetchone()
            disease_id = disease['id'] if disease else None

            location = request.form.get('location', 'Unknown')
            lat_str = request.form.get('latitude')
            lon_str = request.form.get('longitude')
            try:
                latitude = float(lat_str) if lat_str else None
            except (ValueError, TypeError):
                latitude = None
            try:
                longitude = float(lon_str) if lon_str else None
            except (ValueError, TypeError):
                longitude = None

            utc_now = datetime.now(UTC_TZ)

            cur.execute("""
                INSERT INTO scans (
                    user_id, disease_id, image_path, confidence,
                    predicted_name, location, latitude, longitude, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (session['user_id'], disease_id, filepath, confidence,
                  disease_name, location, latitude, longitude, utc_now))
            mysql.connection.commit()
            scan_id = cur.lastrowid

            if location and location != 'Unknown':
                session['barangay'] = location
                cur.execute("UPDATE users SET barangay = %s WHERE id = %s", (location, session['user_id']))
                mysql.connection.commit()
            cur.close()

            flash('✅ Scan completed successfully!')
            return redirect(url_for('result', scan_id=scan_id))

        except Exception as e:
            print(f"Scan processing error: {e}")
            flash('An error occurred during scan processing. Please try again.', 'danger')
            return redirect(url_for('scan'))

    return render_template('scan.html', barangays=barangays, user_barangay=user_barangay,
                           error=None, error_type=None, guidance=None, image_preview=None)

# ---- API endpoints ----
@app.route('/api/barangays')
def api_barangays():
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT name, latitude, longitude FROM barangay_coordinates ORDER BY name")
        barangays = cur.fetchall()
        cur.close()
        return jsonify(barangays)
    except Exception as e:
        print(f"API barangays error: {e}")
        return jsonify([])

@app.route('/api/barangay-coordinates/<barangay_name>')
def get_barangay_coordinates(barangay_name):
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT latitude, longitude FROM barangay_coordinates WHERE name = %s", (barangay_name,))
        result = cur.fetchone()
        cur.close()
        if result:
            return jsonify({'latitude': result['latitude'], 'longitude': result['longitude']})
        return jsonify({'error': 'Barangay not found'}), 404
    except Exception as e:
        print(f"API coordinates error: {e}")
        return jsonify({'error': 'Server error'}), 500

@app.route('/api/scans')
def api_scans():
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT id, user_id, disease_id, confidence, predicted_name, location, latitude, longitude, created_at FROM scans")
        scans = cur.fetchall()
        cur.close()

        result = []
        for s in scans:
            if s['created_at']:
                local_dt = utc_to_local(s['created_at'])
                iso_str = local_dt.isoformat() if local_dt else None
            else:
                iso_str = None

            result.append({
                'id': s['id'],
                'confidence': s['confidence'],
                'predicted_name': s['predicted_name'],
                'location': s['location'],
                'latitude': s['latitude'],
                'longitude': s['longitude'],
                'created_at': iso_str,
            })
        return jsonify(result)
    except Exception as e:
        print(f"API scans error: {e}")
        return jsonify([])

# ---- Result ----
@app.route('/result/<int:scan_id>')
def result(scan_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT s.*,
                   d.name_en as disease_name,
                   d.description_en as description,
                   d.symptoms_en as symptoms,
                   d.causes_en as causes,
                   d.prevention_en as prevention,
                   d.treatment_en as treatment
            FROM scans s
            LEFT JOIN diseases d ON s.disease_id = d.id
            WHERE s.id = %s AND s.user_id = %s
        """, (scan_id, session['user_id']))
        scan = cur.fetchone()
        cur.close()

        if not scan:
            flash('Scan not found.')
            return redirect(url_for('dashboard'))

        if scan.get('created_at'):
            scan['created_at'] = utc_to_local(scan['created_at'])

        images = []
        if scan.get('disease_id'):
            images = get_disease_images(scan['disease_id'])

        if session.get('language') == 'tl':
            for field in ['description', 'symptoms', 'causes', 'prevention', 'treatment']:
                if scan.get(field):
                    scan[field] = translate_with_cache(scan[field] or '')

        return render_template('result.html', scan=scan, images=images)
    except Exception as e:
        print(f"Result error: {e}")
        flash('Unable to load scan result.', 'danger')
        return redirect(url_for('dashboard'))

# ---- History ----
@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    view = request.args.get('view', 'recent')
    scans = []
    recent_count = 0
    archived_count = 0

    try:
        cur = mysql.connection.cursor()

        cur.execute("""
            SELECT COUNT(*) as count
            FROM scans
            WHERE user_id = %s AND created_at >= DATE_SUB(NOW(), INTERVAL 90 DAY)
        """, (session['user_id'],))
        recent_count = cur.fetchone()['count']

        cur.execute("""
            SELECT COUNT(*) as count
            FROM scans
            WHERE user_id = %s AND created_at < DATE_SUB(NOW(), INTERVAL 90 DAY)
        """, (session['user_id'],))
        archived_count = cur.fetchone()['count']

        if view == 'archive':
            query = """
                SELECT s.*, d.name_en as disease_name
                FROM scans s
                LEFT JOIN diseases d ON s.disease_id = d.id
                WHERE s.user_id = %s AND s.created_at < DATE_SUB(NOW(), INTERVAL 90 DAY)
                ORDER BY s.created_at DESC
            """
        else:
            query = """
                SELECT s.*, d.name_en as disease_name
                FROM scans s
                LEFT JOIN diseases d ON s.disease_id = d.id
                WHERE s.user_id = %s AND s.created_at >= DATE_SUB(NOW(), INTERVAL 90 DAY)
                ORDER BY s.created_at DESC
            """

        cur.execute(query, (session['user_id'],))
        scans = cur.fetchall()
        cur.close()

        for scan in scans:
            if scan.get('created_at'):
                scan['created_at'] = utc_to_local(scan['created_at'])

        today = get_local_today()

        if session.get('language') == 'tl':
            for scan in scans:
                scan['disease_name'] = translate_with_cache(scan['disease_name'] or '')

        return render_template('history.html',
                               scans=scans,
                               view=view,
                               recent_count=recent_count,
                               archived_count=archived_count,
                               today=today)
    except Exception as e:
        print(f"History error: {e}")
        flash('Unable to load scan history.', 'danger')
        return render_template('history.html', scans=[], view=view, recent_count=0, archived_count=0, today=get_local_today())

# ---- Delete scan ----
@app.route('/scan/delete/<int:scan_id>', methods=['POST'])
def delete_scan(scan_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT image_path FROM scans WHERE id = %s AND user_id = %s", (scan_id, session['user_id']))
        scan = cur.fetchone()

        if scan:
            if scan['image_path']:
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], scan['image_path'].replace('static/uploads/', ''))
                if os.path.exists(image_path):
                    os.remove(image_path)
            cur.execute("DELETE FROM scans WHERE id = %s", (scan_id,))
            mysql.connection.commit()
            flash('Scan deleted.', 'success')
        else:
            flash('Scan not found or permission denied.', 'danger')

        cur.close()
    except Exception as e:
        print(f"Delete scan error: {e}")
        flash('An error occurred while deleting the scan.', 'danger')

    return redirect(request.referrer or url_for('history'))

# ---- Delete all archived scans ----
@app.route('/scan/delete-archived', methods=['POST'])
def delete_all_archived():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            DELETE FROM scans
            WHERE user_id = %s AND created_at < DATE_SUB(NOW(), INTERVAL 90 DAY)
        """, (session['user_id'],))
        deleted = cur.rowcount
        mysql.connection.commit()
        cur.close()
        flash(f'Deleted {deleted} archived scans.', 'success')
    except Exception as e:
        print(f"Delete archived error: {e}")
        flash('An error occurred while deleting archived scans.', 'danger')

    return redirect(url_for('history', view='archive'))

# ---- Weather ----
@app.route('/weather')
def weather():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    display_location = None

    try:
        if lat is not None and lon is not None:
            try:
                nominatim_url = "https://nominatim.openstreetmap.org/reverse"
                params = {'lat': lat, 'lon': lon, 'format': 'json', 'zoom': 18, 'accept-language': 'en'}
                headers = {'User-Agent': 'AGAP-PALAY/1.0'}
                response = requests.get(nominatim_url, params=params, headers=headers, timeout=5)
                data = response.json()
                if data and 'address' in data:
                    address = data['address']
                    barangay = (address.get('village') or address.get('suburb') or
                                address.get('neighbourhood') or address.get('city_district'))
                    city = address.get('city') or address.get('town') or address.get('municipality')
                    if barangay and city:
                        display_location = f"Barangay {barangay}, {city}"
                    elif barangay:
                        display_location = f"Barangay {barangay}"
                    elif city:
                        display_location = city
                    else:
                        display_location = "Alaminos City"
                else:
                    display_location = "Alaminos City"
            except Exception as e:
                print(f"Reverse geocoding error: {e}")
                cur = mysql.connection.cursor()
                cur.execute("SELECT barangay FROM users WHERE id = %s", (session['user_id'],))
                user = cur.fetchone()
                cur.close()
                user_barangay = user['barangay'] if user and user['barangay'] else None
                display_location = f"Barangay {user_barangay}, Alaminos City" if user_barangay else "Alaminos City"
        else:
            cur = mysql.connection.cursor()
            cur.execute("SELECT barangay FROM users WHERE id = %s", (session['user_id'],))
            user = cur.fetchone()
            cur.close()
            user_barangay = user['barangay'] if user and user['barangay'] else None
            display_location = f"Barangay {user_barangay}, Alaminos City" if user_barangay else "Alaminos City"
            lat = 16.1558
            lon = 119.9808

        forecast_url = "http://api.openweathermap.org/data/2.5/forecast"
        forecast_params = {
            'lat': lat,
            'lon': lon,
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric'
        }
        response = requests.get(forecast_url, params=forecast_params, timeout=5)
        data = response.json()
        if data.get('cod') != '200':
            flash(f"Forecast API error: {data.get('message', 'Unknown error')}", 'danger')
            return redirect(url_for('dashboard'))

        daily = defaultdict(list)
        for entry in data['list']:
            date_str = entry['dt_txt'].split()[0]
            daily[date_str].append(entry)

        today = datetime.now(LOCAL_TZ).date()
        forecast_data = []

        for date_str in sorted(daily.keys()):
            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
            if date_obj <= today:
                continue
            entries = daily[date_str]
            temps = [e['main']['temp'] for e in entries]
            min_temp = min(temps)
            max_temp = max(temps)
            weathers = [e['weather'][0]['main'] for e in entries]
            most_common_weather = Counter(weathers).most_common(1)[0][0]
            icon = entries[0]['weather'][0]['icon']
            day_name = date_obj.strftime('%A')
            forecast_data.append({
                'date': date_str,
                'day': day_name,
                'min_temp': min_temp,
                'max_temp': max_temp,
                'weather': most_common_weather,
                'icon': icon
            })
            if len(forecast_data) >= 5:
                break

        first_day_key = list(daily.keys())[0]
        current = daily[first_day_key][0]
        temp = current['main']['temp']
        humidity = current['main']['humidity']
        rainfall = current.get('rain', {}).get('3h', 0)
        condition = current['weather'][0]['description']
        condition_icon = current['weather'][0]['icon']

        risk = 'high' if humidity > 80 else ('moderate' if humidity > 60 else 'low')

        try:
            cur = mysql.connection.cursor()
            cur.execute("""
                INSERT INTO weather_logs (user_id, temperature, humidity, rainfall, latitude, longitude, city, risk_level, weather_condition)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (session['user_id'], temp, humidity, rainfall, lat, lon, display_location, risk, condition))
            mysql.connection.commit()
            cur.close()
        except Exception as e:
            print(f"Weather log error: {e}")

        generate_weather_advisory(temp, humidity, rainfall, condition, user_id=session['user_id'])

        return render_template('weather.html',
                               display_location=display_location,
                               forecast=forecast_data,
                               temp=temp,
                               humidity=humidity,
                               rainfall=rainfall,
                               risk=risk,
                               condition=condition,
                               condition_icon=condition_icon,
                               lat=lat,
                               lon=lon)
    except Exception as e:
        print(f"Weather route error: {e}")
        flash('Unable to fetch weather data. Please try again later.', 'danger')
        return redirect(url_for('dashboard'))

# ---- Disease Library ----
@app.route('/diseases')
def disease_list():
    page = request.args.get('page', 1, type=int)
    per_page = 12
    offset = (page - 1) * per_page
    search = request.args.get('q', '')
    category = request.args.get('category', '')

    try:
        cur = mysql.connection.cursor()

        count_query = """
            SELECT COUNT(*) as total
            FROM diseases d
            LEFT JOIN categories c ON d.category_id = c.id
            WHERE d.status = 'published'
        """
        count_params = []
        if search:
            count_query += " AND (d.name_en LIKE %s OR d.description_en LIKE %s)"
            count_params.extend([f'%{search}%', f'%{search}%'])
        if category:
            count_query += " AND c.category_name = %s"
            count_params.append(category)

        cur.execute(count_query, count_params)
        total = cur.fetchone()['total']

        query = """
            SELECT
                d.id,
                d.name_en as name,
                d.description_en as description,
                c.category_name,
                (SELECT image_path FROM disease_images
                 WHERE disease_id = d.id AND is_primary = TRUE
                 ORDER BY display_order LIMIT 1) AS primary_image,
                (SELECT GROUP_CONCAT(image_path ORDER BY display_order SEPARATOR ',')
                 FROM disease_images
                 WHERE disease_id = d.id
                 LIMIT 3) AS all_images
            FROM diseases d
            LEFT JOIN categories c ON d.category_id = c.id
            WHERE d.status = 'published'
        """
        params = []
        if search:
            query += " AND (d.name_en LIKE %s OR d.description_en LIKE %s)"
            params.extend([f'%{search}%', f'%{search}%'])
        if category:
            query += " AND c.category_name = %s"
            params.append(category)

        query += " ORDER BY d.created_at DESC LIMIT %s OFFSET %s"
        params.extend([per_page, offset])

        cur.execute(query, params)
        diseases = cur.fetchall()

        for d in diseases:
            if d.get('all_images'):
                d['images'] = d['all_images'].split(',')
            else:
                d['images'] = []

        cur.execute("SELECT category_name FROM categories")
        categories = [row['category_name'] for row in cur.fetchall()]
        cur.close()

        if session.get('language') == 'tl':
            for d in diseases:
                d['name'] = translate_with_cache(d['name'] or '')
                d['description'] = translate_with_cache(d['description'] or '')

        total_pages = (total + per_page - 1) // per_page
        return render_template(
            'disease/list.html',
            diseases=diseases,
            categories=categories,
            page=page,
            total_pages=total_pages,
            search=search,
            selected_category=category
        )
    except Exception as e:
        print(f"Disease list error: {e}")
        flash('Unable to load disease library.', 'danger')
        return render_template('disease/list.html', diseases=[], categories=[], page=1, total_pages=0)

# ---- Disease Detail ----
@app.route('/disease/<int:disease_id>')
def disease_detail(disease_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        cur = mysql.connection.cursor()
        cur.execute("""
    SELECT d.id, d.name_en as name, d.description_en as description,
           d.symptoms_en as symptoms, d.causes_en as causes,
           d.prevention_en as prevention, d.treatment_en as treatment,
           c.category_name, d.created_at
        FROM diseases d
        LEFT JOIN categories c ON d.category_id = c.id
        WHERE d.id = %s AND d.status='published'
    """, (disease_id,))
        disease = cur.fetchone()
        cur.close()

        if not disease:
            flash('Disease not found.')
            return redirect(url_for('disease_list'))

        images = get_disease_images(disease_id)

        if session.get('language') == 'tl':
            for field in ['description', 'symptoms', 'causes', 'prevention', 'treatment']:
                if disease.get(field):
                    disease[field] = translate_with_cache(disease[field] or '')

        return render_template('disease/detail.html', disease=disease, images=images)
    except Exception as e:
        print(f"Disease detail error: {e}")
        flash('Unable to load disease details.', 'danger')
        return redirect(url_for('disease_list'))

# ---- Profile ----
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        cur = mysql.connection.cursor()
        if request.method == 'POST':
            first_name = request.form.get('first_name', '').strip()
            middle_name = request.form.get('middle_name', '').strip() or None
            last_name = request.form.get('last_name', '').strip()
            suffix = request.form.get('suffix', '').strip() or None
            email = request.form.get('email', '').strip()
            barangay = request.form.get('barangay')
            language = request.form.get('language_preference', 'en')
            new_password = request.form.get('new_password')
            confirm = request.form.get('confirm_password')

            if not first_name or not last_name or not email:
                flash('First name, last name, and email are required.')
                return redirect(url_for('profile'))

            if new_password and new_password != confirm:
                flash('Passwords do not match.')
                return redirect(url_for('profile'))

            cur.execute("""
                UPDATE users SET
                    first_name=%s, middle_name=%s, last_name=%s, suffix=%s,
                    email=%s, barangay=%s, language_preference=%s
                WHERE id=%s
            """, (first_name, middle_name, last_name, suffix, email, barangay, language, session['user_id']))

            if new_password:
                hashed = generate_password_hash(new_password)
                cur.execute("UPDATE users SET password=%s WHERE id=%s", (hashed, session['user_id']))

            mysql.connection.commit()
            cur.close()

            session['first_name'] = first_name
            session['last_name'] = last_name
            session['language'] = language
            session['barangay'] = barangay
            session.permanent = True
            session.modified = True
            flash('Profile updated successfully.')
            return redirect(url_for('profile'))

        cur.execute("SELECT first_name, middle_name, last_name, suffix, email, barangay, language_preference FROM users WHERE id=%s", (session['user_id'],))
        user = cur.fetchone()
        cur.close()
        return render_template('profile.html', user=user, barangays=BARANGAYS, get_full_name=get_full_name)
    except Exception as e:
        print(f"Profile error: {e}")
        flash('An error occurred while updating profile.', 'danger')
        return redirect(url_for('profile'))

# ============================================================
# ADMIN ROUTES
# ============================================================
@app.route('/admin')
@admin_required
def admin_dashboard():
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT COUNT(*) as total_users FROM users WHERE role='farmer'")
        total_users = cur.fetchone()['total_users']
        cur.execute("SELECT COUNT(*) as total_scans FROM scans")
        total_scans = cur.fetchone()['total_scans']
        cur.execute("SELECT COUNT(*) as total_diseases FROM diseases")
        total_diseases = cur.fetchone()['total_diseases']

        cur.execute("""
            SELECT al.*, u.first_name, u.middle_name, u.last_name, u.suffix FROM activity_logs al
            JOIN users u ON al.user_id = u.id
            ORDER BY al.created_at DESC LIMIT 10
        """)
        logs = cur.fetchall()
        for log in logs:
            if log.get('created_at'):
                log['created_at'] = utc_to_local(log['created_at'])
            log['fullname'] = get_full_name(log)

        cur.execute("""
            SELECT a.*, u.first_name, u.middle_name, u.last_name, u.suffix
            FROM advisories a
            LEFT JOIN users u ON a.user_id = u.id
            WHERE a.created_at = (
                SELECT MAX(created_at)
                FROM advisories a2
                WHERE DATE(a2.created_at) = DATE(a.created_at)
            )
            ORDER BY a.created_at DESC
            LIMIT 10
        """)
        advisories = cur.fetchall()
        for adv in advisories:
            if adv.get('created_at'):
                adv['created_at'] = utc_to_local(adv['created_at'])
            adv['fullname'] = get_full_name(adv) if adv.get('first_name') else "System"

        cur.close()
        return render_template('admin/dashboard.html',
                               total_users=total_users,
                               total_scans=total_scans,
                               total_diseases=total_diseases,
                               logs=logs,
                               advisories=advisories)
    except Exception as e:
        print(f"Admin dashboard error: {e}")
        flash('Unable to load admin dashboard.', 'danger')
        return render_template('admin/dashboard.html', total_users=0, total_scans=0, total_diseases=0, logs=[], advisories=[])

@app.route('/admin/users')
@admin_required
def admin_users():
    try:
        search = request.args.get('search', '')
        barangay_filter = request.args.get('barangay', '')

        cur = mysql.connection.cursor()
        query = """
            SELECT id, first_name, middle_name, last_name, suffix, email, role,
                   barangay, language_preference, is_verified, created_at
            FROM users
            WHERE role = 'farmer'
        """
        params = []
        if search:
            query += " AND (first_name LIKE %s OR last_name LIKE %s OR email LIKE %s)"
            params.extend([f'%{search}%', f'%{search}%', f'%{search}%'])
        if barangay_filter:
            query += " AND barangay = %s"
            params.append(barangay_filter)
        query += " ORDER BY created_at DESC"
        cur.execute(query, params)
        users = cur.fetchall()
        for u in users:
            if u.get('created_at'):
                u['created_at'] = utc_to_local(u['created_at'])
            u['fullname'] = get_full_name(u)
        cur.close()

        return render_template('admin/users.html', users=users, barangays=BARANGAYS,
                               search=search, barangay_filter=barangay_filter)
    except Exception as e:
        print(f"Admin users error: {e}")
        flash('Unable to load user list.', 'danger')
        return render_template('admin/users.html', users=[], barangays=BARANGAYS)

@app.route('/admin/diseases')
@admin_required
def admin_diseases():
    try:
        search = request.args.get('search', '')
        category = request.args.get('category', '')

        cur = mysql.connection.cursor()
        query = """
            SELECT d.*, c.category_name,
           (SELECT image_path FROM disease_images WHERE disease_id = d.id AND is_primary = TRUE LIMIT 1) AS primary_image
          FROM diseases d
           LEFT JOIN categories c ON d.category_id = c.id
            WHERE 1=1
             AND d.name_en NOT IN ('Not Rice Leaf', 'Insect Damage')
         """
        params = []
        if search:
            query += " AND (d.name_en LIKE %s OR d.description_en LIKE %s)"
            params.extend([f'%{search}%', f'%{search}%'])
        if category:
            query += " AND d.category_id = %s"
            params.append(category)
        query += " ORDER BY d.created_at DESC"
        cur.execute(query, params)
        diseases = cur.fetchall()
        for d in diseases:
            if d.get('created_at'):
                d['created_at'] = utc_to_local(d['created_at'])
            d['author_name'] = get_full_name(d) if d.get('first_name') else "Unknown"

        cur.execute("SELECT id, category_name FROM categories ORDER BY category_name")
        categories = cur.fetchall()
        cur.close()

        return render_template('admin/diseases.html', diseases=diseases, categories=categories)
    except Exception as e:
        print(f"Admin diseases error: {e}")
        flash('Unable to load disease list.', 'danger')
        return render_template('admin/diseases.html', diseases=[], categories=[])

# ---- Admin Add Disease ----
@app.route('/admin/disease/add', methods=['GET', 'POST'])
@admin_required
def admin_disease_add():
    if request.method == 'POST':
        try:
            name_en = request.form.get('name_en', '').strip()
            desc_en = request.form.get('description_en', '')
            symptoms_en = request.form.get('symptoms_en', '')
            causes_en = request.form.get('causes_en', '')
            prevention_en = request.form.get('prevention_en', '')
            treatment_en = request.form.get('treatment_en', '')
            category_id = request.form.get('category_id')
            status = request.form.get('status', 'published')

            if not name_en:
                flash('Disease name is required.')
                return redirect(url_for('admin_disease_add'))

            cur = mysql.connection.cursor()
            cur.execute("""
                INSERT INTO diseases (
                 name_en, description_en, symptoms_en, causes_en,
                 prevention_en, treatment_en, category_id, status
              ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (name_en, desc_en, symptoms_en, causes_en,
            prevention_en, treatment_en, category_id, status))
            disease_id = cur.lastrowid

            images = request.files.getlist('images')
            for i, img in enumerate(images[:3]):
                if img and allowed_file(img.filename):
                    ext = img.filename.rsplit('.', 1)[1].lower()
                    filename = f"{uuid.uuid4()}.{ext}"
                    img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    cur.execute("""
                        INSERT INTO disease_images (disease_id, image_path, is_primary, display_order)
                        VALUES (%s, %s, %s, %s)
                    """, (disease_id, filename, i == 0, i))

            mysql.connection.commit()
            cur.execute("INSERT INTO activity_logs (user_id, action, description) VALUES (%s, %s, %s)",
                        (session['user_id'], 'CREATE', f'Created disease “{name_en}”'))
            mysql.connection.commit()
            cur.close()

            flash('Disease added successfully.')
            return redirect(url_for('admin_diseases'))
        except Exception as e:
            print(f"Add disease error: {e}")
            flash('An error occurred while adding the disease.', 'danger')
            return redirect(url_for('admin_disease_add'))

    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT id, category_name FROM categories")
        categories = cur.fetchall()
        cur.close()
        return render_template('admin/edit_disease.html', disease=None, categories=categories)
    except Exception as e:
        print(f"Add disease GET error: {e}")
        flash('Unable to load form.', 'danger')
        return redirect(url_for('admin_diseases'))

# ---- Admin Edit Disease ----
@app.route('/admin/disease/edit/<int:disease_id>', methods=['GET', 'POST'])
@admin_required
def admin_edit_disease(disease_id):
    try:
        cur = mysql.connection.cursor()

        if request.method == 'POST':
            name_en = request.form.get('name_en', '').strip()
            desc_en = request.form.get('description_en', '')
            symptoms_en = request.form.get('symptoms_en', '')
            causes_en = request.form.get('causes_en', '')
            prevention_en = request.form.get('prevention_en', '')
            treatment_en = request.form.get('treatment_en', '')
            category_id = request.form.get('category_id')
            if category_id == '':
                category_id = None
            else:
                category_id = int(category_id)
            status = request.form.get('status', 'published')

            cur.execute("""
                UPDATE diseases SET
                    name_en=%s, description_en=%s, symptoms_en=%s, causes_en=%s,
                    prevention_en=%s, treatment_en=%s, category_id=%s, status=%s
                WHERE id=%s
            """, (name_en, desc_en, symptoms_en, causes_en,
                  prevention_en, treatment_en, category_id, status, disease_id))

            new_images = request.files.getlist('images')
            cur.execute("SELECT COALESCE(MAX(display_order), -1) + 1 as next_order FROM disease_images WHERE disease_id = %s", (disease_id,))
            next_order = cur.fetchone()['next_order']

            for i, img in enumerate(new_images[:3]):
                if img and allowed_file(img.filename):
                    ext = img.filename.rsplit('.', 1)[1].lower()
                    filename = f"{uuid.uuid4()}.{ext}"
                    img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    cur.execute("""
                        INSERT INTO disease_images (disease_id, image_path, is_primary, display_order)
                        VALUES (%s, %s, %s, %s)
                    """, (disease_id, filename, False, next_order + i))

            primary_id = request.form.get('primary_image')
            if primary_id:
                cur.execute("UPDATE disease_images SET is_primary = FALSE WHERE disease_id = %s", (disease_id,))
                cur.execute("UPDATE disease_images SET is_primary = TRUE WHERE id = %s", (primary_id,))

            delete_ids = request.form.getlist('delete_images')
            for img_id in delete_ids:
                cur.execute("SELECT image_path FROM disease_images WHERE id = %s", (img_id,))
                img = cur.fetchone()
                if img:
                    path = os.path.join(app.config['UPLOAD_FOLDER'], img['image_path'])
                    if os.path.exists(path):
                        os.remove(path)
                    cur.execute("DELETE FROM disease_images WHERE id = %s", (img_id,))

            mysql.connection.commit()
            cur.execute("INSERT INTO activity_logs (user_id, action, description) VALUES (%s, %s, %s)",
                        (session['user_id'], 'UPDATE', f'Updated disease “{name_en}”'))
            mysql.connection.commit()
            cur.close()

            flash('Disease updated successfully.')
            return redirect(url_for('admin_diseases'))

        cur.execute("SELECT * FROM diseases WHERE id=%s", (disease_id,))
        disease = cur.fetchone()
        if not disease:
            flash('Disease not found.')
            return redirect(url_for('admin_diseases'))

        cur.execute("SELECT id, category_name FROM categories")
        categories = cur.fetchall()
        cur.execute("SELECT * FROM disease_images WHERE disease_id = %s ORDER BY display_order", (disease_id,))
        images = cur.fetchall()
        cur.close()

        return render_template('admin/edit_disease.html', disease=disease, categories=categories, images=images)
    except Exception as e:
        print(f"Edit disease error: {e}")
        flash('An error occurred while editing the disease.', 'danger')
        return redirect(url_for('admin_diseases'))

# ---- Admin Delete Disease ----
@app.route('/admin/disease/delete/<int:disease_id>', methods=['POST'])
@admin_required
def admin_disease_delete(disease_id):
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT name_en FROM diseases WHERE id=%s", (disease_id,))
        disease = cur.fetchone()
        if disease:
            name = disease['name_en']
            cur.execute("SELECT image_path FROM disease_images WHERE disease_id = %s", (disease_id,))
            images = cur.fetchall()
            for img in images:
                path = os.path.join(app.config['UPLOAD_FOLDER'], img['image_path'])
                if os.path.exists(path):
                    os.remove(path)
            cur.execute("DELETE FROM diseases WHERE id=%s", (disease_id,))
            mysql.connection.commit()
            cur.execute("INSERT INTO activity_logs (user_id, action, description) VALUES (%s, %s, %s)",
                        (session['user_id'], 'DELETE', f'Deleted disease “{name}”'))
            mysql.connection.commit()
            flash('Disease deleted.')
        else:
            flash('Disease not found.')
        cur.close()
    except Exception as e:
        print(f"Delete disease error: {e}")
        flash('An error occurred while deleting the disease.', 'danger')

    return redirect(url_for('admin_diseases'))

# ---- Admin Reports ----
@app.route('/admin/reports')
@admin_required
def admin_reports():
    try:
        cur = mysql.connection.cursor()

        cur.execute("""
            SELECT d.name_en, COALESCE(COUNT(s.id), 0) as scan_count
            FROM diseases d
            LEFT JOIN scans s ON d.id = s.disease_id
            GROUP BY d.id
            ORDER BY scan_count DESC
        """)
        disease_stats = cur.fetchall()
        disease_labels = [row['name_en'] for row in disease_stats]
        disease_counts = [row['scan_count'] for row in disease_stats]

        cur.execute("SELECT COUNT(*) as total FROM scans")
        total_scans = cur.fetchone()['total']

        cur.execute("SELECT COUNT(DISTINCT user_id) as count FROM scans")
        unique_farmers = cur.fetchone()['count'] or 0

        cur.execute("SELECT COUNT(DISTINCT disease_id) as count FROM scans WHERE disease_id IS NOT NULL")
        diseases_detected = cur.fetchone()['count'] or 0

        cur.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN latitude IS NOT NULL AND longitude IS NOT NULL THEN 1 ELSE 0 END) as geotagged
            FROM scans
        """)
        location_data_raw = cur.fetchone()
        total = location_data_raw['total'] or 1
        geotagged = location_data_raw['geotagged'] or 0
        geotagged_percent = round((geotagged / total) * 100, 1)
        location_data = [geotagged, total - geotagged]

        cur.execute("""
            SELECT DATE(created_at) as date, COUNT(*) as count
            FROM scans
            WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
            GROUP BY DATE(created_at)
            ORDER BY date ASC
        """)
        trend_data_raw = cur.fetchall()
        trend_labels = [row['date'].strftime('%m/%d') for row in trend_data_raw]
        trend_data = [row['count'] for row in trend_data_raw]

        cur.execute("""
            SELECT s.location AS barangay, COUNT(s.id) as scan_count
            FROM scans s
            WHERE s.location IS NOT NULL AND s.location != ''
            GROUP BY s.location
            ORDER BY scan_count DESC
            LIMIT 10
        """)
        barangay_stats = cur.fetchall()
        barangay_labels = [row['barangay'] for row in barangay_stats]
        barangay_counts = [row['scan_count'] for row in barangay_stats]

        cur.execute("""
            SELECT u.first_name, u.middle_name, u.last_name, u.suffix, u.barangay, COUNT(s.id) as scan_count
            FROM scans s
            JOIN users u ON s.user_id = u.id
            GROUP BY s.user_id
            ORDER BY scan_count DESC
            LIMIT 5
        """)
        top_farmers = cur.fetchall()
        for farmer in top_farmers:
            farmer['fullname'] = get_full_name(farmer)

        cur.execute("""
            SELECT d.name_en, AVG(s.confidence) as avg_confidence
            FROM scans s
            JOIN diseases d ON s.disease_id = d.id
            WHERE s.confidence IS NOT NULL
            GROUP BY d.id
            ORDER BY avg_confidence DESC
        """)
        confidence_stats = cur.fetchall()

        cur.execute("""
            SELECT 
                location AS barangay,
                COUNT(DISTINCT CONCAT(user_id, DATE(created_at))) AS total_reports,
                COUNT(DISTINCT user_id) AS unique_farmers
            FROM scans
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
              AND confidence >= 0.85
              AND predicted_name != 'Healthy'
              AND disease_id IS NOT NULL
              AND location IS NOT NULL
            GROUP BY location
            ORDER BY total_reports DESC
        """)
        barangay_risk_summary = cur.fetchall()

        cur.execute("""
            SELECT 
                location AS barangay,
                predicted_name AS disease,
                COUNT(DISTINCT CONCAT(user_id, DATE(created_at))) AS report_count,
                COUNT(DISTINCT user_id) AS unique_farmers
            FROM scans
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
              AND confidence >= 0.85
              AND predicted_name != 'Healthy'
              AND disease_id IS NOT NULL
              AND location IS NOT NULL
            GROUP BY location, predicted_name
        """)
        barangay_disease_breakdown = cur.fetchall()

        cur.close()

        return render_template(
            'admin/reports.html',
            disease_stats=disease_stats,
            disease_labels=disease_labels,
            disease_counts=disease_counts,
            total_scans=total_scans,
            unique_farmers=unique_farmers,
            diseases_detected=diseases_detected,
            geotagged_percent=geotagged_percent,
            location_data=location_data,
            trend_labels=trend_labels,
            trend_data=trend_data,
            barangay_labels=barangay_labels,
            barangay_counts=barangay_counts,
            top_farmers=top_farmers,
            confidence_stats=confidence_stats,
            barangay_risk_summary=barangay_risk_summary,
            barangay_disease_breakdown=barangay_disease_breakdown
        )
    except Exception as e:
        print(f"Admin reports error: {e}")
        flash('Unable to load reports.', 'danger')
        return redirect(url_for('admin_dashboard'))

# ---- Admin Edit User ----
@app.route('/admin/user/edit/<int:user_id>', methods=['GET', 'POST'])
@admin_required
def admin_edit_user(user_id):
    try:
        cur = mysql.connection.cursor()

        if request.method == 'POST':
            barangay = request.form.get('barangay')
            language = request.form.get('language_preference', 'en')
            is_verified = int(request.form.get('is_verified', 0))  # 0 or 1

            cur.execute("""
                UPDATE users SET
                    barangay = %s,
                    language_preference = %s,
                    is_verified = %s
                WHERE id = %s
            """, (barangay, language, is_verified, user_id))

            mysql.connection.commit()
            cur.execute("""
                INSERT INTO activity_logs (user_id, action, description)
                VALUES (%s, %s, %s)
            """, (session['user_id'], 'UPDATE', f'Updated user ID {user_id} (verified: {is_verified})'))
            mysql.connection.commit()
            cur.close()

            flash('User updated successfully.')
            return redirect(url_for('admin_users'))

        # GET request – fetch user data (including is_verified)
        cur.execute("""
            SELECT id, first_name, middle_name, last_name, suffix, email, role,
                   language_preference, barangay, is_verified
            FROM users WHERE id = %s
        """, (user_id,))
        user = cur.fetchone()
        cur.close()

        if not user:
            flash('User not found.')
            return redirect(url_for('admin_users'))

        return render_template('admin/edit_user.html', user=user, barangays=BARANGAYS)
    except Exception as e:
        print(f"Edit user error: {e}")
        flash('An error occurred while editing the user.', 'danger')
        return redirect(url_for('admin_users'))

# ---- Admin Delete User ----
@app.route('/admin/user/delete/<int:user_id>', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    if user_id == session['user_id']:
        flash('You cannot delete your own account.')
        return redirect(url_for('admin_users'))

    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT first_name, last_name FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()
        if user:
            name = f"{user['first_name']} {user['last_name']}".strip()
            cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
            mysql.connection.commit()
            cur.execute("""
                INSERT INTO activity_logs (user_id, action, description)
                VALUES (%s, %s, %s)
            """, (session['user_id'], 'DELETE', f'Deleted user “{name}”'))
            mysql.connection.commit()
            flash('User deleted successfully.')
        else:
            flash('User not found.')
        cur.close()
    except Exception as e:
        print(f"Delete user error: {e}")
        flash('An error occurred while deleting the user.', 'danger')

    return redirect(url_for('admin_users'))

# ---- Admin Map ----
@app.route('/admin/map')
@admin_required
def admin_map():
    try:
        cur = mysql.connection.cursor()

        cur.execute("SELECT id, name_en FROM diseases ORDER BY name_en")
        diseases = cur.fetchall()

        
        barangays = sorted(BARANGAYS)

        query = """
            SELECT 
                s.id, 
                s.latitude, 
                s.longitude, 
                s.predicted_name, 
                s.confidence,
                s.created_at,
                s.location AS scan_location,
                u.first_name, u.middle_name, u.last_name, u.suffix,
                u.barangay AS user_barangay,
                d.name_en AS disease_name,
                d.id AS disease_id
            FROM scans s
            JOIN users u ON s.user_id = u.id
            LEFT JOIN diseases d ON s.disease_id = d.id
            WHERE s.latitude IS NOT NULL AND s.longitude IS NOT NULL
        """
        params = []
        disease_filter = request.args.get('disease', '')
        barangay_filter = request.args.get('barangay', '')

        if disease_filter:
            query += " AND s.disease_id = %s"
            params.append(disease_filter)
        if barangay_filter:
            query += " AND s.location = %s"
            params.append(barangay_filter)

        query += " ORDER BY s.created_at DESC"
        cur.execute(query, params)
        scans = cur.fetchall()
        for scan in scans:
            if scan.get('created_at'):
                scan['created_at'] = utc_to_local(scan['created_at'])
            scan['user_name'] = get_full_name(scan)

        cur.execute("""
            SELECT 
                location AS barangay,
                COUNT(DISTINCT CONCAT(user_id, DATE(created_at))) AS total_reports,
                COUNT(DISTINCT user_id) AS unique_farmers
            FROM scans
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
              AND confidence >= 0.85
              AND predicted_name != 'Healthy'
              AND disease_id IS NOT NULL
              AND location IS NOT NULL
            GROUP BY location
        """)
        risk_summary = cur.fetchall()

        cur.execute("""
            SELECT 
                location AS barangay,
                predicted_name AS disease,
                COUNT(DISTINCT CONCAT(user_id, DATE(created_at))) AS report_count,
                COUNT(DISTINCT user_id) AS unique_farmers
            FROM scans
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
              AND confidence >= 0.85
              AND predicted_name != 'Healthy'
              AND disease_id IS NOT NULL
              AND location IS NOT NULL
            GROUP BY location, predicted_name
        """)
        disease_risks = cur.fetchall()

        cur.close()

        return render_template(
            'admin/map.html',
            scans=scans,
            diseases=diseases,
            barangays=barangays,
            selected_disease=disease_filter,
            selected_barangay=barangay_filter,
            risk_summary=risk_summary,
            disease_risks=disease_risks
        )
    except Exception as e:
        print(f"Admin map error: {e}")
        flash('Unable to load admin map.', 'danger')
        return redirect(url_for('admin_dashboard'))

# ============================================================
# CONTEXT PROCESSOR
# ============================================================
@app.context_processor
def utility_processor():
    return dict(get_risk_level_from_counts=get_risk_level_from_counts)

@app.context_processor
def inject_user():
    """Make current user's full name available globally."""
    if 'user_id' in session:
        first = session.get('first_name', '')
        last = session.get('last_name', '')
        full = f"{first} {last}".strip()
        return {'current_user_fullname': full or 'User'}
    return {'current_user_fullname': None}
# ============================================================
# API ENDPOINTS
# ============================================================
@app.route('/api/recent-advisories')
def api_recent_advisories():
    if 'user_id' not in session:
        return jsonify({'advisories': []})
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT title, message, severity, created_at
            FROM advisories
            WHERE (user_id IS NULL OR user_id = %s) AND is_read = 0
            ORDER BY created_at DESC
            LIMIT 1
        """, (session['user_id'],))
        advisories = cur.fetchall()
        cur.close()
        return jsonify({'advisories': advisories})
    except Exception as e:
        print(f"API advisories error: {e}")
        return jsonify({'advisories': []})

# ============================================================
# DEBUG ROUTES
# ============================================================
@app.route('/debug-session')
def debug_session():
    return dict(session)

@app.route('/debug-model')
def debug_model():
    if model is None:
        return "Model not loaded"
    try:
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        preds = model.predict(dummy, verbose=0)
        return f"Predictions shape: {preds.shape}, number of classes: {preds.shape[1]}"
    except Exception as e:
        return f"Model debug error: {e}"

@app.route('/test-email')
def test_email():
    try:
        msg = Message(
            subject='Test Email from AGAP-PALAY',
            recipients=['agappalay@gmail.com'],
            body='This is a test email to verify SMTP configuration.'
        )
        mail.send(msg)
        return 'Test email sent successfully!'
    except Exception as e:
        return f'Error: {str(e)}'

# ============================================================
# CUSTOM TEMPLATE FILTER
# ============================================================
@app.template_filter('date_label')
def date_label_filter(dt, today):
    if not dt:
        return ''
    if hasattr(dt, 'date'):
        d = dt.date()
    else:
        d = dt
    if d == today:
        return 'Today'
    elif d == today - timedelta(days=1):
        return 'Yesterday'
    else:
        return d.strftime('%B %d, %Y')

# ============================================================
# PRE‑WARM CACHE AND RUN
# ============================================================
with app.app_context():
    prewarm_translation_cache()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)