import os
from flask import Flask, request, render_template, redirect, url_for, flash, session, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO
import time
from supabase import create_client, Client
from functools import wraps
from dotenv import load_dotenv
from datetime import datetime # Import modul datetime

# Muat variabel lingkungan dari file .env
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.secret_key = os.environ.get('FLASK_SECRET_KEY') 
UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']

# --- Konfigurasi Supabase ---
SUPABASE_URL = os.environ.get("SUPABASE_URL") 
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") 

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase client berhasil diinisialisasi.")
    except Exception as e:
        print(f"❌ Gagal menginisialisasi Supabase client: {e}. Pastikan URL dan KEY benar.")
else:
    print("❌ Variabel lingkungan SUPABASE_URL atau SUPABASE_KEY tidak diatur. Autentikasi dan database Supabase tidak akan berfungsi.")
    print("Pastikan file .env ada dan berisi FLASK_SECRET_KEY, SUPABASE_URL, SUPABASE_KEY.")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

yolov8_model = None
try:
    yolov8_model_path = 'models/yolov8_gender_detect_v1/weights/best.pt'
    yolov8_model = YOLO(yolov8_model_path)
    print("✅ Model YOLOv8 berhasil dimuat.")
except Exception as e:
    print(f"❌ Gagal memuat model YOLOv8: {e}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Anda perlu login untuk mengakses halaman ini.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def predict_gender_yolov8(image_path):
    if yolov8_model is None:
        return "Error: Model YOLOv8 tidak tersedia.", None, None

    try:
        start_time = time.perf_counter()

        img = cv2.imread(image_path)
        if img is None:
            return "Error: Gagal membaca gambar.", None, None

        results = yolov8_model(img, verbose=False)

        predicted_gender_label = "Tidak Terdeteksi"
        highest_confidence = 0.0

        class_names = yolov8_model.names

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                confidence = confidences[i]
                class_id = class_ids[i]
                
                if confidence > 0.5:
                    current_detected_label = class_names[class_id]
                    
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        if current_detected_label == 'bitemark_pria':
                            predicted_gender_label = 'Pria'
                        elif current_detected_label == 'bitemark_wanita':
                            predicted_gender_label = 'Wanita'
        
        end_time = time.perf_counter()
        prediction_duration = end_time - start_time

        return predicted_gender_label, highest_confidence, prediction_duration

    except Exception as e:
        return f"Error pada prediksi YOLOv8: {str(e)}", None, None

@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if supabase is None:
        flash("Sistem autentikasi tidak aktif. Konfigurasi Supabase belum lengkap.", 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        display_name = request.form['display_name']

        try:
            res = supabase.auth.sign_up(
                {"email": email, "password": password, "data": {"display_name": display_name}} 
            )
            
            print(f"DEBUG (Register): Supabase sign_up response: {res}")

            user_obj_from_res = None
            if res:
                user_obj_from_res = res.user if hasattr(res, 'user') else (res.data.user if hasattr(res.data, 'user') else None)
            
            if user_obj_from_res:
                current_user_response = supabase.auth.get_user() 
                print(f"DEBUG (Register): current_user_response from get_user(): {current_user_response}")
                if current_user_response and hasattr(current_user_response, 'user') and current_user_response.user:
                    session['user'] = current_user_response.user.dict()
                    print(f"DEBUG (Register): Session updated with get_user() data: {session['user']}")
                else:
                    session['user'] = user_obj_from_res.dict() 
                    print(f"DEBUG (Register): Session updated with fallback data: {session['user']}")

                flash('Registrasi berhasil! Silakan periksa email Anda untuk verifikasi.', 'success')
                return redirect(url_for('login'))
            else:
                error_message = 'Registrasi gagal.'
                if res and hasattr(res, 'error') and res.error:
                    error_message = f"Registrasi gagal: {res.error.message}"
                    if "User already registered" in res.error.message or "duplicate key value violates unique constraint" in res.error.message:
                        error_message = "Email sudah terdaftar. Silakan login atau gunakan email lain."
                elif res is None:
                    error_message = "Registrasi gagal: Respon dari server Supabase kosong atau tidak terduga. Periksa koneksi atau kredensial Supabase Anda."
                flash(error_message, 'error')
        except Exception as e:
            flash(f'Terjadi kesalahan saat registrasi: {str(e)}', 'error')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if supabase is None:
        flash("Sistem autentikasi tidak aktif. Konfigurasi Supabase belum lengkap.", 'error')
        return render_template('login.html')

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        try:
            res = supabase.auth.sign_in_with_password({"email": email, "password": password})
            
            print(f"DEBUG (Login): Supabase sign_in response: {res}")

            user_obj_from_res = None
            if res:
                user_obj_from_res = res.user if hasattr(res, 'user') else (res.data.user if hasattr(res.data, 'user') else None)

            if user_obj_from_res:
                current_user_response = supabase.auth.get_user() 
                print(f"DEBUG (Login): current_user_response from get_user(): {current_user_response}")
                if current_user_response and hasattr(current_user_response, 'user') and current_user_response.user:
                    session['user'] = current_user_response.user.dict()
                    print(f"DEBUG (Login): Session updated with get_user() data: {session['user']}")
                    flash('Login berhasil!', 'success')
                    return redirect(url_for('upload_file'))
                else:
                    session['user'] = user_obj_from_res.dict()
                    print(f"DEBUG (Login): Session updated with fallback data: {session['user']}")
                    flash('Login berhasil (tapi info metadata mungkin tertunda).', 'info')
                    return redirect(url_for('upload_file'))
            else:
                flash('Email atau password salah.', 'error')
        except Exception as e:
            flash(f'Terjadi kesalahan saat login: {str(e)}', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """Menangani logout pengguna."""
    if supabase:
        try:
            if 'user' in session:
                supabase.auth.sign_out()
                flash('Anda telah logout.', 'info')
            else:
                flash('Anda tidak sedang login.', 'info')
        except Exception as e:
            flash(f'Gagal logout: {str(e)}', 'error')
    session.pop('user', None) # Hapus user dari sesi Flask
    return redirect(url_for('login'))

# --- Rute Utama Webapp (Membutuhkan Login) ---
@app.route('/', methods=['GET', 'POST'])
@login_required
def upload_file():
    """Menangani unggahan file dan prediksi gender."""
    prediction_result = None
    confidence_score = None
    uploaded_image_url = None
    uploaded_file_name = None
    prediction_time = None
    
    user = session.get('user')
    
    user_metadata = user.get('user_metadata', {})
    user_display_name = user_metadata.get('display_name', user.get('email', 'Tamu'))

    print(f"DEBUG (Main): Session user data: {session.get('user')}") 
    print(f"DEBUG (Main): User metadata from session: {user_metadata}") 
    print(f"DEBUG (Main): Final user_display_name: {user_display_name}") 


    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Tidak ada bagian file.', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('Tidak ada file yang dipilih.', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            
            uploaded_image_url = url_for('serve_uploaded_file', filename=filename) 
            uploaded_file_name = filename

            prediction_result, confidence_score, prediction_time = predict_gender_yolov8(file_path)
            
            if "Error" in prediction_result:
                flash(prediction_result, 'error')
                prediction_result = None
                confidence_score = None
                prediction_time = None
            
            # Simpan Riwayat Prediksi ke Supabase
            if supabase and user and "Error" not in (prediction_result or ""):
                try:
                    user_id = user.get('id')
                    
                    print(f"DEBUG (Supabase Insert): User object from session: {user}")
                    print(f"DEBUG (Supabase Insert): User ID for insert: {user_id}")

                    if user_id:
                        data_to_insert = {
                            "user_id": user_id,
                            "image_url": uploaded_image_url,
                            "method_used": "YOLOv8",
                            "predicted_gender": prediction_result,
                            "confidence": float(f"{confidence_score:.4f}") if confidence_score is not None else None,
                            "prediction_time_sec": round(prediction_time, 4) if prediction_time is not None else None
                        }
                        print(f"DEBUG (Supabase Insert): Data to insert: {data_to_insert}")

                        response = supabase.table("predictions_history").insert(data_to_insert).execute()
                        
                        print(f"DEBUG (Supabase Insert): Type of response: {type(response)}")
                        print(f"DEBUG (Supabase Insert): Dir of response: {dir(response)}")

                        if response and hasattr(response, 'data') and response.data:
                            print(f"Riwayat prediksi disimpan: {response.data}")
                            flash('Prediksi berhasil disimpan ke riwayat!', 'success')
                        elif response and hasattr(response, 'error') and response.error:
                            print(f"Gagal menyimpan riwayat prediksi: {response.error.message}")
                            flash(f'Gagal menyimpan riwayat: {response.error.message}', 'error')
                        else:
                            print("Gagal menyimpan riwayat prediksi: Respon tidak terduga dari Supabase.")
                            print(f"DEBUG (Supabase Insert): Respon lengkap: {response}")
                            flash('Gagal menyimpan riwayat: Respon tidak terduga dari Supabase.', 'error')
                    else:
                        print("User ID tidak ditemukan di sesi. Tidak dapat menyimpan riwayat prediksi.")
                        flash('User ID tidak ditemukan. Tidak dapat menyimpan riwayat prediksi.', 'error')

                except Exception as e:
                    print(f"Error menyimpan riwayat prediksi ke Supabase: {str(e)}")
                    flash(f'Error saat menyimpan riwayat: {str(e)}', 'error')
        else:
            flash('Tipe file tidak diizinkan. Hanya PNG, JPG, JPEG.', 'error')
            return redirect(request.url)

    return render_template('index.html', 
                           prediction_result=prediction_result, 
                           confidence_score=confidence_score,
                           uploaded_image_url=uploaded_image_url,
                           uploaded_file_name=uploaded_file_name,
                           prediction_time=prediction_time,
                           user_display_name=user_display_name)

# --- Rute Riwayat Prediksi ---
@app.route('/history')
@login_required
def history():
    """Menampilkan riwayat prediksi pengguna."""
    history_data = []
    user = session.get('user')
    user_display_name = user.get('user_metadata', {}).get('display_name', user.get('email', 'Tamu'))

    if supabase and user:
        try:
            user_id = user.get('id')
            if user_id:
                # Ambil data riwayat dari Supabase untuk user_id yang sedang login
                # Urutkan berdasarkan created_at terbaru
                response = supabase.table("predictions_history").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
                
                print(f"DEBUG (History): Supabase history response: {response}") # DEBUG
                
                if response and hasattr(response, 'data') and response.data:
                    # Konversi string created_at menjadi objek datetime
                    for record in response.data:
                        if 'created_at' in record and isinstance(record['created_at'], str):
                            # Supabase mengembalikan ISO 8601 string, parse dengan fromisoformat
                            record['created_at'] = datetime.fromisoformat(record['created_at'].replace('Z', '+00:00'))
                    history_data = response.data
                elif response and hasattr(response, 'error') and response.error:
                    flash(f'Gagal memuat riwayat: {response.error.message}', 'error')
                else:
                    flash('Gagal memuat riwayat: Respon tidak terduga dari Supabase.', 'error')
            else:
                flash('User ID tidak ditemukan. Tidak dapat memuat riwayat.', 'error')
        except Exception as e:
            flash(f'Error saat memuat riwayat: {str(e)}', 'error')
    else:
        flash('Sistem database tidak aktif atau Anda belum login.', 'error')

    return render_template('history.html', 
                           history_data=history_data, 
                           user_display_name=user_display_name)


if __name__ == '__main__':
    if os.environ.get('FLASK_ENV') == 'production':
        # Gunicorn atau Waitress akan menjalankan aplikasi ini
        # Tidak perlu app.run() di sini
        pass 
    else:
        # Mode pengembangan lokal
        if not os.environ.get('FLASK_SECRET_KEY'):
            print("\n!!! PERINGATAN: FLASK_SECRET_KEY tidak diatur. Sesi tidak akan aman. Pastikan ada di .env !!!\n")
        if not os.environ.get('SUPABASE_URL') or not os.environ.get('SUPABASE_KEY'):
            print("\n!!! PERINGATAN: Kredensial Supabase tidak dimuat. Fitur DB/Auth tidak akan berfungsi. Pastikan ada di .env !!!\n")
        app.run(debug=True, host='0.0.0.0')