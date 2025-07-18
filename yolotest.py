import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import time

def multi_testing_yolov8_gender():
    # --- Konfigurasi ---
    # Path ke model YOLOv8 yang sudah Anda latih
    # Pastikan ini adalah path yang benar ke file best.pt dari pelatihan gender Anda
    YOLO_MODEL_PATH = 'models/yolov8_gender_detect_v12/weights/best.pt'
    
    # Path ke folder utama yang berisi data uji Anda
    # Ini harus menunjuk ke folder 'images/test' yang berisi subfolder 'bitemark_pria' dan 'bitemark_wanita'
    TEST_DATA_ROOT = 'yolostuff/test' 
    
    # Ambang batas confidence untuk deteksi YOLOv8. Deteksi di bawah ini akan diabaikan.
    CONFIDENCE_THRESHOLD = 0.5 # Bisa disesuaikan, misal 0.25 jika ingin melihat lebih banyak deteksi

    try:
        model = YOLO(YOLO_MODEL_PATH)
        print(f"✅ Model YOLOv8 berhasil dimuat dari: {YOLO_MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error memuat model YOLOv8: {str(e)}")
        print("Pastikan PATH ke file 'best.pt' sudah benar dan model telah dilatih.")
        return 0

    jml_uji = 0
    jml_benar = 0
    hasil_prediksi = []

    # Dapatkan nama kelas dari model untuk mapping ID ke nama
    class_names = model.names 

    # Dapatkan daftar subfolder di dalam TEST_DATA_ROOT
    ground_truth_folders = [f for f in os.listdir(TEST_DATA_ROOT) if os.path.isdir(os.path.join(TEST_DATA_ROOT, f))]
    
    if not ground_truth_folders:
        print(f"⚠️ Tidak ditemukan subfolder gender di dalam {TEST_DATA_ROOT}.")
        print("Pastikan struktur folder uji Anda seperti: images/test/bitemark_pria/ & images/test/bitemark_wanita/")
        return 0

    print(f"\nMemulai pengujian pada folder: {TEST_DATA_ROOT}")
    print("--------------------------------------------------")

    for ground_truth_folder_name in ground_truth_folders:
        ground_truth_label = ground_truth_folder_name.lower() 

        current_gender_folder_path = os.path.join(TEST_DATA_ROOT, ground_truth_folder_name)
        
        images_in_folder = [f for f in os.listdir(current_gender_folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"\n--- Memproses folder: {ground_truth_folder_name} ({len(images_in_folder)} gambar) ---")

        for nama_citra in images_in_folder:
            bacacitra_path = os.path.join(current_gender_folder_path, nama_citra)
            img = cv2.imread(bacacitra_path)

            if img is None:
                print(f"❌ Gagal membaca gambar: {bacacitra_path}")
                continue

            try:
                start_time = time.perf_counter()

                # Lakukan inferensi dengan YOLOv8
                results = model(img, imgsz=640, verbose=False) # imgsz=640 harus konsisten dengan ukuran pelatihan

                predicted_gender_label = "Tidak Terdeteksi"
                highest_confidence = 0.0 # Default confidence jika tidak ada deteksi

                # Proses hasil deteksi untuk mendapatkan prediksi gender terbaik
                for r in results:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confidences = r.boxes.conf.cpu().numpy() # Ambil confidence scores
                    class_ids = r.boxes.cls.cpu().numpy().astype(int)

                    for i in range(len(boxes)):
                        confidence = confidences[i] # Confidence dari deteksi ini
                        class_id = class_ids[i]
                        
                        if confidence >= CONFIDENCE_THRESHOLD: # Gunakan >= agar deteksi di ambang batas disertakan
                            current_detected_label = class_names[class_id]
                            
                            # Pilih deteksi dengan confidence tertinggi sebagai hasil prediksi
                            if confidence > highest_confidence:
                                highest_confidence = confidence
                                predicted_gender_label = current_detected_label

                end_time = time.perf_counter()
                durasi = end_time - start_time # Waktu prediksi untuk satu gambar

                jml_uji += 1
                
                # Bandingkan prediksi dengan ground truth
                is_correct = (predicted_gender_label == ground_truth_label)
                
                if is_correct:
                    jml_benar += 1
                    status = "Benar"
                    print(f"✅ {nama_citra} | GT: {ground_truth_label} | Prediksi: {predicted_gender_label} (Conf: {highest_confidence:.2f}) | Waktu: {durasi:.4f}s")
                else:
                    status = "Salah"
                    print(f"❌ {nama_citra} | GT: {ground_truth_label} | Prediksi: {predicted_gender_label} (Conf: {highest_confidence:.2f}) | Waktu: {durasi:.4f}s")

                hasil_prediksi.append({
                    "Nama File": nama_citra,
                    "Ground Truth": ground_truth_label,
                    "Prediksi": predicted_gender_label,
                    "Confidence": round(highest_confidence, 4), # Simpan confidence dengan 4 angka desimal
                    "Benar": status,
                    "Waktu Proses (detik)": round(durasi, 4) # Simpan waktu dengan 4 angka desimal
                })

            except Exception as e:
                print(f"❌ Error saat memproses {bacacitra_path}: {str(e)}")
                hasil_prediksi.append({
                    "Nama File": nama_citra,
                    "Ground Truth": ground_truth_label,
                    "Prediksi": "Error",
                    "Confidence": None, # Jika ada error, confidence tidak tersedia
                    "Benar": "Tidak",
                    "Waktu Proses (detik)": "Error"
                })
                continue

    print("\n-----------------------------------")
    if jml_uji > 0:
        akurasi = (jml_benar / jml_uji) * 100
        print(f"--- Hasil Pengujian Keseluruhan ---")
        print(f"Jumlah Gambar Uji: {jml_uji}")
        print(f"Jumlah Prediksi Benar: {jml_benar}")
        print(f"Akurasi Model: {akurasi:.2f}%")
        print("-----------------------------------")
    else:
        print("⚠️ Tidak ada gambar yang berhasil diproses untuk pengujian.")
        akurasi = 0

    # Simpan hasil prediksi ke file CSV
    df = pd.DataFrame(hasil_prediksi)
    output_csv_path = 'hasil_pengujian_gender_yolov8.csv'
    df.to_csv(output_csv_path, index=False)
    print(f"📄 Hasil pengujian detail disimpan ke '{output_csv_path}'")

    return akurasi

if __name__ == "__main__":
    multi_testing_yolov8_gender()