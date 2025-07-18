from ultralytics import YOLO
import os

# Tentukan path ke file data.yaml Anda yang baru (untuk gender)
# Pastikan file ini ada di root folder proyek_bitemark_yolo Anda
data_yaml_path = os.path.abspath('bitemark_data.yaml')

# Tambahkan blok ini untuk multiprocessing di Windows
if __name__ == '__main__':

    model = YOLO('yolov8s   .pt') # Contoh menggunakan model small untuk akurasi yang lebih baik


    results = model.train(
        data=data_yaml_path,
        epochs=300, # Meningkatkan epochs karena ini tugas yang lebih kompleks
        imgsz=640,
        batch=8, # Mungkin perlu disesuaikan dengan VRAM GPU, jika error coba kurangi
        name='yolov8_gender_detect_v1', # Nama folder output yang baru
        workers=0 # Direkomendasikan untuk Windows jika sering ada DataLoader worker error
    )

    print("\nPelatihan deteksi gender selesai!")
    print(f"Hasil disimpan di: runs/detect/{results.save_dir}")