from ultralytics import YOLO
import os

# Tentukan path ke file data.yaml Anda yang baru (untuk gender)
# Pastikan file ini ada di root folder proyek_bitemark_yolo Anda
data_yaml_path = os.path.abspath('bitemark_data.yaml')

# Tambahkan blok ini untuk multiprocessing di Windows
if __name__ == '__main__':
    # Pilih model YOLOv8.
    # Mulai dengan 'yolov8n.pt' (nano) atau 'yolov8s.pt' (small)
    # Jika Anda telah melatih model sebelumnya dan ingin melanjutkan,
    # Anda bisa memuat weights dari model yang sudah dilatih (misal: 'runs/detect/yolov8_bitemark_v127/weights/best.pt')
    # model = YOLO('yolov8n.pt') # Untuk melatih dari awal dengan weights default
    model = YOLO('yolov8s   .pt') # Contoh menggunakan model small untuk akurasi yang lebih baik

    # Latih model
    # data: path ke file .yaml dataset Anda (yang baru)
    # epochs: jumlah iterasi pelatihan. Mulai dengan 100-300, bisa ditingkatkan jika perlu.
    # imgsz: ukuran gambar input untuk model. Umumnya 640.
    # batch: jumlah gambar yang diproses sekaligus. Sesuaikan dengan memori GPU Anda.
    # name: nama folder untuk menyimpan hasil pelatihan (weights, plots, dll.)
    #       Ubah nama ini agar tidak menimpa hasil pelatihan sebelumnya.
    # workers: Jumlah proses untuk memuat data. Di Windows, kadang perlu 0 untuk menghindari RuntimeError.
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