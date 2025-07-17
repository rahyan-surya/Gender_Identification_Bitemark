import os
import shutil
import random

# Path ke folder Anda
base_dir = 'yolostuff' # Pastikan ini sesuai dengan nama folder proyek Anda
images_raw_dir = os.path.join(base_dir, 'images_raw')
labels_raw_dir = os.path.join(base_dir, 'labelimg_raw')

# Buat struktur folder output
output_images_train = os.path.join(base_dir, 'images', 'train')
output_images_val = os.path.join(base_dir, 'images', 'val')
output_labels_train = os.path.join(base_dir, 'labels', 'train')
output_labels_val = os.path.join(base_dir, 'labels', 'val')

# Buat folder jika belum ada
os.makedirs(output_images_train, exist_ok=True)
os.makedirs(output_images_val, exist_ok=True)
os.makedirs(output_labels_train, exist_ok=True)
os.makedirs(output_labels_val, exist_ok=True)

# Dapatkan daftar semua file gambar
image_files = [f for f in os.listdir(images_raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Acak daftar file
random.shuffle(image_files)

# Tentukan rasio split (misal: 80% train, 20% val)
train_split_ratio = 0.8
num_train_images = int(len(image_files) * train_split_ratio)

train_images = image_files[:num_train_images]
val_images = image_files[num_train_images:]

print(f"Total gambar: {len(image_files)}")
print(f"Gambar training: {len(train_images)}")
print(f"Gambar validation: {len(val_images)}")

# Pindahkan gambar dan label ke folder yang sesuai
def move_files(file_list, image_dest_dir, label_dest_dir):
    for img_file in file_list:
        # Pindahkan gambar
        shutil.copy(os.path.join(images_raw_dir, img_file), os.path.join(image_dest_dir, img_file))

        # Pindahkan file label yang sesuai
        label_file_name = os.path.splitext(img_file)[0] + '.txt'
        src_label_path = os.path.join(labels_raw_dir, label_file_name)
        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, os.path.join(label_dest_dir, label_file_name))
        else:
            print(f"Peringatan: File label tidak ditemukan untuk {img_file}")

print("\nMemindahkan file training...")
move_files(train_images, output_images_train, output_labels_train)

print("Memindahkan file validation...")
move_files(val_images, output_images_val, output_labels_val)

print("\nDataset berhasil disiapkan!")