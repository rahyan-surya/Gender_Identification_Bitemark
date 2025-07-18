from ultralytics import YOLO
import os

data_yaml_path = os.path.abspath('bitemark_data.yaml')

if __name__ == '__main__':

    model = YOLO('yolov8s   .pt') 


    results = model.train(
        data=data_yaml_path,
        epochs=300,
        imgsz=640,
        batch=8, 
        name='yolov8_gender_detect_v1', 
        workers=0 
    )

    print("\nPelatihan deteksi gender selesai!")
    print(f"Hasil disimpan di: runs/detect/{results.save_dir}")