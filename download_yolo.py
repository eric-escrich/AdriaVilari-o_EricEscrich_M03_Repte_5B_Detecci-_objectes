import os
import requests

def download_file(url, path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print(f"✅ Descarregat: {path}")
    else:
        print(f"❌ Error descarregant {url}")

# Directori YOLO
yolo_dir = "yolo"
os.makedirs(yolo_dir, exist_ok=True)

# URLs dels fitxers
files = {
    "yolov3.cfg": "https://opencv-tutorial.readthedocs.io/en/latest/_downloads/10e685aad953495a95c17bfecd1649e5/yolov3.cfg",
    "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
    "coco.names": "https://opencv-tutorial.readthedocs.io/en/latest/_downloads/a9fb13cbea0745f3d11da9017d1b8467/coco.names",
}

# Descarregar fitxers
for filename, url in files.items():
    download_file(url, os.path.join(yolo_dir, filename))

print("🎉 Tot descarregat correctament!")
