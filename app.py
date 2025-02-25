from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

# Configurar Flask
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
YOLO_FOLDER = "yolo"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

# Carregar YOLO
weights_path = os.path.join(YOLO_FOLDER, "yolov3.weights")
config_path = os.path.join(YOLO_FOLDER, "yolov3.cfg")
labels_path = os.path.join(YOLO_FOLDER, "coco.names")

with open(labels_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

def detect_objects(image_path):
    """Funció per detectar objectes amb YOLO"""
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Convertir imatge per YOLO
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Obtenir noms de les capes de sortida
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Passar la imatge per la xarxa
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    # Analitzar la sortida
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                box = detection[:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar supressió de no màxims
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Dibuixar resultats
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    processed_path = os.path.join(PROCESSED_FOLDER, os.path.basename(image_path))
    cv2.imwrite(processed_path, image)

    return processed_path

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return "No image uploaded", 400

        file = request.files["image"]
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        processed_path = detect_objects(filepath)

        return render_template("index.html", uploaded_image=filename, processed_image=os.path.basename(processed_path))

    return render_template("index.html")

@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/static/processed/<filename>")
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
