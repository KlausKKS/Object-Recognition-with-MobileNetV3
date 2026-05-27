# This model allows Object Recognition based on a mobilenetv3.keras model.
# It can save single and sequential pictues and correct results.
# It allows changing the magnification and the measurement of the size of the Objects
import cv2
import numpy as np
import os
import time
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# === Setup ===
os.chdir(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = "mobilenet_model_v3.keras"
CLASSES_CSV = "path_to_your_training_data.csv"
RULES_CSV = "path_to_your_size_file.csv"
IMG_SIZE = (224, 224)

BILD_SAVE_DIR = "bilder"
KORREKTUR_DIR = "korrigierte_daten"
AUFNAHME_DIR = "aufnahmen"
os.makedirs(BILD_SAVE_DIR, exist_ok=True)
os.makedirs(KORREKTUR_DIR, exist_ok=True)
os.makedirs(AUFNAHME_DIR, exist_ok=True)

# Kalibrierung für jedes Objektiv (Pixel zu mm)
kalibrierfaktor = {
    "10x": 0.728,
    "20x": 0.363,
    "40x": 0.181,
    "63x": 0.115
}

# === CSVs laden ===
def load_labels(csv_path):
    df = pd.read_csv(csv_path, sep=";")
    return dict(zip(df["label_id"].astype(int), df["class_name"]))

def load_rules(csv_path):
    df = pd.read_csv(csv_path, sep=";")
    return {row["klasse"]: {"min": row["min_um_hoehe"], "max": row["max_um_hoehe"]} for _, row in df.iterrows()}

LABELS = load_labels(CLASSES_CSV)
RULES = load_rules(RULES_CSV)

model = load_model(MODEL_PATH)
print(f"📊 {len(LABELS)} Klassen geladen")
print(f"📐 Modell geladen: {MODEL_PATH}")

# === Globale Variablen ===
zoom_factor = 1.0  # Zoom-Faktor für das Bild
objektiv = "10x"   # Standard-Objektiv
print(f"🔭 Aktuelles Objektiv: {objektiv}")

original_clicks = []

# Feste Fenstergröße
WINDOW_WIDTH, WINDOW_HEIGHT = 1920, 1080
WINDOW_TITLE = "Zoom +/-, 1-4=Objektive, c=Korrektur, s=Speichern, a=Reihenaufnahme, r=Messung zuruecksetzen, ESC=Ende"

# Globale Variablen für die Anzeige der Messung
show_measurement = False
measurement_text = ""
measurement_line = None
measurement_color = (0, 255, 0)  # Standard: Grün
measurement_valid = False
measured_size = 0.0
last_detected_class = None

# === Bildvorverarbeitung ===
def preprocess_frame(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = img.astype("float32")
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)

def classify_top2(frame):
    preds = model.predict(preprocess_frame(frame), verbose=0)[0]
    top = preds.argsort()[-2:][::-1]
    return [(LABELS.get(i, f"ID {i}"), preds[i]) for i in top]

# === Mouse-Callback-Funktion ===
def zoom_mouse_callback(event, x, y, flags, param):
    global original_clicks, zoom_factor, show_measurement, measurement_text, measurement_line, objektiv, measurement_color, measurement_valid, measured_size, last_detected_class, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        scaled_frame = cv2.resize(frame, None, fx=zoom_factor, fy=zoom_factor)
        x_offset = max(0, (scaled_frame.shape[1] - WINDOW_WIDTH) // 2)
        y_offset = max(0, (scaled_frame.shape[0] - WINDOW_HEIGHT) // 2)

        original_x = int((x + x_offset) / zoom_factor)
        original_y = int((y + y_offset) / zoom_factor)
        original_clicks.append((original_x, original_y))
        print(f"Klick bei: ({original_x:.1f}, {original_y:.1f})")

        if len(original_clicks) == 2:
            (x1_click, y1_click), (x2_click, y2_click) = original_clicks
            distance_pixels = np.sqrt((x2_click - x1_click)**2 + (y2_click - y1_click)**2)
            skalierungsfaktor = kalibrierfaktor[objektiv]
            distance_mm = distance_pixels * skalierungsfaktor
            measured_size = distance_mm

            measurement_text = f"Laenge: {distance_mm:.2f} um"

            if last_detected_class in RULES:
                min_size = RULES[last_detected_class]["min"]
                max_size = RULES[last_detected_class]["max"]
                measurement_color = (0, 255, 0) if min_size <= distance_mm <= max_size else (0, 0, 255)
                measurement_valid = min_size <= distance_mm <= max_size
            else:
                measurement_color = (0, 255, 255)
                measurement_valid = False

            line_x1 = int((x1_click * zoom_factor) - x_offset)
            line_y1 = int((y1_click * zoom_factor) - y_offset)
            line_x2 = int((x2_click * zoom_factor) - x_offset)
            line_y2 = int((y2_click * zoom_factor) - y_offset)
            measurement_line = (line_x1, line_y1, line_x2, line_y2)
            show_measurement = True
            print(f"Laenge ({objektiv}): {distance_mm:.2f} um")
            original_clicks = []

# === Hauptprogramm ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Fehler: Kamera nicht gefunden")
    exit()

cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.setMouseCallback(WINDOW_TITLE, zoom_mouse_callback)

aufnahme_modus = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fehler: Kein Bild von der Kamera empfangen")
        break

    scaled_frame = cv2.resize(frame, None, fx=zoom_factor, fy=zoom_factor)
    vis_roi = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)

    x_offset = max(0, (scaled_frame.shape[1] - WINDOW_WIDTH) // 2)
    y_offset = max(0, (scaled_frame.shape[0] - WINDOW_HEIGHT) // 2)

    if scaled_frame.shape[1] <= WINDOW_WIDTH and scaled_frame.shape[0] <= WINDOW_HEIGHT:
        vis_roi[y_offset:y_offset + scaled_frame.shape[0], x_offset:x_offset + scaled_frame.shape[1]] = scaled_frame
    else:
        start_x = max(0, (scaled_frame.shape[1] - WINDOW_WIDTH) // 2)
        start_y = max(0, (scaled_frame.shape[0] - WINDOW_HEIGHT) // 2)
        vis_roi = scaled_frame[start_y:start_y + WINDOW_HEIGHT, start_x:start_x + WINDOW_WIDTH]

    cv2.putText(vis_roi, "Klick mit der Maus fuer eine Laengenmessung", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
    cv2.putText(vis_roi, f"Objektiv: {objektiv}", (20, WINDOW_HEIGHT - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)

    classification_results = classify_top2(frame)
    last_detected_class = classification_results[0][0] if classification_results else None

    for i, (label, prob) in enumerate(classification_results):
        y_pos = 80 + i * 60
        if show_measurement and label in RULES:
            min_size = RULES[label]["min"]
            max_size = RULES[label]["max"]
            color = (0, 255, 0) if min_size <= measured_size <= max_size else (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.putText(vis_roi, f"{label} ({prob:.2f})", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 2)

    if show_measurement and measurement_line is not None:
        line_x1, line_y1, line_x2, line_y2 = measurement_line
        cv2.line(vis_roi, (line_x1, line_y1), (line_x2, line_y2), measurement_color, 2)
        cv2.putText(vis_roi, measurement_text, (20, 80 + len(classification_results) * 60 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, measurement_color, 2)

    cv2.imshow(WINDOW_TITLE, vis_roi)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('+'):  # Zoom erhöhen
        zoom_factor += 0.2
        print(f"Bild-Zoom erhöht: {zoom_factor:.1f}x")
    elif key == ord('-'):  # Zoom verringern
        if zoom_factor > 0.4:
            zoom_factor -= 0.2
            print(f"Bild-Zoom verringert: {zoom_factor:.1f}x")
    elif key == ord('1'):
        objektiv = "10x"
        show_measurement = False
        print(f"🔭 Objektiv gewechselt zu: {objektiv}")
    elif key == ord('2'):
        objektiv = "20x"
        show_measurement = False
        print(f"🔭 Objektiv gewechselt zu: {objektiv}")
    elif key == ord('3'):
        objektiv = "40x"
        show_measurement = False
        print(f"🔭 Objektiv gewechselt zu: {objektiv}")
    elif key == ord('4'):
        objektiv = "63x"
        show_measurement = False
        print(f"🔭 Objektiv gewechselt zu: {objektiv}")
    elif key == ord('s'):
        bildname = input("Dateiname (ohne Endung) eingeben: ").strip()
        if not bildname:
            bildname = time.strftime("%Y%m%d-%H%M%S")
        base_path = os.path.join(BILD_SAVE_DIR, bildname)
        cv2.imwrite(base_path + "_HD.jpg", frame)
        cv2.imwrite(base_path + "_annotiert.jpg", vis_roi)
        print("📸 Bilder gespeichert (HD und annotiert)")
    elif key == ord('c'):
        print("🔽 Klasse auswählen:")
        klassen = sorted(set(LABELS.values()))
        for i, k in enumerate(klassen):
            print(f"{i+1}. {k}")
        try:
            auswahl = int(input("Nummer eingeben: "))
            if 1 <= auswahl <= len(klassen):
                neues_label = klassen[auswahl - 1]
                zielordner = os.path.join(KORREKTUR_DIR, neues_label.replace(" ", "_"))
                os.makedirs(zielordner, exist_ok=True)
                vorhandene = [
                    int(f.split(".")[0]) for f in os.listdir(zielordner)
                    if f.endswith(".jpg") and f.split(".")[0].isdigit()
                ]
                neue_nummer = max(vorhandene) + 1 if vorhandene else 1
                zielpfad = os.path.join(zielordner, f"{neue_nummer:03d}.jpg")
                cv2.imwrite(zielpfad, frame)
                print(f"✏️ Korrektur gespeichert: {zielpfad}")
            else:
                print("❌ Ungültige Eingabe")
        except Exception as e:
            print("❌ Fehlerhafte Eingabe:", e)
    elif key == ord('a'):
        aufnahme_modus = True
        print("📁 Aufnahmemodus gestartet – SPACE = Bild, ESC = Ende")
    elif key == ord('r'):
        show_measurement = False
        original_clicks = []

cap.release()
cv2.destroyAllWindows()
