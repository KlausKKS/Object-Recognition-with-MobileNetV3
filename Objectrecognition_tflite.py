import sys
print("Aktueller Python-Interpreter:", sys.executable)
import cv2
print("OpenCV Version:", cv2.__version__)
import numpy as np
import os
import time
import datetime
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# === Setup ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "mobilenet_model_v3_224.keras")

CLASSES_CSV = os.path.join(
    BASE_DIR,
    "training_data",
    "Classes_alle.csv"
)

RULES_CSV = os.path.join(
    BASE_DIR,
    "training_data",
    "Abmessungen.csv"
)

BILD_SAVE_DIR = os.path.join(BASE_DIR, "bilder")

KORREKTUR_DIR = os.path.join(
    BASE_DIR,
    "korrigierte_daten"
)

AUFNAHME_DIR = os.path.join(
    BASE_DIR,
    "aufnahmen"
)
IMG_SIZE = (224, 224)
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

MODEL_PATH = os.path.join(BASE_DIR, "mobilenet_model_v3_224.tflite")

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
INPUT_SHAPE = input_details[0]["1, 224, 224, 3"]  # z. B. [1, 224, 224, 3]
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
def crop_and_resize(img, target_size=(224, 224)):
    h, w = img.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped = img[start_y:start_y + min_dim, start_x:start_x + min_dim]
    return cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)

def preprocess_frame(frame):
    img = crop_and_resize(frame, (INPUT_SHAPE[1], INPUT_SHAPE[2]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Keras-Modell wurde mit RGB trainiert
    img = img.astype("float32")
    return np.expand_dims(img, axis=0)

def classify_top2(frame):
    input_data = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])[0]
    top = preds.argsort()[-2:][::-1]
    return [(LABELS.get(i, f"ID {i}"), float(preds[i])) for i in top]

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
                measurement_color = (0, 255, 0) if min_size <= distance_mm <= max_size else (255, 255, 0)
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
cap = cv2.VideoCapture(0, cv2.CAP_D_SHOW)
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

    # Zoom anwenden
    scaled_frame = cv2.resize(frame, None, fx=zoom_factor, fy=zoom_factor)

    # Sichtbaren Ausschnitt (vis_roi) berechnen
    x_offset = max(0, (scaled_frame.shape[1] - WINDOW_WIDTH) // 2)
    y_offset = max(0, (scaled_frame.shape[0] - WINDOW_HEIGHT) // 2)

    if scaled_frame.shape[1] <= WINDOW_WIDTH and scaled_frame.shape[0] <= WINDOW_HEIGHT:
        vis_roi = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        vis_roi[y_offset:y_offset + scaled_frame.shape[0], x_offset:x_offset + scaled_frame.shape[1]] = scaled_frame
    else:
        start_x = max(0, (scaled_frame.shape[1] - WINDOW_WIDTH) // 2)
        start_y = max(0, (scaled_frame.shape[0] - WINDOW_HEIGHT) // 2)
        vis_roi = scaled_frame[start_y:start_y + WINDOW_HEIGHT, start_x:start_x + WINDOW_WIDTH]

    # Objekterkennung auf dem sichtbaren Ausschnitt (vis_roi) durchführen
    classification_results = classify_top2(vis_roi)

    last_detected_class = classification_results[0][0] if classification_results else None

    # Rest deines Codes (Anzeige, Tasteneingaben, etc.)
    cv2.putText(vis_roi, "Klick mit der Maus fuer eine Laengenmessung", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
    cv2.putText(vis_roi, f"Objektiv: {objektiv}", (20, WINDOW_HEIGHT - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)

    if classification_results and classification_results[0][1] >= 0.3:
        for i, (label, prob) in enumerate(classification_results):
            y_pos = 80 + i * 60
            if show_measurement and label in RULES:
                min_size = RULES[label]["min"]
                max_size = RULES[label]["max"]
                color = (0, 255, 0) if min_size <= measured_size <= max_size else (255, 255, 0)
            else:
                color = (0, 255, 0)
            cv2.putText(vis_roi, f"{label} ({prob:.2f})", (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 2)
    else:
        cv2.putText(vis_roi, "Keine Erkennung", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 2)

    if show_measurement and measurement_line is not None:
        line_x1, line_y1, line_x2, line_y2 = measurement_line
        cv2.line(vis_roi, (line_x1, line_y1), (line_x2, line_y2), measurement_color, 2)
        cv2.putText(vis_roi, measurement_text, (20, 80 + len(classification_results) * 60 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, measurement_color, 2)

    cv2.imshow(WINDOW_TITLE, vis_roi)

    # Tasteneingaben (Zoom, Objektivwechsel, etc.)
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

    # Taste ESC zum Beenden des Aufnahmemodus oder des Programms
    if key == 27:  # ESC
        if aufnahme_modus:
            aufnahme_modus = False
            print("Aufnahmemodus beendet")
        else:
            break  # Programm beenden, wenn ESC außerhalb des Aufnahmemodus gedrückt wird

    # Logik für den Aufnahmemodus (Leertaste)
    if aufnahme_modus and key == ord(' '):  # Leertaste
        bildname = f"bild_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        base_path = os.path.join(AUFNAHME_DIR, bildname)
        cv2.imwrite(base_path + "_HD.jpg", frame)
        print(f"💾 Bild gespeichert: {base_path}")
    elif key == ord('r'):
        show_measurement = False
        original_clicks = []

cap.release()
cv2.destroyAllWindows()
