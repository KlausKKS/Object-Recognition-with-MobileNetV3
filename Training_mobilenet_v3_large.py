# Uses: venv with Python 3.10, Tensorflow 2.16.1,numpy 1.26.4, pandas 2.2.2 pillow

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil

# 📁 Konfiguration
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAINING_DATA_DIR = os.path.join(
    BASE_DIR,
    "training_data",
    "Dataset_Vorverarbeitet_224"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "mobilenet_model_v3_224.keras"
)
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 35
LEARNING_RATE = 1e-4
PATIENCE = 3

# 🧹 Checkpoints entfernen
checkpoint_path = os.path.join(TRAINING_DATA_DIR, ".ipynb_checkpoints")
if os.path.exists(checkpoint_path):
    shutil.rmtree(checkpoint_path)
    print("🧹 '.ipynb_checkpoints' entfernt.")

# 📈 Bildaugmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    TRAINING_DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    TRAINING_DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation',
    shuffle=False
)

# 🔢 Klassen bestimmen
NUM_CLASSES = len(train_generator.class_indices)
print(f"✅ {NUM_CLASSES} Klassen erkannt")

# ⚖️ Class Weights
y_train = train_generator.classes
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights_array))
print("✅ Class Weights berechnet:", class_weights_dict)

# 🧠 Modell bauen mit MobileNetV3
def get_model(num_classes):
    base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = get_model(NUM_CLASSES)

# ⏱️ Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, min_delta=0.001,restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3, verbose=1, min_lr=1e-7)


# 🏋️‍♂️ Training starten
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weights_dict
)
# =============================================
# CONFUSION MATRIX NACH DEM TRAINING
# =============================================
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(model, val_generator, class_names):
    """
    Berechnet und zeigt die Confusion Matrix für die Validierungsdaten.
    Funktioniert mit 1D-Labels (Klassenindizes) und 2D-Labels (One-Hot-Encoded).
    """
    # Lade alle Validierungsdaten
    val_generator.reset()
    val_images = []
    val_labels = []
    for _ in range(len(val_generator)):
        images, labels = next(val_generator)
        val_images.append(images)
        val_labels.append(labels)
    val_images = np.concatenate(val_images)
    val_labels = np.concatenate(val_labels)

    # Berechne Vorhersagen
    predictions = model.predict(val_images)
    predicted_classes = np.argmax(predictions, axis=1)

    # Konvertiere val_labels zu Klassenindizes (falls One-Hot-Encoded)
    if len(val_labels.shape) > 1 and val_labels.shape[1] > 1:
        true_classes = np.argmax(val_labels, axis=1)  # One-Hot-Encoded
    else:
        true_classes = val_labels  # Bereits Klassenindizes

    # Berechne Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)

    # Plotte die Confusion Matrix
    plt.figure(figsize=(25, 20))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix (Validierung)')
    plt.xlabel('Vorhergesagte Klasse')
    plt.ylabel('Wahre Klasse')
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(
    os.path.join(BASE_DIR, "confusion_matrix_224.png"),
    dpi=300,
    bbox_inches="tight"
)
    plt.show()

    # Drucke den Classification Report
    print("\n=== Classification Report ===")
    print(classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names,
        digits=4
    ))

# Hole die Klassennamen aus dem Generator
class_names = list(val_generator.class_indices.keys())

# Ruf die Funktion auf
plot_confusion_matrix(model, val_generator, class_names)
# 💾 Speichern
model.save(MODEL_PATH)
print(f"✅ Modell gespeichert unter: {MODEL_PATH}")

# 📊 Trainingsverlauf anzeigen
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(
    os.path.join(BASE_DIR, "training_plot_v3.png")
)
    plt.show()

plot_training_history(history)
