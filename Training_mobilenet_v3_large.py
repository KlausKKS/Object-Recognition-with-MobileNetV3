# ============================================================
# MobileNetV3Large Training
# Python 3.10
# TensorFlow 2.16.1
# ============================================================

import os
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# VERSIONEN AUSGEBEN
# ============================================================

print("=" * 60)
print("UMGEBUNG")
print("=" * 60)

print("TensorFlow:", tf.__version__)

try:
    import keras
    print("Keras:", keras.__version__)
except Exception as e:
    print("Keras-Version konnte nicht gelesen werden:", e)

print("NumPy:", np.__version__)
print()


# ============================================================
# KONFIGURATION
# ============================================================

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


print("=" * 60)
print("KONFIGURATION")
print("=" * 60)

print("Trainingsdaten:", TRAINING_DATA_DIR)
print("Modell:", MODEL_PATH)
print("Bildgröße:", IMG_SIZE)
print("Batch Size:", BATCH_SIZE)
print("Epochen:", EPOCHS)
print("Learning Rate:", LEARNING_RATE)
print()


# ============================================================
# PRÜFEN, OB TRAININGSDATEN VORHANDEN SIND
# ============================================================

if not os.path.isdir(TRAINING_DATA_DIR):
    raise FileNotFoundError(
        f"Trainingsverzeichnis nicht gefunden:\n{TRAINING_DATA_DIR}"
    )


# ============================================================
# .ipynb_checkpoints ENTFERNEN
# ============================================================

checkpoint_path = os.path.join(
    TRAINING_DATA_DIR,
    ".ipynb_checkpoints"
)

if os.path.exists(checkpoint_path):
    shutil.rmtree(checkpoint_path)
    print("🧹 '.ipynb_checkpoints' entfernt.")


# ============================================================
# BILD-AUGMENTATION
# ============================================================

train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input,

    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,

    validation_split=0.2,

    fill_mode="nearest"
)


# ============================================================
# TRAINING GENERATOR
# ============================================================

train_generator = train_datagen.flow_from_directory(
    TRAINING_DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="training",
    shuffle=True
)


# ============================================================
# VALIDIERUNGS-GENERATOR
# ============================================================

val_generator = train_datagen.flow_from_directory(
    TRAINING_DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="validation",
    shuffle=False
)


# ============================================================
# KLASSEN
# ============================================================

NUM_CLASSES = len(train_generator.class_indices)

print()
print("=" * 60)
print("KLASSEN")
print("=" * 60)

print(f"✅ {NUM_CLASSES} Klassen erkannt")
print()

for class_name, class_id in sorted(
    train_generator.class_indices.items(),
    key=lambda x: x[1]
):
    print(f"{class_id:3d}: {class_name}")

print()


# ============================================================
# CLASS WEIGHTS
# ============================================================

y_train = train_generator.classes

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights_dict = dict(
    enumerate(class_weights_array)
)

print("=" * 60)
print("CLASS WEIGHTS")
print("=" * 60)

print(class_weights_dict)
print()


# ============================================================
# MODELL
# ============================================================

def get_model(num_classes):

    print("=" * 60)
    print("MODELL WIRD ERSTELLT")
    print("=" * 60)

    base_model = MobileNetV3Large(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # MobileNet zunächst einfrieren
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)

    x = Dense(
        128,
        activation="relu"
    )(x)

    x = Dropout(
        0.4
    )(x)

    output = Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(
        inputs=base_model.input,
        outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )

    return model


model = get_model(NUM_CLASSES)

print()
print("✅ Modell erstellt")
print()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=PATIENCE,
    min_delta=0.001,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.4,
    patience=3,
    verbose=1,
    min_lr=1e-7
)


# ============================================================
# TRAINING
# ============================================================

print("=" * 60)
print("TRAINING START")
print("=" * 60)

history = model.fit(
    train_generator,

    validation_data=val_generator,

    epochs=EPOCHS,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    class_weight=class_weights_dict
)

print()
print("=" * 60)
print("TRAINING BEENDET")
print("=" * 60)
print()


# ============================================================
# MODELL SPEICHERN
# ============================================================

print("=" * 60)
print("MODELL WIRD GESPEICHERT")
print("=" * 60)

model.save(MODEL_PATH)

print(f"✅ Modell gespeichert unter:")
print(MODEL_PATH)
print()


# ============================================================
# ENTSCHEIDENDER TEST:
# GESPEICHERTES MODELL WIEDER LADEN
# ============================================================

print("=" * 60)
print("LADE-TEST")
print("=" * 60)

try:

    test_model = load_model(
        MODEL_PATH
    )

    print("✅ GESPEICHERTES MODELL ERFOLGREICH GELADEN!")

except Exception as e:

    print()
    print("❌ FEHLER BEIM WIEDERLADEN DES MODELLS")
    print()
    print(e)

    raise


# ============================================================
# VALIDIERUNG
# ============================================================

print()
print("=" * 60)
print("VALIDIERUNG")
print("=" * 60)

val_generator.reset()

predictions = test_model.predict(
    val_generator,
    verbose=1
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

true_classes = val_generator.classes


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

class_names = list(
    val_generator.class_indices.keys()
)

plt.figure(
    figsize=(25, 20)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title(
    "Confusion Matrix (Validierung)"
)

plt.xlabel(
    "Vorhergesagte Klasse"
)

plt.ylabel(
    "Wahre Klasse"
)

plt.xticks(
    rotation=90,
    fontsize=6
)

plt.yticks(
    rotation=0,
    fontsize=6
)

plt.tight_layout()

confusion_path = os.path.join(
    BASE_DIR,
    "confusion_matrix_224.png"
)

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

print()
print(f"✅ Confusion Matrix gespeichert:")
print(confusion_path)

plt.show()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names,
        digits=4
    )
)


# ============================================================
# TRAININGSVERLAUF
# ============================================================

plt.figure(
    figsize=(12, 5)
)

plt.subplot(1, 2, 1)

plt.plot(
    history.history["loss"],
    label="Train Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Val Loss"
)

plt.title("Loss")
plt.legend()


plt.subplot(1, 2, 2)

plt.plot(
    history.history["accuracy"],
    label="Train Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Val Accuracy"
)

plt.title("Accuracy")
plt.legend()


plt.tight_layout()

training_plot_path = os.path.join(
    BASE_DIR,
    "training_plot_v3.png"
)

plt.savefig(
    training_plot_path
)

print()
print(f"✅ Trainingsverlauf gespeichert:")
print(training_plot_path)

plt.show()


# ============================================================
# ABSCHLUSS
# ============================================================

print()
print("=" * 60)
print("🎉 TRAINING UND MODELLTEST ERFOLGREICH ABGESCHLOSSEN")
print("=" * 60)
print()
print("Modell:")
print(MODEL_PATH)
print()
print("Dieses Modell wurde nach dem Speichern erfolgreich")
print("mit load_model() wieder geladen.")
print()