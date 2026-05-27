# Verkehrszeichen-Erkennung mit YOLOv8

Echtzeit-Erkennung deutscher Verkehrszeichen per Webcam – aufgebaut auf YOLOv8 Fine-Tuning
mit dem [GTSDB (German Traffic Sign Detection Benchmark)](https://benchmark.ini.rub.de/gtsdb_news.html).

## Ergebnisse

### YOLOv8s – Kaggle T4 GPU, 100 Epochen *(bisherig)*

| Metrik     | Wert  |
|------------|-------|
| mAP50      | 0.534 |
| mAP50-95   | 0.437 |
| Precision  | 0.607 |
| Recall     | 0.521 |

Trainiert mit `kaggle_train.ipynb` (YOLOv8s, GPU T4, 100 Epochen, Copy-Paste-Augmentierung).

### YOLOv8n – Lokal CPU, 50 Epochen *(Baseline)*

| Metrik     | Wert  |
|------------|-------|
| mAP50      | 0.354 |
| mAP50-95   | 0.277 |
| Precision  | 0.453 |
| Recall     | 0.331 |

Trainiert mit `train.py` auf Apple M1 Pro, Early Stopping bei Epoch 30.

> Der GTSDB-Trainingsset enthält nur ~600 Szenenbilder für 43 Klassen. Viele seltene Klassen
> haben 1–3 Validierungsbeispiele. `augment_copypaste.py` gleicht dieses Ungleichgewicht aus.

## Datensatz

**GTSDB** – 43 Klassen, ~900 Bilder mit Bounding-Box-Annotationen.

- Tempolimits (20, 30, 50, 60, 70, 80, 100, 120 km/h)
- Vorfahrtsregeln, Verbotsschilder, Warnschilder, Gebotsschilder

## Installation

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python setup.py               # Prüft Abhängigkeiten & GPU
```

## Workflow

### 1. Datensatz vorbereiten

Datensatz von https://benchmark.ini.rub.de/gtsdb_news.html herunterladen,
dann entpacken nach `archive(2)/TrainIJCNN2013/TrainIJCNN2013/` (PPM-Bilder + `gt.txt`).

```bash
python convert_dataset.py
```

Konvertiert PPM → JPEG und Annotationen → YOLO-Format mit 80/20 Train/Val-Split.

### 1b. Datensatz-Augmentierung (empfohlen)

Viele Klassen haben im GTSDB nur wenige Beispiele. Dieser Schritt klebt GTSRB-Crops auf
Szenenbilder, bis jede Klasse mindestens 150 Annotationen hat:

```bash
python augment_copypaste.py
```

Überspringen, wenn nur der Basis-Workflow getestet werden soll.

### 2. Training

```bash
# Empfohlen: yolov8m, optimierte Augmentierung
python train_improved.py

# Mit Hyperparameter-Autotuning (langsamer, aber bessere Ergebnisse)
python train_improved.py --tune

# Noch größeres Modell
python train_improved.py --model yolov8l

# Schnelles Basis-Training
python train.py
```

Gewichte landen in `runs/detect/verkehrszeichen_v2/weights/best.pt`.

### 2b. Training auf Kaggle (kostenlose GPU)

Falls kein leistungsfähiger PC verfügbar ist:

1. `kaggle_train.ipynb` auf [kaggle.com](https://kaggle.com) importieren
2. Dataset `german-traffic-sign-detection-benchmark-gtsdb` (safabouguezzi) hinzufügen
3. Accelerator **GPU T4 x2** aktivieren
4. **Run All** – dauert ca. 20–40 Minuten
5. Gewichte unter *Output* → `best_weights.zip` herunterladen

### 3. Trainingsverläufe visualisieren

```bash
python plot_results.py                          # Standard: verkehrszeichen_v1
python plot_results.py --run verkehrszeichen_v2 # Anderer Lauf
```

Öffnet eine interaktive HTML-Datei mit Loss-, Precision-, Recall- und mAP-Kurven.

### 4. Evaluierung

```bash
python predict.py --weights best.pt --val          # Gesamtmetriken
python predict.py --weights best.pt --per-class    # AP50 pro Klasse (zeigt schwache Klassen)
python predict.py --weights best.pt --export       # ONNX-Export für schnellere Inferenz
python predict.py --weights best.pt --image testbild.jpg
python predict.py --weights best.pt --video testvideo.mp4
```

### 5. Webcam-Demo

```bash
python webcam_demo.py --weights runs/detect/verkehrszeichen_v2/weights/best.pt
# oder mit ONNX (schneller):
python webcam_demo.py --weights best.onnx
```

| Taste | Aktion              |
|-------|---------------------|
| `q`   | Beenden             |
| `s`   | Screenshot speichern|
| `+`   | Konfidenz erhöhen   |
| `-`   | Konfidenz verringern|

## Projektstruktur

```
Verkehrschild Erkennung/
├── archive(2)/
│   └── TrainIJCNN2013/TrainIJCNN2013/  ← GTSDB entpackt hier hin
├── data/
│   ├── images/
│   │   ├── train/       ← JPEGs (nach convert_dataset.py)
│   │   └── val/
│   └── labels/
│       ├── train/       ← YOLO-Annotationen (.txt)
│       └── val/
├── dataset.yaml         ← YOLO Datensatz-Konfiguration
├── convert_dataset.py   ← GTSDB → YOLO Konvertierung
├── augment_copypaste.py ← Klassen-Balancierung via Copy-Paste
├── train.py             ← Basis Fine-Tuning (yolov8n)
├── train_improved.py    ← Empfohlenes Training (yolov8m, Autotuning)
├── kaggle_train.ipynb   ← Kaggle-Notebook (kostenlose GPU)
├── predict.py           ← Evaluierung, Per-Klassen-Analyse, ONNX-Export
├── plot_results.py      ← Trainingsverläufe visualisieren
├── webcam_demo.py       ← Echtzeit-Demo
├── setup.py             ← Umgebungs-Check
└── requirements.txt
```

## Technischer Hintergrund

YOLOv8 (You Only Look Once v8) ist ein Single-Stage-Detektor: Bild → CNN → Grid →
Bounding Boxes + Klassen in einem einzigen Forward Pass. Das ermöglicht Echtzeiterkennung
bei >30 FPS auf moderner Hardware.

Das Fine-Tuning nutzt Transfer Learning: vortrainierte ImageNet-Gewichte werden durch
GTSDB-spezifisches Training angepasst. Dadurch reichen deutlich weniger Daten als bei
Training von Grund auf.

**Vergleich zu Anomalie-Detektion (Autoencoder):**
- Autoencoder: kein Labeling nötig, erkennt "unbekannte" Fehler
- YOLOv8: präzise Lokalisierung, benötigt annotierte Bounding Boxes
- Beide Ansätze ergänzen sich je nach Anwendungsfall
