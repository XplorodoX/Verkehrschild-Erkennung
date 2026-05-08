# Verkehrszeichen-Erkennung mit YOLOv8

Echtzeit-Erkennung deutscher Verkehrszeichen per Webcam – aufgebaut auf YOLOv8 Fine-Tuning
mit dem [GTSDB (German Traffic Sign Detection Benchmark)](https://benchmark.ini.rub.de/gtsdb_news.html).

## Ergebnisse

| Metrik     | Wert  |
|------------|-------|
| mAP50      | –     |
| mAP50-95   | –     |
| Precision  | –     |
| Recall     | –     |

*(Wird nach dem Training ergänzt)*

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
dann entpacken nach `data/raw/` (PPM-Bilder + `gt.txt`).

```bash
python convert_dataset.py
```

Konvertiert PPM → JPEG und Annotationen → YOLO-Format mit 80/20 Train/Val-Split.

### 2. Training

```bash
python train.py                      # Standard: yolov8n, 50 Epochen
python train.py --model yolov8s      # Größeres Modell
python train.py --epochs 100         # Mehr Epochen
```

Gewichte landen in `runs/detect/verkehrszeichen_v1/weights/best.pt`.

### 3. Evaluierung

```bash
python predict.py --weights runs/detect/verkehrszeichen_v1/weights/best.pt --val
python predict.py --weights best.pt --image testbild.jpg
python predict.py --weights best.pt --video testvideo.mp4
```

### 4. Webcam-Demo

```bash
python webcam_demo.py --weights runs/detect/verkehrszeichen_v1/weights/best.pt
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
├── data/
│   ├── raw/             ← GTSDB entpackt hier hin
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── dataset.yaml         ← YOLO Datensatz-Konfiguration
├── convert_dataset.py   ← GTSDB → YOLO Konvertierung
├── train.py             ← Fine-Tuning
├── predict.py           ← Evaluierung & Videoverarbeitung
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
