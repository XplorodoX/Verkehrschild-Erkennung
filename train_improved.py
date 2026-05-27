"""
Verbessertes Training: yolov8m, 100 Epochen, optimierte Augmentierung für Verkehrszeichen.

Wichtigste Unterschiede zu train.py:
  - Größeres Modell (Standard: yolov8m ~26M Parameter)
  - copy_paste=0.3  (YOLOs eingebaute Copy-Paste-Aug.)
  - Kein horizontales/vertikales Spiegeln (Schilder sind nicht symmetrisch)
  - Mehr Farb- und Beleuchtungsvariation (Wetter, Tageszeit)
  - Höherer patience-Wert damit Early Stopping nicht zu früh greift
  - Kleinere Lernrate für stabileres Fine-Tuning

Aufruf:
    python train_improved.py
    python train_improved.py --model yolov8l   # noch größer
    python train_improved.py --resume          # nach Unterbrechung fortsetzen
    python train_improved.py --tune            # Hyperparameter-Autotuning vor dem Training (benötigt ray[tune])
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="yolov8m",  help="yolov8m | yolov8l | yolov8x")
    p.add_argument("--epochs",  type=int, default=100)
    p.add_argument("--batch",   type=int, default=8,
                   help="Kleiner Batch wegen großem Modell auf MPS/CPU")
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--name",    default="verkehrszeichen_v2")
    p.add_argument("--patience",type=int, default=20)
    p.add_argument("--resume",  action="store_true",
                   help="Letztes Checkpoint fortsetzen")
    p.add_argument("--tune",    action="store_true",
                   help="Hyperparameter-Autotuning vor dem Training (benötigt ray[tune])")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset_yaml = Path("dataset.yaml")
    if not dataset_yaml.exists():
        raise FileNotFoundError("dataset.yaml nicht gefunden.")

    if args.resume:
        last = Path(f"runs/detect/{args.name}/weights/last.pt")
        if not last.exists():
            raise FileNotFoundError(f"Kein Checkpoint gefunden: {last}")
        print(f"Setze Training fort von: {last}")
        model = YOLO(str(last))
    else:
        model_file = f"{args.model}.pt"
        print(f"Lade vortrainiertes Modell: {model_file}")
        model = YOLO(model_file)

    print(f"\nVerbessertes Training:")
    print(f"  Modell        : {args.model}")
    print(f"  Epochen       : {args.epochs}  (patience={args.patience})")
    print(f"  Batch         : {args.batch}")
    print(f"  Bildgröße     : {args.imgsz}")
    print(f"  Name          : {args.name}\n")

    if args.tune:
        print("Starte Hyperparameter-Autotuning (50 Iterationen) ...")
        model.tune(
            data=str(dataset_yaml),
            epochs=30,
            iterations=50,
            imgsz=args.imgsz,
            batch=args.batch,
        )
        print("Tuning abgeschlossen – starte Training mit besten Parametern.\n")

    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=args.patience,
        resume=args.resume,

        # --- Lernrate ---
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,

        # --- Augmentierung speziell für Schilder ---
        flipud=0.0,
        fliplr=0.0,

        # Farbe & Beleuchtung
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,

        # Geometrie
        degrees=12.0,
        translate=0.1,
        scale=0.6,
        shear=3.0,
        perspective=0.0003,

        # Mosaic & Copy-Paste
        mosaic=1.0,
        copy_paste=0.3,

        # Sonstiges
        mixup=0.1,
        close_mosaic=10,
    )

    best = Path(f"runs/detect/{args.name}/weights/best.pt")
    print(f"\nTraining abgeschlossen! Beste Gewichte: {best}")
    print(f"\nEvaluierung       : python predict.py --weights {best} --val")
    print(f"Per-Klassen-Analyse: python predict.py --weights {best} --per-class")
    print(f"ONNX exportieren  : python predict.py --weights {best} --export")
    print(f"Webcam-Demo       : python webcam_demo.py --weights {best}")


if __name__ == "__main__":
    main()
