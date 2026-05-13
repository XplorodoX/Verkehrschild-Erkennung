"""
Echtzeit-Webcam-Demo für die Verkehrszeichen-Erkennung.

Aufruf:
    python webcam_demo.py --weights runs/detect/verkehrszeichen_v1/weights/best.pt
    python webcam_demo.py --weights best.pt --camera 1  # externe Kamera

Steuerung:
    q  – Beenden
    s  – Screenshot speichern
    +  – Konfidenz erhöhen
    -  – Konfidenz verringern
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Pfad zur best.pt")
    p.add_argument("--camera",  type=int, default=0,   help="Kamera-Index (Standard: 0)")
    p.add_argument("--conf",    type=float, default=0.25, help="Startkonfidenz")
    p.add_argument("--imgsz",   type=int, default=960)
    return p.parse_args()


def draw_overlay(frame, fps: float, conf: float, detections: int, best_conf: float) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (320, 92), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Konfidenz: {conf:.2f}  (+/-)", (8, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Detektionen: {detections}  Best: {best_conf:.2f}", (8, 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if detections == 0:
        cv2.putText(frame, "Keine Erkennung", (w - 210, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def main() -> None:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"[FEHLER] Gewichte nicht gefunden: {weights}")
        print("Erst trainieren: python train.py")
        return

    model = YOLO(str(weights))
    cap   = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"[FEHLER] Kamera {args.camera} nicht verfügbar.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    conf   = args.conf
    prev_t = time.time()

    print("Demo gestartet. Tasten: q=Beenden  s=Screenshot  +=Konfidenz+  -=Konfidenz-")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[FEHLER] Kein Frame empfangen.")
            break

        results = model(frame, conf=conf, imgsz=args.imgsz, verbose=False)
        result = results[0]
        annotated = result.plot().copy()
        detections = 0 if result.boxes is None else len(result.boxes)
        best_conf = 0.0 if result.boxes is None or len(result.boxes) == 0 else float(result.boxes.conf.max().item())

        now = time.time()
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now

        draw_overlay(annotated, fps, conf, detections, best_conf)
        cv2.imshow("Verkehrszeichen-Erkennung  |  q=Quit  s=Screenshot", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fname, annotated)
            print(f"Screenshot gespeichert: {fname}")
        elif key == ord("+"):
            conf = min(conf + 0.05, 0.95)
            print(f"Konfidenz: {conf:.2f}")
        elif key == ord("-"):
            conf = max(conf - 0.05, 0.05)
            print(f"Konfidenz: {conf:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
