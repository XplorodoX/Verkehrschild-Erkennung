"""
Einmaliges Setup-Skript: prüft die Abhängigkeiten und richtet die Umgebung ein.

Aufruf:
    python setup.py
"""

import subprocess
import sys
from pathlib import Path


REQUIRED = [
    "ultralytics",
    "opencv-python",
    "Pillow",
    "pandas",
    "matplotlib",
    "torch",
    "torchvision",
]


def check_and_install() -> None:
    import importlib
    missing = []
    for pkg in REQUIRED:
        module = pkg.split("-")[0].lower().replace("-", "_")
        if module == "opencv":
            module = "cv2"
        elif module == "pillow":
            module = "PIL"
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Installiere fehlende Pakete: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
    else:
        print("Alle Abhängigkeiten sind bereits installiert.")


def check_gpu() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"GPU erkannt: {name}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("Apple Silicon MPS erkannt (M1/M2/M3)")
        else:
            print("Keine GPU gefunden – Training läuft auf CPU (langsamer).")
            print("Tipp: Google Colab mit kostenloser T4-GPU nutzen.")
    except ImportError:
        pass


def check_dataset() -> None:
    archive = Path("archive(2)/TrainIJCNN2013/TrainIJCNN2013")
    ppms = list(archive.glob("*.ppm")) if archive.exists() else []
    gt   = archive / "gt.txt"

    print(f"\nDatensatz-Status:")
    if ppms and gt.exists():
        print(f"  {len(ppms)} PPM-Bilder gefunden in {archive}")
        print(f"  gt.txt gefunden")
        print(f"  → Bereit für Konvertierung: python convert_dataset.py")
    else:
        print("  Noch kein GTSDB-Datensatz gefunden.")
        print("  Datensatz laden unter:")
        print("  https://benchmark.ini.rub.de/gtsdb_news.html")
        print("  Dann entpacken nach: archive(2)/TrainIJCNN2013/TrainIJCNN2013/")


def main() -> None:
    print("=== Verkehrszeichen-Erkennung: Setup ===\n")
    check_and_install()
    check_gpu()
    check_dataset()
    print("\nSetup abgeschlossen!")
    print("\nWorkflow:")
    print("  1. Datensatz nach archive(2)/TrainIJCNN2013/TrainIJCNN2013/ entpacken")
    print("  2. python convert_dataset.py")
    print("  3. python augment_copypaste.py        # optional, balanciert seltene Klassen")
    print("  4. python train_improved.py           # empfohlen; oder: python train.py")
    print("  5. python plot_results.py             # Trainingsverläufe visualisieren")
    print("  6. python predict.py --weights runs/detect/verkehrszeichen_v2/weights/best.pt --val")
    print("  7. python webcam_demo.py --weights runs/detect/verkehrszeichen_v2/weights/best.pt")


if __name__ == "__main__":
    main()
