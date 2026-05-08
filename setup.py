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
    raw = Path("data/raw")
    ppms = list(raw.glob("*.ppm"))
    gt   = raw / "gt.txt"

    print(f"\nDatensatz-Status:")
    if ppms and gt.exists():
        print(f"  {len(ppms)} PPM-Bilder gefunden")
        print(f"  gt.txt gefunden")
        print(f"  → Bereit für Konvertierung: python convert_dataset.py")
    else:
        print("  Noch kein GTSDB-Datensatz gefunden.")
        print("  Datensatz laden unter:")
        print("  https://benchmark.ini.rub.de/gtsdb_news.html")
        print("  Dann entpacken nach: data/raw/")


def main() -> None:
    print("=== Verkehrszeichen-Erkennung: Setup ===\n")
    check_and_install()
    check_gpu()
    check_dataset()
    print("\nSetup abgeschlossen!")
    print("\nWorkflow:")
    print("  1. Datensatz nach data/raw/ entpacken")
    print("  2. python convert_dataset.py")
    print("  3. python train.py")
    print("  4. python predict.py --weights runs/detect/verkehrszeichen_v1/weights/best.pt --val")
    print("  5. python webcam_demo.py --weights runs/detect/verkehrszeichen_v1/weights/best.pt")


if __name__ == "__main__":
    main()
