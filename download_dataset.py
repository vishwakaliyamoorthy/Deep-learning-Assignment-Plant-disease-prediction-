"""
Download the PlantVillage dataset (colour leaf images, 38 classes, ~54,300 images).

Source: https://github.com/spMohanty/PlantVillage-Dataset  (folder raw/color)
It uses a git sparse checkout so ONLY the colour images are downloaded (~1 GB).

Usage:   python download_dataset.py
Result:  dataset/PlantVillage-Dataset/raw/color/<class_name>/*.JPG

If git is not installed, download manually instead (see README.md, "Option B").
"""
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/spMohanty/PlantVillage-Dataset.git"
TARGET = Path("dataset") / "PlantVillage-Dataset"
COLOR = TARGET / "raw" / "color"


def run(cmd, cwd=None):
    print(">", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def count_images(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def main():
    if COLOR.exists() and count_images(COLOR) > 50000:
        print(f"Dataset already present: {count_images(COLOR)} images in {COLOR}")
        return
    if shutil.which("git") is None:
        sys.exit("git is not installed. Install git (https://git-scm.com) or use the manual "
                 "download option in README.md.")

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if not TARGET.exists():
        run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, str(TARGET)])
    run(["git", "sparse-checkout", "set", "raw/color"], cwd=TARGET)

    n = count_images(COLOR)
    classes = len([d for d in COLOR.iterdir() if d.is_dir()])
    print(f"\nDone. {n} images in {classes} classes -> {COLOR}")
    print("Now run:  python train.py --quick")


if __name__ == "__main__":
    main()
