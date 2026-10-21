from pathlib import Path
from datasets import load_dataset
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "places365"

NUM_IMAGES = 10

DATASET_ID = "ljnlonoljpiljm/places365-256px"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Places365 sample to {OUTPUT_DIR}")

    dataset = load_dataset(
        DATASET_ID,
        split="train",
        streaming=True,
    )

    saved = 0

    for i, sample in enumerate(dataset):
        if saved >= NUM_IMAGES:
            break

        image = sample["image"].convert("RGB")
        image = image.resize((256, 256), Image.Resampling.LANCZOS)

        output_path = OUTPUT_DIR / f"places365_{saved:03d}.jpg"
        image.save(output_path, quality=95)

        print(f"Saved: {output_path}")
        saved += 1

    print(f"\nFinished. {saved} images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()