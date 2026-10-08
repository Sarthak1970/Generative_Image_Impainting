from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT/"data"/"celeba"

def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    print(f"Downloading CelebA dataset to {OUTPUT_DIR}")

    dataset=load_dataset("flwrlabs/celeba",
                         split="test",
                         streaming=True)


    num_images=10

    for i,sample in enumerate(dataset):
        if i>=num_images:
            break


        image=sample["image"]
        output_path=OUTPUT_DIR/f"celeba_{i:03d}.png"

        image.save(output_path)
        print(f"Saved: {output_path}")

    print()
    print(f"Finished. Images save to: {OUTPUT_DIR}")

if __name__=="__main__":
    main()