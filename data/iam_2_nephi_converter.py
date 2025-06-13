import os
import shutil
from tqdm import tqdm
import argparse


def preprocess_iam_dataset(iam_base_path, output_path):
    """
    Preprocesses the IAM Handwriting Database.

    This function reads the original IAM dataset structure, parses the ground truth
    files, and organizes the data into 'train' and 'validation' sets suitable
    for the CRNN training script. It copies the required word-level images and
    creates a 'labels.txt' file in each output directory.

    Args:
        iam_base_path (str): The path to the root of the downloaded IAM dataset.
                             This directory should contain 'words.txt', the 'words'
                             image folder, and the train/val/test split files.
        output_path (str): The path where the processed 'train' and 'validation'
                           folders will be created.
    """
    print("Starting IAM Dataset Preprocessing...")

    # --- 1. Define File Paths ---
    words_txt_path = os.path.join(iam_base_path, 'ascii', 'words.txt')
    words_img_path = os.path.join(iam_base_path, 'words')

    # Paths to the official split files
    train_split_path = os.path.join(iam_base_path, 'trainset.txt')
    val1_split_path = os.path.join(iam_base_path, 'validationset1.txt')
    val2_split_path = os.path.join(iam_base_path, 'validationset2.txt')
    test_split_path = os.path.join(iam_base_path, 'testset.txt')

    # --- 2. Create Output Directories ---
    train_output_path = os.path.join(output_path, 'train')
    val_output_path = os.path.join(output_path, 'validation')
    os.makedirs(train_output_path, exist_ok=True)
    os.makedirs(val_output_path, exist_ok=True)
    print(f"Output directories created at: {output_path}")

    # --- 3. Read Split Definitions ---
    try:
        with open(train_split_path, 'r') as f:
            train_ids = {line.strip() for line in f}
        # We will combine both validation sets into one
        with open(val1_split_path, 'r') as f:
            val_ids = {line.strip() for line in f}
        with open(val2_split_path, 'r') as f:
            val_ids.update(line.strip() for line in f)
        with open(test_split_path, 'r') as f:
            test_ids = {line.strip() for line in f}
    except FileNotFoundError as e:
        print(f"Error: Split file not found. {e}")
        print("Please ensure 'trainset.txt', 'validationset1.txt', etc., are in the iam_base_path.")
        return

    print(f"Loaded {len(train_ids)} train IDs, {len(val_ids)} validation IDs, and {len(test_ids)} test IDs.")

    # --- 4. Parse words.txt and Process Data ---
    train_labels = []
    val_labels = []

    with open(words_txt_path, 'r') as f:
        for line in tqdm(f.readlines(), desc="Parsing words.txt"):
            if line.startswith('#'):
                continue

            parts = line.strip().split()
            word_id = parts[0]

            # Check segmentation quality
            segmentation_ok = parts[1] == 'ok'
            if not segmentation_ok:
                continue

            # Reconstruct text label (can contain spaces)
            text_label = ' '.join(parts[8:])

            # Determine which split this word belongs to
            word_base_id = '-'.join(word_id.split('-')[:2])

            if word_base_id in train_ids:
                target_dir = train_output_path
                label_list = train_labels
            elif word_base_id in val_ids:
                target_dir = val_output_path
                label_list = val_labels
            else:
                # Skip test set images or any others not in splits
                continue

            # Construct source image path
            # e.g., words/a01/a01-000u/a01-000u-00-00.png
            form_id_parts = word_id.split('-')
            form_folder = form_id_parts[0]
            sub_folder = f"{form_id_parts[0]}-{form_id_parts[1]}"
            source_img_path = os.path.join(words_img_path, form_folder, sub_folder, f"{word_id}.png")

            # Define destination path
            dest_img_path = os.path.join(target_dir, f"{word_id}.png")

            # Copy image and store label
            if os.path.exists(source_img_path):
                shutil.copyfile(source_img_path, dest_img_path)
                label_list.append(f"{word_id}.png {text_label}")

    # --- 5. Write labels.txt for each split ---
    with open(os.path.join(train_output_path, 'labels.txt'), 'w') as f:
        f.write('\n'.join(train_labels))

    with open(os.path.join(val_output_path, 'labels.txt'), 'w') as f:
        f.write('\n'.join(val_labels))

    print("\nPreprocessing complete!")
    print(f"Training data: {len(train_labels)} images and labels.txt created in {train_output_path}")
    print(f"Validation data: {len(val_labels)} images and labels.txt created in {val_output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess the IAM Handwriting Database.")
    parser.add_argument('--iam_path', type=str, required=True,
                        help="Path to the root of the downloaded IAM dataset.")
    parser.add_argument('--output_path', type=str, required=True,
                        help="Path where the processed 'train' and 'validation' folders will be created.")

    args = parser.parse_args()

    preprocess_iam_dataset(args.iam_path, args.output_path)
