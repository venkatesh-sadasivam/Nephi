import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
import os
import argparse
from tqdm import tqdm
import string
import jiwer  # For WER and CER calculation. Install with: pip install jiwer


# =====================================================================================
# 1. Utility: Label Converter for CTC Loss
#
# This class handles the conversion between text strings and integer sequences,
# which is required by the CTC Loss function. It manages an "alphabet" of
# characters and includes the special 'blank' token used by CTC.
# =====================================================================================
class CTCLabelConverter(object):
    """Convert between text-label and text-index for CTC Loss"""

    def __init__(self, alphabet):
        self.alphabet = alphabet
        # The first index (0) is reserved for the CTC blank character
        self.dict = {char: i + 1 for i, char in enumerate(self.alphabet)}
        self.blank_idx = 0

    def encode(self, text):
        """
        Encodes a list of text strings into a single tensor of integer labels.

        Args:
            text (list[str]): list of text labels of a batch
        Returns:
            torch.IntTensor: concatenated text index for CTC Loss, e.g., [S, U, P, E, R] -> [19, 21, 16, 5, 18]
            torch.IntTensor: length of each text label in a batch
        """
        length = [len(s) for s in text]
        chars = ''.join(text)
        indices = []
        for char in chars:
            try:
                indices.append(self.dict[char])
            except KeyError:
                # Handle characters not in the alphabet by skipping them
                # A more robust solution might map them to an <UNK> token
                print(f"Warning: Character '{char}' not in alphabet. Skipping.")
        return (torch.IntTensor(indices), torch.IntTensor(length))

    def decode(self, text_index, length):
        """
        Decodes a sequence of integer predictions back into a text string.
        This is the core logic for interpreting the model's output. It removes
        consecutive repeated characters and the blank token.

        Args:
            text_index (torch.LongTensor): A BxT tensor of integer predictions from the model.
            length (torch.IntTensor): A B-element tensor with the lengths of each prediction.
        Returns:
            list[str]: A list of decoded text strings.
        """
        texts = []
        for i, l in enumerate(length):
            t = text_index[i, :l]

            # Core CTC decoding logic
            char_list = []
            for j in range(l):
                # If not a blank token and not a repeat of the previous character
                if t[j] != self.blank_idx and (j == 0 or t[j] != t[j - 1]):
                    # Check if index is valid before accessing alphabet
                    if t[j] - 1 < len(self.alphabet):
                        char_list.append(self.alphabet[t[j] - 1])
            text = ''.join(char_list)
            texts.append(text)
        return texts


# =====================================================================================
# 2. Dataset and Dataloader
#
# This class defines how to load and preprocess the handwriting data.
# It assumes a specific dataset structure:
# - A root directory containing image files.
# - A 'labels.txt' file inside the root directory where each line maps an
#   image filename to its ground truth text, separated by a space.
#   Example line in labels.txt: `word_001.png This is a sample.`
# =====================================================================================
class CustomDataset(Dataset):
    def __init__(self, root, img_h, img_w, transform=None):
        self.root = root
        self.transform = transform
        self.img_h = img_h
        self.img_w = img_w
        self.labels = self._load_labels()
        self.image_paths = list(self.labels.keys())

    def _load_labels(self):
        labels = {}
        labels_path = os.path.join(self.root, 'labels.txt')
        with open(labels_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    image_file, text = parts
                    labels[image_file] = text
        return labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_file = self.image_paths[index]
        image_path = os.path.join(self.root, image_file)

        # Load image, convert to grayscale, and resize
        try:
            image = Image.open(image_path).convert('L')  # Grayscale
            # Maintain aspect ratio while resizing
            w, h = image.size
            ratio = w / float(h)
            new_w = int(ratio * self.img_h)
            image = image.resize((new_w, self.img_h), Image.BICUBIC)
        except IOError:
            print(f'Corrupted image for {image_file}')
            # Return a placeholder if image is corrupt
            image = Image.new('L', (self.img_w, self.img_h), 'black')
            image_file = self.image_paths[0]  # use a valid label

        label = self.labels[image_file]

        if self.transform:
            image = self.transform(image)

        # Pad width if necessary
        c, h, w = image.shape
        if w < self.img_w:
            padding = torch.zeros((c, self.img_h, self.img_w - w))
            image = torch.cat([image, padding], dim=2)
        elif w > self.img_w:
            image = image[:, :, :self.img_w]  # Crop if wider

        return image, label


# =====================================================================================
# 3. Model Architecture: Convolutional Recurrent Neural Network (CRNN)
#
# This is the implementation of the Nephi model. It consists of three parts:
# - CNN: Extracts visual features from the input image.
# - RNN (LSTM): Processes the sequence of features, capturing context.
# - FC Layer: Maps the RNN output to character probabilities.
# =====================================================================================
class CRNN(nn.Module):
    def __init__(self, img_h, nc, nclass, nh):
        """
        Args:
            img_h (int): The height of the input image.
            nc (int): The number of input channels (1 for grayscale).
            nclass (int): The number of output classes (alphabet size + 1 for blank).
            nh (int): The size of the hidden state of the LSTM.
        """
        super(CRNN, self).__init__()
        assert img_h % 16 == 0, 'img_h has to be a multiple of 16'

        # --- CNN Part ---
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),  # 64x16xW/2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),  # 128x8xW/4
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),  # 256x8xW/4
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),  # 256x4xW/4
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),  # 512x4xW/4
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),  # 512x2xW/4
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True)  # 512x1xW/4
        )

        # --- RNN Part ---
        self.rnn = nn.Sequential(
            nn.LSTM(512, nh, bidirectional=True),
            nn.LSTM(nh * 2, nh, bidirectional=True)
        )

        # --- FC Layer ---
        self.fc = nn.Linear(nh * 2, nclass)

    def forward(self, x):
        # CNN forward pass
        conv = self.cnn(x)

        # Reshape for RNN: (batch, channels, height, width) -> (width, batch, channels*height)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)  # (b, c, w)
        conv = conv.permute(2, 0, 1)  # (w, b, c)

        # RNN forward pass
        rnn_out, _ = self.rnn(conv)

        # FC forward pass
        output = self.fc(rnn_out)

        return output  # (seq_len, batch, n_classes)


# =====================================================================================
# 4. Main Training and Validation Logic
# =====================================================================================
def main(opt):
    # --- Setup ---
    if not os.path.exists(opt.saved_model_dir):
        os.makedirs(opt.saved_model_dir)

    # Device setup (CPU or GPU)
    device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")
    if opt.cuda and torch.cuda.is_available():
        print(f"Using {torch.cuda.device_count()} GPUs!")
        torch.backends.cudnn.benchmark = True

    # --- Data Loading ---
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize for grayscale images
    ])

    train_dataset = CustomDataset(root=opt.train_root, img_h=opt.img_h, img_w=opt.img_w, transform=transform)
    # Filter out empty labels which cause issues with CTC
    train_dataset.image_paths = [p for p in train_dataset.image_paths if len(train_dataset.labels[p]) > 0]
    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.workers, pin_memory=True
    )

    val_dataset = CustomDataset(root=opt.val_root, img_h=opt.img_h, img_w=opt.img_w, transform=transform)
    val_dataset.image_paths = [p for p in val_dataset.image_paths if len(val_dataset.labels[p]) > 0]
    val_loader = DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.workers, pin_memory=True
    )

    # --- Model & Converter ---
    with open(opt.alphabet_path, 'r', encoding='utf-8') as f:
        alphabet = f.read().strip()

    converter = CTCLabelConverter(alphabet)
    num_classes = len(converter.alphabet) + 1  # +1 for blank token

    model = CRNN(img_h=opt.img_h, nc=1, nclass=num_classes, nh=opt.nh)

    # --- Multi-GPU Setup ---
    if opt.cuda and torch.cuda.device_count() > 1:
        print("Enabling DataParallel for Multi-GPU.")
        model = nn.DataParallel(model)

    model.to(device)

    # --- Loss and Optimizer ---
    criterion = nn.CTCLoss(blank=converter.blank_idx, reduction='mean', zero_infinity=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # --- Training Loop ---
    print("Starting Training...")

    for epoch in range(opt.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{opt.epochs} [Training]")

        for i, (images, texts) in enumerate(pbar):
            optimizer.zero_grad()

            images = images.to(device)
            text, length = converter.encode(texts)
            text = text.to(device)
            length = length.to(device)

            # Skip batch if it's empty after filtering for valid characters
            if text.numel() == 0:
                continue

            batch_size = images.size(0)
            preds = model(images)
            preds_log_softmax = F.log_softmax(preds, dim=2)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)

            cost = criterion(preds_log_softmax, text, preds_size, length)

            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            train_loss += cost.item()
            pbar.set_postfix({'loss': cost.item()})

        print(f"Epoch {epoch + 1}, Average Train Loss: {train_loss / len(train_loader):.4f}")

        # --- Validation Loop with Metrics ---
        model.eval()
        with torch.no_grad():
            print("\nRunning Validation...")
            val_loss = 0
            n_correct = 0
            n_total = 0
            ground_truths = []
            predictions = []

            for images, texts in tqdm(val_loader, desc="[Validating]"):
                images = images.to(device)
                text, length = converter.encode(texts)
                text = text.to(device)
                length = length.to(device)

                if text.numel() == 0:
                    continue

                batch_size = images.size(0)
                preds = model(images)  # (seq_len, batch, n_classes)
                preds_log_softmax = F.log_softmax(preds, dim=2)
                preds_size = torch.IntTensor([preds.size(0)] * batch_size)

                cost = criterion(preds_log_softmax, text, preds_size, length)
                val_loss += cost.item()

                # Decode predictions for metrics
                _, preds_index = preds.max(2)
                preds_index = preds_index.transpose(1, 0)  # B x T
                preds_str = converter.decode(preds_index, preds_size)

                ground_truths.extend(texts)
                predictions.extend(preds_str)

                for pred, gt in zip(preds_str, texts):
                    if pred == gt:
                        n_correct += 1
                n_total += batch_size

            # Calculate metrics
            avg_val_loss = val_loss / len(val_loader)
            accuracy = n_correct / n_total
            wer = jiwer.wer(ground_truths, predictions)
            cer = jiwer.cer(ground_truths, predictions)

            print("-" * 80)
            print(f"Validation Results - Epoch {epoch + 1}")
            print(f"  Average Loss: {avg_val_loss:.4f}")
            print(f"  Accuracy:     {accuracy:.4f} ({n_correct}/{n_total})")
            print(f"  Word Error Rate (WER): {wer:.4f}")
            print(f"  Char Error Rate (CER): {cer:.4f}")
            print("-" * 80)

        # Save model checkpoint
        model_path = os.path.join(opt.saved_model_dir, f"{opt.name}_epoch_{epoch + 1}.pth")
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), model_path)
        else:
            torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root', required=True, help='path to training dataset')
    parser.add_argument('--val_root', required=True, help='path to validation dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--img_h', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--img_w', type=int, default=100, help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
    parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Adam')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--alphabet_path', type=str, default='./alphabet.txt', help='path to alphabet file')
    parser.add_argument('--saved_model_dir', type=str, default='./saved_models', help='Where to save trained models')
    parser.add_argument('--name', type=str, default='nephi_crnn', help='Name for the saved model files')

    opt = parser.parse_args()
    print("Configuration:")
    print(opt)

    # --- Create a dummy alphabet.txt for demonstration ---
    if not os.path.exists(opt.alphabet_path):
        print(f"Creating a sample alphabet file at {opt.alphabet_path}")
        # A standard English alphabet plus some punctuation
        alphabet = string.ascii_letters + string.digits + string.punctuation + ' '
        with open(opt.alphabet_path, 'w', encoding='utf-8') as f:
            f.write(alphabet)

    main(opt)
