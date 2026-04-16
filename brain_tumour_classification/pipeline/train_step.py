import os
import argparse
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.cnn import BrainTumourCNN
from config import CLASS_NAMES

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

# ── Args ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--processed_data", type=str)
parser.add_argument("--model_output",   type=str)
parser.add_argument("--num_epochs",     type=int,   default=25)
parser.add_argument("--learning_rate",  type=float, default=1e-4)
parser.add_argument("--batch_size",     type=int,   default=32)
parser.add_argument("--val_split",      type=float, default=0.2)
args = parser.parse_args()

os.makedirs(args.model_output, exist_ok=True)

# ── Check flags ──────────────────────────────────────────────────
no_new_data_flag = os.path.join(args.processed_data, "NO_NEW_DATA")
if os.path.exists(no_new_data_flag):
    print("✅ No new data detected — skipping everything.")
    exit(0)

def read_flag(flag_name):
    flag_path = os.path.join(args.processed_data, flag_name)
    if os.path.exists(flag_path):
        with open(flag_path) as f:
            return f.read().strip() == "true"
    return False

training_changed = read_flag("training_changed.flag")
testing_changed  = read_flag("testing_changed.flag")

print(f"Training data changed: {training_changed}")
print(f"Testing data changed:  {testing_changed}")

for flag, val in [("training_changed.flag", training_changed), ("testing_changed.flag", testing_changed)]:
    with open(os.path.join(args.model_output, flag), "w") as f:
        f.write("true" if val else "false")

if not training_changed and not testing_changed:
    print("✅ Neither split changed — skipping.")
    exit(0)


# ── Dataset ──────────────────────────────────────────────────────
class TensorDataset(Dataset):
    def __init__(self, root_dir: str, split: str):
        self.samples = []
        split_dir = os.path.join(root_dir, split)
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"[WARN] Not found: {class_dir}")
                continue
            for pt_path in glob.glob(os.path.join(class_dir, "*.pt")):
                self.samples.append(pt_path)
        print(f"{split}: {len(self.samples)} samples")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        data = torch.load(self.samples[idx], weights_only=True)
        return data["tensor"], data["label"]


# ── Training functions ───────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct      += outputs.argmax(1).eq(labels).sum().item()
        total        += labels.size(0)
    return running_loss / total, 100.0 * correct / total


def val_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            loss           = criterion(outputs, labels)
            running_loss  += loss.item() * images.size(0)
            preds          = outputs.argmax(1)
            correct       += preds.eq(labels).sum().item()
            total         += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / total, 100.0 * correct / total, all_preds, all_labels


# ── Setup ────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model     = BrainTumourCNN(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()


# ── Train if training data changed ───────────────────────────────
if training_changed:
    print("\n── Training ─────────────────────────────────────")

    full_train = TensorDataset(args.processed_data, "Training")
    full_train_size = len(full_train)

    if full_train_size < 2:
        raise ValueError(
            f"Training requires at least 2 samples to create non-empty train/validation splits, "
            f"but found {full_train_size} sample(s)."
        )
    if not 0 < args.val_split < 1:
        raise ValueError(
            f"val_split must be strictly between 0 and 1, got {args.val_split}."
        )

    val_size = int(args.val_split * full_train_size)
    val_size = max(1, min(val_size, full_train_size - 1))
    train_size = full_train_size - val_size
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(
            torch.randint(0, 10000, (1,)).item()
        )
    )
    print(f"  Train split: {train_size}  |  Val split: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    optimizer = optim.Adam([
        {"params": model.model.fc.parameters(), "lr": args.learning_rate * 10},
        {"params": [p for n, p in model.model.named_parameters() if "fc" not in n], "lr": args.learning_rate},
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    mlflow.start_run()
    mlflow.log_params({
        "num_epochs":       args.num_epochs,
        "learning_rate":    args.learning_rate,
        "batch_size":       args.batch_size,
        "val_split":        args.val_split,
        "optimizer":        "Adam",
        "scheduler":        "ReduceLROnPlateau",
        "training_changed": True,
        "testing_changed":  testing_changed,
    })

    best_val_acc  = 0.0
    best_val_loss = float("inf")

    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc            = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, preds, labels = val_epoch(model, val_loader,    criterion, device)

        scheduler.step(val_loss)

        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
        }, step=epoch)

        print(f"Epoch {epoch:>3}  train={train_loss:.4f}/{train_acc:.2f}%  "
              f"val={val_loss:.4f}/{val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.model_output, "best_model.pt"))
            print(f"  ✅ New best saved (val_acc={best_val_acc:.2f}%)")

    mlflow.log_metrics({
        "best_val_acc":  best_val_acc,
        "best_val_loss": best_val_loss,
    })
    mlflow.end_run()

    with open(os.path.join(args.model_output, "metrics.txt"), "w") as f:
        f.write(str(round(best_val_acc, 4)))

    with open(os.path.join(args.model_output, "model_trained.flag"), "w") as f:
        f.write("true")

    print(f"\n── Complete ─────────────────────────────────────")
    print(f"  Best val acc: {best_val_acc:.2f}%")

# ── Testing changed only — no training needed ────────────────────
elif testing_changed and not training_changed:
    print("\n── Testing data changed but training unchanged ───")
    print("   Deploy step will handle re-inference via batch endpoint.")
    with open(os.path.join(args.model_output, "model_trained.flag"), "w") as f:
        f.write("false")
