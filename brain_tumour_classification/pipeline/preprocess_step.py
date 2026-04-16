import os
import glob
import argparse
import json
import torch
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLASS_NAMES, DEFAULT_IMAGE_SIZE

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}

# ── Args ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--raw_data",                type=str)
parser.add_argument("--processed_data",          type=str)
parser.add_argument("--manifest_file_training",  type=str, default="")
parser.add_argument("--manifest_file_testing",   type=str, default="")
parser.add_argument("--n_augmentations",         type=int, default=3)
parser.add_argument("--debug_limit",             type=int, default=0)
args = parser.parse_args()

# ── Transforms ──────────────────────────────────────────────────
base_transform = transforms.Compose([
    transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

augment_transform = transforms.Compose([
    transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Print raw data structure ─────────────────────────────────────
print(f"\n── Raw data path: {args.raw_data}")
for root, dirs, files in os.walk(args.raw_data):
    depth = root.replace(args.raw_data, "").count(os.sep)
    if depth > 4:
        continue
    print("  " * depth + os.path.basename(root) + f"/  ({len(files)} files)")


# ── Helper: build manifest for a split ───────────────────────────
def build_manifest(split):
    manifest = set()
    for class_name in CLASS_NAMES:
        in_dir = os.path.join(args.raw_data, split, class_name)
        paths = (
            glob.glob(os.path.join(in_dir, "*.jpg"))  +
            glob.glob(os.path.join(in_dir, "*.jpeg")) +
            glob.glob(os.path.join(in_dir, "*.png"))
        )
        if args.debug_limit > 0:
            paths = paths[:args.debug_limit]
        for path in paths:
            manifest.add(os.path.relpath(path, os.path.join(args.raw_data, split)))
    return manifest


def load_previous_manifest(manifest_file):
    if manifest_file and os.path.exists(manifest_file):
        try:
            with open(manifest_file, "r") as f:
                return set(json.loads(f.read()))
        except Exception as e:
            print(f"[WARN] Could not parse manifest {manifest_file}: {e}")
    return set()


# ── Step 1: Build current manifests ─────────────────────────────
current_training = build_manifest("Training")
current_testing  = build_manifest("Testing")

print(f"\nCurrent training images: {len(current_training)}")
print(f"Current testing images:  {len(current_testing)}")

if args.debug_limit > 0:
    print(f"  ⚠️  Debug mode: limited to {args.debug_limit} files per class")

# ── Step 2: Load previous manifests & compute diffs ──────────────
previous_training = load_previous_manifest(args.manifest_file_training)
previous_testing  = load_previous_manifest(args.manifest_file_testing)

print(f"\nPrevious training manifest: {len(previous_training)} files")
print(f"Previous testing manifest:  {len(previous_testing)} files")

training_new     = current_training - previous_training
training_removed = previous_training - current_training
testing_new      = current_testing - previous_testing
testing_removed  = previous_testing - current_testing

training_changed = bool(training_new or training_removed)
testing_changed  = bool(testing_new or testing_removed)

print(f"\nTraining — new: {len(training_new)}, removed: {len(training_removed)}, changed: {training_changed}")
print(f"Testing  — new: {len(testing_new)}, removed: {len(testing_removed)}, changed: {testing_changed}")

# ── Step 3: If nothing changed, exit early ───────────────────────
if not training_changed and not testing_changed:
    print("\n✅ No changes in raw data — skipping preprocessing.")
    os.makedirs(args.processed_data, exist_ok=True)
    with open(os.path.join(args.processed_data, "NO_NEW_DATA"), "w") as f:
        f.write("no new data")
    exit(0)


# ── Step 4: Process files ────────────────────────────────────────
def process_split(split_name, manifest, apply_augmentation):
    """Process all files in the given manifest for a split."""
    saved   = 0
    skipped = 0

    for rel_path in manifest:
        full_path  = os.path.join(args.raw_data, split_name, rel_path)
        parts      = rel_path.split(os.sep)
        class_name = parts[0]
        stem       = os.path.splitext(parts[1])[0]

        out_dir = os.path.join(args.processed_data, split_name, class_name)
        os.makedirs(out_dir, exist_ok=True)

        try:
            with Image.open(full_path) as img:
                img.verify()

            with Image.open(full_path) as img:
                img = img.convert("RGB")

                # Base tensor
                out_path = os.path.join(out_dir, f"{stem}.pt")
                if not os.path.exists(out_path):
                    torch.save({
                        "tensor": base_transform(img),
                        "label": CLASS_TO_IDX[class_name]
                    }, out_path)
                    saved += 1

                # Augmented tensors (training only)
                if apply_augmentation:
                    for i in range(args.n_augmentations):
                        aug_path = os.path.join(out_dir, f"{stem}_aug{i}.pt")
                        if not os.path.exists(aug_path):
                            torch.save({
                                "tensor": augment_transform(img),
                                "label": CLASS_TO_IDX[class_name]
                            }, aug_path)
                            saved += 1

        except Exception as e:
            print(f"[WARN] Skipping {full_path}: {e}")
            skipped += 1

    return saved, skipped


# Always process both splits so the output folder is complete.
print("\nProcessing training files...")
train_saved, train_skipped = process_split("Training", current_training, apply_augmentation=True)
print(f"  Saved: {train_saved}, Skipped: {train_skipped}")

print("\nProcessing testing files...")
test_saved, test_skipped = process_split("Testing", current_testing, apply_augmentation=False)
print(f"  Saved: {test_saved}, Skipped: {test_skipped}")

# ── Step 5: Write manifests and flags ────────────────────────────
with open(os.path.join(args.processed_data, "manifest_training.json"), "w") as f:
    json.dump(sorted(current_training), f)

with open(os.path.join(args.processed_data, "manifest_testing.json"), "w") as f:
    json.dump(sorted(current_testing), f)

with open(os.path.join(args.processed_data, "training_changed.flag"), "w") as f:
    f.write("true" if training_changed else "false")

with open(os.path.join(args.processed_data, "testing_changed.flag"), "w") as f:
    f.write("true" if testing_changed else "false")

print(f"\n── Preprocessing complete ──────────────────────")
print(f"  Training tensors saved: {train_saved}")
print(f"  Testing tensors saved:  {test_saved}")
print(f"  Training manifest:      {len(current_training)} files")
print(f"  Testing manifest:       {len(current_testing)} files")
print(f"  Training changed:       {training_changed}")
print(f"  Testing changed:        {testing_changed}")
