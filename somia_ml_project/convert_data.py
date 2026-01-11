import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_dir = sys.argv[1]
H = int(sys.argv[2])
W = int(sys.argv[3])
C = int(sys.argv[4])
output_file = sys.argv[5]
correct_data = int(sys.argv[6])  # 1 = correct, 0 = no correction

X = []
T = []
skipped_images = []
files = os.listdir(image_dir)
print("=" * 60)
print(f"Processing images from: {image_dir}")
print(f"Target shape: {H}x{W} with {C} channel(s)")
print(f"Correction enabled: {correct_data == 1}")
print("=" * 60)
for fname in files:
    if not fname.lower().endswith(".png"):
        continue

    path = os.path.join(image_dir, fname)

    # Extract label from filename
    try:
        label = int(fname.split("-")[0])
    except:
        skipped_images.append((fname, "invalid_label"))
        continue

    # Load image safely
    try:
        img = Image.open(path)
    except:
        skipped_images.append((fname, "corrupted_file"))
        continue

    try:
        # Convert to required channels
        if C == 1:
            img = img.convert("L")
        elif C == 3:
            img = img.convert("RGB")

        # Resize
        img = img.resize((W, H))

        # Convert to NumPy
        img_array = np.asarray(img, dtype=np.float32)

        # Check for blank/corrupted images (mostly black or white)
        if correct_data == 1:
            img_mean = np.mean(img_array)
            img_std = np.std(img_array)

            # Skip nearly blank images (std < 5)
            if img_std < 5:
                skipped_images.append((fname, "blank_image"))
                continue

            # Skip images with extreme values (potential corruption)
            if img_mean < 10 or img_mean > 245:
                skipped_images.append((fname, "extreme_values"))
                continue

        # Normalize
        img_array /= 255.0

        # Ensure correct shape
        if C == 1:
            img_array = img_array.reshape(H, W, 1)

        # Skip invalid images
        if img_array.shape != (H, W, C):
            skipped_images.append((fname, "wrong_shape"))
            if correct_data == 1:
                continue

        X.append(img_array)
        T.append(label)

    except Exception as e:
        skipped_images.append((fname, str(e)))
        if correct_data == 1:
            continue

X = np.array(X)
T = np.array(T)

print("\n" + "=" * 60)
print("DATA ANALYSIS")
print("=" * 60)
print(f"Final dataset shape: {X.shape}")
print(f"Labels shape: {T.shape}")
print(f"Unique classes: {len(np.unique(T))}")
print(f"Images skipped: {len(skipped_images)}")

if len(skipped_images) > 0 and correct_data == 1:
    print("\nSkipped images breakdown:")
    skip_reasons = {}
    for fname, reason in skipped_images:
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
    for reason, count in sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count}")

# Print class distribution
print("\nClass distribution:")
unique, counts = np.unique(T, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  Class {int(label)}: {count} images")

# Print pixel statistics
print("\nPixel statistics:")
print(f"  Mean: {np.mean(X):.4f}")
print(f"  Std: {np.std(X):.4f}")
print(f"  Min: {np.min(X):.4f}")
print(f"  Max: {np.max(X):.4f}")

print("=" * 60)
np.savez(output_file, X=X, T=T)
print(f"Saved data to: {output_file}")
print("=" * 60)
