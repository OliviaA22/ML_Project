import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image



image_dir = sys.argv[1]
h = int(sys.argv[2])
w = int(sys.argv[3])
c = int(sys.argv[4]) # Asumming 1 Channels (for grayscale)
output_file = sys.argv[5]
mode = sys.argv[6]


print(f"Processing images from {image_dir} to size {h}x{w}x{c}...")


images = []
labels = []
processed_count = 0
skipped_count = 0

# Get list of all files in directory
files = sorted(os.listdir(image_dir))

# Load all images
for file in files:
    file_path = os.path.join(image_dir, file)

    label = int(file.split('-')[0]) # Extract label from file format:


    

    # --- Image Processing ---
    if mode == '1':
        # MODE 1: Robust checking (The "First Time" run)
        try:
            with Image.open(file_path) as img:
                img.load() # Force load to check for file corruption
                            
            # Resize if dimensions don't match expected size
            if img.size != (w, h):
                img = img.resize((w, h))

            # Convert to numpy array
            img_array = np.array(img)

            # Validate shape
            if img_array.shape[:2] != (h, w):
                skipped_count += 1
                raise ValueError(f"Image dimension not corresponding after resize.")
            
                
            images.append(img_array)
            labels.append(label)
            processed_count += 1

        except Exception as e:
            # In Mode 1, we expect corruption, so we catch it and skip
            print(f"Correction: Removing corrupted file {file} -> {e}")
            skipped_count += 1

    else:
        # MODE 0: Fast processing (The "Already Corrected" run)
        # We assume files are valid to save time.
        with Image.open(file_path) as img:
            
            img = img.resize((w, h))
            img_array = np.array(img)
            
            images.append(img_array)
            labels.append(label)
            processed_count += 1
        
X = np.array(images)
y = np.array(labels)

# Normalize
X = X.astype('float32') / 255.0

# Reshape (N, H, W, C)
X = X.reshape(-1, h, w, c)
print(f"--- Done ---")
print(f"Processed: {processed_count}")
if mode == '1':
    print(f"Corrupted/Skipped: {skipped_count}")
print(f"Saving to {output_file}...")


np.savez(output_file, images=X, labels=y)



