import sys
import os
import numpy as np
from PIL import Image



image_dir = sys.argv[1]
h = int(sys.argv[2])
w = int(sys.argv[3])
c = int(sys.argv[4])
output_file = sys.argv[5]
mode = sys.argv[6]



images = []
labels = []

files = os.listdir(image_dir)

for file_name in files:
    try:
        file_path = os.path.join(image_dir, file_name)
        label = int(file_name.split("-")[0])

        img = Image.open(file_path)
        img = img.resize((w, h))
        img_array = np.asarray(img)
        if mode == '1':
            white_rows = np.all(img_array == 255, axis=1)
            img_array = img_array[~white_rows]
            rows_missing = h - img_array.shape[0]
            padding = np.zeros((rows_missing, w), dtype=img_array.dtype)
            img_array = np.vstack([img_array, padding])

        img_array = img_array / 255

        images.append(img_array)
        labels.append(label)
    except Exception as e:
        print(f"Skipping file {file_name}: {e}")
        continue

images = np.array(images, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

 # Problem 2:
if mode == '1':
    unique_labels, counts = np.unique(labels, return_counts=True)
    max_count = counts.max()

    new_images = [images]
    new_labels = [labels]

    for label, count in zip(unique_labels, counts):
        if count < max_count:
            diff = max_count - count
            class_indices = np.where(labels == label)[0]

            new_indices = np.random.choice(class_indices, diff, replace=True)

            new_images.append(images[new_indices])
            new_labels.append(labels[new_indices])
    images = np.concatenate(new_images)
    labels = np.concatenate(new_labels)

np.savez(output_file, images=images, labels=labels)

print("Saved cleaned dataset to:", output_file)
print("Final image shape:", images.shape)
