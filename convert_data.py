import os
import pandas as pd
import numpy as np ;
import requests ;
import matplotlib.pyplot as plt ;
from pathlib import Path
from PIL import Image
from itertools import islice



folder = Path("fashion_mnist")
images = []

for img_path in folder.glob("*.png"):  
    img = Image.open(img_path)
    images.append(img)



# Option 1:
plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(img[i], cmap='gray')  # no reshape
    plt.axis('off')
plt.suptitle("First 5 Images")
plt.show()



from itertools import islice
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

folder = Path("fashion_mnist")
images = []

