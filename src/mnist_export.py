import urllib.request
import gzip
import numpy as np
import os

def download(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)

base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
files = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
]

for f in files:
    download(base + f, f)

with gzip.open("train-images-idx3-ubyte.gz", "rb") as f:
    f.read(16)
    train_images = np.frombuffer(f.read(), np.uint8).reshape(-1, 784).astype(np.float32) / 255.0

with gzip.open("train-labels-idx1-ubyte.gz", "rb") as f:
    f.read(8)
    train_labels = np.frombuffer(f.read(), np.uint8)

train_images.tofile("mnist_train_images.bin")
train_labels.tofile("mnist_train_labels.bin")

print(f"Train images: {train_images.shape}")
print(f"Train labels: {train_labels.shape}")
print("Done.")