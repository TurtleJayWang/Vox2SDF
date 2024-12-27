import torch
import numpy as np
import pickle
import time

start = time.time()
end = time.time()

tensor = torch.randn(250000 + 128 * 128 * 128)
array = tensor.numpy()

print("Torch write")
with open("tensor.pkl", "b+a") as f:
    pickle.dump(tensor, f)
print("Numpy write")
np.save("array.npz", array)

print("Torch read")
start = time.time()
with open("tensor.pkl", "b+r") as f:
    tensor = pickle.load(f)
end = time.time()
print(end - start)
print("Numpy read")
start = time.time()
array = np.load("array.npz.npy")
end = time.time()
print(end - start)
