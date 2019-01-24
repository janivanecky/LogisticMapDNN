import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Tweakable variables
iterations = 2
R = 3.82843
cosine = True
freq = 1.0

def activation_fn(x):
    # Logistic map might diverge if values are not within (0,1) range (or R > 4),
    # so let's squash input values to (0, 1) range, either by `clamp` or `cos`.
    if cosine:
        x = torch.cos(x * freq) * 0.5 + 0.5
    else:
        x = torch.clamp(x, 0, 1)

    def logistic_map(x):
        return R * x * (1 - x)

    for _ in range(iterations):
        x = logistic_map(x)
    return x

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.affine1 = torch.nn.Linear(2, 64)
        self.affine2 = torch.nn.Linear(64, 64)
        self.affine3 = torch.nn.Linear(64, 64)
        self.affine4 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = activation_fn(self.affine1(x))
        x = activation_fn(self.affine2(x))
        x = activation_fn(self.affine3(x))
        x = activation_fn(self.affine4(x))
        return x

# Set up model and input data.
model = Model().cuda()
xs = np.linspace(-5, 5, 512)
ys = np.linspace(-5, 5, 512)
xs, ys = np.meshgrid(xs, ys)
xy_space = np.stack((xs, ys), axis=2) # HW
xy_space = torch.tensor(xy_space).float().cuda(0)

# Help.
print("""
Key Controls:
===
Q: Logistic map iteration count--
W: Logistic map iteration count++
===
A: Logistic map R multiplier - 0.01
S: Logistic map R multiplier +1 0.01
===
Z: If cosine is used before logistic map, this decreases cosine frequency by 0.01
X: If cosine is used before logistic map, this increases cosine frequency by 0.01
===
R: Reset DNN model (re-initialize weights)
T: Toggle cosine to be used before logistic map
P: Print current variable values
O: Save current image to 'img.png'
ESC: Quit
""")

# App loop.
while(True):
    img = model(xy_space).cpu().detach().numpy()
    cv2.imshow('image', img)

    # Controls handling.
    k = cv2.waitKey(16) & 0xEFFFFF
    if k == ord('q'):
        iterations = max(iterations - 1, 0)
    elif k == ord('w'):
        iterations += 1
    elif k == ord('a'):
        R -= 0.01
    elif k == ord('s'):
        R += 0.01
    elif k == ord('z'):
        freq -= 0.01
    elif k == ord('x'):
        freq += 0.01
    elif k == ord('r'):
        model = Model().cuda()
    elif k == ord('t'):
        cosine = not cosine
    elif k == ord('o'):
        cv2.imwrite('img.png', (img * 255).astype(np.uint8))
    elif k == ord('p'):
        print("Logistic Map Iterations: {}".format(iterations))
        print("Logistic Map R: {}".format(R))
        print("Cosine: {}".format(cosine))
        print("Frequency: {}".format(freq))
    elif k == 27: # Esc
        break