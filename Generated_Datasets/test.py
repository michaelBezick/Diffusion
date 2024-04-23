import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

best_image = torch.load("./best_topology.pt")

best_image = torch.squeeze(best_image).numpy()

best_image = (best_image + 1) / 2

best_image *= 255

best_image = best_image.astype(np.uint8)

fig, ax = plt.subplots()
ax.imshow(best_image)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
plt.savefig("./best_topology.png", bbox_inches='tight')
#
# image = Image.fromarray(best_image, mode="L")
# image.save("./best_topology.jpg", quality=100)
