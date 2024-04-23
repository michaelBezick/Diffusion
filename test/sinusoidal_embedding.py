import matplotlib
import torch
import matplotlib.pyplot as plt
import numpy as np
import math as m
from matplotlib.colors import BoundaryNorm
import torch.nn as nn

class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = m.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        # embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        interleaved = torch.empty(time.size(0), self.dim)
        interleaved[:, 0::2] = embeddings.sin()
        interleaved[:, 1::2] = embeddings.cos()
        return interleaved

batch_size = 100
encoding_size = 100
scale = 1000
encoder = SinusoidalPositionalEmbeddings(encoding_size)
positions = torch.linspace(0.5, 1, batch_size) * scale
print(positions)
torch.set_printoptions(sci_mode=False)
encoded = encoder(positions)
print(encoded[0, :])

boundaries = np.linspace(0, 1, 10)
# cmap = cm.get_cmap(matplotlib.colormaps["viridis"], 10)
cmap = matplotlib.colormaps["viridis"]

norm = BoundaryNorm(boundaries, cmap.N, clip=True)

fig, ax = plt.subplots()
new_vectors = torch.zeros((encoding_size, batch_size))
for i in range(batch_size):
    new_vectors[:, i] = encoded[i, :]

print(new_vectors)
im = ax.imshow(new_vectors, cmap=cmap, norm=norm, origin='lower')
# cbar = fig.colorbar(im, ax=ax, boundaries=boundaries, ticks=boundaries)
# cbar.set_label('Value Interval')

# Add labels and title
# ax.set_title('Sinusoidal Positional Encodings')
# ax.set_xlabel('FOM')
# ax.set_ylabel('Positional Encoding')

ax.set_yticklabels([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_xticks([])

# Show the plot
plt.savefig("Sinusoidal Positional Encodings", dpi=300, bbox_inches='tight')
