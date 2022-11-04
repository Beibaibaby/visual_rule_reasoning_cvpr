import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

fig = plt.figure(figsize=(4.5, 6.9))
heights = (5, 2)
outer = gridspec.GridSpec(
        2, 1,
        wspace=0.2,
        hspace=0.2,
        height_ratios=heights)

img = np.load('distribute_nine/RAVEN_0_train.npz')['image']
shapes = [(3, 3), (2, 4)]
for j in range(2):
    inner = gridspec.GridSpecFromSubplotSpec(
            *shapes[j],
            subplot_spec=outer[j],
            wspace=0.1, hspace=0.1)
    for i in range(9):
        if i != 8:
            ax = plt.Subplot(fig, inner[i // shapes[j][1], i % shapes[j][1]])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(img[i+j*8, :, :], plt.get_cmap('gist_heat'))
            fig.add_subplot(ax)
fig.savefig("view.pdf")
