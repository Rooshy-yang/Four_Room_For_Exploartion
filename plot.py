
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dir_ = 'outputs/2022-09-01/{}/0.npy'
names = ('ourc', 'baseline', 'ourd_state')


grid_width = 13
grid_height = 13
f, ax = plt.subplots(ncols=len(names))

for idx, name in enumerate(names):
    state = np.load(dir_.format(name)).astype(np.int32)
    print('computing ', name)
    position = np.zeros([grid_width, grid_height])
    for _, value in enumerate(state):
        row = value // grid_height
        col = value % grid_height
        position[row, col] += 1
    np.save('{}_position_data.npy'.format(name), position)
    sns.heatmap(position, ax=ax[idx])

plt.show()
figure = ax.get_figure()
figure.savefig('heat_maps.jpg')
plt.close()
