import numpy as np
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes

def data_visualize(dataset, t):
    """
    Choose t continous time points in data and visualize the chosen points. Note that some datasets have more than one
    channel.
    param:
        dataset: dataset to visualize
        t: the number of timestamps to visualize
    """
    # data: (time_steps, channels) 2d-array
    data = dataset.data.squeeze(0)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=1)
    # randomly choose t-steps to show
    ridx = np.random.randint(0, dataset.data_stamp.size-t)
    data_show = data[ridx:ridx+t, :]
    data_stamp = dataset.data_stamp[ridx:ridx+t]
    # for every channels(features), draw a line, save a pic
    for c in range(len(dataset.data_cols)):
        channel = dataset.data_cols[c]
        data_channel = data_show[:, c]
        plt.plot(data_stamp, data_channel, label=channel)
        plt.legend()
        plt.savefig(f'./pics/{channel}.jpg')
        plt.close()
    # for all channels(features), draw multiple lines in a pic
    # WARNING: No way to show these lines together clearly, they are too far away
    # ax = brokenaxes(ylims=((0.1, 0.2), (0.7, 0.9), (1.5, 1.75)), despine=False)
    ax = brokenaxes(despine=False)
    for c in range(len(dataset.data_cols)):
        channel = dataset.data_cols[c]
        data_channel = data_show[:, c]
        ax.plot(data_stamp, data_channel, label=channel)
    ax.legend()
    plt.savefig(f'./pics/all.jpg')
