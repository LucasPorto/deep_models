import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from scipy.stats import norm
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from vae import VariationalAutoencoder

std_z0, std_z1 = np.random.multivariate_normal([0, 0], np.eye(2), size=500).T

def visualize_distributions(vae: VariationalAutoencoder, X, y, epoch):
    X = X.numpy()
    X_stratified, _, y_stratified, _ = train_test_split(X, y, train_size=1000, stratify=y)

    encoder = vae.encoder
    decoder = vae.decoder

    z, _, _ = encoder(X_stratified)
    z = z[:, :, 0].numpy()

    plt.figure()
    g = sns.JointGrid(std_z0, std_z1, xlim=(-4,4), ylim=(-4,4))
    g.plot_joint(sns.kdeplot, kernel='gau', bw=1.0)
    g.plot_marginals(sns.distplot, fit=norm, kde=False, hist=False,
                     fit_kws={'color': 'blue'})
    g.set_axis_labels('z[0]', 'z[1]')
    c_palette = sns.color_palette("bright", 10)

    g.x = z[:, 0]
    g.y = z[:, 1]
    g.plot_marginals(sns.distplot, kde=True,
                     hist=False,
                     color='r', kde_kws={'alpha': 0.5})

    for label in range(np.unique(y_stratified).shape[0]):
        g.x = z[y_stratified == label, 0]
        g.y = z[y_stratified == label, 1]
        g.plot_joint(plt.scatter, color=c_palette[label])

    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    g.savefig('outputs/posterior_{:04d}.jpg'.format(epoch))

    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    samples = np.linspace(-3, 3, n)
    x, y = np.meshgrid(samples, -samples)
    grid = np.stack((x, y)).transpose((2, 1, 0)).reshape(-1, 2)
    x_decoded = decoder.predict(grid)
    x_decoded = x_decoded.reshape(n, n, digit_size, digit_size)

    for i_x in range(n):
        for i_y in range(n):
            digit = x_decoded[i_x, i_y]
            figure[i_y * digit_size: (i_y + 1) * digit_size,
            i_x * digit_size: (i_x + 1) * digit_size] = digit

    f = plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(samples, 1)
    sample_range_y = np.round(-samples, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    f.savefig('outputs/recon_{:04d}.jpg'.format(epoch))

    plt.close('all')


