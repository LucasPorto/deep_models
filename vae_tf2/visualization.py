import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model

std_z1, std_z2 = np.random.multivariate_normal([0, 0], np.eye(2), size=10000).T

def visualize_distributions(encoder: Model, X, y, epoch):
    X = X.numpy()
    X_stratified, _, y_stratified, _ = train_test_split(X, y, train_size=1000, stratify=y)

    z, _, _ = encoder(X_stratified)
    z = z[:, :, 0].numpy()

    g = sns.jointplot(std_z1, std_z2, kind='kde')
    g.set_axis_labels('z_0', 'z_1')
    c_palette = sns.color_palette("bright", 10)

    g.x = z[:, 0]
    g.y = z[:, 1]
    g.plot_marginals(sns.distplot, kde=True,
                     hist=False,
                     color='r', kde_kws={'alpha': 0.5})

    for label in range(np.unique(y_stratified).shape[0]):
        g.x = z[y_stratified == label, 0]
        g.y = z[y_stratified == label, 1]
        g.plot_joint(plt.scatter, color=c_palette[label], s=1.5)


    g.savefig('outputs/dist_vis_{:04d}.jpg'.format(epoch))
    plt.close('all')


