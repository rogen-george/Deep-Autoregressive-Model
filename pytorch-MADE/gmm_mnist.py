import matplotlib.pyplot as plt
#import seaborn as sns; sns.set()
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import pickle
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-path', required=True, type=str, help="Path to binarized_mnist.npz")
    args = parser.parse_args()

    print("loading binarized mnist from", args.data_path)
    mnist = np.load(args.data_path)
    xtr, xte = mnist['train_data'], mnist['valid_data']
    print("type(xtr): ", type(xtr))
    print("xtr.shape: ", xtr.shape)

    pca = PCA(0.99, whiten=True)
    data = pca.fit_transform(xtr)

    # Use of 110 as components number
    gmm = GaussianMixture(110, covariance_type='full', random_state=0)
    gmm.fit(data)
    print(gmm.converged_)

    # Generate new data
    data_new = gmm.sample(2)
    # use the inverse transform of the PCA object to construct the new digits
    digits_new = pca.inverse_transform(data_new[0])
    plot_digits(digits_new[0])




def plot_digits(data):
    plt.imsave("gmm_mnist_sample.png", data.reshape(28,28), cmap="gray_r")



if __name__ == '__main__':
    main()
