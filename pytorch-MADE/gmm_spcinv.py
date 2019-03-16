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

    print("loading binarized space invaders dataset", args.data_path)
    flag = True
    pickle_file = open(args.data_path,'rb')
    count = 0
    try:
        while True:
            count += 1
            if count > 1000:
                break
            imgarray = pickle.load(pickle_file)
            imgarray = imgarray.flatten()
            imgarray = imgarray.reshape([1, imgarray.shape[0]])
            if flag:
                spcinv_data = imgarray
                flag = False
            else:
                spcinv_data = np.vstack((spcinv_data, imgarray))
    except EOFError:
        pass

    print("shape", spcinv_data.shape)
    train_split_index = spcinv_data.shape[0]
    train_split_index = int(train_split_index * 4 / 5)
    xtr = spcinv_data[0:train_split_index,:]
    xte = spcinv_data[train_split_index:,:]
    print("xtr.shape: ", xtr.shape)
    print("xte.shape: ", xte.shape)

    pca = PCA(0.99, whiten=True)
    data = pca.fit_transform(xtr)

    # Use of 10 as components number
    gmm = GaussianMixture(110, covariance_type='full', random_state=0)
    gmm.fit(data)
    print(gmm.converged_)

    # Generate new data
    data_new = gmm.sample(2)
    # use the inverse transform of the PCA object to construct the new digits
    digits_new = pca.inverse_transform(data_new[0])
    plot_digits(digits_new[0])




def plot_digits(data):
    plt.imsave("gmm_atari_sample.png", data.reshape(210,160), cmap="gray_r")



if __name__ == '__main__':
    main()
