# Deep-Autoregressive-Model

Deep, generative autoencoder capable of learning hierarchies of distributed representations from data.

Xavier initialization

A nice explanation of Xavier initialization for weights

http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization

Paper for this idea - http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

KL Divergence

In mathematical statistics, the Kullback–Leibler divergence (also called relative entropy) is a measure of how one probability distribution is different from a second, reference probability distribution.

Reparameterization Trick

This is the process of representing a normal distribution with a learned mean and standard deviation. The mean and standard deviation are learned using back propagation in a neural network. Let the learned standard deviation be u and the mean be m. Then we can convert this into a normal distribution by multiplying the u with a sample taken from a unit normal distribution and adding the mean m. ie. If the sample is x   We can represent it as a normal distribution by  x * u + m.


