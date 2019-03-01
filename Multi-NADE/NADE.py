import numpy as np
import theano
import theano.tensor as T

from weights_initializer import WeightsInitializer
from momentums import DecreasingLearningRate, AdaDelta, AdaGrad, RMSProp, Adam, Adam_paper


class NADE(object):
    def __init__(self, dataset,
                 learning_rate=0.001,
                 decrease_constant=0,
                 hidden_size=500,
                 random_seed=1234,
                 batch_size=1,
                 hidden_activation=T.nnet.sigmoid,
                 use_cond_mask=False,
                 direct_input_connect="None",
                 direct_output_connect=False,
                 momentum="None",
                 dropout_rate=0,
                 weights_initialization="Uniform",
                 mask_distribution=0,
                 tied=False):

    #def __init__(self, random_seed, l_rate=None, input=None, input_size=400, hidden_size=200, tied=False):
        input_size = dataset['input_size']
        self.hidden_activation = hidden_activation

        class SeedGenerator(object):
        # This class purpose is to maximize randomness and still keep reproducibility
            def __init__(self, random_seed):
                self.rng = np.random.mtrand.RandomState(random_seed)

            def get(self):
                return self.rng.randint(42424242)
        seed_generator = SeedGenerator(random_seed)
        weights_initialization = getattr(WeightsInitializer(seed_generator.get()), weights_initialization)  # Get the weights initializer by string name

        # Initialize layers
        self.W = theano.shared(value=weights_initialization((input_size, hidden_size)), name='W', borrow=True)
        self.b = theano.shared(value=np.zeros(hidden_size, dtype=theano.config.floatX), name='bhid', borrow=True)
        self.b_prime = theano.shared(value=np.zeros(input_size, dtype=theano.config.floatX), name='bvis', borrow=True)

        self.parameters = [self.W, self.b, self.b_prime]

        # Init tied W
        if tied:
            self.W_prime = self.W
        else:
            self.W_prime = theano.shared(value=weights_initialization((input_size, hidden_size)), name='W_prime', borrow=True)
            self.parameters.append(self.W_prime)

        # Initialize momentum
        if momentum == "None":
            self.momentum = DecreasingLearningRate(learning_rate, decrease_constant)
        elif momentum == "adadelta":
            self.momentum = AdaDelta(decay=decrease_constant, epsilon=learning_rate)
        elif momentum == "adagrad":
            self.momentum = AdaGrad(learning_rate=learning_rate)
        elif momentum == "rmsprop":
            self.momentum = RMSProp(learning_rate=learning_rate, decay=decrease_constant)
        elif momentum == "adam":
            self.momentum = Adam(learning_rate=learning_rate)
        elif momentum == "adam_paper":
            self.momentum = Adam_paper(learning_rate=learning_rate)

        # The loss function
        input = T.matrix(name="input")
        nll, output = self.get_nll(input)
        loss = nll.mean()

        # How to update the parameters
        parameters_gradient = T.grad(loss, self.parameters)
        updates = self.momentum.get_updates(zip(self.parameters, parameters_gradient))

        #
        # Functions to train and use the model
        index = T.lscalar()
        self.learn = theano.function(name='learn',
                                     #inputs=[index, current_iteration],
                                     inputs=[index],
                                     outputs=loss,
                                     updates=updates,
                                     givens={input: dataset['train']['data'][index * batch_size:(index + 1) * batch_size]},
                                     on_unused_input='ignore')  # ignore for when dropout is absent

        self.use = theano.function(name='use',
                                   inputs=[input],
                                   outputs=output,
                                   on_unused_input='ignore')  # ignore for when dropout is absent

        # Test functions
        self.valid_log_prob = theano.function(name='valid_log_prob',
                                              inputs=[index],
                                              outputs=nll,
                                              givens={input: dataset['valid']['data'][index * batch_size:(index + 1) * batch_size]},
                                              on_unused_input='ignore')  # ignore for when dropout is absent
        self.train_log_prob = theano.function(name='train_log_prob',
                                              inputs=[index],
                                              outputs=nll,
                                              givens={input: dataset['train']['data'][index * batch_size:(index + 1) * batch_size]},
                                              on_unused_input='ignore')  # ignore for when dropout is absent
        self.test_log_prob = theano.function(name='test_log_prob',
                                             inputs=[index],
                                             outputs=nll,
                                             givens={input: dataset['test']['data'][index * batch_size:(index + 1) * batch_size]},
                                             on_unused_input='ignore')  # ignore for when dropout is absent

    def get_nll(self, input):
        input_times_W = input.T[:, :, None] * self.W[:, None, :]

        #acc_input_times_W = T.concatenate([T.zeros_like(input_times_W[[0]]), T.cumsum(input_times_W, axis=0)[:-1]], axis=0)
        # Hack for no GPUSplit
        acc_input_times_W = T.cumsum(input_times_W, axis=0)
        #acc_input_times_W = T.roll(acc_input_times_W, 1, axis=1???)  # USES Join internally too
        acc_input_times_W = T.set_subtensor(acc_input_times_W[1:], acc_input_times_W[:-1])
        acc_input_times_W = T.set_subtensor(acc_input_times_W[0, :], 0.0)

        acc_input_times_W += self.b[None, None, :]
        h = self.hidden_activation(acc_input_times_W)

        pre_output = T.sum(h * self.W_prime[:, None, :], axis=2) + self.b_prime[:, None]
        output = T.nnet.sigmoid(pre_output)
        nll = T.sum(T.nnet.softplus(-input.T * pre_output + (1 - input.T) * pre_output), axis=0)
        return nll, output
