# MLPython datasets wrapper
import os
import time as t
import theano
import numpy as np
import mlpython.mlproblems.generic as mlpb
import mlpython.datasets.store as dataset_store
import mlpython.datasets.binarized_mnist as mldataset
from utils import get_done_text

class Dataset(object):

    @staticmethod
    def get_permutation(input_size):
        # Generate dataset of all possible permutations for size input_size
        import itertools
        print("Generating dataset of all possible permutations for size ({0}) input_size ...".format(input_size))
        dataset = []
        for i in itertools.product("01", repeat=input_size):
            dataset.append(i)
        fake_dataset = {'input_size': input_size}
        fake_dataset['valid'] = {'data': theano.shared(value=np.asarray(dataset, dtype=theano.config.floatX), borrow=True)}
        fake_dataset['train'] = {'data': theano.shared(value=np.asarray(dataset[:int(len(dataset) * 0.10)], dtype=theano.config.floatX), borrow=True)}
        fake_dataset['test'] = {'data': theano.shared(value=np.asarray(dataset, dtype=theano.config.floatX), borrow=True)}
        print("Done, len() of {0}".format(len(dataset)))
        return fake_dataset

    @staticmethod
    def get_fake(in_size=4, dataset_size=1):
        fake_dataset = {'input_size': in_size}
        fake_dataset['valid'] = {'data': theano.shared(value=np.zeros((dataset_size, in_size), dtype=theano.config.floatX), borrow=True), 'length': dataset_size}
        fake_dataset['train'] = {'data': theano.shared(value=np.zeros((dataset_size, in_size), dtype=theano.config.floatX), borrow=True), 'length': dataset_size}
        fake_dataset['test'] = {'data': theano.shared(value=np.zeros((dataset_size, in_size), dtype=theano.config.floatX), borrow=True), 'length': dataset_size}
        return fake_dataset

    @staticmethod
    def _subSample(dataset):
        # TAKE SAME SAMPLE AS Marco ####
        rng2 = np.random.mtrand.RandomState(1234)
        percent = 0.1
        idx = rng2.choice(np.arange(len(dataset)), size=int(len(dataset) * percent), replace=False)
        dataset = mlpb.SubsetProblem(dataset, subset=idx)

    @staticmethod
    def _clean(dataset):
        data = []
        for i in dataset:
            data.append(i)
        return np.asarray(data, dtype=theano.config.floatX)

    @staticmethod
    def get(dataset_name):
        # List of datasets that works with the current model ?
        datasets = ['adult',
                    'binarized_mnist',
                    'connect4',
                    'dna',
                    'mushrooms',
                    'nips',
                    'ocr_letters',
                    'rcv1',
                    'rcv2_russ',
                    'web']

        # Setup dataset env
        if dataset_name not in datasets:
            raise ValueError('Dataset unknown: ' + dataset_name)
        # mldataset = __import__('mlpython.datasets.' + dataset_name, globals(), locals(), [dataset_name], -1)
        datadir = os.path.join(os.getenv("MLPYTHON_DATASET_REPO"), dataset_name)

        # Verify if dataset exist and if not, download it
        if(not os.path.exists(datadir)):
            dataset_store.download(dataset_name)

        print('### Loading dataset [{0}] ...'.format(dataset_name))
        start_time = t.time()

        all_data = mldataset.load(datadir, load_to_memory=True)
        train_data, train_metadata = all_data['train']

        if dataset_name == 'binarized_mnist' or dataset_name == 'nips':
            trainset = mlpb.MLProblem(train_data, train_metadata)
        else:
            trainset = mlpb.SubsetFieldsProblem(train_data, train_metadata)

        trainset.setup()

        valid_data, valid_metadata = all_data['valid']

        validset = trainset.apply_on(valid_data, valid_metadata)

        test_data, test_metadata = all_data['test']
        testset = trainset.apply_on(test_data, test_metadata)

        # Cleaning up, packaging and theanized
        full_dataset = {'input_size': trainset.metadata['input_size']}

        trainset_theano = theano.shared(value=Dataset._clean(trainset), borrow=True)
        validset_theano = theano.shared(value=Dataset._clean(validset), borrow=True)
        testset_theano = theano.shared(value=Dataset._clean(testset), borrow=True)

        full_dataset['train'] = {'data': trainset_theano, 'length': all_data['train'][1]['length']}
        full_dataset['valid'] = {'data': validset_theano, 'length': all_data['valid'][1]['length']}
        full_dataset['test'] = {'data': testset_theano, 'length': all_data['test'][1]['length']}

        print("(Dim:{0} Train:{1} Valid:{2} Test:{3})".format(trainset.metadata['input_size'], full_dataset['train']['length'], full_dataset['valid']['length'], full_dataset['test']['length']))
        print(get_done_text(start_time), "###")
        return full_dataset
