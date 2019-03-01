#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import sys
import os
import time as t
import numpy as np
import theano
import theano.sandbox.softsign

from NADE import NADE

from dataset import Dataset
import utils


def get_done_text(start_time):
    return "DONE in {:.4f} seconds.".format(t.time() - start_time)


def get_mean_error_and_std(model, error_fnc, nb_batches):
    nll = []
    for index in range(nb_batches):
        nll += [error_fnc(index)]
    losses = np.concatenate(nll)
    return round(losses.mean(), 6), round(losses.std() / np.sqrt(losses.shape[0]), 6)


def train_model(model, dataset, look_ahead, max_epochs, batch_size, save_model_path=None, trainer_status=None):
    start_training_time = t.time()

    if trainer_status is None:
        trainer_status = {
            "best_valid_error": np.inf,
            "best_epoch": 0,
            "epoch": 0,
            "nb_of_epocs_without_improvement": 0
        }

    print('\n### Training NADE ###')
    while(trainer_status["epoch"] < max_epochs and trainer_status["nb_of_epocs_without_improvement"] < look_ahead):
        trainer_status["epoch"] += 1

        print('Epoch {0} (Batch Size {1})'.format(trainer_status["epoch"], batch_size))
        print('\tTraining   ...')
        start_time = t.time()
        nb_iterations = int(np.ceil(dataset['train']['length'] / batch_size))
        train_err = 0
        for index in range(nb_iterations):
            #current_iteration = ((trainer_status["epoch"] - 1) * nb_iterations) + index
            #train_err += model.learn(index, current_iteration)
            train_err += model.learn(index)
            # print train_err

        print(utils.get_done_text(start_time), " avg NLL: {0:.6f}".format(train_err / nb_iterations))

        print('\tValidating ...')
        start_time = t.time()
        valid_err, valid_err_std = get_mean_error_and_std(model, model.valid_log_prob, batch_size)
        print(utils.get_done_text(start_time), " NLL: {0:.6f}".format(valid_err))

        if valid_err < trainer_status["best_valid_error"]:
            trainer_status["best_valid_error"] = valid_err
            trainer_status["best_epoch"] = trainer_status["epoch"]
            trainer_status["nb_of_epocs_without_improvement"] = 0
            # Save best model
            if save_model_path is not None:
                save_model_params(model, save_model_path)
                utils.save_dict_to_json_file(os.path.join(save_model_path, "trainer_status"), trainer_status)
        else:
            trainer_status["nb_of_epocs_without_improvement"] += 1

    print("### Training", utils.get_done_text(start_training_time), "###")
    total_train_time = t.time() - start_training_time
    return trainer_status["best_epoch"], total_train_time


def build_model(dataset, trainingparams, hyperparams, hidden_size):
    print('\n### Initializing NADE ... ')
    start_time = t.time()
    model = NADE(dataset,
                 learning_rate=trainingparams['learning_rate'],
                 decrease_constant=trainingparams['decrease_constant'],
                 hidden_size=hidden_size,
                 random_seed=hyperparams['random_seed'],
                 batch_size=trainingparams['batch_size'],
                 hidden_activation=activation_functions[hyperparams['hidden_activation']],
                 momentum=trainingparams['momentum'],
                 dropout_rate=trainingparams['dropout_rate'],
                 weights_initialization=hyperparams['weights_initialization'],
                 tied=hyperparams['tied'])
    print(utils.get_done_text(start_time), "###")
    # printParams(model)
    return model


def parse_args(args):
    import argparse

    class GroupedAction(argparse.Action):

        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            super(GroupedAction, self).__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            group = self.container.title
            dest = self.dest
            groupspace = getattr(namespace, group, argparse.Namespace())
            setattr(groupspace, dest, values)
            setattr(namespace, group, groupspace)

    parser = argparse.ArgumentParser(description='Train the NADE model.')

    group_trainer = parser.add_argument_group('train')
    group_trainer.add_argument('dataset_name', action=GroupedAction, default=argparse.SUPPRESS)
    group_trainer.add_argument('learning_rate', type=float, action=GroupedAction, default=argparse.SUPPRESS)
    group_trainer.add_argument('decrease_constant', type=float, action=GroupedAction, default=argparse.SUPPRESS)
    group_trainer.add_argument('max_epochs', type=lambda x: np.inf if x == "-1" else int(x), help="If -1 will run until convergence.", action=GroupedAction, default=argparse.SUPPRESS)
    group_trainer.add_argument('batch_size', type=int, action=GroupedAction, default=argparse.SUPPRESS)
    group_trainer.add_argument('look_ahead', type=int, action=GroupedAction, default=argparse.SUPPRESS)
    group_trainer.add_argument('momentum', choices=['None', 'adadelta', 'adagrad', 'rmsprop', 'adam', 'adam_paper'], action=GroupedAction, default=argparse.SUPPRESS)
    group_trainer.add_argument('dropout_rate', type=float, action=GroupedAction, default=argparse.SUPPRESS)

    group_model = parser.add_argument_group('model')
    group_model.add_argument('hidden_size', type=int, action=GroupedAction, default=argparse.SUPPRESS)
    group_model.add_argument('random_seed', type=int, action=GroupedAction, default=argparse.SUPPRESS)
    group_model.add_argument('hidden_activation', choices=activation_functions.keys(), action=GroupedAction, default=argparse.SUPPRESS)
    group_model.add_argument('weights_initialization', action=GroupedAction, default=argparse.SUPPRESS)
    group_model.add_argument('tied', metavar="tied", type=eval, choices=[False, True], action=GroupedAction, default=argparse.SUPPRESS)

    parser.add_argument("--force", required=False, action='store_true', help="Override the already trained model if it exists insteat of resuming training.")
    parser.add_argument("--name", required=False, help="Set the name of the expirement insted of hasing it from the arguments.")

    args = parser.parse_args()

    return args


def save_model_params(model, model_path):
    np.savez_compressed(os.path.join(model_path, "params"), model.parameters, model.momentum.parameters)


def load_model_params(model, model_path):
    for i, param in enumerate(np.load(os.path.join(model_path, "params.npz"))['arr_0']):
        model.parameters[i].set_value(param.get_value())

    for i, param in enumerate(np.load(os.path.join(model_path, "params.npz"))['arr_1']):
        model.momentum.parameters[i].set_value(param.get_value())

activation_functions = {
    "sigmoid": theano.tensor.nnet.sigmoid,
    "hinge": lambda x: theano.tensor.maximum(x, 0.0),
    "softplus": theano.tensor.nnet.softplus,
    "tanh": theano.tensor.tanh,
    "softsign": theano.sandbox.softsign.softsign,
    "brain": lambda x: theano.tensor.maximum(theano.tensor.log(theano.tensor.maximum(x + 1, 1)), 0.0)
}

if __name__ == '__main__':
    resume_mode = False

    #
    # Pars args from the shell
    args = parse_args(sys.argv)
    dataset_name = args.train.dataset_name
    hyperparams = vars(args.model)
    trainingparams = vars(args.train)

    #
    # Set the name of the experiment (remove the --force from the args to make sure it will generate the same uid)
    if '--force' in sys.argv:
        sys.argv.remove('--force')
    experiment_name = args.name if args.name is not None else utils.generate_uid_from_string(' '.join(sys.argv))

    #
    # Creating the experiments folder or resuming experiment
    save_path_experiment = os.path.join('./experiments/', experiment_name)
    if os.path.isdir(save_path_experiment):
        if not args.force:
            print("### Resuming experiment ({0}). ###\n".format(experiment_name))
            loaded_hyperparams = utils.load_dict_from_json_file(os.path.join(save_path_experiment, "hyperparams"))
            loaded_trainingparams = utils.load_dict_from_json_file(os.path.join(save_path_experiment, "trainingparams"))

            if loaded_trainingparams != trainingparams or loaded_hyperparams != hyperparams:
                print("The arguments provided are different than the one saved. Use --force if you are certain.\nQuitting.")
                exit()

            resume_mode = True

    else:
        os.makedirs(save_path_experiment)
        utils.save_dict_to_json_file(os.path.join(save_path_experiment, "hyperparams"), hyperparams)
        utils.save_dict_to_json_file(os.path.join(save_path_experiment, "trainingparams"), trainingparams)

    #
    # LOAD DATASET ####
    dataset = Dataset.get(dataset_name)
    if trainingparams['batch_size'] == -1:
        trainingparams['batch_size'] = dataset['train']['length']

    #
    # INITIALIZING LEARNER ####
    model = build_model(dataset, trainingparams, hyperparams, hyperparams['hidden_size'])

    trainer_status = None

    # Sadly ignoring the pretraining if it was in that phase last time it saved
    if resume_mode:
        # TODO Add the save and load of masks(or not ... assuming the seeding of random will give the same mask), adadelta stuff
        load_model_params(model, save_path_experiment)
        trainer_status = utils.load_dict_from_json_file(os.path.join(save_path_experiment, "trainer_status"))

    #
    # TRAINING LEARNER ####
    best_epoch, total_train_time = train_model(model, dataset, trainingparams['look_ahead'], trainingparams['max_epochs'], trainingparams['batch_size'], save_path_experiment, trainer_status)

    #
    # Loading best model
    load_model_params(model, save_path_experiment)

    #
    # EVALUATING BEST MODEL ####
    model_evaluation = {}
    print('\n### Evaluating best model from Epoch {0} ###'.format(best_epoch))
    for log_prob_func_name in ['test', 'valid', 'train']:
        nb_batches = int(np.ceil(dataset[log_prob_func_name]['length'] / trainingparams['batch_size']))
        model_evaluation[log_prob_func_name] = get_mean_error_and_std(model, model.__dict__['{}_log_prob'.format(log_prob_func_name)], nb_batches)
        print("\tBest {0} error is : {1:.6f} ± {2:.6f}".format(log_prob_func_name.upper(), *model_evaluation[log_prob_func_name]))

    #
    # WRITING RESULTS #####
    model_info = [trainingparams['learning_rate'], trainingparams['decrease_constant'], hyperparams['hidden_size'], hyperparams['random_seed'], hyperparams['hidden_activation'], hyperparams['tied'], trainingparams['max_epochs'], best_epoch, trainingparams['look_ahead'], trainingparams['batch_size'], trainingparams['momentum'], trainingparams['dropout_rate'], hyperparams['weights_initialization'], float(model_evaluation['train'][0]), float(model_evaluation['train'][1]), float(model_evaluation['valid'][0]), float(model_evaluation['valid'][1]), float(model_evaluation['test'][0]), float(model_evaluation['test'][1]), total_train_time]
    utils.write_result(dataset_name, model_info, experiment_name)
