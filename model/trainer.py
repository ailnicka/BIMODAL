"""
Implementation of different training methods
"""

import numpy as np
from sklearn.model_selection import KFold, train_test_split, ShuffleSplit
import pandas as pd
import configparser
from model.bimodal import BIMODAL
from model.one_hot_encoder import SMILESEncoder
from sklearn.utils import shuffle
import os
from model.helper import clean_molecule, check_model, check_molecules
from ast import literal_eval

np.random.seed(1)


class Trainer():

    def __init__(self, experiment_name='ForwardRNN'):

        self._encoder = SMILESEncoder()

        # Read all parameter from the .ini file
        self._config = configparser.ConfigParser()
        self._config.read('experiments/' + experiment_name + '.ini')

        self._model_type = self._config['MODEL']['model']
        self._experiment_name = experiment_name
        self._hidden_units = int(self._config['MODEL']['hidden_units'])
        try:
            self._start_model = self._config['MODEL']['start_model']
        except KeyError:
            self._start_model = None

        self._file_name = 'data/' + self._config['DATA']['data']
        self._encoding_size = int(self._config['DATA']['encoding_size'])
        self._molecular_size = int(self._config['DATA']['molecular_size'])

        self._epochs = int(self._config['TRAINING']['epochs'])
        self._n_folds = int(self._config['TRAINING']['n_folds'])
        self._learning_rate = float(self._config['TRAINING']['learning_rate'])
        self._batch_size = int(self._config['TRAINING']['batch_size'])
        try:
            self._freeze_layer = literal_eval(self._config['TRAINING']['freeze_layer'])
        except KeyError:
            self._freeze_layer = []

        self._samples = int(self._config['EVALUATION']['samples'])
        self._T = float(self._config['EVALUATION']['temp'])
        self._starting_token = self._encoder.encode([self._config['EVALUATION']['starting_token']])
        try:
            self._period = self._config['EVALUATION']['period']
        except KeyError:
            self._period = 1

        if self._model_type == 'BIMODAL':
            if self._start_model:
                self._model = BIMODAL(self._molecular_size, self._encoding_size,
                                  self._learning_rate, self._hidden_units,
                                  self._start_model, self._freeze_layer)  # only when restarting from existing model
            else:
                self._model = BIMODAL(self._molecular_size, self._encoding_size,
                                  self._learning_rate, self._hidden_units)
        else:
            raise NotImplementedError("No longer allowing other models than BIMODAL")

        self._data = self._encoder.encode_from_file(self._file_name)

    def run(self, stor_dir='evaluation/'):
        if self._n_folds == 0:
            self.run_without_validation(stor_dir)
        else:
            self.run_with_validation(stor_dir)

    def run_without_validation(self, stor_dir='evaluation/'):
        '''Training without validation on complete data'''

        # Create directories
        if not os.path.exists(stor_dir + '/' + self._experiment_name + '/models'):
            os.makedirs(stor_dir + '/' + self._experiment_name + '/models')

        if not os.path.exists(stor_dir + '/' + self._experiment_name + '/molecules'):
            os.makedirs(stor_dir + '/' + self._experiment_name + '/molecules')

        if not os.path.exists(stor_dir + '/' + self._experiment_name + '/statistic'):
            os.makedirs(stor_dir + '/' + self._experiment_name + '/statistic')

        # Compute labels
        label = np.argmax(self._data, axis=-1).astype(int)

        # Store total Statistics
        tot_stat = []

        # Shuffle data before training (Data reshaped from (N_samples, N_augmentation, molecular_size, encoding_size)
        # to  (all_SMILES, molecular_size, encoding_size))
        self._data, label = shuffle(self._data.reshape(-1, self._molecular_size, self._encoding_size),
                                    label.reshape(-1, self._molecular_size))

        for i in range(self._epochs):
            print('Epoch:', i)

            # Train model
            statistic = self._model.train(self._data, label, epochs=1, batch_size=self._batch_size)
            tot_stat.append(statistic.tolist())

            # Store statistic
            store_stat = np.array(tot_stat).reshape(i + 1, -1)
            pd.DataFrame(np.array(store_stat)).to_csv(
                stor_dir + '/' + self._experiment_name + '/statistic/stat.csv',
                header=None)

            # save model and sample molecules only over period requested by the user
            if i % self._period == 0:
                # Store model
                self._model.save(
                    stor_dir + '/' + self._experiment_name + '/models/model_epochs_' + str(i))

                # Sample new molecules
                new_molecules = []
                for s in range(self._samples):
                    mol = self._encoder.decode(self._model.sample(self._starting_token, self._T))
                    new_molecules.append(clean_molecule(mol[0], self._model_type))

                # Store new molecules
                new_molecules = np.array(new_molecules)
                pd.DataFrame(new_molecules).to_csv(
                    stor_dir + '/' + self._experiment_name + '/molecules/molecule_epochs_' + str(
                        i) + '.csv', header=None)

    def run_with_validation(self, stor_dir='../evaluation/'):
        '''Training with validation and store data'''

        # Create directories
        if not os.path.exists(stor_dir + '/' + self._experiment_name + '/models'):
            os.makedirs(stor_dir + '/' + self._experiment_name + '/models')

        if not os.path.exists(stor_dir + '/' + self._experiment_name + '/molecules'):
            os.makedirs(stor_dir + '/' + self._experiment_name + '/molecules')

        if not os.path.exists(stor_dir + '/' + self._experiment_name + '/statistic'):
            os.makedirs(stor_dir + '/' + self._experiment_name + '/statistic')

        if not os.path.exists(stor_dir + '/' + self._experiment_name + '/validation'):
            os.makedirs(stor_dir + '/' + self._experiment_name + '/validation')

        # Compute labels
        label = np.argmax(self._data, axis=-1).astype(int)

        # Split data into train and test data
        train_data, test_data, train_label, test_label = train_test_split(self._data, label, test_size= 1.0/self._n_folds,
                                                                          random_state=1, shuffle=True)

        # Store total Statistics
        tot_stat = []

        # Store validation loss
        tot_loss = []


        for i in range(self._epochs):
            print('Epoch:', i)


            # Train model (Data reshaped from (N_samples, N_augmentation, molecular_size, encoding_size)
            # to  (all_SMILES, molecular_size, encoding_size))
            statistic = self._model.train(train_data.reshape(-1, self._molecular_size, self._encoding_size),
                                          train_label.reshape(-1, self._molecular_size), epochs=1,
                                          batch_size=self._batch_size)
            tot_stat.append(statistic.tolist())

            # Test model on validation set
            tot_loss.append(
                self._model.validate(test_data.reshape(-1, self._molecular_size, self._encoding_size),
                                     test_label.reshape(-1, self._molecular_size)))

            # Store statistic
            store_stat = np.array(tot_stat).reshape(i + 1, -1)
            pd.DataFrame(np.array(store_stat)).to_csv(
                stor_dir + '/' + self._experiment_name + '/statistic/stat.csv',
                header=None)

            # Store validation data
            pd.DataFrame(np.array(tot_loss).reshape(-1, 1)).to_csv(
                stor_dir + '/' + self._experiment_name + '/validation/val.csv',
                header=None)

            # save model and sample molecules only over period requested by the user
            if i % self._period == 0:
                # Store model
                self._model.save(
                    stor_dir + '/' + self._experiment_name + '/models/model_epochs_' + str(i))

                # Sample new molecules
                new_molecules = []
                for s in range(self._samples):
                    mol = self._encoder.decode(self._model.sample(self._starting_token, self._T))
                    new_molecules.append(clean_molecule(mol[0], self._model_type))

                # Store new molecules
                new_molecules = np.array(new_molecules)
                pd.DataFrame(new_molecules).to_csv(
                    stor_dir + '/' + self._experiment_name + '/molecules/molecule_epochs_' + str(
                        i) + '.csv', header=None)

