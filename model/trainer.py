"""
Implementation of different training methods
"""

import numpy as np
from sklearn.model_selection import KFold, train_test_split, ShuffleSplit
import pandas as pd
import configparser
import sys
sys.path.append("./model")
from bimodal import BIMODAL
from one_hot_encoder import SMILESEncoder
from sklearn.utils import shuffle
import os
from helper import clean_molecule, check_model
from ast import literal_eval
from rdkit import Chem

np.random.seed(1)

GENERATION_TYPES = ['both', 'random', 'beam_search']

class Trainer():

    def __init__(self, experiment_name='BIMODAL'):

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
            self._period = int(self._config['EVALUATION']['period'])
        except KeyError:
            self._period = 1

        try:
            self._generation_type = self._config['EVALUATION']['generation_type']
            if self._generation_type not in GENERATION_TYPES:
                raise KeyError
        except KeyError:
            self._generation_type = 'both'

        try:
            self._beam_width = int(self._config['EVALUATION']['beam_width'])
        except KeyError:
            self._beam_width = 15

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

        # Read data
        if os.path.isfile(self._file_name + '.csv'):
            self._data = pd.read_csv(self._file_name + '.csv', header=None).values
        elif os.path.isfile(self._file_name + '.tar.xz'):
            # Skip first line since empty and last line since nan
            self._data = pd.read_csv(self._file_name + '.tar.xz', compression='xz', header=None).values[1:-1]
        else:
            print('CAN NOT READ DATA')
            sys.exit()
        print("Init done")

    def sample(self, stor_dir='evaluation/'):
        model_name = self._start_model.split('/')[-1]
        if self._generation_type in ['both', 'random']:
            filename = stor_dir + '/' + self._experiment_name + '/molecules/molecule_'+model_name+'.csv'
            self.sample_random(filename)

        if self._generation_type in ['both', 'beam_search']:
            filename = stor_dir + '/' + self._experiment_name + '/molecules/molecule_beam'+model_name+'.csv'
            self.beam_search(filename)

    def run(self, stor_dir='evaluation/', restart=False):

        if self._n_folds == 0:
            self.run_without_validation(stor_dir, restart)
        else:
            self.run_with_validation(stor_dir, restart)

    def _check_restart(self, stor_dir='evaluation/'):
        # With restart read existing files

        tmp_stat_file = pd.read_csv(stor_dir + '/' + self._experiment_name + '/statistic/stat.csv',
            header=None).to_numpy()
        if self._n_folds != 0:
            tmp_val_file = pd.read_csv(stor_dir + '/' + self._experiment_name + '/validation/val.csv',
                        header=None).to_numpy()
        else:
            tmp_val_file = None

        # Check how many epochs are finished
        finished_epoch = -1
        for epoch in range(self._epochs):
            if check_model(self._experiment_name, stor_dir, epoch) and (tmp_stat_file.shape[0] >= epoch):
                finished_epoch = epoch
            else:
                break
        # Load model from last finished epoch
        if finished_epoch >= 0:
            self._model = BIMODAL(self._molecular_size, self._encoding_size,
                                  self._learning_rate, self._hidden_units,
                                  stor_dir + '/' + self._experiment_name + '/models/model_epochs_' + str(finished_epoch))
        print(f'Restarting from epoch {finished_epoch}')

        return finished_epoch, tmp_stat_file, tmp_val_file

    def run_without_validation(self, stor_dir='evaluation/', restart=False):
        '''Training without validation on complete data'''

        # Create directories
        if not os.path.exists(stor_dir + '/' + self._experiment_name + '/models'):
            os.makedirs(stor_dir + '/' + self._experiment_name + '/models')

        if not os.path.exists(stor_dir + '/' + self._experiment_name + '/molecules'):
            os.makedirs(stor_dir + '/' + self._experiment_name + '/molecules')

        if not os.path.exists(stor_dir + '/' + self._experiment_name + '/statistic'):
            os.makedirs(stor_dir + '/' + self._experiment_name + '/statistic')

        # Store total Statistics
        tot_stat = []

        print("Data before reshape", str(self._data.shape))
        # Shuffle data before training (Data reshaped from (N_samples, N_augmentation)
        # to  (all_SMILES))
        self._data = shuffle(self._data.reshape(-1))
        print("Data after reshape", str(self._data.shape))

        if restart:
            last_epoch, tmp_stat, _ = self._check_restart(stor_dir)
            if last_epoch == -1:
                restart = False

        for i in range(self._epochs):
            print('Epoch:', i)
            if restart:
                if last_epoch >= i:
                    tot_stat.append(tmp_stat[i, 1:].reshape(1, -1).tolist())
                    continue
                else:
                    restart = False
            # Train model
            statistic = self._model.train(self._data, epochs=1, batch_size=self._batch_size)
            tot_stat.append(statistic.tolist())

            # Store statistic
            store_stat = np.array(tot_stat).reshape(i + 1, -1)
            pd.DataFrame(np.array(store_stat)).to_csv(
                stor_dir + '/' + self._experiment_name + '/statistic/stat.csv',
                header=None)

            # save model and sample molecules only over period requested by the user
            if (i+1) % self._period == 0:
                self.store_n_sample(stor_dir, i)

    def run_with_validation(self, stor_dir='../evaluation/', restart = False):
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

        print(f"Start preparing data of shape {self._data.shape}")
        # Split data into train and test data
        train_data, test_data = train_test_split(self._data, test_size= 1.0/self._n_folds, random_state=1, shuffle=True)

        print("Data before reshape", str(train_data.shape), str(test_data.shape))
        train_data = train_data.reshape(-1)
        test_data = test_data.reshape(-1)
        print("Data after reshape", str(train_data.shape), str(test_data.shape))

        # Store total Statistics
        tot_stat = []

        # Store validation loss
        tot_loss = []

        if restart:
            last_epoch, tmp_stat, tmp_val = self._check_restart(stor_dir)
            if last_epoch == -1:
                restart = False

        for i in range(self._epochs):
            print('Epoch:', i)

            if restart:
                if last_epoch >= i:
                    tot_stat.append(tmp_stat[i, 1:].reshape(1, -1).tolist())
                    tot_loss.append(tmp_val[i, 1])
                    continue
                else:
                    restart = False

            statistic = self._model.train(train_data, epochs=1,
                                          batch_size=self._batch_size)
            tot_stat.append(statistic.tolist())

            # Test model on validation set
            tot_loss.append(
                self._model.validate(test_data))

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
            if (i+1) % self._period == 0:
                print("Saving model and generating molecules.")
                self.store_n_sample(stor_dir, i)

    def sample_random(self, filename):
        new_molecules = []
        for s in range(self._samples):
            mol = self._model.sample(self._starting_token, self._T)
            mol = self._encoder.decode([mol[0]])
            new_molecules.append(clean_molecule(mol[0], self._model_type))
        print(f'Generated {len(new_molecules)} molecules with random sampling')
        new_molecules, _ = self.check_chemistry(new_molecules)
        print(f'Chemistry check survived {len(new_molecules)} molecules')
        new_molecules = np.array(new_molecules)
        pd.DataFrame(new_molecules).to_csv(filename, header=False, index=False)

    def beam_search(self, filename):
        molecules, scores = self._model.beam_search(self._starting_token, self._beam_width)
        molecules = self._encoder.decode(np.array(molecules).squeeze())
        print(f'Generated {len(molecules)} molecules with beam search')
        molecules = [clean_molecule(mol, self._model_type) for mol in molecules]
        molecules, score_idx = self.check_chemistry(molecules)
        scores = [scores[i] for i in score_idx]
        print(f'Chemistry check survived {len(molecules)} molecules')
        pd.DataFrame(dict(molecules=molecules, scores=scores)).to_csv(filename, index=False)

    def store_n_sample(self, stor_dir, epoch):
        self._model.save(
            stor_dir + '/' + self._experiment_name + '/models/model_epochs_' + str(epoch))

        if self._generation_type in ['both', 'random']:
            filename = stor_dir + '/' + self._experiment_name + '/molecules/molecule_epochs_' + str(epoch) + '.csv'
            self.sample_random(filename)

        if self._generation_type in ['both', 'beam_search']:
            filename = stor_dir + '/' + self._experiment_name + '/molecules/molecule_beam_epochs_' + str(epoch) + '.csv'
            self.beam_search(filename)

    @staticmethod
    def check_chemistry(molecules):
        correct_molecules = []
        correct_idx = []
        for i, m in enumerate(molecules):
            m = Chem.MolFromSmiles(m)
            if m is not None:
                m = Chem.MolToSmiles(m, canonical=True)
                if len(m) > 1:
                    correct_molecules.append(m)
                    correct_idx.append(i)
        return correct_molecules, correct_idx


