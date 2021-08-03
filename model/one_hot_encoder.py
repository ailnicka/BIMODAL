"""
Implementation of one-hot-encoder for SMILES strings
"""
import pandas as pd
import numpy as np
import os
import sys


class SMILESEncoder():

    def __init__(self):
        # Allowed tokens (adapted from default dictionary)
        self._tokens = np.sort(['#', '=',
                                '\\', '/', '%', '@', '+', '-', '.',
                                '(', ')', '[', ']',
                                '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
                                'A', 'B', 'E', 'C', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'S', 'T', 'V',
                                'Z',
                                'a', 'b', 'c', 'd', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's', 't'
                                ])

        num_tokens = len(self._tokens)

        self._id_matrix = np.eye(num_tokens, dtype=int)
        self._idx_range = np.array(range(num_tokens), dtype=int)
        self._token_to_idx = {v: i for i, v in enumerate(self._tokens)}

        self._encode_many = np.vectorize(self._encode_one, signature=f'()->(K,{num_tokens})')
        self._decode_many = np.vectorize(self._decode_one, signature=f'(K,{num_tokens})->()')

    def _encode_one(self, molecule):
        return np.array([self._id_matrix[self._token_to_idx[char]] for char in molecule])

    def _decode_one(self, encoding):
        return ''.join(self._tokens[np.dot(encoding, self._idx_range).astype(int)])

    def encode_from_file(self, name='data'):
        '''One-hot-encoding from .csv file
        :param name:    name of data file
        :return:    encoded data (data size, molecule size, allowed token size)
        '''

        # Read data
        if os.path.isfile(name + '.csv'):
            data = pd.read_csv(name + '.csv', header=None).values
        elif os.path.isfile(name + '.tar.xz'):
            # Skip first line since empty and last line since nan
            data = pd.read_csv(name + '.tar.xz', compression='xz', header=None).values[1:-1]
        else:
            print('CAN NOT READ DATA')
            sys.exit()

        print(f"Encoding data {data.shape}")
        data = self.encode(data)
        print(f"Encoded data {data.shape}")

        return data

    def encode(self, data):
        '''One-hot-encoding
        :param data:         input data (sample size,)
        :return one_hot:     encoded data (sample size, molecule size, allowed token size)
        '''

        return self._encode_many(data)

    def decode(self, one_hot):
        '''Decode one-hot encoding to SMILES
        :param one_hot:    one_hot data (sample size, molecule size, allowed token size)
        :return data:      SMILES (sample size,)
        '''

        return self._decode_many(one_hot)
