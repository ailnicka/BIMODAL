"""
Implementation of BIMODAL to generate SMILES
"""

import numpy as np
import torch
import torch.nn as nn
import sys
import os
sys.path.append("./model")
from bidir_lstm import BiDirLSTM
from one_hot_encoder import SMILESEncoder


torch.manual_seed(1)
np.random.seed(5)


class BIMODAL:

    def __init__(self, molecule_size=7, encoding_dim=55, lr=.01, hidden_units=128, name=None, layers_to_freeze=[]):
        """Build new model or load model by name
        :param name:    model name
        :param layers_to_freeze: which layers shall be frozen during training: applies to model loaded from file
        """
        self._molecule_size = molecule_size
        self._input_dim = encoding_dim
        self._output_dim = encoding_dim
        self._layer = 2
        self._hidden_units = hidden_units

        # Learning rate
        self._lr = lr

        # Check availability of GPUs
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if name is None:
            self._lstm = BiDirLSTM(input_dim=self._input_dim, hidden_dim=self._hidden_units, layers=self._layer)

        else:
            print(os.getcwd())
            self._lstm = torch.load(name+'.dat', map_location=self._device)
            if len(layers_to_freeze) != 0:
                for layer in layers_to_freeze:
                    for name, param in self._lstm.named_parameters():
                        if str(layer) in name:
                            param.requires_grad = False

        if torch.cuda.is_available():
            self._lstm = self._lstm.cuda()

        # SMILES Encoder
        self._encoder = SMILESEncoder()
        # Adam optimizer: only take into optimisation the un-frozen layers
        self._optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,  self._lstm.parameters()), lr=self._lr, betas=(0.9, 0.999))
        # Cross entropy loss
        self._loss = nn.CrossEntropyLoss(reduction='mean')

    def print_model(self):
        '''Print name and shape of all tensors'''
        for name, p in self._lstm.state_dict().items():
            print(name)
            print(p.shape)

    def train(self, data, epochs=1, batch_size=1):
        '''Train the model
        :param  data:   data array (n_samples, molecule_size, encoding_length)
        :param  label:  label array (n_samples, molecule_size)
        :param  epochs: number of epochs for the training
        :param  batch_size: batch size for the training
        :return statistic:  array storing computed losses (epochs, batch size)
        '''

        # Number of samples
        n_samples = data.shape[0]

        # Calculate number of batches per epoch
        if (n_samples % batch_size) is 0:
            n_iter = n_samples // batch_size
        else:
            n_iter = n_samples // batch_size + 1

        # To store losses
        statistic = np.zeros((epochs, n_iter))

        # Prepare model for training
        self._lstm.train()

        # Iteration over epochs
        for i in range(epochs):

            # Iteration over batches
            for n in range(n_iter):
                # Set gradient to zero for batch
                self._optimizer.zero_grad()

                # Store losses in list
                losses = []

                # Compute indices used as batch
                batch_start = n * batch_size
                batch_end = min((n + 1) * batch_size, n_samples)

                # Reset model with correct batch size
                self._lstm.new_sequence(batch_end - batch_start, self._device)

                # encode current batch
                batch_data = data[batch_start:batch_end]
                batch_data = self._encoder.encode(batch_data)
                batch_label = np.argmax(batch_data, axis=-1).astype(int)
                batch_data = np.swapaxes(batch_data, 0, 1)

                batch_data = torch.from_numpy(batch_data.astype('float32')).to(self._device)
                batch_label = torch.from_numpy(batch_label).to(self._device)

                # Initialize start and end position of sequence read by the model
                start = self._molecule_size // 2
                end = start + 1

                for j in range(self._molecule_size - 1):
                    self._lstm.new_sequence(batch_end - batch_start, self._device)

                    # Select direction for next prediction
                    if j % 2 == 0:
                        dir = 'right'
                    else:
                        dir = 'left'

                    # Predict next token
                    pred = self._lstm(batch_data[start:end], dir, self._device)

                    # Compute loss and extend sequence read by the model
                    if j % 2 == 0:
                        loss = self._loss(pred, batch_label[:, end])
                        end += 1

                    else:
                        loss = self._loss(pred, batch_label[:, start - 1])
                        start -= 1

                    # Append loss of current position
                    losses.append(loss.item())

                    # Accumulate gradients
                    # (NOTE: This is more memory-efficient than summing the loss and computing the final gradient for the sum)
                    loss.backward()

                # Store statistics: loss per token (middle token not included)
                statistic[i, n] = np.sum(losses) / (self._molecule_size - 1)

                # Perform optimization step
                self._optimizer.step()

        return statistic

    def validate(self, data, batch_size=128):
        ''' Validation of model and compute error
        :param data:    test data (n_samples, molecule_size, encoding_size)
        :param label:   label data (n_samples_molecules_size)
        :param batch_size:  batch size for validation
        :return:            mean loss over test data
        '''


        # Use train mode to get loss consistent with training
        self._lstm.train()

        # Gradient is not compute to reduce memory requirements
        with torch.no_grad():
            # Number of samples
            n_samples = data.shape[0]

            # Initialize loss for complete validation set
            tot_loss = 0

            # Calculate number of batches per epoch
            if (n_samples % batch_size) is 0:
                n_iter = n_samples // batch_size
            else:
                n_iter = n_samples // batch_size + 1

            for n in range(n_iter):

                # Compute indices used as batch
                batch_start = n * batch_size
                batch_end = min((n + 1) * batch_size, n_samples)

                # Data used in this batch
                # encode current batch
                batch_data = data[batch_start:batch_end]
                batch_data = self._encoder.encode(batch_data)
                batch_label = np.argmax(batch_data, axis=-1).astype(int)
                batch_data = np.swapaxes(batch_data, 0, 1)

                batch_data = torch.from_numpy(batch_data.astype('float32')).to(self._device)
                batch_label = torch.from_numpy(batch_label).to(self._device)

                # Initialize loss for molecule
                molecule_loss = 0

                # Reset model with correct batch size and device
                self._lstm.new_sequence(batch_end - batch_start, self._device)

                start = self._molecule_size // 2
                end = start + 1

                for j in range(self._molecule_size - 1):
                    self._lstm.new_sequence(batch_end - batch_start, self._device)

                    # Select direction for next prediction
                    if j % 2 == 0:
                        dir = 'right'
                    if j % 2 == 1:
                        dir = 'left'

                    # Predict next token
                    pred = self._lstm(batch_data[start:end], dir, self._device)

                    # Extend reading of the sequence
                    if j % 2 == 0:
                        loss = self._loss(pred, batch_label[:, end])
                        end += 1

                    if j % 2 == 1:
                        loss = self._loss(pred, batch_label[:, start - 1])
                        start -= 1

                    # Sum loss over molecule
                    molecule_loss += loss.item()

                # Add loss per token to total loss (start token and end token not counted)
                tot_loss += molecule_loss / (self._molecule_size - 1)

            return tot_loss / n_iter

    def sample(self, middle_token, T=1):
        '''Generate new molecule
        :param middle_token:    starting sequence
        :param T:               sampling temperature
        :return molecule:       newly generated molecule (molecule_length, encoding_length)
        '''

        # Prepare model
        self._lstm.eval()

        # Gradient is not compute to reduce memory requirements
        with torch.no_grad():
            # Output array with merged forward and backward directions

            # New sequence
            seq = np.zeros((self._molecule_size, 1, self._output_dim))
            seq[self._molecule_size // 2, 0] = middle_token

            # Create tensor for data and select correct device
            seq = torch.from_numpy(seq.astype('float32')).to(self._device)

            # Define start/end values for reading
            start = self._molecule_size // 2
            end = start + 1

            for j in range(self._molecule_size - 1):
                self._lstm.new_sequence(1, self._device)

                # Select direction for next prediction
                if j % 2 == 0:
                    dir = 'right'
                if j % 2 == 1:
                    dir = 'left'

                pred = self._lstm(seq[start:end], dir, self._device)

                # Compute new token
                token = self.sample_token(np.squeeze(pred.cpu().detach().numpy()), T)

                # Set new token within sequence
                if j % 2 == 0:
                    seq[end, 0, token] = 1.0
                    end += 1

                if j % 2 == 1:
                    seq[start - 1, 0, token] = 1.0
                    start -= 1

        return seq.cpu().numpy().reshape(1, self._molecule_size, self._output_dim)

    def sample_token(self, out, T=1.0):
        ''' Sample token
        :param out: output values from model
        :param T:   sampling temperature
        :return:    index of predicted token
        '''
        # Explicit conversion to float64 avoiding truncation errors
        out = out.astype('float64')

        # Compute probabilities with specific temperature
        out_T = out / T
        p = np.exp(out_T) / np.sum(np.exp(out_T))

        # Generate new token at random
        char = np.random.multinomial(1, p, size=1)
        return np.argmax(char)

    def beam_search(self, middle_token, beam_width=15):
        # based on implementation from molecular_design_with_beam_search

        # Prepare model
        self._lstm.eval()

        # Gradient is not compute to reduce memory requirements
        with torch.no_grad():
            # Output array with merged forward and backward directions

            # New sequence
            seq = np.zeros((self._molecule_size, 1, self._output_dim))
            seq[self._molecule_size // 2, 0] = middle_token

            # Create tensor for data and select correct device
            seq = torch.from_numpy(seq.astype('float32')).to(self._device)

            candidates = [seq]
            scores = [1]*beam_width

            # Define start/end values for reading
            start = self._molecule_size // 2
            end = start + 1

            for j in range(self._molecule_size - 1):
                self._lstm.new_sequence(1, self._device)

                # Select direction for next prediction
                if j % 2 == 0:
                    dir = 'right'
                if j % 2 == 1:
                    dir = 'left'

                current_candidates = []
                current_scores = []

                # grow sequence for all candidates
                for i, x in enumerate(candidates):
                    preds = self._lstm(x[start:end], dir, self._device)
                    preds = np.squeeze(preds.cpu().detach().numpy()).astype('float64')
                    # to exchange linear into only positive in similar style as in random sample
                    preds = np.exp(preds) / np.sum(np.exp(preds))
                    idx_preds_sorted = np.argsort(preds)[::-1][:beam_width]
                    preds_sorted = preds[idx_preds_sorted]

                    # Set new token within sequence for all best tokens
                    for idx_pred in idx_preds_sorted:
                        new_seq = x.clone()
                        if j % 2 == 0:
                            new_seq[end, 0, idx_pred] = 1.0

                        if j % 2 == 1:
                            new_seq[start - 1, 0, idx_pred] = 1.0
                        current_candidates.append(new_seq)
                    # multiply the "probability" of new token by old probability
                    current_scores.extend([a*b for a,b in zip(scores,list(preds_sorted))])

                # Find the k best candidates from the scores
                idx_current_best = np.argsort(current_scores)[::-1][:beam_width]
                candidates = [x for i, x in enumerate(current_candidates) if i in idx_current_best]
                scores = [x for i, x in enumerate(current_scores) if i in idx_current_best]
                # update start and end when all candidates processed on this position
                if j % 2 == 0:
                    end += 1
                if j % 2 == 1:
                    start -= 1
        #         print("Step", j, "width", beam_width,
        #               "Candidates shape", np.array([x.cpu().numpy().reshape(1, self._molecule_size, self._output_dim) for x in candidates]).shape,
        #               "Candidates sum", np.array([x.cpu().numpy().reshape(1, self._molecule_size, self._output_dim) for x in candidates]).sum())
        candidates = [x.cpu().numpy().reshape(1, self._molecule_size, self._output_dim) for x in candidates]
        return candidates, scores

    def save(self, name='test_model'):
        torch.save(self._lstm, name + '.dat')
