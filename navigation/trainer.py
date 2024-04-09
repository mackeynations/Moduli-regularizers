# -*- coding: utf-8 -*-
import torch
import numpy as np
from math import isnan
import gc


#from visualize import save_ratemaps
import os


class Trainer(object):
    def __init__(self, options, model, trajectory_generator, restore=False):
        self.options = options
        self.model = model
        self.Ng = options.Ng
        self.savefile = options.savefile
        self.trajectory_generator = trajectory_generator
        lr = self.options.learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.save_repo = options.save_repo

        self.loss = []
        self.err = []
        self.besterr = np.Inf

        # Set up checkpoints
        self.ckpt_dir = os.path.join(options.save_dir, options.run_ID)
        ckpt_path = os.path.join(self.ckpt_dir, 'most_recent_model.pth')
        if restore and os.path.isdir(self.ckpt_dir) and os.path.isfile(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path))
            print("Restored trained model from {}".format(ckpt_path))
        else:
            if not os.path.isdir(self.ckpt_dir):
                os.makedirs(self.ckpt_dir, exist_ok=True)
            print("Initializing new model from scratch.")
            print("Saving to: {}".format(self.ckpt_dir))

    def train_step(self, inputs, pc_outputs, pos, options):
        ''' 
        Train on one batch of trajectories.

        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''
        self.model.zero_grad()

        loss, err, cat10, cat20, cat50 = self.model.compute_loss(inputs, pc_outputs, pos, options)
        if err < self.besterr and not isnan(loss):
            self.besterr = err
            torch.save(self.model.state_dict(), 'models/bestachieved' + self.savefile + '.pt')

        loss.backward()
        self.optimizer.step()
        
        
        gc.collect()
        torch.cuda.empty_cache()

        return loss.item(), err.item(), cat10.item(), cat20.item(), cat50.item()

    def train(self, options, n_epochs: int = 1000, n_steps=10, save=False):
        ''' 
        Train model on simulated trajectories.

        Args:
            n_steps: Number of training steps
            save: If true, save a checkpoint after each epoch.
        '''

        # Construct generator
        gen = self.trajectory_generator.get_generator()

        # tbar = tqdm(range(n_steps), leave=False)
        for epoch_idx in range(n_epochs):
            for step_idx in range(n_steps):
                inputs, pc_outputs, pos = next(gen)
                loss, err, cat10, cat20, cat50 = self.train_step(inputs, pc_outputs, pos, options)
                self.loss.append(loss)
                self.err.append(err)

                # Log error rate to progress bar
                # tbar.set_description('Error = ' + str(np.int(100*err)) + 'cm')
                if step_idx % 100 == 0:
                    sparsity = (torch.sum(torch.where(torch.abs(self.model.RNN.weight_hh_l0.data) < .001, 1.0, 0.0))/(self.Ng**2)).item()
                    print('Epoch: {}. Loss: {}. Err: {}cm, Sparsity: {:.3f}.'.format(
                        1000*epoch_idx + step_idx,
                        np.round(loss, 3), np.round(100 * err, 2), sparsity))
                    with open(self.save_repo + self.savefile + '.txt', 'a') as the_file:
                        the_file.write('Epoch: {}. Loss: {}. Err: {}cm. Sparsity: {:.3f}\n'.format(1000*epoch_idx + step_idx, np.round(loss, 3), np.round(100 * err, 2), sparsity))

            if save:
                # Save checkpoint
                ckpt_path = os.path.join(self.ckpt_dir, 'epoch_{}.pth'.format(epoch_idx))
                torch.save(self.model.state_dict(), ckpt_path)
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir,
                                                                 'most_recent_model.pth'))

                # Save a picture of rate maps
                #save_ratemaps(self.model, self.trajectory_generator,
                #              self.options, step=epoch_idx)