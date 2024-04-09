# -*- coding: utf-8 -*-
import torch
import numpy as np

import os


class SparseValidator(object):
    def __init__(self, options, model, trajectory_generator, restore=False):
        self.options = options
        self.model = model
        self.Ng = options.Ng
        self.savefile = options.savefile
        self.trajectory_generator = trajectory_generator
        self.save_repo = options.save_repo

        self.loss = []
        self.err = []


    def val_step(self, inputs, pc_outputs, pos, options):
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
        self.model.eval()
        self.model.RNN.weight_hh_l0.data = torch.where(torch.abs(self.model.RNN.weight_hh_l0.data) < .001, 0.0, self.model.RNN.weight_hh_l0.data)

        loss, err, cat10, cat20, cat50 = self.model.compute_loss(inputs, pc_outputs, pos, options)


        return loss.item(), err.item(), cat10.item(), cat20.item(), cat50.item()

    def test(self, options, n_epochs: int = 1000, n_steps=10, save=False):
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
            for step_idx in range(5):
                inputs, pc_outputs, pos = next(gen)
                loss, err, cat10, cat20, cat50 = self.val_step(inputs, pc_outputs, pos, options)
                self.loss.append(loss)
                self.err.append(err)

                # Log error rate to progress bar
                # tbar.set_description('Error = ' + str(np.int(100*err)) + 'cm')
                sparsity = (torch.sum(torch.where(torch.abs(self.model.RNN.weight_hh_l0.data) < .001, 1.0, 0.0))/(self.Ng**2)).item()
                print('Validator: Step: {}. Loss: {}. Err: {}cm, Sparsity: {:3f}.'.format(
                        step_idx,
                        np.round(loss, 3), np.round(100 * err, 2), sparsity))
                with open(self.save_repo + self.savefile + '.txt', 'a') as the_file:
                    the_file.write('Validator: Step: {}. Loss: {}. Err: {}cm. Sparsity: {:3f}\n'.format(step_idx, np.round(loss, 3), np.round(100 * err, 2), sparsity))

            if save:
                # Save checkpoint
                ckpt_path = os.path.join(self.ckpt_dir, 'epoch_{}.pth'.format(epoch_idx))
                torch.save(self.model.state_dict(), ckpt_path)
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir,
                                                                 'most_recent_model.pth'))

                # Save a picture of rate maps
                #save_ratemaps(self.model, self.trajectory_generator,
                #              self.options, step=epoch_idx)
                
                
                
class PercentileSparse(object):
    def __init__(self, options, model, trajectory_generator, percentile, restore=False):
        self.options = options
        self.model = model
        self.Ng = options.Ng
        self.percentile = percentile
        self.savefile = options.savefile
        self.trajectory_generator = trajectory_generator
        self.percentile = percentile
        self.save_repo = options.save_repo
        self.cutoff = self.get_percentiles()

        self.loss = []
        self.err = []


    def get_percentiles(self):
        attr = torch.abs(self.model.RNN.weight_hh_l0.data).cpu().view(-1)
        return np.percentile(attr, self.percentile)
    
    def val_step(self, inputs, pc_outputs, pos, options):
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
        self.model.eval()
        self.model.RNN.weight_hh_l0.data = torch.where(torch.abs(self.model.RNN.weight_hh_l0.data) < self.cutoff, 0.0, self.model.RNN.weight_hh_l0.data)

        loss, err, cat10, cat20, cat50 = self.model.compute_loss(inputs, pc_outputs, pos, options)


        return loss.item(), err.item(), cat10.item(), cat20.item(), cat50.item()

    def test(self, options, n_epochs: int = 1, n_steps=10, save=False):
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
            for step_idx in range(5):
                inputs, pc_outputs, pos = next(gen)
                loss, err, cat10, cat20, cat50 = self.val_step(inputs, pc_outputs, pos, options)
                self.loss.append(loss)
                self.err.append(err)

                # Log error rate to progress bar
                # tbar.set_description('Error = ' + str(np.int(100*err)) + 'cm')
                sparsity = (torch.sum(torch.where(self.model.RNN.weight_hh_l0.data == 0, 1.0, 0.0))/(self.Ng**2)).item()
                print('Percentile {}: Cutoff: {:2f}. Step: {}. Loss: {}. Err: {}cm, Sparsity: {:3f}.'.format(
                        self.percentile, self.cutoff, step_idx,
                        np.round(loss, 3), np.round(100 * err, 2), sparsity))
                with open(self.save_repo + self.savefile + '.txt', 'a') as the_file:
                    the_file.write('Percentile {}: Cutoff: {:.3f}. Step: {}. Loss: {}. Err: {}cm. Sparsity: {:.2f}\n'.format(self.percentile, self.cutoff, step_idx, np.round(loss, 3), np.round(100 * err, 2), sparsity))

            if save:
                # Save checkpoint
                ckpt_path = os.path.join(self.ckpt_dir, 'epoch_{}.pth'.format(epoch_idx))
                torch.save(self.model.state_dict(), ckpt_path)
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir,
                                                                 'most_recent_model.pth'))

                # Save a picture of rate maps
                #save_ratemaps(self.model, self.trajectory_generator,
                #              self.options, step=epoch_idx)