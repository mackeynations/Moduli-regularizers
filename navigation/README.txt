###################################################
###              Read Me                        ###
###################################################


This directory contains all files needed to replicate the Navigation experiments in our paper "Geometric sparsification in recurrent neural nets." (link to be posted) We recommend creating a python environment in conda for our experiments; the list of all packages and versions we used can be found in "requirements.txt". The experiments conducted in our paper can most easily be replicated by running the shell script multiruns.sh. This shell script will train 5 models using a variety of moduli regularizers and inhibitor functions, as well as all control experiments, at a set level of sparsity. The output consists of the following:
 1. The training loss every 100 steps while the model is being sparsified
 2. A sequence of five Validators: these are copies of the final model over 5 x (batch size) routes, which we averaged to report the final model error
 3. The training loss every 100 steps for the "lottery ticket" model with randomized weights, but identical sparse structure
 4. The same sequence of Validators as 2.
In the interests of making this messy output as easy to navigate as possible, we've also provided a Jupyter notebook in the directory 'Graphs' which contains the code we used to analyze data and create tables. To run baseline experiments, set --target_perc to 0. They will also run more quickly if lines 170-193 of the the code are commented out.



###################################################
###         Customizing experiments             ###
###################################################

We'll additionally explain a sample command, so that it can be easily modified as desired:
 python3 main.py --cuda --emsize 650 --nhid 650 --dropout 0.0 --epochs 15 --model RNN_TANH --regularizer torus --regtype 1 --regpower DoG --invert True --permute False --target_perc 90 --savefile torusl1dog$i

The options work as follows:
 * --cuda: activates cuda, if available on your machine. Do not use without cuda
 * --emsize 650: sets the embedding dimension for the list of words
 * --nhid 650: sets the dimension of the RNN hidden state
 * --dropout 0.0: sets dropout in the RNN. We deactivated it for the purpose of running single layer RNNs, results may vary if it is increased
 * --epochs 15: number of training epochs
 * --model RNN_TANH: sets the recurrent structure. RNN_TANH gives an RNN with tanh activation, and similarly for RNN_RELU. LSTM does work, but with naive regularizers. 
 * --regularizer torus: sets the moduli space to be a torus. Other options include klein, torus6, sphere, circle, and standard (for L_1 or L_2 regularization)
 * --regtype 1: sets what type of regularization is being done. Accepts an integer k, returns L_k regularization
 * --regpower DoG: sets the inhibitor function. Accepts square, none, gauss, DoG, ripple, which are as in our paper for square (diffusion), DoG (difference of Gaussians), and ripple (ripple/sinuosoid). none sets the regularizer to be all 0s, and gauss applies a Gaussian function. 
 * --invert True: this exists for historical reasons, and should be included for all DoG tests, and optionally for gauss ones. Replaces the regularizer with max(regularizer) - regularizer. The historical reasons are that I implemented the DoG backwards. 
 * --permute False: the setting for whether regularizing weights should be shuffled, as in our control experiments
 * --target_perc 90: sets the percentile of weights which will be set to 0 by the final epoch. Expects a number between 0 and 100.
 * --savefile torusl1dog: sets the filename where data will be saved
 * --trainembed False: sets whether to train the moduli space embedding, in addition to the model itself. Not extensively tested, may have bugs. We didn't report any models with trainembed set to True in the paper, but did briefly review the possibility in 2.2.1