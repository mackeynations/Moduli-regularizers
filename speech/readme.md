This speech recognition RNN is based on this blog: https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch/.

It implements a multilayer bidirectional RNN (or GRU, or LSTM) which listens to recorded speech and guesses a letter of the alphabet, or nothing if no speech is occurring, along with suitable moduli regularization protocols. 

### Sample terminal commands
The following can run tests of the RNN:
```
python3 main.py  --rnn_dim 512 --RNN_type RNN_tanh --savefile noregrnn
python3 main.py  --rnn_dim 512 --RNN_type LSTM --invert True --regularizer torus --regtype 1 --regpower DoG --savefile torusl1doglstm
python3 main.py  --rnn_dim 512 --n_layers 3 --RNN_type GRU --invert True --regularizer klein --regtype 1 --regpower DoG --savefile kleinl1doggru
```


### Known issues
There is a size mismatch when bidirectional is set to False. In the meantime, don't do that
