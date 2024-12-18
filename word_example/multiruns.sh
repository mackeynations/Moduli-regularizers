#!/bin/bash
for i in {0..4}
do
 python3 main.py --cuda --emsize 650 --nhid 650 --dropout 0.0 --epochs 15 --model RNN_TANH --savefile noreg$i
 python3 main.py --cuda --emsize 650 --nhid 650 --dropout 0.0 --epochs 15 --model RNN_TANH --invert True --regularizer torus6 --regtype 1 --regpower DoG --savefile torus6l1dog$i
 python3 main.py --cuda --emsize 650 --nhid 650 --dropout 0.0 --epochs 15 --model RNN_TANH --invert True --regularizer circle --regtype 1 --regpower DoG --savefile circlel1dog$i
 python3 main.py --cuda --emsize 650 --nhid 650 --dropout 0.0 --epochs 15 --model RNN_TANH --invert True --regularizer torus --regtype 1 --regpower DoG --savefile torusl1dog$i
 python3 main.py --cuda --emsize 650 --nhid 650 --dropout 0.0 --epochs 15 --model RNN_TANH --invert True --regularizer klein --regtype 1 --regpower DoG --savefile kleinl1dog$i
 python3 main.py --cuda --emsize 650 --nhid 650 --dropout 0.0 --epochs 15 --model RNN_TANH --invert True --regularizer sphere --regtype 1 --regpower DoG --savefile spherel1dog$i
 python3 main.py --cuda --emsize 650 --nhid 650 --dropout 0.0 --epochs 15 --model RNN_TANH --invert True --regularizer torus --regtype 1 --regpower DoG --permute True --savefile perml1dog$i
 python3 main.py --cuda --emsize 650 --nhid 650 --dropout 0.0 --epochs 15 --model RNN_TANH --invert True --regularizer torus --regtype 1 --regpower square --savefile torusl1sq$i
 python3 main.py --cuda --emsize 650 --nhid 650 --dropout 0.0 --epochs 15 --model RNN_TANH --invert True --regularizer torus --regtype 1 --regpower square --permute True --savefile perml1sq$i
 python3 main.py --cuda --emsize 650 --nhid 650 --dropout 0.0 --epochs 15 --model RNN_TANH --invert True --regularizer torus --regtype 1 --regpower ripple --savefile torusl1ripple$i
 python3 main.py --cuda --emsize 650 --nhid 650 --dropout 0.0 --epochs 15 --model RNN_TANH --regularizer standard --regtype 1 --savefile l1$i
done