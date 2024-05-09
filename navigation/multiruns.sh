#!/bin/bash
for i in {0..4}
do
 python3 main.py --regularizer torus --regtype 1 --regpower square --savefile torusl1sq$i
 python3 main.py --invert True --regularizer torus --regtype 1 --regpower DoG --savefile torusl1dog$i
 python3 main.py --invert True --regularizer standard --regtype 1 --savefile l1$i
 python3 main.py --savefile noreg$i
 python3 main.py --invert True --regularizer torus --regtype 1 --regpower DoG --permute True --savefile perml1dog$i
 python3 main.py --invert True --regularizer klein --regtype 1 --regpower DoG --savefile kleinl1dog$i
 python3 main.py --invert True --regularizer sphere --regtype 1 --regpower DoG --savefile spherel1dog$i
 python3 main.py --invert True --regularizer circle --regtype 1 --regpower DoG --savefile circlel1dog$i
 python3 main.py --invert True --regularizer torus6 --regtype 1 --regpower DoG --savefile torus6l1dog$i
done