#!/bin/sh

set -xe

gcc -Wall -Wextra -o nn.out NN.c -lm

./nn.out