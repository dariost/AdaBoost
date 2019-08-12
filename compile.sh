#!/bin/bash

g++ src/*.cpp -O3 -march=native -Wall -Wextra -fopenmp -o AdaBoost.elf
