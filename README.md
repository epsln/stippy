# Stippy
## A simple weighted voronoi stippler in Python

Stippy is a CLI tool that can vectorize images using stippling, which can prove useful for technical artists using Plotters or Laser engraving.

# Use
## Basic use
In a shell, run:
```
python main.py input.jpg
```

## Otions 
You can control some parameters using the following flags:

- `-lr` controls the learning rate. Higher numbers means a potentially quicker convergence but migh also cause instability.
- `-n` controls the number of iterations of the Lloyds algorithm to perform. Higher means yields a potentially better result at a cost of compute time. 
- `-p` controls the number of stippling points to use. 

For example
```
python main.py input.jpg -lr 20 -n 100 -p 10000 
```
Will compute 100 iterations of the Lloyds algorithm with 10000 stipple points and a learning rate of 20.
