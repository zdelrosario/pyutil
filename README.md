# pyutil
I do a lot of research prototyping in Python. This repo collects a bunch of utility functions
and recipies that I use frequently. Someone else may find some things in this repo useful; thus
I made it public. Bear in mind that none of this is production code -- it's all very hacky.

## numeric
A collection of utilities intended to be used with numpy.

## tex

- [flatten.py](https://github.com/zdelrosario/pyutil/blob/master/pyutil/tex/flatten.py) parses a tex source file to place all images and code listings in the same directory, and modifies the \graphicspath to match. Useful for paper submissions.

## plotting
A collection of plotting utilities, including a port of matlab's nice (user-made) color set linspecer().
Some recipies for placing figures at a desired location on a computer screen (useful when dealing
with many plots).

## boilerplate
Annoying boilerplate code I need to look up frequently.

## interface
Some esoteric parsing functions, such as loading (particular) tecplot .dat files
