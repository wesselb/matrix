# [Structured Matrices](http://github.com/wesselb/matrix)

[![Build](https://travis-ci.org/wesselb/matrix.svg?branch=master)](https://travis-ci.org/wesselb/matrix)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/matrix/badge.svg?branch=master&service=github)](https://coveralls.io/github/wesselb/matrix?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://wesselb.github.io/matrix)

Structured matrices

*Note:* This package requires Python 3.6 or higher.

## Installation

Before installing the package, please ensure that `gcc` and `gfortran` are 
available.
On OS X, these are both installed with `brew install gcc`;
users of Anaconda may want to instead consider `conda install gcc`.
On Linux, `gcc` is most likely already available, and `gfortran` can be
installed with `apt-get install gfortran`.
Then simply

```bash
pip install backends-matrix
```

## Basic Usage
```python
>>> import lab as B

>>> from matrix import Diagonal

>>> d = Diagonal(B.ones(3))

>>> d
Diagonal matrix with diagonal
  [1. 1. 1.]
  
>>> d + d
Diagonal matrix with diagonal
  [2. 2. 2.]
  
>>> d + B.randn(3, 3)
Dense matrix:
  (3x3 array of data type float64)
  [[ 0.231  1.107 -0.648]
   [-1.199  1.668 -1.344]
   [ 0.545  0.807  0.934]]
```

## Matrix Types

Coming soon.

