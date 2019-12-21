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

>>> from matrix import Diagonal, LowRank

>>> d = Diagonal(B.ones(3))

>>> d
<diagonal matrix: shape=3x3, data type=float64,
 diag=[1. 1. 1.]>
  
>>> 2 * d
<diagonal matrix: shape=3x3, data type=float64,
 diag=[2. 2. 2.]>

>>> 2 + d
<dense matrix: shape=3x3, data type=float64,
 mat=[[3. 2. 2.]
      [2. 3. 2.]
      [2. 2. 3.]]>
  
>>> d + B.randn(3, 3)
<dense matrix: shape=3x3, data type=float64,
 mat=[[ 1.83  -0.799 -2.36 ]
      [-0.09   2.371  0.572]
      [ 0.064  1.721  2.14 ]]>

>>> lr = LowRank(B.ones(3, 1), 2 * B.ones(3, 1))

>>> lr
<low-rank matrix: shape=3x3, dtype=float64, rank=1,
 left=[[1.]
       [1.]
       [1.]],
 right=[[2.]
        [2.]
        [2.]]>

>>> d + lr
<Woodbury matrix: shape=3x3, dtype=float64,
 diag=<diagonal matrix: shape=3x3, dtype=float64,
       diag=[1. 1. 1.]>,
 lr=<low-rank matrix: shape=3x3, dtype=float64, rank=1,
     left=[[1.]
           [1.]
           [1.]],
     right=[[2.]
            [2.]
            [2.]]>>
```

## Matrix Types

All matrix types are subclasses of `AbstractMatrix`.

The following base types are provided:

```
Zero
Dense
Diagonal
Constant
LowerTriangular
UpperTriangular
```

The following composite types are provided:
```
LowRank
Woodbury
Kronecker
```


## Functions

The following functions are added to LAB.
They can be accessed with `B.<function>` where `import lab as B`.

```
dense(a)
fill_diag(a, diag_len)
```
