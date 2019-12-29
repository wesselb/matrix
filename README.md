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

## Example
```python
>>> import lab as B

>>> from matrix import Diagonal, LowRank

>>> d = Diagonal(B.ones(3))

>>> d
<diagonal matrix: shape=3x3, data type=float64,
 diag=[1. 1. 1.]>
  
>>> 2 * d
<diagonal matrix: shape=3x3, data type=float64
 diag=[2. 2. 2.]>

>>> 2 * d + 1
<Woodbury matrix: shape=3x3, dtype=int64
 diag=<diagonal matrix: shape=3x3, dtype=float64
       diag=[2. 2. 2.]>
 lr=<low-rank matrix: shape=3x3, dtype=int64, rank=1, sign=0
     left=[[1]
           [1]
           [1]]
     middle=<diagonal matrix: shape=1x1, dtype=int64
             diag=[1]>>>
  
>>> B.inv(2 * d + 1)
<Woodbury matrix: shape=3x3, dtype=float64
 diag=<diagonal matrix: shape=3x3, dtype=float64
       diag=[0.5 0.5 0.5]>
 lr=<low-rank matrix: shape=3x3, dtype=float64, rank=1, sign=0
     left=<dense matrix: shape=3x1, dtype=float64
           mat=[[0.5]
                [0.5]
                [0.5]]>
     middle=[[-0.4]]
     right=<dense matrix: shape=3x1, dtype=float64
            mat=[[0.5]
                 [0.5]
                 [0.5]]>>>

>>> B.kron(d, 2 * d)
<Kronecker product: shape=9x9, dtype=float64
 left=<diagonal matrix: shape=3x3, dtype=float64
       diag=[1. 1. 1.]>
 right=<diagonal matrix: shape=3x3, dtype=float64
        diag=[2. 2. 2.]>>


>>> B.inv(B.kron(d, 2 * d))
<Kronecker product: shape=9x9, dtype=float64
 left=<diagonal matrix: shape=3x3, dtype=float64
       diag=[1. 1. 1.]>
 right=<diagonal matrix: shape=3x3, dtype=float64
        diag=[0.5 0.5 0.5]>>
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
LowRank (with definiteness: PositiveLowRank, NegativeLowRank)
Woodbury
Kronecker
```


## Functions

The following functions are added to LAB.
They can be accessed with `B.<function>` where `import lab as B`.

```
dense(a)
fill_diag(a, diag_len)
block(*rows)

matmul_diag(a, b, tr_a=False, tr_b=False)
iqf(a, b, c)
iqf_diag(a, b, c)
ratio(a, c)
root(a)
sample(a, num=1)
```
