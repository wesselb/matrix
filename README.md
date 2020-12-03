# [Structured Matrices](http://github.com/wesselb/matrix)

[![CI](https://github.com/wesselb/matrix/workflows/CI/badge.svg?branch=master)](https://github.com/wesselb/matrix/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/matrix/badge.svg?branch=master&service=github)](https://coveralls.io/github/wesselb/matrix?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://wesselb.github.io/matrix)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Structured matrices

## Requirements and Installation

See [the instructions here](https://gist.github.com/wesselb/4b44bf87f3789425f96e26c4308d0adc).
Then simply


```bash
pip install backends-matrix
```

## Example
```python
>>> import lab as B

>>> from matrix import Diagonal

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
 lr=<low-rank matrix: shape=3x3, dtype=int64, rank=1
     left=[[1]
           [1]
           [1]]
     middle=<diagonal matrix: shape=1x1, dtype=int64
             diag=[1]>>>
  
>>> B.inv(2 * d + 1)
<Woodbury matrix: shape=3x3, dtype=float64
 diag=<diagonal matrix: shape=3x3, dtype=float64
       diag=[0.5 0.5 0.5]>
 lr=<low-rank matrix: shape=3x3, dtype=float64, rank=1
     left=<dense matrix: shape=3x1, dtype=float64
           mat=[[0.5]
                [0.5]
                [0.5]]>
     middle=<dense matrix: shape=1x1, dtype=float64
             mat=[[-0.4]]>
     right=<dense matrix: shape=3x1, dtype=float64
            mat=[[0.5]
                 [0.5]
                 [0.5]]>>>

>>> B.inv(B.inv(2 * d + 1))
<Woodbury matrix: shape=3x3, dtype=float64
 diag=<diagonal matrix: shape=3x3, dtype=float64
       diag=[2. 2. 2.]>
 lr=<low-rank matrix: shape=3x3, dtype=float64, rank=1
     left=<dense matrix: shape=3x1, dtype=float64
           mat=[[1.]
                [1.]
                [1.]]>
     middle=<dense matrix: shape=1x1, dtype=float64
             mat=[[1.]]>
     right=<dense matrix: shape=3x1, dtype=float64
            mat=[[1.]
                 [1.]
                 [1.]]>>>

>>> B.inv(B.inv(2 * d + 1)) + 3
<Woodbury matrix: shape=3x3, dtype=float64
 diag=<diagonal matrix: shape=3x3, dtype=float64
       diag=[2. 2. 2.]>
 lr=<low-rank matrix: shape=3x3, dtype=float64, rank=1
     left=[[1.]
           [1.]
           [1.]]
     middle=[[4.]]
     right=[[1.]
            [1.]
            [1.]]>>

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
block(*rows)

matmul_diag(a, b, tr_a=False, tr_b=False)

pd_inv(a)
schur(a)
pd_schur(a)
iqf(a, b, c)
iqf_diag(a, b, c)

ratio(a, c)
root(a)

sample(a, num=1)
```
