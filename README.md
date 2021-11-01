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

>>> d = Diagonal(B.rand(2, 3))  # A batch of diagonal marices

>>> d
<diagonal matrix: batch=(2,), shape=(3, 3), dtype=float64
 diag=[[0.427 0.912 0.622]
       [0.777 0.048 0.808]]>

>>> 2 * d
<diagonal matrix: batch=(2,), shape=(3, 3), dtype=float64
 diag=[[0.854 1.824 1.243]
       [1.553 0.096 1.616]]>
  
>>> 2 * d + 1
<Woodbury matrix: batch=(2,), shape=(3, 3), dtype=float64
 diag=<diagonal matrix: batch=(2,), shape=(3, 3), dtype=float64
       diag=[[0.854 1.824 1.243]
             [1.553 0.096 1.616]]>
 lr=<low-rank matrix: batch=(), shape=(3, 3), dtype=int64, rank=1
     left=[[1]
           [1]
           [1]]
     middle=<diagonal matrix: batch=(), shape=(1, 1), dtype=int64
             diag=[1]>>>

>>> B.inv(2 * d + 1)
<Woodbury matrix: batch=(2,), shape=(3, 3), dtype=float64
 diag=<diagonal matrix: batch=(2,), shape=(3, 3), dtype=float64
       diag=[[ 1.171  0.548  0.804]
             [ 0.644 10.386  0.619]]>
 lr=<low-rank matrix: batch=(2,), shape=(3, 3), dtype=float64, rank=1
     left=<dense matrix: batch=(2,), shape=(3, 1), dtype=float64
           mat=[[[ 1.171]
                 [ 0.548]
                 [ 0.804]]

                [[ 0.644]
                 [10.386]
                 [ 0.619]]]>
     middle=<dense matrix: batch=(2,), shape=(1, 1), dtype=float64
             mat=[[[-0.284]]

                  [[-0.079]]]>
     right=<dense matrix: batch=(2,), shape=(3, 1), dtype=float64
            mat=[[[ 1.171]
                  [ 0.548]
                  [ 0.804]]

                 [[ 0.644]
                  [10.386]
                  [ 0.619]]]>>>

>>> B.inv(B.inv(2 * d + 1))
<Woodbury matrix: batch=(2,), shape=(3, 3), dtype=float64
 diag=<diagonal matrix: batch=(2,), shape=(3, 3), dtype=float64
       diag=[[0.854 1.824 1.243]
             [1.553 0.096 1.616]]>
 lr=<low-rank matrix: batch=(2,), shape=(3, 3), dtype=float64, rank=1
     left=<dense matrix: batch=(2,), shape=(3, 1), dtype=float64
           mat=[[[1.]
                 [1.]
                 [1.]]

                [[1.]
                 [1.]
                 [1.]]]>
     middle=<dense matrix: batch=(2,), shape=(1, 1), dtype=float64
             mat=[[[1.]]

                  [[1.]]]>
     right=<dense matrix: batch=(2,), shape=(3, 1), dtype=float64
            mat=[[[1.]
                  [1.]
                  [1.]]

                 [[1.]
                  [1.]
                  [1.]]]>>>

>>> B.inv(B.inv(2 * d + 1)) - 1
<diagonal matrix: batch=(2,), shape=(3, 3), dtype=float64
 diag=[[0.854 1.824 1.243]
       [1.553 0.096 1.616]]>
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
TiledBlocks
```

## Functions

The following functions are added to LAB.
They can be accessed with `B.<function>` where `import lab as B`.

```
shape_broadcast(*elements)
shape_batch(a, *indices)
shape_batch_broadcast(*elements)
shape_matrix(a, *indices)
shape_matrix_broadcast(*elements)

broadcast_batch_to(a, *batch)

dense(a)
fill_diag(a, diag_len)
block(*rows)
block_diag(*blocks)

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
