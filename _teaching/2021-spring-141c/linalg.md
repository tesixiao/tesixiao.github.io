---
title: "Linear Algebra in Python"
collection: teaching
permalink: /teaching/2021-spring-141c/linalg
---

# STA 141C Big-data and Statistical Computing

## Discussion 1: Linear Algebra in Python

TA: Tesi Xiao

In Python, we usually use [NumPy](https://numpy.org/) to implement the matrix computations for the sake of efficiency. The package `NumPy` provides several powerful classes and methods for numerical computations. 

To be specific, the class `np.ndarray` (`np.array`) is commonly used for matrix computations. Please check out this [manual](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html).

### Vector - 1d array


```python
import numpy as np
vec = np.array([1,2,3])
vec.shape
```




    (3,)



`(3,)` is a tuple with only one element. The 1-d array has only the `length` attribute.


```python
len(vec)
```




    3



### Matrix - 2d array


```python
mat = np.array([[1,2,3],[4,5,9],[7,8,9],[10,11,12]])
mat
```




    array([[ 1,  2,  3],
           [ 4,  5,  9],
           [ 7,  8,  9],
           [10, 11, 12]])




```python
mat.shape
```




    (4, 3)



`(4, 3)` is a tuple with two elements, which corresponds to # of rows, # of columns.

### Tensor - multidimensional array

Tensors are a generalized data structure, which have a wide range of applications in modern machine learning. We will later discuss it in the section of deep neural networks. For example, in [PyTorch](https://pytorch.org/), we will use tensors to encode the inputs and outputs of a model, as well as the model’s parameters. Then, tensors can run on GPUs or other specialized hardware to accelerate computing.



### Matrix Computations

#### 1. Matrix Addition, Substraction, Element-wise Multiplication & Division


```python
A = np.array([[1,2],[3,4]])
B = np.array([[2,3],[4,5]])
A+B
```




    array([[3, 5],
           [7, 9]])




```python
A-B
```




    array([[-1, -1],
           [-1, -1]])




```python
A*B
```




    array([[ 2,  6],
           [12, 20]])




```python
A/B
```




    array([[0.5       , 0.66666667],
           [0.75      , 0.8       ]])



`+`, `-`, `*`, `/` will do element-wise addition and substraction. If the shapes of two arrays are different, this operation cannot be done with an error message. However, the matrix-scalar calculations are acceptable.


```python
A + 1
```




    array([[2, 3],
           [4, 5]])




```python
A * 2
```




    array([[2, 4],
           [6, 8]])



#### 2. Matrix Muplication

When computing the matrix muplication using NumPy and the operator `@`, make sure that the shapes of two operands follow the rules, i.e., `(m,p) @ (p,n)` $\rightarrow$ `(m,n)` .

- Matrix-Vector Product (2darray-1darray Product)


```python
b = np.array([1,1,1])
mat @ b
```




    array([ 6, 18, 24, 33])




```python
c = np.array([1,1,1,1])
c @ mat
```




    array([22, 26, 33])



As long as two shapes follow the matrix multiplication rules, it returns a 1d-array.

- Matrix-Matrix Product (2darray-2darray Product)


```python
b = b.reshape(-1,1)
b, b.shape
```




    (array([[1],
            [1],
            [1]]), (3, 1))




```python
mat @ b
```




    array([[ 6],
           [18],
           [24],
           [33]])




```python
A @ B
```




    array([[10, 13],
           [22, 29]])



It returns a 2d-array following the multiplication rules.

- Matrix Transpose


```python
mat.T
```




    array([[ 1,  4,  7, 10],
           [ 2,  5,  8, 11],
           [ 3,  9,  9, 12]])



#### 3. Eigenvalues and Eigenvectors

[`np.linalg.eig`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html)


```python
# set a random seed
np.random.seed(2021)
# Create a 3 by 3 random matrix
A = np.random.rand(3,3)

# Compute eigenvalues and eigenvectors
eigenvals, eigenvecs = np.linalg.eig(A)

eigenvals
```




    array([1.46993771, 0.28721889, 0.50822548])




```python
eigenvecs
```




    array([[-0.55986774, -0.76336389, -0.14005302],
           [-0.54059029,  0.42411145, -0.16626493],
           [-0.62794128, -0.48724229,  0.97608459]])



Note that only if all the eigenvectors are linearly independent, we have the eigendecomposition $A = U\Sigma U^\top$.

#### 4. Sigular Value Decomposition (SVD)

[`np.linalg.svd`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)

$A_{m\times n} = U_{m\times m} S_{m\times n} V^\top_{n\times n}$


```python
# Create a 5 by 3 random matrix
A = np.random.randn(5, 3)
# Singular Value Decomposition
U, S, Vt = np.linalg.svd(A, full_matrices=True)
```


```python
U, U.shape
```




    (array([[ 0.22073128, -0.03043587,  0.16712615, -0.89000162,  0.36099491],
            [-0.11471826,  0.92761524, -0.33552023, -0.09536938,  0.06856048],
            [ 0.22007922,  0.12998267,  0.32382857,  0.43547724,  0.80010267],
            [ 0.68167637, -0.16826726, -0.70307225,  0.0763847 ,  0.08281451],
            [-0.65192015, -0.30560473, -0.51021515, -0.057678  ,  0.46686145]]),
     (5, 5))




```python
S, S.shape
```




    (array([3.36166521, 2.40725415, 1.48282724]), (3,))



`S` here is a 1d-array containing non-zero diagonal entries.


```python
Vt, Vt.shape
```




    (array([[-0.5160804 ,  0.28883446, -0.80637193],
            [ 0.19164251,  0.95649989,  0.21995704],
            [ 0.83482583, -0.04101963, -0.5489838 ]]), (3, 3))



#### 5. Norms

[`np.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)`(x, ord=None, axis=None, keepdims=False)`

Below are several useful norms in this class

| ord  | matrix norm                     | vector norm |
| ---- | ------------------------------- | ----------- |
| None | Frobenius norm                  | 2-norm      |
| 1    | max(sum(abs(x), axis=0))        | 1-norm      |
| 2    | 2-norm (largest singular value) | 2-norm      |
| inf  | max(sum(abs(x), axis=1))        | max(abs(x)) |
| 0    | –                               | sum(x != 0) |


```python
np.linalg.norm(A) # Frobenius norm
```




    4.392543930507632




```python
np.linalg.norm(b, 1) # L1-norm for vector b 
```




    3.0

