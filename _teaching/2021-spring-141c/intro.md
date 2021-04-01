---
title: "Introduction to Python Programming"
collection: teaching
permalink: /teaching/2021-spring-141c/intro
---


# STA 141C Big-data and Statistical Computing

## Discussion 1: Introduction to Python Programming

TA: Tesi Xiao


### Course Overview

- *Python* is the **only** programming language used in this class. If you are unfamiliar with the basic syntax (like the control flow tools), please quickly go over *Section 1-6* in [Python tutorials](https://docs.python.org/3/tutorial/index.html) in the first several weeks. Also, a sense of objected-oriented programming (OOP) is preferred. If you want to learn OOP in Python, take time to read *Section 9: Classes*.

- The course will cover:
  + Python Programming: Data Manipulation, Parallel Computing, Visualization, etc.
  + Two Algorithms: Power Method, Gradient Descent.
  + Unsupervised Learning: Clustering, Dimension Reduction, etc.
  + Supervised Learning: Regression, SVM, Neural Networks, etc.

- The main focus of this course is to understand two algorithms (Power Method and Gradient Descent), which are widely used to study (un)-supervised learning, and learn how to implement machine learning techniques to analyze the large-scale real world data in Python.

- This is not a machine learning or deep learning course. We will not cover too much technical details. 

### Python Environment Setup

For this class, installing [Anaconda](https://www.anaconda.com/) for setting up your programming environment is strongly recommended.

Several tools you may use for programming:

- [Jupyter Notebook](https://jupyter.org/): a great platform to organize your project with Markdown and Python. We will use this to submit homework assignments.
- [PyCharm](https://www.jetbrains.com/pycharm/): a great IDE for Python.


(Adanced) Text Editors you can use for coding:

- [Vim](https://www.vim.org/)
- [Sublime Text 3](https://www.sublimetext.com/)
- [Visual Studio Code](https://code.visualstudio.com/)

[Markdown Syntax](https://www.markdownguide.org/basic-syntax/)
[Markdown Cheatsheet](https://www.markdownguide.org/cheat-sheet/)

### Python Basic Syntax

Please quick review all the listed commands.

- Operators: `+`, `-`, `*`, `**`, `/`, `//`, `=`, `==`, `is`, `in`, ...
- Control flow tools: `if...elif...else...`, `for`, `while`, `break`, `continue`, `pass`, ...

### Python Basic Data Structures

To warm up, we summarize several basic data structures in Python. Below are four commonly-used structures.

#### 1. Lists


```python
# list
a = []
print("Empty: "+str(a))

a.append('a')
a.append('c')
print("After appending: "+str(a))

a.insert(1,'b')
print("After inserting: "+str(a))

a.pop()
print("After popping: "+str(a))

# .... other methods for Python basic lists (sort, remove, ...)

# index starts from 0 (unlike R)
print("a[0] = "+str(a[0]))
print("a[1] = "+str(a[1]))
```

    Empty: []
    After appending: ['a', 'c']
    After inserting: ['a', 'b', 'c']
    After popping: ['a', 'b']
    a[0] = a
    a[1] = b


#### 2. Tuples: immutable lists which are faster and consume less memory


```python
b = (1,2,2,3)

# count the number of occurence of a value
b.count(2)
```




    2



#### 3. Dictinaries (Hashmap)


```python
# dictionaries contain a mapping from keys to values (fast)
d = {'first':'string value', 'second':[1,2]}

print("keys:"+str(d.keys()))
print("values:"+str(d.values()))
```

    keys:dict_keys(['first', 'second'])
    values:dict_values(['string value', [1, 2]])


#### 4. Sets


```python
a = set([1, 2, 3, 4])
b = set([3, 4, 5, 6])

print("a | b: "+ str(a | b ))# Union

print("a & b: "+ str(a & b )) # Intersection
```

    a | b: {1, 2, 3, 4, 5, 6}
    a & b: {3, 4}


### Advanced Data Structures from Packages

For example,

- [numpy.array](https://numpy.org/devdocs/user/quickstart.html)

- [pandas.Series, pandas.DataFrame](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html)

### Module and Package

*Python* is **open**. Python is developed under an OSI-approved open source license, making it freely usable and distributable, even for commercial use. Everyone can contribute to this community, such as developing useful modules.




```python
import math

math.sqrt(2)

import math as m

m.sqrt(2)
```




    1.4142135623730951




```python
from math import sqrt

# from math import *

sqrt(2)
```




    1.4142135623730951



Below are some useful modules you will use in this class.

- [numpy](https://numpy.org/): The fundamental package for scientific computing with Python 
- [scikit-learn](https://scikit-learn.org/stable/): Machine Learning in Python
- [matploblib](https://matplotlib.org/): Visualization with Python 
- [seaborn](https://seaborn.pydata.org/): Statistical data visualization
- [statsmodels](https://www.statsmodels.org/stable/index.html): statistical models, hypothesis tests, and data exploration
- [PyTorch](https://pytorch.org/): An open source machine learning framework that accelerates the path from research prototyping to production deployment


### Function

You can define your own function or call functions from modules.


```python
# define your own functions
def myfunction(a):
    b = a
    return b
```


```python
# Functions from the module
math.sqrt(3)

np.array([1,2,3])
```




    array([1, 2, 3])



### Class, Object and its methods (OOP)

Object-oriented programming (OOP) is a programming paradigm based on the concept of "objects", which can contain data and code: data in the form of fields (often known as attributes or properties), and code, in the form of procedures (often known as methods). 


```python
import numpy as np
# Create a np.array object
ar = np.array([1,2,3])

# Check all the attributes and methods of this object
dir(ar)
```




    ['T',
     '__abs__',
     '__add__',
     '__and__',
     '__array__',
     '__array_finalize__',
     '__array_function__',
     '__array_interface__',
     '__array_prepare__',
     '__array_priority__',
     '__array_struct__',
     '__array_ufunc__',
     '__array_wrap__',
     '__bool__',
     '__class__',
     '__complex__',
     '__contains__',
     '__copy__',
     '__deepcopy__',
     '__delattr__',
     '__delitem__',
     '__dir__',
     '__divmod__',
     '__doc__',
     '__eq__',
     '__float__',
     '__floordiv__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__getitem__',
     '__gt__',
     '__hash__',
     '__iadd__',
     '__iand__',
     '__ifloordiv__',
     '__ilshift__',
     '__imatmul__',
     '__imod__',
     '__imul__',
     '__index__',
     '__init__',
     '__init_subclass__',
     '__int__',
     '__invert__',
     '__ior__',
     '__ipow__',
     '__irshift__',
     '__isub__',
     '__iter__',
     '__itruediv__',
     '__ixor__',
     '__le__',
     '__len__',
     '__lshift__',
     '__lt__',
     '__matmul__',
     '__mod__',
     '__mul__',
     '__ne__',
     '__neg__',
     '__new__',
     '__or__',
     '__pos__',
     '__pow__',
     '__radd__',
     '__rand__',
     '__rdivmod__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__rfloordiv__',
     '__rlshift__',
     '__rmatmul__',
     '__rmod__',
     '__rmul__',
     '__ror__',
     '__rpow__',
     '__rrshift__',
     '__rshift__',
     '__rsub__',
     '__rtruediv__',
     '__rxor__',
     '__setattr__',
     '__setitem__',
     '__setstate__',
     '__sizeof__',
     '__str__',
     '__sub__',
     '__subclasshook__',
     '__truediv__',
     '__xor__',
     'all',
     'any',
     'argmax',
     'argmin',
     'argpartition',
     'argsort',
     'astype',
     'base',
     'byteswap',
     'choose',
     'clip',
     'compress',
     'conj',
     'conjugate',
     'copy',
     'ctypes',
     'cumprod',
     'cumsum',
     'data',
     'diagonal',
     'dot',
     'dtype',
     'dump',
     'dumps',
     'fill',
     'flags',
     'flat',
     'flatten',
     'getfield',
     'imag',
     'item',
     'itemset',
     'itemsize',
     'max',
     'mean',
     'min',
     'nbytes',
     'ndim',
     'newbyteorder',
     'nonzero',
     'partition',
     'prod',
     'ptp',
     'put',
     'ravel',
     'real',
     'repeat',
     'reshape',
     'resize',
     'round',
     'searchsorted',
     'setfield',
     'setflags',
     'shape',
     'size',
     'sort',
     'squeeze',
     'std',
     'strides',
     'sum',
     'swapaxes',
     'take',
     'tobytes',
     'tofile',
     'tolist',
     'tostring',
     'trace',
     'transpose',
     'var',
     'view']



Also, check out the [documentations](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) for detailed intructions of all these methods.


```python
# Call the min() method
ar.min() # ar.max()
```




    1

