import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
plt.style.use('./deeplearning.mplstyle')

Now let's call the compute_model_output function and plot the output..


def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples
      w,b (scalar)    : model parameters
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb
############################################################################
tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
############################################################################
Computing Cost


def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.

    Args:
      x (ndarray (m,)): Data, m examples
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost

x_train = np.array([1.0, 2.0])           #(size in 1000 square feet)
y_train = np.array([300.0, 500.0])

plt_intuition(x_train,y_train)

# !/usr/bin/env python
# coding: utf-8

# # Optional Lab: Python, NumPy and Vectorization
# A brief introduction to some of the scientific computing used in this course. In particular the NumPy scientific computing package and its use with python.
#
# # Outline
# - [&nbsp;&nbsp;1.1 Goals](#toc_40015_1.1)
# - [&nbsp;&nbsp;1.2 Useful References](#toc_40015_1.2)
# - [2 Python and NumPy <a name='Python and NumPy'></a>](#toc_40015_2)
# - [3 Vectors](#toc_40015_3)
# - [&nbsp;&nbsp;3.1 Abstract](#toc_40015_3.1)
# - [&nbsp;&nbsp;3.2 NumPy Arrays](#toc_40015_3.2)
# - [&nbsp;&nbsp;3.3 Vector Creation](#toc_40015_3.3)
# - [&nbsp;&nbsp;3.4 Operations on Vectors](#toc_40015_3.4)
# - [4 Matrices](#toc_40015_4)
# - [&nbsp;&nbsp;4.1 Abstract](#toc_40015_4.1)
# - [&nbsp;&nbsp;4.2 NumPy Arrays](#toc_40015_4.2)
# - [&nbsp;&nbsp;4.3 Matrix Creation](#toc_40015_4.3)
# - [&nbsp;&nbsp;4.4 Operations on Matrices](#toc_40015_4.4)
#

# In[ ]:


import numpy as np  # it is an unofficial standard to use np for numpy
import time

# <a name="toc_40015_1.1"></a>
# ## 1.1 Goals
# In this lab, you will:
# - Review the features of NumPy and Python that are used in Course 1

# <a name="toc_40015_1.2"></a>
# ## 1.2 Useful References
# - NumPy Documentation including a basic introduction: [NumPy.org](https://NumPy.org/doc/stable/)
# - A challenging feature topic: [NumPy Broadcasting](https://NumPy.org/doc/stable/user/basics.broadcasting.html)
#

# <a name="toc_40015_2"></a>
# # 2 Python and NumPy <a name='Python and NumPy'></a>
# Python is the programming language we will be using in this course. It has a set of numeric data types and arithmetic operations. NumPy is a library that extends the base capabilities of python to add a richer data set including more numeric types, vectors, matrices, and many matrix functions. NumPy and python  work together fairly seamlessly. Python arithmetic operators work on NumPy data types and many NumPy functions will accept python data types.
#

# <a name="toc_40015_3"></a>
# # 3 Vectors
# <a name="toc_40015_3.1"></a>
# ## 3.1 Abstract
# <img align="right" src="./images/C1_W2_Lab04_Vectors.PNG" style="width:340px;" >Vectors, as you will use them in this course, are ordered arrays of numbers. In notation, vectors are denoted with lower case bold letters such as $\mathbf{x}$.  The elements of a vector are all the same type. A vector does not, for example, contain both characters and numbers. The number of elements in the array is often referred to as the *dimension* though mathematicians may prefer *rank*. The vector shown has a dimension of $n$. The elements of a vector can be referenced with an index. In math settings, indexes typically run from 1 to n. In computer science and these labs, indexing will typically run from 0 to n-1.  In notation, elements of a vector, when referenced individually will indicate the index in a subscript, for example, the $0^{th}$ element, of the vector $\mathbf{x}$ is $x_0$. Note, the x is not bold in this case.
#

# <a name="toc_40015_3.2"></a>
# ## 3.2 NumPy Arrays
#
# NumPy's basic data structure is an indexable, n-dimensional *array* containing elements of the same type (`dtype`). Right away, you may notice we have overloaded the term 'dimension'. Above, it was the number of elements in the vector, here, dimension refers to the number of indexes of an array. A one-dimensional or 1-D array has one index. In Course 1, we will represent vectors as NumPy 1-D arrays.
#
#  - 1-D array, shape (n,): n elements indexed [0] through [n-1]
#

# <a name="toc_40015_3.3"></a>
# ## 3.3 Vector Creation
#

# Data creation routines in NumPy will generally have a first parameter which is the shape of the object. This can either be a single value for a 1-D result or a tuple (n,m,...) specifying the shape of the result. Below are examples of creating vectors using these routines.

# In[ ]:


# NumPy routines which allocate memory and fill arrays with value
a = np.zeros(4);
print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.zeros((4,));
print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.random_sample(4);
print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# Some data creation routines do not take a shape tuple:

# In[ ]:


# NumPy routines which allocate memory and fill arrays with value but do not accept shape as input argument
a = np.arange(4.);
print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.rand(4);
print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# values can be specified manually as well.

# In[ ]:


# NumPy routines which allocate memory and fill with user specified values
a = np.array([5, 4, 3, 2]);
print(f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([5., 4, 3, 2]);
print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# These have all created a one-dimensional vector  `a` with four elements. `a.shape` returns the dimensions. Here we see a.shape = `(4,)` indicating a 1-d array with 4 elements.

# <a name="toc_40015_3.4"></a>
# ## 3.4 Operations on Vectors
# Let's explore some operations using vectors.
# <a name="toc_40015_3.4.1"></a>
# ### 3.4.1 Indexing
# Elements of vectors can be accessed via indexing and slicing. NumPy provides a very complete set of indexing and slicing capabilities. We will explore only the basics needed for the course here. Reference [Slicing and Indexing](https://NumPy.org/doc/stable/reference/arrays.indexing.html) for more details.
# **Indexing** means referring to *an element* of an array by its position within the array.
# **Slicing** means getting a *subset* of elements from an array based on their indices.
# NumPy starts indexing at zero so the 3rd element of an vector $\mathbf{a}$ is `a[2]`.

# In[ ]:


# vector indexing operations on 1-D vectors
a = np.arange(10)
print(a)

# access an element
print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")

# access the last element, negative indexes count from the end
print(f"a[-1] = {a[-1]}")

# indexs must be within the range of the vector or they will produce and error
try:
    c = a[10]
except Exception as e:
    print("The error message you'll see is:")
    print(e)

# <a name="toc_40015_3.4.2"></a>
# ### 3.4.2 Slicing
# Slicing creates an array of indices using a set of three values (`start:stop:step`). A subset of values is also valid. Its use is best explained by example:

# In[ ]:


# vector slicing operations
a = np.arange(10)
print(f"a         = {a}")

# access 5 consecutive elements (start:stop:step)
c = a[2:7:1];
print("a[2:7:1] = ", c)

# access 3 elements separated by two
c = a[2:7:2];
print("a[2:7:2] = ", c)

# access all elements index 3 and above
c = a[3:];
print("a[3:]    = ", c)

# access all elements below index 3
c = a[:3];
print("a[:3]    = ", c)

# access all elements
c = a[:];
print("a[:]     = ", c)

# <a name="toc_40015_3.4.3"></a>
# ### 3.4.3 Single vector operations
# There are a number of useful operations that involve operations on a single vector.

# In[ ]:


a = np.array([1, 2, 3, 4])
print(f"a             : {a}")
# negate elements of a
b = -a
print(f"b = -a        : {b}")

# sum all elements of a, returns a scalar
b = np.sum(a)
print(f"b = np.sum(a) : {b}")

b = np.mean(a)
print(f"b = np.mean(a): {b}")

b = a ** 2
print(f"b = a**2      : {b}")

# <a name="toc_40015_3.4.4"></a>
# ### 3.4.4 Vector Vector element-wise operations
# Most of the NumPy arithmetic, logical and comparison operations apply to vectors as well. These operators work on an element-by-element basis. For example
# $$ c_i = a_i + b_i $$

# In[ ]:


a = np.array([1, 2, 3, 4])
b = np.array([-1, -2, 3, 4])
print(f"Binary operators work element wise: {a + b}")

# Of course, for this to work correctly, the vectors must be of the same size:

# In[ ]:


# try a mismatched vector operation
c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print("The error message you'll see is:")
    print(e)

# <a name="toc_40015_3.4.5"></a>
# ### 3.4.5 Scalar Vector operations
# Vectors can be 'scaled' by scalar values. A scalar value is just a number. The scalar multiplies all the elements of the vector.

# In[ ]:


a = np.array([1, 2, 3, 4])

# multiply a by a scalar
b = 5 * a
print(f"b = 5 * a : {b}")


# <a name="toc_40015_3.4.6"></a>
# ### 3.4.6 Vector Vector dot product
# The dot product is a mainstay of Linear Algebra and NumPy. This is an operation used extensively in this course and should be well understood. The dot product is shown below.

# <img src="./images/C1_W2_Lab04_dot_notrans.gif" width=800>

# The dot product multiplies the values in two vectors element-wise and then sums the result.
# Vector dot product requires the dimensions of the two vectors to be the same.

# Let's implement our own version of the dot product below:
#
# **Using a for loop**, implement a function which returns the dot product of two vectors. The function to return given inputs $a$ and $b$:
# $$ x = \sum_{i=0}^{n-1} a_i b_i $$
# Assume both `a` and `b` are the same shape.

# In[ ]:


def my_dot(a, b):
    """
   Compute the dot product of two vectors

    Args:
      a (ndarray (n,)):  input vector
      b (ndarray (n,)):  input vector with same dimension as a

    Returns:
      x (scalar):
    """
    x = 0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x


# In[ ]:


# test 1-D
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
print(f"my_dot(a, b) = {my_dot(a, b)}")

# Note, the dot product is expected to return a scalar value.
#
# Let's try the same operations using `np.dot`.

# In[ ]:


# test 1-D
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
c = np.dot(a, b)
print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape} ")
c = np.dot(b, a)
print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(a, b).shape = {c.shape} ")

# Above, you will note that the results for 1-D matched our implementation.

# <a name="toc_40015_3.4.7"></a>
# ### 3.4.7 The Need for Speed: vector vs for loop
# We utilized the NumPy  library because it improves speed memory efficiency. Let's demonstrate:

# In[ ]:


np.random.seed(1)
a = np.random.rand(10000000)  # very large arrays
b = np.random.rand(10000000)

tic = time.time()  # capture start time
c = np.dot(a, b)
toc = time.time()  # capture end time

print(f"np.dot(a, b) =  {c:.4f}")
print(f"Vectorized version duration: {1000 * (toc - tic):.4f} ms ")

tic = time.time()  # capture start time
c = my_dot(a, b)
toc = time.time()  # capture end time

print(f"my_dot(a, b) =  {c:.4f}")
print(f"loop version duration: {1000 * (toc - tic):.4f} ms ")

del (a);
del (b)  # remove these big arrays from memory

# So, vectorization provides a large speed up in this example. This is because NumPy makes better use of available data parallelism in the underlying hardware. GPU's and modern CPU's implement Single Instruction, Multiple Data (SIMD) pipelines allowing multiple operations to be issued in parallel. This is critical in Machine Learning where the data sets are often very large.

# <a name="toc_12345_3.4.8"></a>
# ### 3.4.8 Vector Vector operations in Course 1
# Vector Vector operations will appear frequently in course 1. Here is why:
# - Going forward, our examples will be stored in an array, `X_train` of dimension (m,n). This will be explained more in context, but here it is important to note it is a 2 Dimensional array or matrix (see next section on matrices).
# - `w` will be a 1-dimensional vector of shape (n,).
# - we will perform operations by looping through the examples, extracting each example to work on individually by indexing X. For example:`X[i]`
# - `X[i]` returns a value of shape (n,), a 1-dimensional vector. Consequently, operations involving `X[i]` are often vector-vector.
#
# That is a somewhat lengthy explanation, but aligning and understanding the shapes of your operands is important when performing vector operations.

# In[ ]:


# show common Course 1 example
X = np.array([[1], [2], [3], [4]])
w = np.array([2])
c = np.dot(X[1], w)

print(f"X[1] has shape {X[1].shape}")
print(f"w has shape {w.shape}")
print(f"c has shape {c.shape}")

# <a name="toc_40015_4"></a>
# # 4 Matrices
#

# <a name="toc_40015_4.1"></a>
# ## 4.1 Abstract
# Matrices, are two dimensional arrays. The elements of a matrix are all of the same type. In notation, matrices are denoted with capitol, bold letter such as $\mathbf{X}$. In this and other labs, `m` is often the number of rows and `n` the number of columns. The elements of a matrix can be referenced with a two dimensional index. In math settings, numbers in the index typically run from 1 to n. In computer science and these labs, indexing will run from 0 to n-1.
# <figure>
#     <center> <img src="./images/C1_W2_Lab04_Matrices.PNG"  alt='missing'  width=900><center/>
#     <figcaption> Generic Matrix Notation, 1st index is row, 2nd is column </figcaption>
# <figure/>

# <a name="toc_40015_4.2"></a>
# ## 4.2 NumPy Arrays
#
# NumPy's basic data structure is an indexable, n-dimensional *array* containing elements of the same type (`dtype`). These were described earlier. Matrices have a two-dimensional (2-D) index [m,n].
#
# In Course 1, 2-D matrices are used to hold training data. Training data is $m$ examples by $n$ features creating an (m,n) array. Course 1 does not do operations directly on matrices but typically extracts an example as a vector and operates on that. Below you will review:
# - data creation
# - slicing and indexing

# <a name="toc_40015_4.3"></a>
# ## 4.3 Matrix Creation
# The same functions that created 1-D vectors will create 2-D or n-D arrays. Here are some examples
#

# Below, the shape tuple is provided to achieve a 2-D result. Notice how NumPy uses brackets to denote each dimension. Notice further than NumPy, when printing, will print one row per line.
#

# In[ ]:


a = np.zeros((1, 5))
print(f"a shape = {a.shape}, a = {a}")

a = np.zeros((2, 1))
print(f"a shape = {a.shape}, a = {a}")

a = np.random.random_sample((1, 1))
print(f"a shape = {a.shape}, a = {a}")

# One can also manually specify data. Dimensions are specified with additional brackets matching the format in the printing above.

# In[ ]:


# NumPy routines which allocate memory and fill with user specified values
a = np.array([[5], [4], [3]]);
print(f" a shape = {a.shape}, np.array: a = {a}")
a = np.array([[5],  # One can also
              [4],  # separate values
              [3]]);  # into separate rows
print(f" a shape = {a.shape}, np.array: a = {a}")

# <a name="toc_40015_4.4"></a>
# ## 4.4 Operations on Matrices
# Let's explore some operations using matrices.

# <a name="toc_40015_4.4.1"></a>
# ### 4.4.1 Indexing
#

# Matrices include a second index. The two indexes describe [row, column]. Access can either return an element or a row/column. See below:

# In[ ]:


# vector indexing operations on matrices
a = np.arange(6).reshape(-1, 2)  # reshape is a convenient way to create matrices
print(f"a.shape: {a.shape}, \na= {a}")

# access an element
print(
    f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar\n")

# access a row
print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")

# It is worth drawing attention to the last example. Accessing a matrix by just specifying the row will return a *1-D vector*.

# **Reshape**
# The previous example used [reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html) to shape the array.
# `a = np.arange(6).reshape(-1, 2) `
# This line of code first created a *1-D Vector* of six elements. It then reshaped that vector into a *2-D* array using the reshape command. This could have been written:
# `a = np.arange(6).reshape(3, 2) `
# To arrive at the same 3 row, 2 column array.
# The -1 argument tells the routine to compute the number of rows given the size of the array and the number of columns.
#

# <a name="toc_40015_4.4.2"></a>
# ### 4.4.2 Slicing
# Slicing creates an array of indices using a set of three values (`start:stop:step`). A subset of values is also valid. Its use is best explained by example:

# In[ ]:


# vector 2-D slicing operations
a = np.arange(20).reshape(-1, 10)
print(f"a = \n{a}")

# access 5 consecutive elements (start:stop:step)
print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

# access 5 consecutive elements (start:stop:step) in two rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

# access all elements
print("a[:,:] = \n", a[:, :], ",  a[:,:].shape =", a[:, :].shape)

# access all elements in one row (very common usage)
print("a[1,:] = ", a[1, :], ",  a[1,:].shape =", a[1, :].shape, "a 1-D array")
# same as
print("a[1]   = ", a[1], ",  a[1].shape   =", a[1].shape, "a 1-D array")

# <a name="toc_40015_5.0"></a>
# ## Congratulations!
# In this lab you mastered the features of Python and NumPy that are needed for Course 1.

# In[ ]:


def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """

    g = 1 / (1 + np.exp(-z))

    return g

# Plot z vs sigmoid(z)
fig,ax = plt.subplots(1,1,figsize=(5,3))
ax.plot(z_tmp, y, c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)

##################################################################################################################
##################################################################################################################
##################################################################################################################

def compute_cost_logistic(X, y, w, b):
    """
    Computes cost

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      cost (scalar): cost
    """

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)

    cost = cost / m
    return cost

"""X.shape[0] ifadesi ile veri setindeki örnek sayısı (m) bulunur.

cost isimli bir değişken başlangıçta 0.0 olarak ayarlanır. Bu değişken, tüm örnekler üzerinden toplam maliyeti tutmak için kullanılacak.

Bir for döngüsü başlatılır ve bu döngü veri setindeki her bir örnek için döner (m kadar).

Her bir döngü iterasyonunda, np.dot(X[i],w) + b ifadesi ile her bir örneğin ağırlıklı toplamı hesaplanır. Bu işlem, modelin tahminini (z_i) üretir.

sigmoid(z_i) fonksiyonu, z_i değerini bir olasılık değerine dönüştürür. Sigmoid fonksiyonu, girdi olarak aldığı herhangi bir gerçek sayıyı 0 ile 1 arasında bir değere dönüştürür, böylece bu değer bir olasılık olarak yorumlanabilir.

-y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i) ifadesi, iki sınıf için çapraz entropi maliyetini hesaplar. Burada y[i] gerçek etiket (0 veya 1), f_wb_i ise modelin tahmini olasılıktır. Maliyet, modelin tahmininin gerçek değerden ne kadar farklı olduğunu ölçer. Gerçek değer ile tahmin arasındaki uyumsuzluk ne kadar büyükse, maliyet o kadar yüksek olur.

Tüm örnekler için maliyet toplandıktan sonra, bu toplam örnek sayısına (m) bölünerek ortalama maliyet hesaplanır.

Fonksiyon, hesaplanan ortalama maliyeti geri döndürür.

"""

##################################################################################################################
##################################################################################################################
##################################################################################################################

def compute_gradient_logistic(X, y, w, b):
    """
    Computes the gradient for logistic regression

    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape
    dj_dw = np.zeros((n,))  # (n,)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)  # (n,)(n,)=scalar
        err_i = f_wb_i - y[i]  # scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i, j]  # scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw / m  # (n,)
    dj_db = dj_db / m  # scalar

    return dj_db, dj_dw

"""Girdiler olarak, veri seti X, hedef değerler y, model parametreleri w ve sabit terim b alınır.

Fonksiyon öncelikle m örneklemin sayısını ve her bir örnekte n özelliğin sayısını X matrisinin şeklinden (X.shape) çıkararak alır.

dj_dw ve dj_db sıfır olarak başlatılır; bunlar sırasıyla w ve b parametreleri için gradyanın toplamını tutacak olan değişkenlerdir.

Her bir örnek (i) için:

f_wb_i değişkeni, modelin i örneği için tahmini olasılığını hesaplar. Bu, X[i] (özellik vektörü) ile w (ağırlık vektörü) arasındaki nokta çarpımının b ile toplanması ve sonucun sigmoid fonksiyonundan geçirilmesiyle yapılır.
err_i, modelin tahmini ile gerçek hedef değer y[i] arasındaki farkı (f_wb_i - y[i]) hesaplar.
dj_dw[j] için, j her bir özellik indeksi için, bu hata err_i ile X[i,j] (o örnekteki j özelliğinin değeri) çarpılır ve dj_dw[j]'ye eklenir.
dj_db için, hata err_i doğrudan dj_db'ye eklenir.
Her bir gradyan toplamı örnek sayısı m ile bölünür, böylece gradyanın ortalama değeri hesaplanır."""


def compute_gradient_logistic(X, y, w, b):
    """
    Computes the gradient for logistic regression

    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape
    dj_dw = np.zeros((n,))  # (n,)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)  # (n,)(n,)=scalar
        err_i = f_wb_i - y[i]  # scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i, j]  # scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw / m  # (n,)
    dj_db = dj_db / m  # scalar

    return dj_db, dj_dw
####################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################

def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    """
    Performs batch gradient descent

    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(compute_cost_logistic(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return w, b, J_history  # return final w,b and J history for graphing

Let's run gradient descent on our data set.

w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters)
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")

Let's plot the results of gradient descent:¶

fig,ax = plt.subplots(1,1,figsize=(5,4))
# plot the probability
plt_prob(ax, w_out, b_out)

# Plot the original data
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
ax.axis([0, 4, 0, 3.5])
plot_data(X_train,y_train,ax)

# Plot the decision boundary
x0 = -b_out/w_out[0]
x1 = -b_out/w_out[1]
ax.plot([0,x0],[x1,0], c=dlc["dlblue"], lw=1)
plt.show()

