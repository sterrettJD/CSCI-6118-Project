#!/usr/bin/env python
# coding: utf-8

# # Tools for Analysis

# ## A Brief Introduction to Python
# 
# This resource assumes Python is already installed, and
# that the user has some familiarity with Python basics such as:
# 
# - Variables 
#     - Assigning variables
#     - Basic variable types (lists, dicts, strings, booleans, tuples)
# - Control flow
#     - `for` loops
#     - `while` loops
#     - `if`/`elif`/`else` statements
# - Logical checks
#     - E.g. `==`, `>`, `<`, etc
# - Functions
#     - Defining functions
#     - Calling functions
# - Methods and attributes
#     - Calling methods and attributes
#     - (We will likely not define any methods or attributes, but it will be good to understand how to use them)
# - Comments
# 
# If you need more information regarding those, see these resources:
# - [W3 Schools](https://www.w3schools.com/python/python_variables.asp)
#     - Work through Python modules

# ## Handling Data
# 
# ### NumPy
# 
# - NumPy (Numerical Python) is an open source Python library used in many areas of science and engineering.\
# It utilizes multidimensional array and matrix data structures, similarly to MATLAB. NumPy is an object-oriented\
# coding package that uses `ndarray`, and n-dimensional array object on which methods can operate. 
# 
# 
# #### Using NumPy on Your Computer
# 
# - NumPy must both be installed and imported.
#     - First, to install NumPy, open a Terminal window on your computer and type the following command: `pip install numpy`
#     - If you have Anaconda installed, you can install NumPy using the command: `conda install numpy`
#         - Note that Conda is the package manager for Anaconda whereas pip is the package manager for Python. You can use pip           if you do not have Anaconda installed.
#         
#     - Now that you have installed NumPy, you need to import it. To import this package, type the following command
#       into your Terminal window: `import numpy as np`. The imported name is shortened to `np` as a convention and for
#       better readability of code in NumPy. 
# 
# #### What's an Array?
# 
# - An array is sort of like a list in Python, but its size is initialized ahead of time. This means that the array
#   can only hold as many elements as its initial size - it cannot be appended like a list can. An array is a homogenous
#   data structure (all of the elements in an array are of the same data type). Each element in an array is numbered
#   consequetively, beginning with 0. An array can be multidimensional. 
#   
# - One-dimensional array can be indexed, sliced, and interated over similarly to lists in Python. Here are some examples of     one-dimensional arrays and their functionality: 

# In[1]:


import numpy as np


# In[2]:


# This array is manually initialized. It has a length of 7. Each element is filled with an int. 
array_one = [1, 2, 3, 4, 5, 6, 7]
print(array_one, '\n')

# This array is initialized using a for loop. The loop begins with i as 0 and finishes with i as 5 
# (i is the index in the array). Each index has a value of 3.
array_two = [3 for i in range(5)]
print(array_two, '\n')

# This array uses the object-oriented funtionality of NumPy by calling the empty() method. The empty()
# method takes the length of the array and element type as inputs. In this case, it returns an empty array of 
# length 10 which can be filled with strings.
array_three = np.empty(10, dtype=str)
print(array_three, '\n')

# This array demonstrates the algebraic functionality of NumPy. It outputs an array of length 4 with
# each element as the given string.
array_four = ['Hello World']*4
print(array_four, '\n')


# - Here are some examples of you can initialize multi-dimensional arrays:

# In[3]:


# Initialize an array of length 10 in dimension 1 and length 3 in dimension 2. 
# The array is filled with random integers in the range (0, 100).
multi_array_one = np.random.randint(100, size=(10,3))
print(multi_array_one, '\n')

# Initialize another array of length 10 in dimension 1 and length 3 in dimension 2. 
# The array is filled with random integers in the range (0, 100) and each element is then
# multiplied by 5.
multi_array_two = np.random.randint(100, size=(10,3))*5
print(multi_array_two, '\n')

# Initialize yet another array of length 10 in dimension 1 and length 3 in dimension 2,
# except this time each element is manually inputted.
multi_array_three = np.array([[55, 48, 40], [93, 58, 58],
                              [83, 57, 49], [39, 49, 34],
                              [92, 74, 20], [49, 37, 95],
                              [48, 30, 27], [60, 27, 59],
                              [29, 10, 44], [73, 93, 85]])
print(multi_array_three, '\n')


# #### Accessing Elements in an Array
# 
# - Let's say you have a large dataset and you want to extract data points below a certain value:
# 

# In[4]:


# Initialize a dataset, this time it's a random array of integers in the range (0, 1000).
# The array has a suze of 7x5.
data_to_analyze = np.random.randint(1000, size=(5, 7))
print(data_to_analyze, '\n')

# Set the value threshold. This is the maximum value you want to extract from your dataset.
value_threshold = 500
# Initialize an empty list which will eventually contain all datapoints you want to keep. 
output_data = []

# Iterate through the dataset and append kept datapoints to the output list:
for row in data_to_analyze:
    for element in row:
        if element < value_threshold:
            output_data.append(element)
print(output_data)


# - We can also use `np.where()` to give us the indices where some logical statement is true.

# In[5]:


indices_to_keep = np.where(data_to_analyze < value_threshold)
print(indices_to_keep, "\n")

output_data = data_to_analyze[indices_to_keep]
print(output_data, "\n")


# - Note that the above example simply creates a list of data that you want to keep derived from a larger dataset.
#   Let's say that you still want to keep the datapoints only below a given threshold, but you wish to retain the same size       dataset as the original:

# In[6]:


print(data_to_analyze, '\n')

# Set the value threshold. This is the maximum value you want to extract from your dataset.
value_threshold = 500
# Initialize an empty array which will eventually contain all datapoints you want to keep. 
output_data = np.empty((5,7), dtype=float)

# Iterate through the dataset and append kept datapoints to the output list:
for i, row in enumerate(data_to_analyze):
    for j, element in enumerate(row):
        if element < value_threshold:
            output_data[i,j] = element
        else:
            output_data[i,j] = np.NaN
    
print(output_data)


# ### Pandas
# 
# #### Sources: 
# - [NumPy: The Absolute Basics](https://numpy.org/doc/stable/user/absolute_beginners.html)
# - [NumPy: Create Random Set of Rows From 2D Array](https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-114.php)

# ## Plotting
# 
# ### Matplotlib
# 
# ### Seaborn

# ## Statistical analysis
# 
# ### SciPy
# 
# ### Scikit-bio

# ## Network analysis
# 
# ### Networkx

# In[ ]:




