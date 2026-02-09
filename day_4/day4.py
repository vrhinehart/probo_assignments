import numpy as np
"""
General form for covariance matrix for 3-variable system:
[Var(x),   Cov(x,y), Cov(x,z)
 Cov(y,x), Var(y),   Cov(y,z)
 Cov(z,x), Cov(z,y), Var(z) ]
"""

def covariance(m: np.ndarray):
    """
    Returns a covariance matrix of m
    
    m: 2-d np.ndarray where each row is a variable and each column is an obeservation
    """
    assert len(m.shape) == 2
    vars = m.shape[0]
    length = m.shape[1]

    output = np.empty((vars, vars))

    for row in range(vars):
        for col in range(vars):
            var_a = m[row,:]
            var_b = m[col,:]
            output[row, col] = sample_cov(var_a, var_b)

    return output.copy()

def sample_cov(a, b):
    """
    Computes the scalar sample covariance between two vectors
    a: the first vector
    b: the other vector
    """
    n = len(a)

    a = np.array(a)
    b = np.array(b)
    assert a.shape == b.shape
    assert len(a.shape) == 1

    output = 1/(n - 1) * np.sum((a - np.mean(a)) * (b - np.mean(b)))
    return output

x = [-2.1, -1,  4.3]
y = [3,  1.1,  0.12]
X = np.stack((x, y), axis=0)
print("Covariance:")
print("My version:")
print(covariance(X))
print("\nNumpy:")
print(np.cov(X))

def corrcoef(m: np.ndarray):
    """
    Constructs a normalized correlation coefficient matrix

    M: a 2-dimensional numpy array where rows are variables and cols are observations
    """

    assert len(m.shape) == 2
    vars = m.shape[0]
    length = m.shape[1]

    output = np.empty((vars, vars))
    c = covariance(m)

    for row in range(vars):
        for col in range(vars):
            c_ij = c[row, col]
            c_ii = c[row, row]
            c_jj = c[col, col]
            output[row, col] = c_ij / np.sqrt(c_ii * c_jj)

    return output.copy()

print("\n\nCorrcoef:")
print("My version:")
print(corrcoef(X))
print("\nNumpy:")
print(np.corrcoef(X))

# Covariance:
# My version:
# [[11.71       -4.286     ]
#  [-4.286       2.14413333]]

# Numpy:
# [[11.71       -4.286     ]
#  [-4.286       2.14413333]]


# Corrcoef:
# My version:
# [[ 1.         -0.85535781]
#  [-0.85535781  1.        ]]

# Numpy:
# [[ 1.         -0.85535781]
#  [-0.85535781  1.        ]]
