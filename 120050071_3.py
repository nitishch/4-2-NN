import math
import numpy as np

def initialise_fs(dims):
    """Initialise a structure that is of suitable shape so that fs, delfs
can fit in the return type. It is basically nested lists. Structure is
initialised with 0s. 

For example, if dims = (3,2,1), output will be
[[0,0,0],[0,0],[0]]."""
    return [[0] * a for a in dims]

def initialise_ws(dims):
    """Initialise a structure that is of suitable shape so that ws, can
fit in the return type. It is basically nested lists. Structure is
initialised with 0s. Aim is that w[l][i][j] must return the weight
between ith node in lth layer and jth node in l+1 st layer. (Since the
nodes are relatively small in our case, it is better to sacrifice
efficiency for simplicity in implementation)

For example, if dims = (3,2,1), output will be
[[[0,0],[0,0],[0,0]],[[0],[0]]].
"""
    return [[[n/1000 + i/100 + j/10000 for j in range(dims[i + 1])] for n in range(dims[i])] for i in range(len(dims) - 1)]

def g(x):
    try:
        return 1/(1 + math.exp(-x))
    except OverflowError:
        if x < 0:
            return 0.00000001
        else:
            return 0.99999999


def feed_forward(input_values, dims, weights):
    """Given the input values, f and weights, this will update f and return f after propagating the given input"""
    f = initialise_fs(dims)
    if len(input_values) != len(f[0]):
        raise Exception('input vector and neural network input size do not match')
    f[0] = input_values
    for l in range(1, len(f)):
        f[l] = [g(sum((f[l - 1][i] * weights[l - 1][i][j] for i in range(len(f[l - 1]))))) for j in range(len(f[l]))]
        # We can't put the outer loop as a list comprehension too
        # because we want to use the updated values of f in next
        # computations. Anyway, it is easier to read this way
    return f

def doefs(f, output, weights):
    """Given f after feed forward, expected output output and weights weights of the neural network returns the doe(E)/doe(f) values for all the nodes

    Remember we are doing only Stochastic gradient
"""
    doefs = initialise_fs([len(x) for x in f]) # This is a turnabout way of saying dims
    if f[-1][0] == 0:
        f[-1][0] = 0.00000001
    if f[-1][0] == 1:
        f[-1][0] = 0.99999999
    doefs[-1][0] = - output/f[-1][0] + (1 - output)/(1 - f[-1][0])
    for l in range(len(f) - 2, -1, -1):
        doefs[l] = [sum((doefs[l + 1][j] * f[l + 1][j] * (1 - f[l + 1][j]) * weights[l][i][j] for j in range(len(f[l + 1]))))
                   for i in range(len(f[l]))]
    return doefs

def gradient(doefs, f, weights):
    gradients = initialise_ws([len(x) for x in f])
    for l in range(len(weights)):
        for i in range(len(weights[l])):
            for j in range(len(weights[l][i])):
                gradients[l][i][j] = doefs[l + 1][j] * f[l + 1][j] * (1 - f[l + 1][j]) * f[l][i]
    return gradients


def update(weights, grad, eta = 1):
    for i in range(len(weights)):
        if not isinstance(weights[i], list):
            weights[i] -= eta * grad[i]
        else:
            update(weights[i], grad[i], eta)
    return weights


train_data = np.genfromtxt('Train.csv', delimiter = ',', dtype = float)
X = train_data[:, 0:-2]
y = train_data[:, -1]
X_test = np.genfromtxt('TestX.csv', delimiter = ',', dtype = float)
X_test = X_test[:, 0:-1]

def normalise(X):
    """X is an ndarray. We have to mean and variance normalise it"""
    mean = sum(X)/np.shape(X)[0] # This is the mean row. Mean of each
                                 # column is present in this
    X = X - mean                 # Surprisingly this subtracts columns
                                 # in the order we want them
    mean_of_squares = sum(X * X)/np.shape(X)[0]
    X = X/np.sqrt(mean_of_squares)
    return X, mean, mean_of_squares

X,mean,variance = normalise(X)



dims = (np.shape(X)[1], 10, 1)
weights = initialise_ws(dims)
does = []
grad = []



def take_input(X_row, y_row,weights):
    fs = feed_forward(X_row, dims, weights)
    does = doefs(fs, y_row, weights)
    grad = gradient(does, fs, weights)
    weights = update(weights, grad, 0.05)



for i in range(10):
    for i in range(np.shape(X)[0]):
        take_input(np.ndarray.tolist(X[i, :]), y[i], weights)

correct = 0
collection = [0] * np.shape(X)[0]
for i in range(np.shape(X)[0]):
    output = feed_forward(np.ndarray.tolist(X[i, :]), dims, weights)[-1][0]
    collection[i] = output

o = [int(x >= 0.5) for x in collection]
print('accuracy on training set is {0}%'.format(sum(o == y) * 100/np.shape(X)[0]))



X_test = X_test - mean
X_test = X_test/np.sqrt(variance)

printable = [0] * np.shape(X_test)[0]
for i in range(np.shape(X_test)[0]):
    output = feed_forward(np.ndarray.tolist(X_test[i, :]), dims, weights)[-1][0]
    printable[i] = output

sub = [int(x >= 0.5) for x in printable]

output_file = open('TestY.csv', 'w')
output_file.write("Id,Label\n")
for i in range(len(sub)):
    output_file.write(str(i) + "," + str(sub[i]) + "\n")
output_file.close()