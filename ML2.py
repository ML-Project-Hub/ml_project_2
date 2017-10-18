
# coding: utf-8

# In[8]:

import numpy as np
syn_input_data = np.genfromtxt('Info/input.csv', delimiter=',')
syn_output_data = np.genfromtxt(
'Info/output.csv', delimiter=',').reshape([-1, 1])
letor_input_data = np.genfromtxt(
'Info/Querylevelnorm_X.csv', delimiter=',')
letor_output_data = np.genfromtxt(
'Info/Querylevelnorm_t.csv', delimiter=',').reshape([-1, 1])


# In[9]:

def compute_design_matrix(X, centers, spreads):
    # use broadcast
    basis_func_outputs = np.exp(
    np.sum(
    np.matmul(X - centers, spreads) * (X - centers),
    axis=2
    ) / (-2)
    ).T
    # insert ones to the 1st col
    return np.insert(basis_func_outputs, 0, 1, axis=1)


# In[10]:

def closed_form_sol(L2_lambda, design_matrix, output_data):
    return np.linalg.solve(
    L2_lambda * np.identity(design_matrix.shape[1]) +
    np.matmul(design_matrix.T, design_matrix),
    np.matmul(design_matrix.T, output_data)
    ).flatten()


# In[11]:

def SGD_sol(learning_rate,minibatch_size,num_epochs,L2_lambda,design_matrix,output_data):
    N, _ = design_matrix.shape
    # You can try different mini-batch size size
    # Using minibatch_size = N is equivalent to standard gradient descent
    # Using minibatch_size = 1 is equivalent to stochastic gradient descent
    # In this case, minibatch_size = N is better
    weights = np.zeros([1, 4])
    # The more epochs the higher training accuracy. When set to 1000000,
    # weights will be very close to closed_form_weights. But this is unnecessary


# In[ ]:

for epoch in range(num_epochs):
    for i in range(N / minibatch_size):
        lower_bound = i * minibatch_size
        upper_bound = min((i+1)*minibatch_size, N)
        Phi = design_matrix[lower_bound : upper_bound, :]
        t = output_data[lower_bound : upper_bound, :]
        E_D = np.matmul((np.matmul(Phi, weights.T)-t).T,Phi)
        E = (E_D + L2_lambda * weights) / minibatch_size
        weights = weights - learning_rate * E
        print np.linalg.norm(E)
        #return weights.flatten()


# In[ ]:

N, D = input_data.shape
# Assume we use 3 Gaussian basis functions M = 3
# shape = [M, 1, D]
centers = np.array([np.ones((D))*1, np.ones((D))*0.5, np.ones((D))*1.5])
centers = centers[:, np.newaxis, :]
# shape = [M, D, D]
spreads = np.array([np.identity(D), np.identity(D), np.identity(D)]) * 0.5
# shape = [1, N, D]
X = input_data[np.newaxis, :, :]
design_matrix = compute_design_matrix(X, centers, spreads)


# In[ ]:

# Closed-form solution
print closed_form_sol(L2_lambda=0.1,design_matrix=design_matrix,output_data=output_data)
# Gradient descent solution
print SGD_sol(learning_rate=1,minibatch_size=N,num_epochs=10000,L2_lambda=0.1,design_matrix=design_matrix,output_data=output_data)


# In[ ]:



