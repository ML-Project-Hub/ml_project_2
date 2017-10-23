
# coding: utf-8

# In[1]:

import numpy as np
import random
from sklearn.cluster import KMeans
syn_input_data = np.genfromtxt('Info/input.csv', delimiter=',')
syn_output_data = np.genfromtxt('Info/output.csv', delimiter=',').reshape([-1, 1])
letor_input_data = np.genfromtxt('Info/Querylevelnorm_X.csv', delimiter=',')
letor_output_data = np.genfromtxt('Info/Querylevelnorm_t.csv', delimiter=',').reshape([-1, 1])


# In[ ]:

def shuffle(input_data):
    indices = [i for i in range(len(input_data))]
    random.shuffle(indices)
    return indices
    
def split_data(input_data,indices):
    length = len(indices)
    training_data = [input_data[indices[i]] for i in range(int(length*0.8))]
    validation_data = [input_data[indices[i]] for i in range(int(length*0.8),int(length*0.9))]
    test_data = [input_data[indices[i]] for i in range(int(length*0.9),length)]
    total_data = [input_data[indices[i]] for i in range(length)]
    return np.array(training_data),np.array(validation_data),np.array(test_data),np.array(total_data)


# In[2]:

def k_means(k,input_data):
    kmeans = KMeans(n_clusters=k, random_state=None, precompute_distances=True, n_init=30).fit(input_data)
    dic = {}
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    for i in range(len(labels)):
        try:
            dic[labels[i]] += [input_data[i]]
        except:
            dic[labels[i]] = [input_data[i]]

    for i in dic.keys():
        listty = np.matrix(dic[i])
        dic[i] = {}
        dic[i]["val"] = listty
        dic[i]["center"] = list(centers[i])
        dic[i]["spread"] = np.linalg.pinv(np.matrix(np.cov(listty.T)))
    return dic


# In[ ]:

##Given in ppt
def compute_design_matrix(X, centers, spreads):
    # use broadcast
    basis_func_outputs = np.exp(np.sum(np.matmul(X - centers, spreads) * (X - centers),axis=2)/(-2)).T
    # insert ones to the 1st col
    return np.insert(basis_func_outputs, 0, 1, axis=1)


# In[ ]:

def design_mat_gen(total_data, m_value):
    dic = k_means(m_value-1,total_data)
    centers = [dic[i]["center"] for i in dic.keys()]
    centers = np.array([list(i) for i in centers])
    spreads = [dic[i]["spread"].tolist() for i in dic.keys()]
    spreads = np.array(spreads)
    design_matrix = compute_design_matrix(np.array(total_data[np.newaxis, :, :]),np.array(centers[:, np.newaxis, :]),spreads)
    return design_matrix


# In[3]:

##Given in ppt
def closed_form_sol(L2_lambda, design_matrix, output_data):
    return np.array(np.linalg.solve(
    L2_lambda * np.identity(design_matrix.shape[1]) +
    np.matmul(design_matrix.T, design_matrix),
    np.matmul(design_matrix.T, output_data)
    ).flatten().tolist())


# In[4]:

##Given in ppt
def SGD_sol(learning_rate,minibatch_size,num_epochs,L2_lambda,design_matrix,output_data):
    N, _ = design_matrix.shape
    # You can try different mini-batch size size
    # Using minibatch_size = N is equivalent to standard gradient descent
    # Using minibatch_size = 1 is equivalent to stochastic gradient descent
    # In this case, minibatch_size = N is better
    weights = np.zeros([1, len(design_matrix[0])])
    # The more epochs the higher training accuracy. When set to 1000000,
    # weights will be very close to closed_form_weights. But this is unnecessary
    for epoch in range(num_epochs):
        for i in range(N / minibatch_size):
            lower_bound = i * minibatch_size
            upper_bound = min((i+1)*minibatch_size, N)
            Phi = design_matrix[lower_bound : upper_bound, :]
            t = output_data[lower_bound : upper_bound, :]
            E_D = np.matmul((np.matmul(Phi, weights.T)-t).T,Phi)
            E = (E_D + L2_lambda * weights) / minibatch_size
            weights = weights - learning_rate * E
            #print np.linalg.norm(E)
    return weights.flatten()


# In[5]:

def error(output_data,w,dm,lambda_val):
    error = 1/2.0*sum([(output_data[i]-sum(w.T*dm[i]))**2 for i in range(len(output_data))]).tolist()[0] + lambda_val*sum(1/2.0*w.T*w)
    return error

def rms_error(error,test_data_len):
    rms = np.sqrt(2.0*error/test_data_len)
    return rms


# In[ ]:




# In[7]:

#Synthetic or LeToR data
input_data = letor_input_data
output_data = letor_output_data

indices = shuffle(input_data)
mat1,mat2,mat3,tot_in = split_data(input_data,indices)
out1,out2,out3,tot_out = split_data(output_data,indices)


# In[8]:

print "CLOSED FORM"


# In[9]:

#Lambda Hyperparameter
lambda_value = 0.1

#Early stop parameter
p = 10
j = 0
m = 2

#Initial Minimum error found during training validation cycle
min_err_val = np.inf

#Early stopping
while (j + 2 < p):
    #compute design matrix for hyper parameter M = m
    design_matrix = design_mat_gen(tot_in,m)
    
    #split into training and validation
    design_mat_training = design_matrix[:int(len(design_matrix)*0.8)]
    design_mat_validation = design_matrix[int(len(design_matrix)*0.8):int(len(design_matrix)*0.9)]
    
    #compute min w by the closed form solution for that particular lambda and M
    w = closed_form_sol(lambda_value,design_mat_training,out1)
    
    #MSE plus the regularization error
    error_min = error(out2,w,design_mat_validation,lambda_value)
    
    #saving min error state
    if (error_min < min_err_val):
        j = 0
        min_err_val = error_min
        min_m = m
        min_dm = design_matrix
        min_w = w
    else:
        j +=1
    
    print "Latest Optimal M:", min_m, " Current M:",m
    m += 1


# In[10]:

mat2_dm = min_dm[int(len(design_matrix)*0.8):int(len(design_matrix)*0.9)]
mat3_dm = min_dm[int(len(design_matrix)*0.9):]

print "Minimum error found in Validation set is: ", min_err_val
print "Minimum RMS error found in Validation set is: ", rms_error(min_err_val,len(mat2_dm))

error_test = rms_error(error(out3,min_w,mat3_dm,lambda_value),len(mat3_dm))

print "RMS in test data is: ", error_test


# In[ ]:




# In[ ]:

print "STOCHASTIC GRADIENT DESCENT"


# In[ ]:

#Early stop parameter
p = 10
j = 0

#Hyperparameters
lambda_value = 0.1
learn_rate = 0.01
m = 2

#Initial Minimum error found during training validation cycle
min_err_val = np.inf

#Early stopping
while (j + 2 < p):
    #compute design matrix for hyper parameter M = m
    design_matrix = design_mat_gen(tot_in,m)
    N, D = mat1.shape
    #split into training and validation
    design_mat_training = design_matrix[:int(len(design_matrix)*0.8)]
    design_mat_validation = design_matrix[int(len(design_matrix)*0.8):int(len(design_matrix)*0.9)]
    
    #Gradient Descent solution for min w
    w_sgd = SGD_sol(learning_rate=learn_rate,minibatch_size=N,num_epochs=10000,L2_lambda=lambda_value,design_matrix=design_mat_training,output_data=out1)
    w_sgd = np.array(w_sgd.tolist())
    
    #MSE plus the regularization error
    error_min = error(out2,w_sgd,design_mat_validation,lambda_value)
    
    #saving min error state
    if (error_min < min_err_val):
        j = 0
        min_err_val = error_min
        min_m = m
        min_dm = design_matrix
        min_w = w_sgd
    else:
        j += 1
        
    print "Latest Optimal M:", min_m, " Current M:",m
    m += 1


# In[ ]:

mat2_dm = min_dm[int(len(design_matrix)*0.8):int(len(design_matrix)*0.9)]
mat3_dm = min_dm[int(len(design_matrix)*0.9):]

print "Minimum error found in Validation set is: ", min_err_val
print "Minimum RMS error found in Validation set is: ", rms_error(min_err_val,len(mat2_dm))

error_test = rms_error(error(out3,min_w,mat3_dm,lambda_value),len(mat3_dm))

print "RMS in test data is: ", error_test


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

"""

input_data = letor_input_data
output_data = letor_output_data
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


# Closed-form solution
print closed_form_sol(L2_lambda=0.1,design_matrix=design_matrix,output_data=output_data)
# Gradient descent solution
print SGD_sol(learning_rate=1,minibatch_size=N,num_epochs=10000,L2_lambda=0.1,design_matrix=design_matrix,output_data=output_data)

"""


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



