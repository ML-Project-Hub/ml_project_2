
# coding: utf-8

# In[73]:

#Python 2.7.13
import numpy as np
import random
from sklearn.cluster import KMeans
syn_input_data = np.genfromtxt('Info/input.csv', delimiter=',')
syn_output_data = np.genfromtxt('Info/output.csv', delimiter=',').reshape([-1, 1])
letor_input_data = np.genfromtxt('Info/Querylevelnorm_X.csv', delimiter=',')
letor_output_data = np.genfromtxt('Info/Querylevelnorm_t.csv', delimiter=',').reshape([-1, 1])


# In[74]:

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


# In[75]:

#For setting hyperparameters muj and sigmaj with respect to M - number of basis functions
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
        cluster_members = np.matrix(dic[i])
        dic[i] = {}
        dic[i]["val"] = cluster_members
        dic[i]["center"] = list(centers[i])
        dic[i]["spread"] = np.linalg.pinv(np.matrix(np.cov(cluster_members.T)))
    return dic


# In[76]:

##Given in ppt
def compute_design_matrix(X, centers, spreads):
    # use broadcast
    basis_func_outputs = np.exp(np.sum(np.matmul(X - centers, spreads) * (X - centers),axis=2)/(-2)).T
    # insert ones to the 1st col
    return np.insert(basis_func_outputs, 0, 1, axis=1)


# In[77]:

def design_mat_gen(total_data, m_value):
    dic = k_means(m_value-1,total_data)
    centers = [dic[i]["center"] for i in dic.keys()]
    centers = np.array([list(i) for i in centers])
    spreads = [dic[i]["spread"].tolist() for i in dic.keys()]
    spreads = np.array(spreads)
    design_matrix = compute_design_matrix(np.array(total_data[np.newaxis, :, :]),np.array(centers[:, np.newaxis, :]),spreads)
    return design_matrix


# In[78]:

##Given in ppt
def closed_form_sol(L2_lambda, design_matrix, output_data):
    return np.array(np.linalg.solve(
    L2_lambda * np.identity(design_matrix.shape[1]) +
    np.matmul(design_matrix.T, design_matrix),
    np.matmul(design_matrix.T, output_data)
    ).flatten().tolist())


# In[79]:

##Given in ppt
def SGD_sol(learning_rate,minibatch_size,num_epochs,L2_lambda,design_matrix,output_data):
    N, _ = design_matrix.shape
    # You can try different mini-batch size size
    # Using minibatch_size = N is equivalent to standard gradient descent
    # Using minibatch_size = 1 is equivalent to stochastic gradient descent
    # In this case, minibatch_size = N is better
    early_stop_error = 1
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
            if(epoch == 0):
                prev = np.linalg.norm(E)
                now = np.linalg.norm(E)
            else:
                prev = now
                now = np.linalg.norm(E)
                early_stop_error = prev - now
                
        #Early Stopping
        if (early_stop_error < 0.0001):
            break
                
            
    return weights.flatten()


# In[80]:

def error(output_data,w,dm,lambda_val):
    error = 1/2.0*sum([(output_data[i]-sum(w.T*dm[i]))**2 for i in range(len(output_data))]).tolist()[0] + lambda_val*sum(1/2.0*w.T*w)
    return error

def rms_error(error,test_data_len):
    rms = np.sqrt(2.0*error/test_data_len)
    return rms


# In[ ]:




# In[81]:

#Synthetic or LeToR data
##################################
###Change syn to letor here####
is_syn = True

if(is_syn):
    input_data = syn_input_data
    output_data = syn_output_data
else:
    input_data = letor_input_data
    output_data = letor_output_data
##################################


indices = shuffle(input_data)
mat1,mat2,mat3,tot_in = split_data(input_data,indices)
out1,out2,out3,tot_out = split_data(output_data,indices)


# In[ ]:




# In[ ]:

if(len(tot_in.T) == 10):
    print "CLOSED FORM FOR SYNTHETIC DATA"
elif (len(tot_in.T) == 46):
    print "CLOSED FORM FOR LETOR DATA"


# In[ ]:

#Lambda Hyperparameter
lambda_value = 0.1
m = 2 #incremented every turn


p = 6
j = 0


#Initial Minimum error (infinity) found during training validation cycle
min_err_val = np.inf

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
    m += 1
print "Optimal M:", min_m


# In[ ]:

mat2_dm = min_dm[int(len(design_matrix)*0.8):int(len(design_matrix)*0.9)]
mat3_dm = min_dm[int(len(design_matrix)*0.9):]

print "Minimum error found in Validation set: ", min_err_val
print "Minimum RMS error found in Validation set: ", rms_error(min_err_val,len(mat2_dm))

error_test = rms_error(error(out3,min_w,mat3_dm,lambda_value),len(mat3_dm))

print "RMS in test data is: ", error_test


# In[ ]:

print "Min w:"
print min_w


# In[ ]:

print "Design Matrix corresponding to min W:"
print min_dm


# In[ ]:

predicted_y = np.array([i[0] for i in np.matmul(np.matrix(mat3_dm.tolist()),np.matrix(min_w.tolist()).T).tolist()])
print "Predicted Y:"
print predicted_y
print "Original Y:"
print out3.T


# In[ ]:

scaled_y = [round(i,0) for i in predicted_y]
count = 0
for i in range(len(scaled_y)):
    if out3[i] == scaled_y[i]:
        count+=1
        
print "Number of records matching in test data after rounding to the nearest decimal:",count,
print "out of", len(out3)


# In[ ]:




# In[ ]:

if(len(tot_in.T) == 10):
    print "SGD FOR SYNTHETIC DATA"
elif (len(tot_in.T) == 46):
    print "SGD FOR LETOR DATA"


# In[ ]:

#Hyperparameters
lambda_value = 0.1
learn_rate = 0.01
m = 2 #incremented every turn


p = 6
j = 0


#Initial Minimum error (infinity) found during training validation cycle
min_err_val = np.inf

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
    m += 1
print "Optimal M:", min_m


# In[ ]:

mat2_dm = min_dm[int(len(design_matrix)*0.8):int(len(design_matrix)*0.9)]
mat3_dm = min_dm[int(len(design_matrix)*0.9):]

print "Minimum error found in Validation set is: ", min_err_val
print "Minimum RMS error found in Validation set is: ", rms_error(min_err_val,len(mat2_dm))

error_test = rms_error(error(out3,min_w,mat3_dm,lambda_value),len(mat3_dm))

print "RMS in test data is: ", error_test


# In[ ]:

print "Min w:"
print min_w


# In[ ]:

predicted_y = np.array([i[0] for i in np.matmul(np.matrix(mat3_dm.tolist()),np.matrix(min_w.tolist()).T).tolist()])
print "Predicted Y:"
print predicted_y

print "Original Y:"
print out3.T


# In[ ]:

scaled_y = [round(i,0) for i in predicted_y]
count = 0
for i in range(len(scaled_y)):
    if out3[i] == scaled_y[i]:
        count+=1
        
print "Number of records matching in test data after rounding to the nearest decimal:",count,
print "out of", len(out3)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



