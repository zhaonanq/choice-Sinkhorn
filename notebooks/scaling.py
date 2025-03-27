#This document implements a fast iterative algorithm
#that computes the maximum likelihood estimate of 
#the Plackett-Luce model, given comparison matrix and frequency of being chosen.

from __future__ import division

import numpy as np
import pandas as pd
import scipy.optimize

# convert ranking data into observation matrix and winning frequency
def ranking_to_choice(nb_items,rankings):
    num_observations = 0
    for ranking in rankings:
        num_observations += len(ranking)-1
    observation_matrix = np.zeros((num_observations,nb_items))
    winning_count = np.zeros(nb_items)
    
    observation = 0
    for ranking in rankings:
        index = list(ranking)
        while len(index) > 1:
            observation_matrix[observation,index] = 1
            winning_count[index[0]] += 1
            del index[0]
            observation += 1
    return observation_matrix, winning_count/num_observations

# implementation of matrix scaling algorithm to perform MLE of Luce-Plackett model.
def iterative_scaling(A,p,max_iter=int(1e10),eps = 1e-8,ground_truth=None):
    (m,n) = A.shape
    q = np.ones(m)/m
    d_0 = np.ones(n)
    j = 0
    errors = []
    #for j in range(max_iter):
    while True:
        j += 1
        d_0_old = d_0
        d_1 = 1/np.dot(A,d_0)
        d_0 = p/np.dot(A.T,d_1*q)
        d_0 = n/np.sum(d_0)*d_0
        if (ground_truth is not None):
            errors.append(np.linalg.norm(d_0-ground_truth))
        #theta = np.log(d_0)-np.log(d_0).sum()/len(d_0)
        #error = np.linalg.norm(theta_star-theta)/np.sqrt(iteration) 
        if max(abs(np.log(d_0_old) - np.log(d_0))) < eps:
        #if max(abs(d_0_old - d_0)) < eps:
            A_new = ((A*d_0.T).T*d_1).T
            return d_0, A_new, j, errors
    raise RuntimeError("Did not converge after {} iterations".format(max_iter))

# Linear RUM scaling using features weighted with winning counts+constrained least squares
def iterative_scaling_RUM(A,p,X,max_iter=int(1e10),eps = 1e-8,ground_truth=None):
    (m,n) = A.shape
    q = np.ones(m)/m
    d_0 = np.ones(n)
    j = 0
    errors = []
    
    projection = X.dot(np.linalg.inv(X.T.dot(X))).dot(X.T)
    
    #for j in range(max_iter):
    while True:
        j += 1
        d_0_old = d_0
        d_1 = 1/np.dot(A,d_0)
        d_t = scipy.optimize.lsq_linear(X.T,p,bounds=(1e-2,np.inf))['x']
        d_0 = d_t/np.dot(A.T,d_1*q)
        projection = X.dot(np.linalg.inv(X.T.dot(X))).dot(X.T)
        d_0 = np.exp(projection.dot(np.log(d_0)))
        d_0 = n/np.sum(d_0)*d_0
        
        if (ground_truth is not None):
            errors.append(np.linalg.norm(d_0-ground_truth))
        #theta = np.log(d_0)-np.log(d_0).sum()/len(d_0)
        #error = np.linalg.norm(theta_star-theta)/np.sqrt(iteration) 
        if max(abs(np.log(d_0_old) - np.log(d_0))) < eps:
        #if max(abs(d_0_old - d_0)) < eps:
            A_new = ((A*d_0.T).T*d_1).T
            return d_0, A_new, j, errors
    raise RuntimeError("Did not converge after {} iterations".format(max_iter))

# linear RUM scaling with one projection onto range space of X
def iterative_scaling_RUM_projection(A,p,X,max_iter=int(1e10),eps = 1e-8,ground_truth=None):
    (m,n) = A.shape
    q = np.ones(m)/m
    d_0 = np.ones(n)
    j = 0
    errors = []
    
    projection = X.dot(np.linalg.inv(X.T.dot(X))).dot(X.T)
    
    while True:
        j += 1
        d_0_old = d_0
        d_1 = 1/np.dot(A,d_0)
        d_t = p/np.dot(A.T,d_1*q)
       
        
        # project log(d_t) onto range space of X
        
        d_0 = np.exp(projection.dot(np.log(d_t)))
        d_0 = n/np.sum(d_0)*d_0
        
        if (ground_truth is not None):
            errors.append(np.linalg.norm(d_0-ground_truth))
        #theta = np.log(d_0)-np.log(d_0).sum()/len(d_0)
        #error = np.linalg.norm(theta_star-theta)/np.sqrt(iteration) 
        if max(abs(np.log(d_0_old) - np.log(d_0))) < eps:
        #if max(abs(d_0_old - d_0)) < eps:
            A_new = ((A*d_0.T).T*d_1).T
            return d_0, A_new, j, errors
    raise RuntimeError("Did not converge after {} iterations".format(max_iter))
    
# linear RUM with alternating projection 
def iterative_scaling_RUM_alternating(A,p,X,max_iter=int(1e10),eps = 1e-9,ground_truth=None):
    (m,n) = A.shape
    q = np.ones(m)/m
    d_0 = np.ones(n)
    j = 0
    errors = []
    
    # define the necessary projection operators
    projection = X.dot(np.linalg.inv(X.T.dot(X))).dot(X.T)
    F = scipy.linalg.null_space(X.T)
    P_F = F.dot(np.linalg.inv(F.T.dot(F))).dot(F.T)
   
    while True:
        j += 1
        d_0_old = d_0
        d_1 = 1/np.dot(A,d_0)
        d_t = p/np.dot(A.T,d_1*q)
        
        # alternating projection
        error = 1
        d_0 = d_t
        
        for i in range(20):
        #while error > 1e-1:
            d_old = d_0
            d_0 = np.exp(projection.dot(np.log(d_0)))

            #print(np.min(P_F.dot(d_0-d_t)+d_t))
            #residual = P_F.dot(d_0-d_t)
            #d_0 = np.maximum(d_t + residual, 1e-4)

            max_deviation = max(abs(P_F.dot(d_0-d_t)-(d_0-d_t)))
            violation = np.min(P_F.dot(d_0-d_t)+d_t)

            if max_deviation < 1e-1: break

            residual = np.maximum(P_F.dot(d_0-d_t), 1e-4)
            d_0 = d_t + residual
            error = max(abs(np.log(d_old) - np.log(d_0)))
            

        print('max deviation', max_deviation)
        print('violation', violation)
        
        d_0 = n/np.sum(d_0)*d_0
        if (ground_truth is not None):
            errors.append(np.linalg.norm(d_0-ground_truth))
        #theta = np.log(d_0)-np.log(d_0).sum()/len(d_0)
        #error = np.linalg.norm(theta_star-theta)/np.sqrt(iteration) 
        if max(abs(np.log(d_0_old) - np.log(d_0))) < eps:
        #if max(abs(d_0_old - d_0)) < eps:
            A_new = ((A*d_0.T).T*d_1).T
            return d_0, A_new, j, errors
    raise RuntimeError("Did not converge after {} iterations".format(max_iter))
    
# linear RUM with Newton iteration to solve the equation
def iterative_scaling_RUM_Newton(A,p,X,max_iter=int(1e10),eps = 1e-8,ground_truth=None):
    (m,n) = A.shape
    q = np.ones(m)/m
    d_0 = np.ones(n)
    j = 0
    errors = []
    #for j in range(max_iter):
    while True:
        j += 1
        d_0_old = d_0
        d_1 = 1/np.dot(A,d_0)
        d_t = p/np.dot(A.T,d_1*q)
                
        d_0 = Newton(d_t,X)
        
        d_0 = n/np.sum(d_0)*d_0
        if (ground_truth is not None):
            errors.append(np.linalg.norm(d_0-ground_truth))
        #theta = np.log(d_0)-np.log(d_0).sum()/len(d_0)
        #error = np.linalg.norm(theta_star-theta)/np.sqrt(iteration) 
        if max(abs(np.log(d_0_old) - np.log(d_0))) < eps:
        #if max(abs(d_0_old - d_0)) < eps:
            A_new = ((A*d_0.T).T*d_1).T
            return d_0, A_new, j, errors
    raise RuntimeError("Did not converge after {} iterations".format(max_iter))
    
    
def Newton(d_t,X):
    
    projection = X.dot(np.linalg.inv(X.T.dot(X))).dot(X.T)
    
    F = scipy.linalg.null_space(X.T)
    
    P_F = F.dot(np.linalg.inv(F.T.dot(F))).dot(F.T)
    
    def function(gamma,projection=projection,P_F=P_F,d_t=d_t):
        
        print(min(d_t+P_F.dot(gamma)))
        
        #d = np.maximum(d_t+P_F.dot(gamma),0.1)
        d = d_t+P_F.dot(gamma)
      
        return projection.dot(np.log(d))-np.log(d)
    
    
    #gamma = scipy.optimize.newton(function,np.zeros((P_F.shape[1],)),maxiter=100)
    
    #d_0 = d_t+P_F.dot(gamma)
        
    d_0 = np.exp(projection.dot(np.log(d_t)))
    
    return d_0
