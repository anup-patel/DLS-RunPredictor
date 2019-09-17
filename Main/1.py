#!/usr/bin/env python
# coding: utf-8

#### Author : Anup Patel (M.tech CSA) 

#### Library Import 

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.pyplot as plt


#### Read File 

data=pd.read_csv("./../data/04_cricket_1999to2011.csv")
data.head()


#### Computing No. of Columns and Data Instances in Dataset:


print("No. of Columns: " +str(len(data.columns)))
print("No. of Data Instances: " +str(len(data)))


#### Feature Extraction 

innings = data['Innings'].values
overs_completed = data['Over'].values
total_overs = data['Total.Overs'].values
overs_remaining = total_overs - overs_completed
innings_total_score = data['Innings.Total.Runs'].values
current_score = data['Total.Runs'].values
runs_remaining = innings_total_score - current_score
wickets_remaining = data['Wickets.in.Hand'].values
print("Features Extracted \n")


# ### Squared Error Loss Function 

def squared_error_loss(params, args):
    squared_loss = []
    L = params[10]
    innings=args[0]
    overs_remaining = args[1]
    runs_remaining = args[2]
    wickets_remaining = args[3]
    loss=0   
    #count=0
    
    for i in range(len(wickets_remaining)):
        if innings[i] == 1:
            if (runs_remaining[i] > 0 and wickets_remaining[i]>0):
                predicted_run = params[wickets_remaining[i]-1] * (1.0 - np.exp((-1*L*overs_remaining[i])/(params[wickets_remaining[i]-1])))
                tmp=(predicted_run - runs_remaining[i])**2
                loss=loss+tmp
                #count+=1
    #print(count) 
    return loss


#### Optimization Function

def optimizer(method_name,innings,overs_remaining,runs_remaining,wickets_remaining):    
    #BFGS Method

    initial_parameters = [10.0, 20.0, 35.0, 50.0, 70.0, 100.0, 140.0, 180.0, 235.0,280.0, 19] #Random Values

    #Minimize Loss function and find optimized Parameters
    parameters = minimize(squared_error_loss, initial_parameters,
                      args=[innings,
                            overs_remaining,
                            runs_remaining,
                            wickets_remaining
                            ],
                      method=method_name)
    optimized_params1, squared_error_loss1 = parameters['x'], parameters['fun']
    #print(optimized_params1)
    #print(squared_error_loss1)
    return parameters['x'], parameters['fun']


#### Optimization

print("Optimizing Parameters .... ")
optimized_params, squared_error_loss= optimizer('BFGS',innings,overs_remaining,
                                                runs_remaining,wickets_remaining)
print("Optimized Parameters Computed")
#optimizer('Powell')
#optimizer('L-BFGS-B')
#optimizer('TNC')
#optimizer('COBYLA')
#Got least Loss by using BFGS Method


#### Loss and Parameters 

print ('Total Squared Error Loss = ' + str(squared_error_loss))
print ('Parameter L :: ' + str(optimized_params[10]))
for i in range(10):
    print ('Parameter Z' + str(i+1) + ' = ' + str(optimized_params[i]))


#### Compute RMSE

print("Root Mean Squared Error :: " + str(np.sqrt(squared_error_loss/len(overs_remaining))))


#### Prediction Function 

def prediction_func(z, l, u):
    return z * (1 - np.exp(-l*u/z))
"""
z=parameter
u=overs remaining
"""


#### Visualisation 

# Plot the resource vs overs used graphs for 10 parameters
plt.figure(figsize=(10,7)) #Fig Size
plt.xlabel('Overs remaining (u)')
plt.ylabel('Percentage of resource remaining')
plt.xlim((0, 50))
plt.ylim((0, 1))
plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
max_resource = prediction_func(optimized_params[9], optimized_params[10], 50)
overs = np.arange(0, 51, 1)
line=[] #Center Line

#For Center Line
for i in range(len(overs)):
    line.append(2*i)
plt.plot(overs, line, color='blue')

#Plot Resources Remaining vs overs Remaining
for i in range(10):
    fraction= prediction_func(optimized_params[i], optimized_params[10], overs)/max_resource
    y=100*fraction
    plt.plot(overs, y, label='Z['+str(i+1)+']')
    plt.legend()
#plt.show()
print("Plot Generated Successfully and saved in folder")
plt.savefig('plot.png')



