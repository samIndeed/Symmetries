# =================================================================================
# =================================================================================
# Script:"clean_up_output"
# Date: 2022-03-08
# Implemented by: Johannes Borgqvist
# Description:
# The script reduces the density of the output vectors and it removes outliers
# =================================================================================
# =================================================================================
# Import Libraries
# =================================================================================
# =================================================================================
import pandas as pd
from hampel import hampel
import numpy as np
# =================================================================================
# =================================================================================
# Functions
# =================================================================================
# =================================================================================
# ---------------------------------------------------------------------------------------
# Function 1: "reduce_density"
# The function takes three inputs:
# 1. x_dense: the dense vector with explanatory variables,
# 2. y_dense: the dense vector with response variables,
# 3. x_sparse: the sparse vector with explanatory variables.
# The function returns one output:
# 1. y_sparse: the sparse vector with response variables.
def reduce_density(x_dense,y_dense,x_sparse):
    # Allocate memory for the sparse vector with explanatory variables we will return
    y_sparse = []
    # Loop over the x-values in the sparse vector
    for x in list(x_sparse):
        # Find the closest index in our dense vector
        index = np.where(x_dense==x_dense[np.abs(x_dense-x).argmin()])[0][0]
        # Find the response variable through interpolation
        if x == x_dense[index]:
            y_sparse.append(y_dense[index])
        else: # We interpolate between the two adjacent values         
            if x < x_dense[index] and x > x_dense[index-1]:
                x_range = [x_dense[index-1], x_dense[index]]
                y_range = [y_dense[index-1], y_dense[index]]
            elif x > x_dense[index] and x < x_dense[index+1]:
                x_range = [x_dense[index], x_dense[index+1]]
                y_range = [y_dense[index], y_dense[index+1]]
            y_sparse.append(np.interp(x, x_range, y_range))
    # Lastly, we return the sparse output as an array
    return np.array(y_sparse)
# ---------------------------------------------------------------------------------------
# Function 2: "remove_outliers"
# The function takes two inputs
# 1. epsilon: numpy array with transformation parameters,
# 2. RMS: numpy array with RMS values,
# 3. fitted_parameters: list with the fitted parameters for each transformation parameter,
# 4. inverse_parameters: list with the inverse parameters for each transformation parameter,
# 5. transformed_data: list with the transformed time series for each transformation parameter,
# The function then returns the same number of outputs where the indices corresponding to
# outliers in the RMS array have been removed. 
def remove_outliers(epsilon,RMS,fitted_parameters,inverse_parameters,transformed_data):
    # Re-cast our array y as a pandas times series
    ts = pd.Series(list(RMS))
    # Calculate the outlier indices using the hampel function
    outlier_indices = hampel(ts, window_size=10, n=3)
    # Delete the outliers from the original time series
    new_epsilon = np.delete(epsilon, outlier_indices)
    new_RMS = np.delete(RMS, outlier_indices)
    # We want to remove these indices from the parameters and the transformed data as well
    # Allocate memory for these ouputs
    new_fitted_parameters = fitted_parameters
    new_inverse_parameters = inverse_parameters
    new_transformed_data = transformed_data
    # Sort the outlier indices in reverse order
    outlier_indices.sort(reverse=True)
    # Loop over the indices in reverse order and remove these elements
    for outlier_index in outlier_indices:
        del new_fitted_parameters[outlier_index]
        del new_inverse_parameters[outlier_index]
        del new_transformed_data[outlier_index]
    # Now, it turns out that some of the indices can be switched when we remove elements.
    # So, we add a sorting step to the epsilon vector as well.
    sorting_matrix = np.array([list(np.arange(len(new_epsilon))), list(new_epsilon)]).T
    # Sort this matrix based on the second column
    sorting_matrix = sorting_matrix[sorting_matrix[:, 1].argsort()]
    # Extract the first column with the indices from this matrix
    #sorted_indices = list(sorting_matrix[:,0])
    sorted_indices = [int(sorting_matrix[i,0]) for i in range(len(new_epsilon))]
    # Now, we can sort the output before we return it
    new_sorted_epsilon = np.array([new_epsilon[i] for i in sorted_indices])
    new_sorted_RMS = np.array([new_RMS[i] for i in sorted_indices])
    new_sorted_fitted_parameters = [new_fitted_parameters[i] for i in sorted_indices]
    new_sorted_inverse_parameters = [new_inverse_parameters[i] for i in sorted_indices]
    new_sorted_transformed_data = [new_transformed_data[i] for i in sorted_indices]
    # Return the output
    return new_sorted_epsilon, new_sorted_RMS, new_sorted_fitted_parameters, new_sorted_inverse_parameters, new_sorted_transformed_data
