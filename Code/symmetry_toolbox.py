# =================================================================================
# =================================================================================
# Script:"symmetry_based_model_selection"
# Date: 2022-02-15
# Implemented by: Johannes Borgqvist
# Description:
# The script involves all the symmetry functionalities used for transforming the
# functions as well as conducting the symmetry based model selection. 
# =================================================================================
# =================================================================================
# Import Libraries
# =================================================================================
# =================================================================================
import read_data  # Home-made
import write_output  # Home-made
import fit_to_data  # Home-made
import numpy as np 
from matplotlib import pyplot as plt
from scipy.optimize import fmin_cobyla # To find the orthogonal distance
# =================================================================================
# =================================================================================
# Functions
# =================================================================================
# =================================================================================
#----------------------------------------------------------------------------------
# FUNCTION 1: PLM_symmetry
# The second symmetry of the power law model (PLM)
def PLM_symmetry(t,R,epsilon):
    t_hat = t * np.exp(epsilon)
    R_hat = R
    return t_hat,R_hat
#----------------------------------------------------------------------------------
# FUNCTION 2: PLM_transformation
# The function plots the action of
# the symmetries of the PLM from
# a given starting point and an end point
def PLM_transformation(t_from,R_from,epsilon,gamma):
    # In order to plot the action of the transformation
    # continuously we will add small plots along the
    # way.
    epsilon_increment = epsilon/10 # Increment rotation angle
    epsilon_temp = epsilon_increment # The current angle we plot with
    # Two lists as output corresponding to the transformation
    t_trans_list = [] # Transformed variable
    R_trans_list = [] # Transformed state
    # Append the starting point
    t_trans_list.append(t_from)
    R_trans_list.append(R_from)
    # Calculate the transformed point of interest
    t_to, R_to = PLM_symmetry(t_from,R_from,epsilon_increment)        
    # Save the new rotated point
    t_trans_list.append(t_to)
    R_trans_list.append(R_to)
    # Re-do this procedure until we have rotated the original point
    # far enough
    while abs(epsilon_temp)<abs(epsilon):
        # Update point we want to rotate
        t_from = t_to
        R_from = R_to
        # Calculate the transformed point of interest
        t_to, R_to = PLM_symmetry(t_from,R_from,epsilon_increment)                    
        # Save the rotated point
        t_trans_list.append(t_to)
        R_trans_list.append(R_to)
        # Change the angle we rotate with
        epsilon_temp += epsilon_increment
    # Convert the lists to arrays
    t_trans = np.asarray(t_trans_list)
    R_trans = np.asarray(R_trans_list)
    # Return the action of the transformation
    return t_trans,R_trans
#----------------------------------------------------------------------------------
# FUNCTION 3: PLM_transformed_solution
# This function returns the transformed solution of the PLM for a given value
# of the parameters A and gamma as well as the transformation parameter epsilon
def PLM_transformed_solution(t,R,epsilon,A,gamma):
    # Allocate memory for the transformed solution
    t_trans = np.asarray([PLM_symmetry(t[index],R[index],epsilon)[0] for index in range(len(t))])
    R_trans = R
    # Return the transformed solution
    return t_trans,R_trans
#----------------------------------------------------------------------------------
# FUNCTION 4: PLM_objective
# This function returns the objective value of the PLM which is used to calculate
# the orthogonal distance between the data point (t_data,R_data) and the solution
# the solution curve (t,R(t)) of the power law model.
def PLM_objective(t,args):
    # Extract the parameters
    A = args[0]
    gamma = args[1]
    # Return the objective value of the PLM
    return A*(t**gamma)
    #return A + gamma*np.log(t)
#----------------------------------------------------------------------------------
# FUNCTION 5: PLM_constraint
# This function is a help function that is necessary for finding the orthogonal distance
# between a data point and a point on a solution curve of the PLM.
def PLM_constraint(Model_Point,*args):
    # Extract the data point
    t,R = Model_Point
    # Pass the constraint to the optimisation
    return PLM_objective(t,args) - R
    #return PLM_objective(t,args) - np.log(R)
#----------------------------------------------------------------------------------
# FUNCTION 6: PLM_transformation_scale
# This function calculates the transformation scale for the PLM model. It calculates
# the value of epsilon that is needed to increase the age from t years to n*t years
# where n>1 is a scaling factor
def PLM_transformation_scale(n):
    return np.log(n)
#----------------------------------------------------------------------------------
# FUNCTION 7: IM_III_symmetry
# The symmetry of the immunological model IM-III
def IM_III_symmetry(t,R,epsilon,tau,alpha):
    # Define t_hat recursively
    t_hat =  np.log(np.log(np.exp(np.exp(-alpha*(t-tau))) - (alpha*np.exp(alpha*tau)*epsilon) ))
    #t_hat =  np.log(np.log(abs(np.exp(np.exp(-alpha*(t-tau))) - (alpha*np.exp(alpha*tau)*epsilon) )))
    t_hat = tau - ((t_hat)/(alpha))
    R_hat = R
    return t_hat,R_hat
#----------------------------------------------------------------------------------
# FUNCTION 8: IM_III_transformation
# The function plots the action of
# the symmetries of the IM-III from
# a given starting point and an end point
def IM_III_transformation(t_from,R_from,epsilon,tau,alpha):
    # In order to plot the action of the transformation
    # continuously we will add small plots along the
    # way.
    epsilon_increment = epsilon/10 # Increment rotation angle
    epsilon_temp = epsilon_increment # The current angle we plot with
    # Two lists as output corresponding to the transformation
    t_trans_list = [] # Transformed variable
    R_trans_list = [] # Transformed state
    # Append the starting point
    t_trans_list.append(t_from)
    R_trans_list.append(R_from)
    # Calculate the transformed point of interest
    t_to, R_to = IM_III_symmetry(t_from,R_from,epsilon_increment,tau,alpha)
    # Save the new rotated point
    t_trans_list.append(t_to)
    R_trans_list.append(R_to)
    # Re-do this procedure until we have rotated the original point
    # far enough
    while abs(epsilon_temp)<abs(epsilon):
        # Update point we want to rotate
        t_from = t_to
        R_from = R_to
        # Calculate the next transformed point
        t_to, R_to = IM_III_symmetry(t_from,R_from,epsilon_increment,tau,alpha)
        # Save the rotated point
        t_trans_list.append(t_to)
        R_trans_list.append(R_to)
        # Change the angle we rotate with
        epsilon_temp += epsilon_increment
    # Convert the lists to arrays
    t_trans = np.asarray(t_trans_list)
    R_trans = np.asarray(R_trans_list)
    # Return the action of the transformation
    return t_trans,R_trans
#----------------------------------------------------------------------------------
# FUNCTION 9: IM_III_transformed_solution
# This function returns the transformed solution of the IM-III for a given value
# of the parameters A, tau and C as well as the transformation parameter epsilon
def IM_III_transformed_solution(t,R,epsilon,tau,alpha):
    # Allocate memory for the transformed solution
    t_trans = np.asarray([IM_III_symmetry(t[index],R[index],epsilon,tau,alpha)[0] for index in range(len(t))])
    R_trans = R
    # Return the transformed solution
    return t_trans,R_trans
#----------------------------------------------------------------------------------
# FUNCTION 10: IM_III_objective
# This function returns the objective value of the IM_III which is used to calculate
# the orthogonal distance between the data point (t_data,R_data) and the solution
# the solution curve (t,R(t)) of the immunological model.
def IM_III_objective(t,args):
    # Extract the parameters
    A = args[0]
    tau = args[1]
    C = args[2]
    alpha = args[3]
    # Define the value of the fixed parameter alpha
    #alpha = 0.044
    # Return the objective value of the IM-III
    return ((A)/(np.exp(np.exp(-alpha*(t-tau)))-C))
    #return np.log(A)-np.log(np.exp(np.exp(-alpha*(t-tau)))-C)
#----------------------------------------------------------------------------------
# FUNCTION 11: IM_III_constraint
# This function is a help function that is necessary for finding the orthogonal distance
# between a data point and a point on a solution curve of the IM-III.
def IM_III_constraint(Model_Point,*args):
    # Extract the data point
    t,R = Model_Point
    # Pass the constraint to the optimisation
    return IM_III_objective(t,args) - R
    #return IM_III_objective(t,args) - np.log(R)
#----------------------------------------------------------------------------------
# FUNCTION 12: IM_III_transformation_scale
# This function calculates the transformation scale for the IM-III model. It calculates
# the value of epsilon that is needed to increase the age from t years to n*t years
# where n>1 is a scaling factor.
def IM_III_transformation_scale(t,n,alpha,tau):
    # Calculate the nominator
    nom = np.exp(np.exp(-alpha*(t-tau))) - np.exp(np.exp(-alpha*((n*t)-tau)))
    # Calculate the denominator
    denom = alpha * np.exp(alpha*tau)
    # Return the transformation scale
    return nom/denom
#----------------------------------------------------------------------------------
# FUNCTION 13: SS_res_model_data
# This function calculates the Sum of Squares (SS) of the residuals between the
# data and the model
def SS_res_model_data(Model_Point,Data_Point):
    # Extract the model points
    t_model,R_model = Model_Point
    # Extract the model points
    t_data,R_data = Data_Point
    # Return the objective value
    return ((t_model - t_data)**2 + (R_model - R_data)**2)

#----------------------------------------------------------------------------------
# FUNCTION 14: symmetry_based_model_selection
# This function performs the symmetry based model selection. It does so by
# taking the following input:
# 1. DATA: t_data, the ages of the patients,
# 2. DATA: R_data, the number of incidences of cancer,
# 3. epsilon_vector, the transformation parameters we transform the data with,
# 4. model_fitting_structure, a structure with parameters, error bounds etc. that is returned from the ODR function,
# 5. model_str, a string indicating which model we consider (i.e. the PLM or the IM-III).
# The function has one output and it is the RMS profile as a function of the epsilon vector.
def symmetry_based_model_selection(t_data,R_data,epsilon_vector,model_fitting_structure,model_str):
    # SETTING UP THE MODEL SELECTION
    # Allocate memory for the output which is the RMS(epsilon)
    # indicating how the fit to transformed data changes as a
    # function of the transformation parameter epsilon
    RMS_transf = []
    # We allocate a vector with transformation parameters in case we abort to early
    epsilon_transf =  []
    # We allocate a list for all the transformed data
    transformed_data = []
    # We allocate a list with all fitted parameters
    fitted_parameters = []
    # We allocate a list with all inverse parameters
    inverse_parameters = []
    # We also define the previous parameters as well as their error bounds
    prev_para = model_fitting_structure.beta
    prev_para_bounds = model_fitting_structure.sd_beta
    # We also save the original parameters
    original_para = model_fitting_structure.beta
    original_bounds = model_fitting_structure.sd_beta
    # We have a logical variable saying whether the fitting was successful or not
    fitting_successful = True
    # Loop over the epsilon vectors and for each epsilon we do the following four steps:
    # 1. Transform the data,
    # 2. Fit the model to the transform data,
    # 3. Inversely transform this model back,
    # 4. Calculate the fit of the inversely transformed data to the original data.
    for epsilon in list(epsilon_vector):
        # Allocate memory for the sum of squares (SS)
        SS = 0
        # For the first data point, we just calculate the fit of the model
        # directly to the data
        if epsilon == 0:
            # Loop over the transformed time series
            for time_series_index in range(len(t_data)):
                # Extract a data point
                Data_point = (t_data[time_series_index],R_data[time_series_index])
                # The fit is calculated by finding the point on the solution curve
                # that is orthogonal to the datapoint, which is done using fmin_cobyla
                # in scipy.
                if model_str == "PLM":
                    # Find the orthogonal point on the solution curve (t,R(t)) of the PLM
                    Model_point = fmin_cobyla(SS_res_model_data, x0=list(Data_point), cons=[PLM_constraint], args=(Data_point,),consargs=(model_fitting_structure.beta[0],model_fitting_structure.beta[1]))          
                elif model_str == "IM-III":    
                    # Find the orthogonal point on the solution curve (t,R(t)) of the IM-III
                    Model_point = fmin_cobyla(SS_res_model_data, x0=list(Data_point), cons=[IM_III_constraint], args=(Data_point,),consargs=(model_fitting_structure.beta[0],model_fitting_structure.beta[1],model_fitting_structure.beta[2],model_fitting_structure.beta[3]))    
                # Add the squared distances to our growing sum of squares (SS)
                SS += SS_res_model_data(Model_point,Data_point)
            # Append the root mean squared calculated based on the SS-value
            RMS_transf.append(np.sqrt(SS/len(t_data)))
            # Also, we save the current epsilon value
            epsilon_transf.append(epsilon)
            # Append the original data
            transformed_data.append([t_data, R_data])
            # Append the fitted parameters and the inverse parameters
            if model_str == "PLM":
                # Fitted parameters 
                fitted_parameters.append((model_fitting_structure.beta[0],model_fitting_structure.beta[1]))
                # Append the parameters of the inversely transformed curve
                inverse_parameters.append((model_fitting_structure.beta[0],model_fitting_structure.beta[1]))                            
            elif model_str == "IM-III":
                # Fitted parameters 
                fitted_parameters.append((model_fitting_structure.beta[0],model_fitting_structure.beta[1],model_fitting_structure.beta[2],model_fitting_structure.beta[3]))
                # Inverse parameters
                inverse_parameters.append((model_fitting_structure.beta[0],model_fitting_structure.beta[1],model_fitting_structure.beta[2],model_fitting_structure.beta[3]))
        else: # Now, we actually transform the data with a transformation parameter epsilon>0
            # CONDUCT THE SYMMETRY BASED METHODOLOGY FOR MODEL SELECTION
            #------------------------------------------------------
            # STEP 1 OUT OF 4: TRANSFORM THE DATA
            if model_str == "PLM":
                t_trans = np.array([PLM_symmetry(t_data[index],R_data[index],epsilon)[0] for index in range(len(t_data))])
                R_trans = np.array([PLM_symmetry(t_data[index],R_data[index],epsilon)[1] for index in range(len(t_data))])
            elif model_str == "IM-III":
                t_trans = np.array([IM_III_symmetry(t_data[index],R_data[index],epsilon,model_fitting_structure.beta[1],model_fitting_structure.beta[3])[0] for index in range(len(t_data))])
                R_trans = np.array([IM_III_symmetry(t_data[index],R_data[index],epsilon,model_fitting_structure.beta[1],model_fitting_structure.beta[3])[1] for index in range(len(t_data))])
            #------------------------------------------------------
            # STEP 2 OUT OF 4: FIT THE CANDIDATE MODEL TO THE TRANSFORMED DATA
            # We start by fitting the models to the data without any special start guesses
            if model_str == "PLM":
                print("PLM original_para")
                print([original_para[0]*np.exp(-original_para[1]*epsilon), original_para[1]])
                # Fit the PLM to the data without any start guesses
                fitting_structure, R_hat, RMS, fitting_successful  = fit_to_data.PE_risk_profiles(t_trans,R_trans,"PLM","ODR",[1,0],[original_para[0]*np.exp(-original_para[1]*epsilon), original_para[1]])
            elif model_str == "IM-III":
                print("IM-III original_para")
                print([original_para[0],original_para[1],original_para[2],original_para[3]])
                C_vec = np.linspace(-1,1,3)
                start_guesses = [[original_para[0],original_para[1],C,original_para[3]] for C in C_vec]
                # Fit the IM-III to the data but without any start guesses
                fitting_structure, R_hat, RMS, fitting_successful  = fit_to_data.PE_risk_profiles(t_trans,R_trans,"IM-III","ODR",[0,0,1,0],start_guesses)                          
            # If fitting was successful, we do the remaining two steps of the algorithm. Otherwise, we break and return our results
            if fitting_successful:                
                #------------------------------------------------------
                # STEP 3 OUT OF 4: INVERSELY TRANSFORM THE MODEL BACK 
                if model_str == "PLM":
                    # Define the parameters of the inversely transformed curve of the PLM (parameters=[A, gamma])
                    parameters_inverse = [fitting_structure.beta[0]*np.exp(fitting_structure.beta[1]*epsilon),fitting_structure.beta[1]]
                elif model_str == "IM-III":
                    # Define the parameters of the inversely transformed curve of the IM-III (parameters=[A, tau, C, alpha])
                    parameters_inverse = [fitting_structure.beta[0],fitting_structure.beta[1],fitting_structure.beta[2] + fitting_structure.beta[3]*np.exp(fitting_structure.beta[3]*fitting_structure.beta[1])*epsilon,fitting_structure.beta[3]]                
                #------------------------------------------------------
                # STEP 4 OUT OF 4: CALCULATE THE FIT OF THE INVERSELY TRANSFORMED CURVE TO THE ORIGINAL DATA
                # Loop over the transformed time series
                for time_series_index in range(len(t_data)):
                    # Extract a data point
                    Data_point = (t_data[time_series_index],R_data[time_series_index])
                    # Update the curve specific parameters that are transformed corresponding to
                    # the parameter A in the case of the PLM and the parameter C in the case of
                    # the IM-III. Also, we find the orthogonal point on the solution curve using
                    # fmin_cobyla
                    if model_str == "PLM":
                        # Find the orthogonal point on the solution curve (t,R(t)) of the PLM
                        Model_point = fmin_cobyla(SS_res_model_data, x0=list(Data_point), cons=[PLM_constraint], args=(Data_point,),consargs=(parameters_inverse[0],parameters_inverse[1]))          
                    elif model_str == "IM-III":    
                        # Find the orthogonal point on the solution curve (t,R(t)) of the IM-III
                        Model_point = fmin_cobyla(SS_res_model_data, x0=list(Data_point), cons=[IM_III_constraint], args=(Data_point,),consargs=(parameters_inverse[0],parameters_inverse[1],parameters_inverse[2],parameters_inverse[3]))    
                    # Add the squared distances to our growing sum of squares (SS)
                    SS += SS_res_model_data(Model_point,Data_point)
                #------------------------------------------------------                
                # Lastly, append the root mean square calculated based on the SS-value
                RMS_transf.append(np.sqrt(SS/len(t_data)))
                # Also add the current epsilon value to our variable
                epsilon_transf.append(epsilon)
                # Append the transformed data
                transformed_data.append([t_trans, R_trans])
                # Append the fitted parameters
                if model_str == "PLM":
                    fitted_parameters.append((fitting_structure.beta[0],fitting_structure.beta[1]))
                    inverse_parameters.append((parameters_inverse[0],parameters_inverse[1]))      
                elif model_str == "IM-III":
                    fitted_parameters.append((fitting_structure.beta[0],fitting_structure.beta[1],fitting_structure.beta[2],fitting_structure.beta[3]))
                    inverse_parameters.append((parameters_inverse[0],parameters_inverse[1],parameters_inverse[2],parameters_inverse[3]))      
            else: # The fitting was unsuccessful, so we abort!
                #break
                continue
    # Lastly, cast the RMS-values as an array before we return
    return np.array(epsilon_transf),np.array(RMS_transf), transformed_data, fitted_parameters, inverse_parameters    

