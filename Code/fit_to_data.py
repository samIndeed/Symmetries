# =================================================================================
# =================================================================================
# Script:"fit_to_data"
# Date: 2022-02-24
# Implemented by: Johannes Borgqvist
# Description:
# The program conducts a simple curve fitting to a time series and returns the
# fitted model.
# =================================================================================
# =================================================================================
# Import Libraries
# =================================================================================
# =================================================================================
import symmetry_toolbox  # Home-made
import fit_to_data  # Home-made
from scipy.odr import *  # For calculating the total sum of squares
from scipy.optimize import fmin_cobyla  # To find the orthogonal distance
import numpy as np  # For the exponential function
# =================================================================================
# =================================================================================
# Functions
# =================================================================================
# =================================================================================
# ---------------------------------------------------------------------------------------
# Function 1: "objective_PLM"
# The function returns the objective value of the power law model and it takes
# the following two inputs:
# 1."t" being the ages in the time series (or the independent variable if you will),
# 2. "parameters" containing the parameters (A,gamma) of the PLM.


def objective_PLM(parameters, t):
    # Extract the parameters to be fitted
    A, gamma = parameters
    # Return the function value
    return A*(t**gamma)
# ---------------------------------------------------------------------------------------
# Function 2: "objective_IM_III"
# The function returns the objective value of the exponential model and it takes
# the following two inputs:
# 1."t" being the ages in the time series (or the independent variable if you will),
# 2. "parameters" containing the parameters (A,tau,C,alpha) of the IM-III.


def objective_IM_III(parameters, t):
    # Extract the parameters to be fitted
    A, tau, C, alpha = parameters
    # Return function value
    return ((A)/(np.exp(np.exp(-alpha*(t-tau)))-C))
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# Function 3: "PE_risk_profiles"
# The function fits one of the two candidate models (the PLM or the IM-III) to experimental
# data. It takes five inputs:
# 1. The vector t being the ages of the patients,
# 2. The vector R being the number of incidences of cancer for a given age,
# 3. The string model_str defining whether it is the PLM or the IM-III that is fitted to the data,
# 4. The string fit_type defining whether it is the standard least square or the Orthogonal Distance Regression (ODR) measure that is used,
# 5. A list with a single start guess for the parameters of the model. If this vector is empty, a multi-shooting technique will be used with multiple start guesses for the parameters.
# The function returns three outputs:
# 1. The structure fitted_model containing the fits, the optimal parameters, the parameter variances etc.
# 2. The vector R_hat containing the simulated incidences with the optimal parameters where each age is given by the data vector t,
# 3. The float number R_squared which quantifies the R_squared fit of the candidate model to the data at hand.
def PE_risk_profiles(t, R, model_str, fit_string, fixed_parameters, start_guesses):
    # Take the logarithm of the data
    #R_log = np.array([np.log(R_temp) for R_temp in list(R)])
    # Define the data at hand as a structure on which
    # the scipy odr package will do the estimation
    # if model_str == "PLM":
    data = Data(t, R)
    # elif model_str == "IM-III":
    #data = Data(t, R_log)
    # Define the number of start guesses we will try. Note that the optimisation for finding
    # the optimal parameters is local, and the cost function is non-convex meaning that there
    # are multiple local optima which can be found for different start guesses. Therefore,
    # we use a multiple shooting technique were we test multiple start guesses and then we
    # save the optimal parameters that resulted in the best fit.
    #num_of_start_guesses = 20
    #num_of_start_guesses = 10
    #num_of_start_guesses = 5
    num_of_start_guesses = 3    
    # We have two models to consider, namely the PLM and the IM-III. In order to do the ODR
    # based model fitting, we need to construct a model object and a start guess for the
    # parameters.
    if model_str == "PLM":  # The PLM
        # Define the model
        model = Model(objective_PLM)
        # If we do not have any start guesses, we do a multiple shooting technique.
        if len(start_guesses) == 0:
            # Define a set of parameter values for the multiple shooting technique
            # where we do many local optimisations starting from these startguesses.
            # In the end, we pick the parameters resulting in the lowest minima. This
            # is because we have a non-convex optimisation problem with many local minima.
            A_vec = np.linspace(
                0.00001, 1, num_of_start_guesses, endpoint=True)
            gamma_vec = np.linspace(1, 10, num_of_start_guesses, endpoint=True)
            # Save all start guesses in a big list which we loop over in the end
            parameter_guesses = [[A, gamma]
                                 for A in A_vec for gamma in gamma_vec]
        else:  # We run with the provided start guesses for the parameters
            parameter_guesses = [start_guesses]

    elif model_str == "IM-III":  # The IM-III
        # Define the model
        model = Model(objective_IM_III)
        # If we do not have any start guesses, we do a multiple shooting technique
        if len(start_guesses) == 0:
            # Define the start guess [A, tau, C, ]
            # Define a set of parameter values for the multiple shooting technique
            # where we do many local optimisations starting from these startguesses.
            # In the end, we pick the parameters resulting in the lowest minima. This
            # is because we have a non-convex optimisation problem with many local minima.
            A_vec = np.linspace(0.01, 50, num_of_start_guesses*3, endpoint=True)
            C_vec = np.linspace(0.5, 1, num_of_start_guesses, endpoint=True)
            tau_vec = np.linspace(30, 40, num_of_start_guesses, endpoint=True)
            #alpha_vec = np.array([0.040, 0.045])
            alpha_vec = np.linspace(0.043, 0.045, num_of_start_guesses, endpoint=True)
            # Save all start guesses in a big list which we loop over in the end
            parameter_guesses = [[A, tau, C, alpha]
                                 for A in A_vec for tau in tau_vec for C in C_vec for alpha in alpha_vec]
        else:  # We run with the provided start guesses for the parameters
            parameter_guesses = start_guesses
    elif model_str == "PLM-II":  # The PLM-II
        # Define the model
        model = Model(objective_PLM_II)
        # If we do not have any start guesses, we do a multiple shooting technique
        if len(start_guesses) == 0:
            # Define the start guess [A, gamma, K]
            # Define a set of parameter values for the multiple shooting technique
            # where we do many local optimisations starting from these startguesses.
            # In the end, we pick the parameters resulting in the lowest minima. This
            # is because we have a non-convex optimisation problem with many local minima.
            A_vec = np.linspace(0.1, 5, num_of_start_guesses, endpoint=True)
            gamma_vec = np.linspace(1, 10, num_of_start_guesses, endpoint=True)
            K_vec = np.linspace(1, 100, num_of_start_guesses, endpoint=True)
            # Save all start guesses in a big list which we loop over in the end
            parameter_guesses = [[A, gamma, K]
                                 for A in A_vec for gamma in gamma_vec for K in K_vec]
        else:  # We run with the provided start guess for the parameters
            parameter_guesses = start_guesses
    # Set an initial value of the RMS so that it will be updated
    RMS = 50000
    # Also, initiate the other two outputs that we return
    fitted_model = 0
    R_hat = np.array([1, 2, 3])
    # Also we have a logical variable, whether the fitting was successful or not
    fitting_successful = False
    # ------------------------------------------------------------------------------------------
    # Now we do this analysis for the remaining start guesses of the parameters
    for parameter_guess in parameter_guesses:
        # Define the model fitting object
        if len(fixed_parameters) == 0:
            # Set up ODR with the model and data.
            odr = ODR(data, model, beta0=parameter_guess, sstol=1e-15, partol=1e-15)
        else:
            # Set up ODR with the model and data and fixed parameters.
            odr = ODR(data, model, beta0=parameter_guess,
                      ifixb=fixed_parameters, sstol=1e-15, partol=1e-15)
        # Define whether we should use standard least square or the fancy ODR fitting
        if fit_string == "LS":  # Least square fitting
            odr.set_job(fit_type=2)
        elif fit_string == "ODR":  # Orthogonal distance regression
            odr.set_job(fit_type=0)
        # Run the regression.
        fitted_model_temp = odr.run()
        # Generate the fitted time series as well in the two cases
        if model_str == "PLM":  # The PLM
            R_hat_temp = np.array(
                [objective_PLM(fitted_model_temp.beta, t_i) for t_i in list(t)])
        elif model_str == "IM-III":  # The IM-III
            R_hat_temp = np.array(
                [objective_IM_III(fitted_model_temp.beta, t_i) for t_i in list(t)])
        elif model_str == "PLM-II":  # The PLM-II
            R_hat_temp = np.array(
                [objective_PLM_II(fitted_model_temp.beta, t_i) for t_i in list(t)])
        # Calculate the RMS
        # Allocate memory for the sum of squares (SS)
        SS = 0
        # Loop over the transformed time series
        for time_series_index in range(len(t)):
            # Extract a data point
            Data_point = (t[time_series_index], R[time_series_index])
            # Update the curve specific parameters that are transformed corresponding to
            # the parameter A in the case of the PLM and the parameter C in the case of
            # the IM-III. Also, we find the orthogonal point on the solution curve using
            # fmin_cobyla
            if model_str == "PLM":
                # Find the orthogonal point on the solution curve (t,R(t)) of the PLM
                Model_point = fmin_cobyla(symmetry_toolbox.SS_res_model_data, x0=list(Data_point), cons=[
                                          symmetry_toolbox.PLM_constraint], args=(Data_point,), consargs=(fitted_model_temp.beta[0], fitted_model_temp.beta[1]))
            elif model_str == "IM-III":
                # Find the orthogonal point on the solution curve (t,R(t)) of the IM-III
                Model_point = fmin_cobyla(symmetry_toolbox.SS_res_model_data, x0=list(Data_point), cons=[symmetry_toolbox.IM_III_constraint], args=(
                    Data_point,), consargs=(fitted_model_temp.beta[0], fitted_model_temp.beta[1], fitted_model_temp.beta[2], fitted_model_temp.beta[3]))
            elif model_str == "PLM-II":
                # Find the orthogonal point on the solution curve (t,R(t)) of the IM-III
                Model_point = fmin_cobyla(symmetry_toolbox.SS_res_model_data, x0=list(Data_point), cons=[symmetry_toolbox.PLM_II_constraint], args=(
                    Data_point,), consargs=(fitted_model_temp.beta[0], fitted_model_temp.beta[1], fitted_model_temp.beta[2]))
            # Add the squared distances to our growing sum of squares (SS)
            SS += symmetry_toolbox.SS_res_model_data(Model_point, Data_point)
        # Lastly, append the root mean squared calculated based on the SS-value
        RMS_temp = np.sqrt(SS/len(t))
        # Lastly, if we obtained a better fit than the current minimal fit, we save that fit instead.
        if RMS_temp < RMS:
            RMS = RMS_temp
            R_hat = R_hat_temp
            fitted_model = fitted_model_temp
            if not fitting_successful:
                fitting_successful = True
    # ------------------------------------------------------------------------------------------
    # Return the fitted_model
    return fitted_model, R_hat, RMS, fitting_successful

