# =================================================================================
# =================================================================================
# Script:"symmetry_analysis_PLM_vs_IM_III.py"
# Date: 2022-03-22
# Implemented by: Johannes Borgqvist
# Description:
# The script re-generates all the results presented in the article.
# =================================================================================
# =================================================================================
# Import Libraries
# =================================================================================
# =================================================================================
import read_data  # Home-made
import write_output  # Home-made
import fit_to_data # Home-made
import symmetry_toolbox # Home-made
import clean_up_output # Home-made
import numpy as np # For the numerical calculations
import math # For mathematics
from matplotlib import pyplot as plt # For plotting
import multiprocessing as mp # For parallelisation
# =================================================================================
# =================================================================================
# Read the data
# =================================================================================
# =================================================================================
# MYELOMA
xlabel_myeloma, ylabel_myeloma, t_myeloma, R_myeloma = read_data.time_series_from_csv("Myeloma_cancer")
# Only ages above 25 as we have zero incidences below
t_myeloma = np.array(list(t_myeloma[24:len(t_myeloma)-1]))
R_myeloma = np.array(list(R_myeloma[24:len(R_myeloma)-1]))
# COLON
xlabel_colon, ylabel_colon, t_colon, R_colon = read_data.time_series_from_csv("Colon_cancer")
# Only ages above 12 as we have zero incidences below
t_colon = np.array(list(t_colon[11:len(t_colon)-1]))
R_colon = np.array(list(R_colon[11:len(R_colon)-1]))
# Chronic Myeloid Leukemia (CML)
xlabel_CML, ylabel_CML, t_CML, R_CML = read_data.time_series_from_csv("CML_cancer")
# Only ages above 10 as we have zero incidences below
t_CML = np.array(list(t_CML[9:len(t_CML)-1]))
R_CML = np.array(list(R_CML[9:len(R_CML)-1]))
# =================================================================================
# =================================================================================
# FIT THE MODELS TO THE DATA
# =================================================================================
# =================================================================================
print("\n\t--------------------------------------------------------------------------------------\n")
print("\n\t\tThe model fitting\n")
print("\n\t--------------------------------------------------------------------------------------\n")
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# Orthonal Distance Regression (ODR)
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# Create a list which we iterate over
input_lists = [[t_myeloma,R_myeloma,"PLM"],[t_myeloma,R_myeloma,"IM-III"],[t_colon,R_colon,"PLM"],[t_colon,R_colon,"IM-III"],[t_CML,R_CML,"PLM"],[t_CML,R_CML,"IM-III"]]
# We do the fitting in parallel (to use all 8 cores, use mp.cpu_count())
#pool = mp.Pool(6)
pool = mp.Pool(mp.cpu_count())
# Do the fitting in parallel
results = pool.starmap(fit_to_data.PE_risk_profiles,[(input_list[0],input_list[1],input_list[2],"ODR",[],[]) for input_list in input_lists])
# Close the pool
pool.close()
# Prompt to the user
print("\t\tFitting is done!")
# Extract and save all the results
# MYELOMA DATA
# PLM 
PLM_fitted_to_myeloma_ODR = results[0][0]
R_hat_PLM_myeloma_ODR = results[0][1]
RMS_PLM_myeloma_ODR = results[0][2]
write_output.save_data_PE("myeloma", "PLM", PLM_fitted_to_myeloma_ODR, RMS_PLM_myeloma_ODR, "ODR")
# IM-III
IM_III_fitted_to_myeloma_ODR = results[1][0]
R_hat_IM_III_myeloma_ODR = results[1][1]
RMS_IM_III_myeloma_ODR = results[1][2]
write_output.save_data_PE("myeloma", "IM-III", IM_III_fitted_to_myeloma_ODR, RMS_IM_III_myeloma_ODR, "ODR")
# COLON CANCER DATA
# PLM 
PLM_fitted_to_colon_ODR = results[2][0]
R_hat_PLM_colon_ODR = results[2][1]
RMS_PLM_colon_ODR = results[2][2]
write_output.save_data_PE("colon", "PLM", PLM_fitted_to_colon_ODR, RMS_PLM_colon_ODR, "ODR")
# IM-III
IM_III_fitted_to_colon_ODR = results[3][0]
R_hat_IM_III_colon_ODR = results[3][1]
RMS_IM_III_colon_ODR = results[3][2]
write_output.save_data_PE("colon", "IM-III", IM_III_fitted_to_colon_ODR, RMS_IM_III_colon_ODR, "ODR")
# CML DATA
# PLM 
PLM_fitted_to_CML_ODR = results[4][0]
R_hat_PLM_CML_ODR = results[4][1]
RMS_PLM_CML_ODR = results[4][2]
write_output.save_data_PE("CML", "PLM", PLM_fitted_to_CML_ODR, RMS_PLM_CML_ODR, "ODR")
# IM-III
IM_III_fitted_to_CML_ODR = results[5][0]
R_hat_IM_III_CML_ODR = results[5][1]
RMS_IM_III_CML_ODR = results[5][2]
write_output.save_data_PE("CML", "IM-III", IM_III_fitted_to_CML_ODR, RMS_IM_III_CML_ODR, "ODR")
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# Plot the data and the fit in Python
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# PLOT OF THE FIT OF THE MODEL TO THE DATA
# Overall properties
fig, axes = plt.subplots(1,3,figsize=(15,5))
#fig, axes = plt.subplots(1,1,figsize=(15,5))
plt.rc('axes', labelsize=15)    # fontsize of the x and y label
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
# Subplot 1a
axes[0].plot(t_myeloma, R_myeloma, '*', color='black', label='Data Myeloma cancer')
axes[0].plot(t_myeloma, R_hat_PLM_myeloma_ODR, '-', color = (103/256,0/256,31/256),label='ODR fit PLM')
axes[0].plot(t_myeloma, R_hat_IM_III_myeloma_ODR, '-', color = (2/256,56/256,88/256),label='ODR fit IM-III')
axes[0].legend()
# Subplot 2a
axes[1].plot(t_colon, R_colon, '*', color='black', label='Data colon cancer')
axes[1].plot(t_colon, R_hat_PLM_colon_ODR, '-', color = (103/256,0/256,31/256),label='ODR fit PLM')
axes[1].plot(t_colon, R_hat_IM_III_colon_ODR, '-', color = (2/256,56/256,88/256),label='ODR fit IM-III')
axes[1].legend()
# Subplot 3a
axes[2].plot(t_CML, R_CML, '*', color='black', label='Data CML')
axes[2].plot(t_CML, R_hat_PLM_CML_ODR, '-', color = (103/256,0/256,31/256),label='ODR fit PLM')
axes[2].plot(t_CML, R_hat_IM_III_CML_ODR, '-', color = (2/256,56/256,88/256),label='ODR fit IM-III')
axes[2].legend()
# add a big axis, hide frame
fig.add_subplot(111, frameon=False)
#hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Age, $t$")
plt.ylabel("Incidence, $R(t)$")
# displaying the title
plt.title("Fit of candidate models to three data sets",fontsize=20, fontweight='bold')
#plt.show()
plt.savefig("../Figures/Fit_of_models_to_cancer_data.png")



# =================================================================================
# =================================================================================
# CALCULATE THE TRANSFORMATION SCALES
# =================================================================================
# What epsilon should we use to transform the age 85 years to 170 years? Here, comes
# the answer in the case of our two models.
# The PLM
epsilon_scale_PLM = symmetry_toolbox.PLM_transformation_scale(2)
# The IM-III
# Myeloma data
epsilon_scale_IM_III_myeloma = symmetry_toolbox.IM_III_transformation_scale(85,2,IM_III_fitted_to_myeloma_ODR.beta[3],IM_III_fitted_to_myeloma_ODR.beta[1])
# Colon data
epsilon_scale_IM_III_colon = symmetry_toolbox.IM_III_transformation_scale(85,2,IM_III_fitted_to_colon_ODR.beta[3],IM_III_fitted_to_colon_ODR.beta[1])
# CML data
epsilon_scale_IM_III_CML = symmetry_toolbox.IM_III_transformation_scale(85,2,IM_III_fitted_to_CML_ODR.beta[3],IM_III_fitted_to_CML_ODR.beta[1])
# Prompt to the user
print("\n\t--------------------------------------------------------------------------------------\n")
print("\n\t\tThe transformation scales increasing the age from 80 years to 160 years\n")
print("\n\t--------------------------------------------------------------------------------------\n")
print("\t\tThe PLM:\tepsilon_PLM\t=\t%0.12f"%(epsilon_scale_PLM))
print("\t\tThe IM-III myeloma:\tepsilon_IM_III_myeloma\t=\t%0.12f"%(epsilon_scale_IM_III_myeloma))
print("\t\tThe IM-III colon:\tepsilon_IM_III_colon\t=\t%0.12f"%(epsilon_scale_IM_III_colon))
print("\t\tThe IM-III CML:\tepsilon_IM_III_CML\t=\t%0.12f"%(epsilon_scale_IM_III_CML))
#epsilon_scale_IM_III_myeloma = symmetry_toolbox.IM_III_transformation_scale(86,2,IM_III_fitted_to_myeloma_ODR.beta[3],IM_III_fitted_to_myeloma_ODR.beta[1])
#print("\t\tThe new IM-III myeloma scale for the plot:\tepsilon_IM_III_myeloma\t=\t%0.12f"%(epsilon_scale_IM_III_myeloma))
# Prompt to the user
print("\n\t--------------------------------------------------------------------------------------\n")
print("\n\t\tThe symmetry based framework for model selection\n")
print("\n\t--------------------------------------------------------------------------------------\n")
print("\t\tThe scales for the framework are the following:")
print("\t\t\tPLM\tAll datasets:\t epsilon_scale\t=\t%0.12f"%(5*epsilon_scale_PLM))
print("\t\t\tIM-III\t myeloma:\t epsilon_scale\t=\t%0.12f"%(2*epsilon_scale_IM_III_myeloma))
print("\t\t\tIM-III\t colon:\t epsilon_scale\t=\t%0.12f"%(2*epsilon_scale_IM_III_colon))
print("\t\t\tIM-III\t CML:\t epsilon_scale\t=\t%0.12f\n\n"%(2*epsilon_scale_IM_III_CML))
# Allocate four epsilon vectors with transformation parameters
#epsilon_vector_PLM = np.array([0.0,30*epsilon_scale_PLM])
#epsilon_vector_PLM = np.array([0.0,epsilon_scale_PLM])
epsilon_vector_PLM = np.linspace(0,2*epsilon_scale_PLM,2,endpoint=True)
#--------------------------------------------------------------------------------------------------------
#epsilon_vector_PLM = np.array([0.0,0.2*epsilon_scale_PLM])
#epsilon_vector_IM_III_myeloma = np.array([0,0.6795*epsilon_scale_IM_III_myeloma])
#epsilon_vector_IM_III_colon = np.array([0,0.89*epsilon_scale_IM_III_colon])
#epsilon_vector_IM_III_CML = np.array([0,0.8286*epsilon_scale_IM_III_CML])
#--------------------------------------------------------------------------------------------------------
#epsilon_vector_IM_III_myeloma = np.array([0,0.745*epsilon_scale_IM_III_myeloma,0.75*epsilon_scale_IM_III_myeloma,0.80*epsilon_scale_IM_III_myeloma,0.85*epsilon_scale_IM_III_myeloma,0.90*epsilon_scale_IM_III_myeloma,0.95*epsilon_scale_IM_III_myeloma,epsilon_scale_IM_III_myeloma,])
#epsilon_vector_IM_III_myeloma = np.array([0,0.5*epsilon_scale_IM_III_myeloma,0.55*epsilon_scale_IM_III_myeloma,0.60*epsilon_scale_IM_III_myeloma,0.65*epsilon_scale_IM_III_myeloma,0.70*epsilon_scale_IM_III_myeloma,0.75*epsilon_scale_IM_III_myeloma,0.80*epsilon_scale_IM_III_myeloma,])
#epsilon_vector_IM_III_colon = np.array([0,0.745*epsilon_scale_IM_III_colon,0.75*epsilon_scale_IM_III_colon,0.80*epsilon_scale_IM_III_colon,0.85*epsilon_scale_IM_III_colon,0.90*epsilon_scale_IM_III_colon,0.95*epsilon_scale_IM_III_colon,epsilon_scale_IM_III_colon])
#epsilon_vector_IM_III_colon = np.array([0,0.5*epsilon_scale_IM_III_colon,0.55*epsilon_scale_IM_III_colon,0.60*epsilon_scale_IM_III_colon,0.65*epsilon_scale_IM_III_colon,0.70*epsilon_scale_IM_III_colon,0.75*epsilon_scale_IM_III_colon,0.80*epsilon_scale_IM_III_colon])
#epsilon_vector_IM_III_CML = np.array([0,0.745*epsilon_scale_IM_III_CML,0.75*epsilon_scale_IM_III_CML,0.80*epsilon_scale_IM_III_CML,0.85*epsilon_scale_IM_III_CML,0.90*epsilon_scale_IM_III_CML,0.95*epsilon_scale_IM_III_CML,epsilon_scale_IM_III_CML])
#--------------------------------------------------------------------------------------------------------
epsilon_vector_IM_III_myeloma = np.linspace(0,epsilon_scale_IM_III_myeloma,2,endpoint=True)
epsilon_vector_IM_III_colon = np.linspace(0,epsilon_scale_IM_III_colon,2,endpoint=True)
epsilon_vector_IM_III_CML = np.linspace(0,epsilon_scale_IM_III_CML,2,endpoint=True)
#--------------------------------------------------------------------------------------------------------
# Create an iterable list
input_lists = [[t_myeloma,R_myeloma,epsilon_vector_PLM,PLM_fitted_to_myeloma_ODR,"PLM"],[t_myeloma,R_myeloma,epsilon_vector_IM_III_myeloma,IM_III_fitted_to_myeloma_ODR,"IM-III"],[t_colon,R_colon,epsilon_vector_PLM,PLM_fitted_to_colon_ODR,"PLM"],[t_colon,R_colon,epsilon_vector_IM_III_colon,IM_III_fitted_to_colon_ODR,"IM-III"],[t_CML,R_CML,epsilon_vector_PLM,PLM_fitted_to_CML_ODR,"PLM"],[t_CML,R_CML,epsilon_vector_IM_III_CML,IM_III_fitted_to_CML_ODR,"IM-III"]]
# We do the fitting in parallel (to use all 8 cores, use mp.cpu_count())
#pool = mp.Pool(6)
pool = mp.Pool(mp.cpu_count())
# Do the model selection in parallel
results = pool.starmap(symmetry_toolbox.symmetry_based_model_selection,[(input_list[0],input_list[1],input_list[2],input_list[3],input_list[4]) for input_list in input_lists])
# Close the pool
pool.close()
# Extract the results
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# Myeloma
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
print("\t\tThe output:")
print("\t\t\tMyeloma:")
# PLM MYELOMA
epsilon_transf_PLM_myeloma = results[0][0] # PLM transformation parameters
RMS_transf_PLM_myeloma = results[0][1] # PLM RMS-values
transformed_data_PLM_myeloma = results[0][2] # PLM Transformed data
fitted_parameters_PLM_myeloma = results[0][3] # PLM fitted parameters
inverse_parameters_PLM_myeloma = results[0][4] # PLM inversely transformed parameters
# Remove outliers
epsilon_transf_PLM_myeloma,RMS_transf_PLM_myeloma, fitted_parameters_PLM_myeloma,inverse_parameters_PLM_myeloma,transformed_data_PLM_myeloma = clean_up_output.remove_outliers(epsilon_transf_PLM_myeloma,RMS_transf_PLM_myeloma,fitted_parameters_PLM_myeloma,inverse_parameters_PLM_myeloma,transformed_data_PLM_myeloma)
# Save the data
write_output.save_data_symmetry_based_model_selection("myeloma","PLM",epsilon_transf_PLM_myeloma,RMS_transf_PLM_myeloma, fitted_parameters_PLM_myeloma,inverse_parameters_PLM_myeloma)
# Reduce density
if len(epsilon_transf_PLM_myeloma) > 200:
    epsilon_transf_PLM_myeloma_sparse = np.linspace(epsilon_transf_PLM_myeloma[0],epsilon_transf_PLM_myeloma[-1],200,endpoint=True)
    RMS_transf_PLM_myeloma_sparse = clean_up_output.reduce_density(epsilon_transf_PLM_myeloma,RMS_transf_PLM_myeloma,epsilon_transf_PLM_myeloma_sparse)
else:
    epsilon_transf_PLM_myeloma_sparse = epsilon_transf_PLM_myeloma
    RMS_transf_PLM_myeloma_sparse = RMS_transf_PLM_myeloma
print("\t\t\tPLM, number of transformation parameters:\t%d"%(int(len(epsilon_transf_PLM_myeloma))))
#----------------------------------------------------------------------------------
# IM-III MYELOMA
epsilon_transf_IM_III_myeloma = results[1][0] # IM-III transformation parameters
RMS_transf_IM_III_myeloma = results[1][1] # IM-III RMS-values
transformed_data_IM_III_myeloma = results[1][2] # IM-III Transformed data
fitted_parameters_IM_III_myeloma = results[1][3] # IM-III fitted parameters
inverse_parameters_IM_III_myeloma = results[1][4] # IM-III inversely transformed parameters
# Remove outliers
epsilon_transf_IM_III_myeloma,RMS_transf_IM_III_myeloma,fitted_parameters_IM_III_myeloma,inverse_parameters_IM_III_myeloma,transformed_data_IM_III_myeloma = clean_up_output.remove_outliers(epsilon_transf_IM_III_myeloma,RMS_transf_IM_III_myeloma,fitted_parameters_IM_III_myeloma,inverse_parameters_IM_III_myeloma,transformed_data_IM_III_myeloma)
# Save the data
write_output.save_data_symmetry_based_model_selection("myeloma","IM-III",epsilon_transf_IM_III_myeloma,RMS_transf_IM_III_myeloma, fitted_parameters_IM_III_myeloma,inverse_parameters_IM_III_myeloma)
# Reduce density
if len(epsilon_transf_IM_III_myeloma) > 200:
    # Reduce the sparsity of these ridiculously dense vectors
    epsilon_transf_IM_III_myeloma_sparse = np.linspace(epsilon_transf_IM_III_myeloma[0],epsilon_transf_IM_III_myeloma[-1],200,endpoint=True)
    RMS_transf_IM_III_myeloma_sparse = clean_up_output.reduce_density(epsilon_transf_IM_III_myeloma,RMS_transf_IM_III_myeloma,epsilon_transf_IM_III_myeloma_sparse)
else:
    epsilon_transf_IM_III_myeloma_sparse = epsilon_transf_IM_III_myeloma
    RMS_transf_IM_III_myeloma_sparse = RMS_transf_IM_III_myeloma
print("\t\t\tIM-III, number of transformation parameters:\t%d"%(int(len(epsilon_transf_IM_III_myeloma))))
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# Colon
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
print("\t\t\tColon cancer:")
# PLM COLON
epsilon_transf_PLM_colon = results[2][0] # PLM transformation parameters
RMS_transf_PLM_colon = results[2][1] # PLM RMS-values
transformed_data_PLM_colon = results[2][2] # PLM Transformed data
fitted_parameters_PLM_colon = results[2][3] # PLM fitted parameters
inverse_parameters_PLM_colon = results[2][4] # PLM inversely transformed parameters
# Remove outliers
epsilon_transf_PLM_colon,RMS_transf_PLM_colon,fitted_parameters_PLM_colon,inverse_parameters_PLM_colon,transformed_data_PLM_colon = clean_up_output.remove_outliers(epsilon_transf_PLM_colon,RMS_transf_PLM_colon,fitted_parameters_PLM_colon,inverse_parameters_PLM_colon,transformed_data_PLM_colon)
# Save the data
write_output.save_data_symmetry_based_model_selection("colon","PLM",epsilon_transf_PLM_colon,RMS_transf_PLM_colon, fitted_parameters_PLM_colon,inverse_parameters_PLM_colon)
# Reduce density
if len(epsilon_transf_PLM_colon) > 200:
    # Reduce the sparsity of these ridiculously dense vectors
    epsilon_transf_PLM_colon_sparse = np.linspace(epsilon_transf_PLM_colon[0],epsilon_transf_PLM_colon[-1],200,endpoint=True)
    RMS_transf_PLM_colon_sparse = clean_up_output.reduce_density(epsilon_transf_PLM_colon,RMS_transf_PLM_colon,epsilon_transf_PLM_colon_sparse)
else:
    epsilon_transf_PLM_colon_sparse = epsilon_transf_PLM_colon
    RMS_transf_PLM_colon_sparse = RMS_transf_PLM_colon
print("\t\t\tPLM, number of transformation parameters:\t%d"%(int(len(epsilon_transf_PLM_colon))))
#----------------------------------------------------------------------------------
# IM-III COLON
epsilon_transf_IM_III_colon = results[3][0] # IM-III transformation parameters
RMS_transf_IM_III_colon = results[3][1] # IM-III RMS-values
transformed_data_IM_III_colon = results[3][2] # IM-III Transformed data
fitted_parameters_IM_III_colon = results[3][3] # IM-III fitted parameters
inverse_parameters_IM_III_colon = results[3][4] # IM-III inversely transformed parameters
# Remove outliers
epsilon_transf_IM_III_colon,RMS_transf_IM_III_colon,fitted_parameters_IM_III_colon,inverse_parameters_IM_III_colon,transformed_data_IM_III_colon = clean_up_output.remove_outliers(epsilon_transf_IM_III_colon,RMS_transf_IM_III_colon,fitted_parameters_IM_III_colon,inverse_parameters_IM_III_colon,transformed_data_IM_III_colon)
# Save the data
write_output.save_data_symmetry_based_model_selection("colon","IM-III",epsilon_transf_IM_III_colon,RMS_transf_IM_III_colon, fitted_parameters_IM_III_colon,inverse_parameters_IM_III_colon)
# Reduce density
if len(epsilon_transf_IM_III_colon) > 200:
    # Reduce the sparsity of these ridiculously dense vectors
    epsilon_transf_IM_III_colon_sparse = np.linspace(epsilon_transf_IM_III_colon[0],epsilon_transf_IM_III_colon[-1],200,endpoint=True)
    RMS_transf_IM_III_colon_sparse = clean_up_output.reduce_density(epsilon_transf_IM_III_colon,RMS_transf_IM_III_colon,epsilon_transf_IM_III_colon_sparse)
else:
    epsilon_transf_IM_III_colon_sparse = epsilon_transf_IM_III_colon
    RMS_transf_IM_III_colon_sparse = RMS_transf_IM_III_colon
print("\t\t\tIM-III, number of transformation parameters:\t%d"%(int(len(epsilon_transf_IM_III_colon))))
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# CML
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
print("\t\t\tCML:")
# PLM CML
epsilon_transf_PLM_CML = results[4][0] # PLM transformation parameters
RMS_transf_PLM_CML = results[4][1] # PLM RMS-values
transformed_data_PLM_CML = results[4][2] # PLM Transformed data
fitted_parameters_PLM_CML = results[4][3] # PLM fitted parameters
inverse_parameters_PLM_CML = results[4][4] # PLM inversely transformed parameters
# Remove outliers
epsilon_transf_PLM_CML,RMS_transf_PLM_CML,fitted_parameters_PLM_CML,inverse_parameters_PLM_CML,transformed_data_PLM_CML = clean_up_output.remove_outliers(epsilon_transf_PLM_CML,RMS_transf_PLM_CML,fitted_parameters_PLM_CML,inverse_parameters_PLM_CML,transformed_data_PLM_CML)
# Save the data
write_output.save_data_symmetry_based_model_selection("CML","PLM",epsilon_transf_PLM_CML,RMS_transf_PLM_CML, fitted_parameters_PLM_CML,inverse_parameters_PLM_CML)
# Reduce density
if len(epsilon_transf_PLM_CML)>200:
    # Reduce the sparsity of these ridiculously dense vectors
    epsilon_transf_PLM_CML_sparse = np.linspace(epsilon_transf_PLM_CML[0],epsilon_transf_PLM_CML[-1],200,endpoint=True)
    RMS_transf_PLM_CML_sparse = clean_up_output.reduce_density(epsilon_transf_PLM_CML,RMS_transf_PLM_CML,epsilon_transf_PLM_CML_sparse)
else:
    epsilon_transf_PLM_CML_sparse = epsilon_transf_PLM_CML
    RMS_transf_PLM_CML_sparse = RMS_transf_PLM_CML
print("\t\t\tPLM, number of transformation parameters:\t%d"%(int(len(epsilon_transf_PLM_CML))))
#----------------------------------------------------------------------------------
# IM-III CML
epsilon_transf_IM_III_CML = results[5][0] # IM-III
RMS_transf_IM_III_CML = results[5][1] # IM-III
transformed_data_IM_III_CML = results[5][2] # IM-III Transformed data
fitted_parameters_IM_III_CML = results[5][3] # IM-III fitted parameters
inverse_parameters_IM_III_CML = results[5][4] # IM-III inversely transformed parameters
# Remove outliers
epsilon_transf_IM_III_CML,RMS_transf_IM_III_CML,fitted_parameters_IM_III_CML,inverse_parameters_IM_III_CML,transformed_data_IM_III_CML = clean_up_output.remove_outliers(epsilon_transf_IM_III_CML,RMS_transf_IM_III_CML,fitted_parameters_IM_III_CML,inverse_parameters_IM_III_CML,transformed_data_IM_III_CML)
# Save the data
write_output.save_data_symmetry_based_model_selection("CML","IM-III",epsilon_transf_IM_III_CML,RMS_transf_IM_III_CML, fitted_parameters_IM_III_CML,inverse_parameters_IM_III_CML)
# Reduce density
if len(epsilon_transf_IM_III_CML) > 200:
    # Reduce the sparsity of these ridiculously dense vectors
    epsilon_transf_IM_III_CML_sparse = np.linspace(epsilon_transf_IM_III_CML[0],epsilon_transf_IM_III_CML[-1],200,endpoint=True)
    RMS_transf_IM_III_CML_sparse = clean_up_output.reduce_density(epsilon_transf_IM_III_CML,RMS_transf_IM_III_CML,epsilon_transf_IM_III_CML_sparse)
else:
    epsilon_transf_IM_III_CML_sparse = epsilon_transf_IM_III_CML
    RMS_transf_IM_III_CML_sparse = RMS_transf_IM_III_CML
print("\t\t\tIM_III, number of transformation parameters:\t%d\n"%(int(len(epsilon_transf_IM_III_CML))))
#----------------------------------------------------------------------------------
# Prompt to the user
print("\t\tSymmetry framework is done!")
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# Plot the symmetry based model selection
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# PLM RESULTS
# Overall properties
fig, axes = plt.subplots(1,1,figsize=(15,5))
plt.rc('axes', labelsize=15)    # fontsize of the x and y label
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
axes.plot(epsilon_transf_PLM_myeloma_sparse,RMS_transf_PLM_myeloma_sparse,'-', color = (103/256,0/256,31/256),label='PLM Myeloma cancer')
axes.plot(epsilon_transf_PLM_colon_sparse,RMS_transf_PLM_colon_sparse,'-', color = (206/256,18/256,86/256),label='PLM Colon cancer')
axes.plot(epsilon_transf_PLM_CML_sparse,RMS_transf_PLM_CML_sparse,'-', color = (223/256,101/256,176/256),label='PLM CML')
# add a big axis, hide frame
fig.add_subplot(111, frameon=False)
#hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Transformation parameter, $\epsilon$")
plt.ylabel("Root mean square, $\mathrm{RMS}(\epsilon)$")
# displaying the title
plt.title("The symmetry based model selection for the PLM",fontsize=20, fontweight='bold')
plt.savefig("../Figures/symmetry_based_model_selection_PLM.png")
# IM-III
# Overall properties
fig, axes = plt.subplots(1,3,figsize=(15,5))
plt.rc('axes', labelsize=15)    # fontsize of the x and y label
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
# Subplot 1: Myeloma
axes[0].plot(epsilon_transf_IM_III_myeloma_sparse,RMS_transf_IM_III_myeloma_sparse,'-', color = (2/256,56/256,88/256),label='IM-III Myeloma cancer')
# Subplot 2: Colon cancer
axes[1].plot(epsilon_transf_IM_III_colon_sparse,RMS_transf_IM_III_colon_sparse,'-', color = (54/256,144/256,192/256),label='IM-III Colon cancer')
# Subplot 3: CML
axes[2].plot(epsilon_transf_IM_III_CML_sparse,RMS_transf_IM_III_CML_sparse,'-', color = (208/256,209/256,230/256),label='IM-III CML')
# add a big axis, hide frame
fig.add_subplot(111, frameon=False)
#hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Transformation parameter, $\epsilon$")
plt.ylabel("Root mean square, $\mathrm{RMS}(\epsilon)$")
# displaying the title
plt.title("The symmetry based model selection for the IM-III",fontsize=20, fontweight='bold')
plt.savefig("../Figures/symmetry_based_model_selection_IM_III.png")
plt.show()    
# Prompt to the user



