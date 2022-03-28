# =================================================================================
# =================================================================================
# Script:"read_data"
# Date: 2022-02-15
# Implemented by: Johannes Borgqvist
# Description:
# The script reads the time series of interest stored in a csv file in
# the data folder with relative path "../Data/". It then returns the t and y values
# that was stored in the provided data file. 
# =================================================================================
# =================================================================================
# Import Libraries
# =================================================================================
# =================================================================================
from pandas import read_csv # Pandas to read the data
#from matplotlib import pyplot
import re # To remove text from the data
# =================================================================================
# =================================================================================
# Functions
# =================================================================================
# =================================================================================
# Function 1: "time_series_from_csv"
# The function takes in the name of a csv file and then it returns the variable
# values in the vector t, the output values in the vector R, the string defining
# the x-label called "xlabel_str" and the string defining the y-label called
# "ylabel_str"
def time_series_from_csv(file_name):
    # Define the data_file located in the "Data" folder
    data_file = "../Data/" + file_name + ".csv"
    # Read the data file
    dataframe = read_csv(data_file, header=None)
    # Remove all NaN numbers from both columns
    dataframe.dropna(subset = [0], inplace=True)
    dataframe.dropna(subset = [1], inplace=True)
    # Extract the actual data
    data = dataframe.values
    # Extract the input and output variables
    x, y = data[:, 2], data[:, 5]
    # Define the labels of the axes
    xlabel_str = str(x[0]) # x axis
    ylabel_str = str(y[0]) # y axis
    # Calculate the number of values
    num_val = len(x) - 1
    # Define the input and the output (only the data)
    t = x[1:num_val-1]
    R = y[1:num_val-1]
    # Loop through these values and convert them
    # to floats
    for index in range(len(t)):
        t[index] = float(re.sub(r"\D", "", t[index]))
        R[index] = float(R[index])            
    # Return the output
    return xlabel_str, ylabel_str, t, R

