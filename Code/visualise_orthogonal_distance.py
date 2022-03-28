# =================================================================================
# =================================================================================
# Script:"visualise_orthogonal_distance.py"
# Date: 2022-02-15
# Original code written by: John Kitchin
# Adapted by: Johannes Borgqvist
# Description:
# The script illustrates the orthogonal distance between a datapoint and a curve.
# This script is entirely based on the script written by John Kitchin that is
# licensed under the creative commons license: CC BY-SA 4.0. 
# This script demonstrates how one can calculate the distance between a point
# and a curve. The original script can be accessed at the following web-page:
# https://kitchingroup.cheme.cmu.edu/blog/2013/02/14/Find-the-minimum-distance-from-a-point-to-a-curve/.
# The key for finding the orthogonal point on a curve is the function "fmin\_cobyla"
# that is part of the scipy package. It is this function we have used to calculate
# the Root Mean Square in terms of the orthogonal distances between the transformed
# data points and the transformed solution curves in the symmetry based procedure
# for model selection.
# =================================================================================
# =================================================================================
# Import Libraries
# =================================================================================
# =================================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cobyla
# =================================================================================
# =================================================================================
# Functions
# =================================================================================
# =================================================================================
# The curve we project onto
def f(x,args):
    #Extract args
    a = args[0]
    b = args[1]
    # Return objective
    return b*(x-a)**2
# The objective function we minimise
def objective(X,P):
    x,y = X
    return ((x - P[0])**2 + (y - P[1])**2)
# The constraint saying that we want to find a
# point on the curve
def c1(X,*args):
    x,y = X
    return f(x,args) - y
# =================================================================================
# =================================================================================
# Find the Orthogonal Distance and plot it
# =================================================================================
# =================================================================================
# The data point
P = (0.5, 2)
# The parameters of the parabola "b*(x-a)**2"
a = 1
b = 2
# Find the point on the parabola using "fmin_cobyla"
X = fmin_cobyla(objective, x0=[0.5,0.5], cons=[c1], args=(P,),consargs=(a,b))
# Prompt to the user
print('The minimum distance is {0:1.2f}'.format(objective(X,P)))
# Verify the vector to this point is normal to the tangent of the curve
# Position vector from curve to point
v1 = np.array(P) - np.array(X)
# Position vector
v2 = np.array([1, 2.0 *b * (X[0]-a)])
print('dot(v1, v2) = ',np.dot(v1, v2))
# ----------------------------------------------------------------------------------
# Visualise this result
# ----------------------------------------------------------------------------------
# A vector for plotting the parbola
x = np.linspace(a-2, a+2, 100)
# A length factor for esthetic reasons: determines how long the tangent is
length_factor = 0.25
# Do the actual plotting
plt.plot(x, f(x,(a,b)), 'r-', label="Solution curve, $R(t)=b*(t-a)^2,\quad a="+str(a)+",\quad b="+str(b)+"$")
plt.plot(P[0], P[1], 'ko', label="Point, P=("+str(P[0])+","+str(P[1])+")")
plt.plot([P[0], X[0]], [P[1], X[1]], 'b-', label='Shortest distance')
plt.plot([X[0]-length_factor, X[0] + length_factor], [X[1] - length_factor*2.0 *b* (X[0]-a), X[1] +length_factor* 2.0 *b* (X[0]-a)], 'k--', label='Tangent')
plt.axis('equal')
plt.xlabel('t')
plt.ylabel('R(t)')
plt.title('Orthogonal distance implementation using fmin_cobyla')
plt.legend(loc='best')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})
plt.savefig("../Figures/illustration_orthogonal_distance.png")
plt.show()
