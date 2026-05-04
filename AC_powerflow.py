"""
Creating an AC powerflow script. This work is based on examples in the Power Systems Analysis and Design textbook by Glover. 
Section 6.3: Iterative Solutions to Nonlinear Algebraic Equations: Newton-Raphson
This is a full AC powerflow solution including all four Jacobians - P-delta, P-V, Q-delta, and Q-V equations
"""

import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ------- 1. READ IN EXCEL VIA DATAFRAME -------------------------------------
# Start by reading in an excel file contaning the network data
excel_input = pd.read_excel(r"/Users/jacksonspaeth/Desktop/Python/PowerSystems/network_data.xlsx", sheet_name=None)

# Create dataframes of the topology input from each tab of the excel
system_base_df = excel_input["Base"]
buses_df = excel_input["Buses"]
lines_df = excel_input["Lines"]
generators_df = excel_input["Generators"]
loads_df = excel_input["Loads"]

#print(buses_df,"\n")
#print(lines_df,"\n")
#print(generators_df,"\n")
#print(loads_df,"\n")


# ------- 2. THE AC POWERFLOW THEORY -------------------------------------

"""
Consider the following set of linear algebraic equations in matrix format:
Ax = y --- where x and y are N vectors and A is an NxN square matrix. Given A and y, we want to solve for x.
We assume that determinant(A) is nonzero, so a unique solution exists. (matrix is not singular)
Gauss elimination is possible to back substitute and solve for all values but not an effective solution for large problems like power system topologies.

An iterative approach is the standard for solving these problems:
First, select an initial guess x(0). Then use x(i+1) = g[x(i)], i = 0,1,2
where x(i) is the ith guess and g is an N vector of functions that specify the iterative method. 
Continue the procedure until the following stopping condition is satisfied:
| [x_k(i+1) - x_k(i)] / x_k(i) | <= e , for all k = 1,2,...,N
    where x_k(i) si the kth component of x(i) and e is a specified tolerance level.
    The following questions are pertinent:
        1. Will the iteration procedure converge to a unique solution?
        2. What is the convergence rate (how many iterations are required)?

Jacobi method uses the "old" values of x(i) at iteration i on the right side of equation to generate "new" value x_k(i+1):
    x_k(i+1) = 1/A_kk[yk - SUM[A_kn*x_n] - SUM[A_kn*x_n]] ---> Please see text Equation (6.2.5) for better detail.
    in matrix format: x(i+1) = Mx(i) + [D^-1]y

Section 6.3 - Iterative Solutions to Nonlinear Algebraic Equations: Newton-Raphson
Iterative methods of section 6.2 can be extended to nonlinear equations.
x(i+1) = x(i) + [D^-1]*[y-f[x(i)]]
    - for linear equations, f(x) = Ax and the above equation reduces to the Jacobi method above.
    - for nonlinear equations, the matrix D must be specified.
        - Newton-Raphson method is based on a Taylor series expansion that neglects higher order terms (see text for explanation)

Newton-Raphson:
    x(i+1) = x(i) + [J^-1(i)][y - f[x(i)]]
        where J(i) is a NxN matrix, whose elements are the partial derivatives of each function with respect to each variable. 
        See text for a nice picture representation of the J matrix
        - The J matrix is called the Jacobian

Newton-Raphson in 4 steps:
    - we don't actually want to calculate the J matrix inverse because that is a costly and hard task when you have a big system
    - Instead, we restructure the math to solve a linear system of equations, which is computationally easier.

    Equations:
        J(i)[delta_x(i)] = delta_y(i)  (6.3.11)
        delta_x(i) = x(i+1) - x(i)     (6.3.12)
        delta_y(i) = y - f[x(i)]       (6.3.13)

    Step 1. Compute delta_y(i). 
    Step 2. Compute J(i)
    Step 3. Use Gauss elimination and back substitution, solve delta_x(i)
    Step 4. Compute x(i+1)

    Explanation:
    Step 1 - Measure how wrong your current guess is (the mismatch)
    Step 2 - Evaluate how sensitive the system is at your current operating point (the Jacobian)
    Step 3 - Use that sensitivity to solve for the correction needed, delta_x
           - Do this by solving J[delta_x] = delta_y for delta_x. Do this instead of computing J^-1
    Step 4 - Apply the correction delta_x to get the new guess.

Section 6.4 - The Power-Flow Problem
    - The powerflow problem is the computation of voltage magnitude and phase angle at each bus in a power system under balanced 3ph steady state conditions
    - As a by-product of this calc, real and reactive power flows and equipment losses can be computed

Section 6.6 - Power-Flow Solution by Newton Raphson
    The Power-Flow Equations:
        P_k = V_k * SUM[Y_kn * V_n * cos(delta_k - delta_n - theta_kn)] ---> Eq (6.6.2)
        Q_k = V_k * SUM[Y_kn * V_n * sin(delta_k - delta_n - theta_kn)] ---> Eq (6.6.3)
"""

# ------- 3. CONSTRUCT THE Y-BUS -------------------------------------

n = 0 #variable to hold the N dimensions of the Y matrix
Y_dict = {}
for index, row in buses_df.iterrows(): # include all buses in the Y_dictionary. Don't exclude the slack bus like in the DC powerflow
    
    Y_dict[row['Bus Name']] = n 
    n = n + 1

print("This is the bus dictionary: ",Y_dict) # this is the dictionary that will be a guide to storing values in the actual Y matrix later on.

# Constructing a Y matrix with NxN zeros
Ybus = np.zeros((n,n), dtype=complex) 
print("Here is the size of the matrix prior to filling it: \n", Ybus)

# Fill the Y matrix with admittance values from the topology.
for index, row in lines_df.iterrows():
    
    # CASE 1: if both buses ARE in B_dict -> update all 4 matrix elements
    if row['Start Bus'] in Y_dict and row['End Bus'] in Y_dict:

        # assign the start bus and end bus
        bus_j = row['Start Bus']
        bus_k = row['End Bus']
        #print(bus_j, bus_k)

        z_jk = complex(row['Resistance (r)'], row['Reactance (x)']) # calc the line total impedance
        y_jk = 1/z_jk # calc line admittance

        # identify the position of each bus on either side of the line in the B_dict
        j = Y_dict[bus_j]
        k = Y_dict[bus_k]

        # use the j and k buses to fill the correct matrix element with the correct admittance value.
        # Diagonal Elements: sum of all admittances connected to bus
        # Off-Diagonal Elements: negative sum of admittances between bus j and bus k
        # Because each loop defines all combinations of Diagonal and Off-Diagonal elements, this allows different lines to add to the same diagonal bus.
        #   - example: Line_1_2 and Line_1_3 will both have a B[1][1] position because they're both connected to Bus_1, but the off diagonals will be different.
        #              Because of the += and matrix B is stored outside of the loop, each loop iteration adds to the previous Y matrix value. 
        Ybus[j][j] += y_jk
        Ybus[k][k] += y_jk
        Ybus[j][k] -= y_jk     # logic is -= instead of just -y_jk in case there are parallel lines between buses.
        Ybus[k][j] -= y_jk

    else:
         print("ERROR: One or more lines is connected to a bus not in the Y bus dictionary. Please check bus inputs.")
         sys.exit(1)


bus_names = list(Y_dict.keys())
Ybus_df = pd.DataFrame(Ybus, index=bus_names, columns=bus_names)
pd.set_option('display.float_format', '{:.4f}'.format)
print("Here is the Y matrix filled with admittance values: \n", Ybus_df)

# ------- 3. CONSTRUCT THE JACOBIAN -------------------------------------
# The partial derivatives of the powerflow equations P and Q will be taken with respect to the phase angle and voltage magnitude for each bus
# This will result in four quadrants of the Jacobian

# identify the PQ and PV buses. This is necessary for constructing the submatrices of the Jacobian, as the # of buses and their types determinethe size of the matrix.
non_slack_dict = {}
PV_dict = {}
PQ_dict = {}
Slack_dict = {}
PQ_column_dict = {} # necessary for 
n = 0
i = 0
for index, row in buses_df.iterrows():
    if row['Bus Type'] == 'Slack':
        Slack_dict[row['Bus Name']] = index
    else:
        non_slack_dict[row['Bus Name']] = n
        if row['Bus Type'] == 'PV':
            PV_dict[row['Bus Name']] = n
        elif row['Bus Type'] == 'PQ':
            PQ_dict[row['Bus Name']] = n
            PQ_column_dict[row['Bus Name']] = i
            i = i + 1
        n = n + 1
print("Number of buses:", len(non_slack_dict)+1)
print("Number of Slack buses:", len(Slack_dict))
print("Number of PV buses:", len(PV_dict))
print("Number of PQ buses:", len(PQ_dict))

print("non-slack buses:", non_slack_dict)
print("PV buses:", PV_dict)
print("PQ buses:", PQ_dict)
print("PQ column buses:", PQ_column_dict)

# Convert Ybus into magnitude and angle values necessary for powerflow equations
Ybus_magnitude = np.abs(Ybus)
Ybus_angle = np.angle(Ybus)
print("Magnitude of the Ybus:\n", Ybus_magnitude)
print("Angle of the Ybus in rad:\n", Ybus_angle)

# Flat start initialization - set all bus voltages to 1.0pu and angles to 0
# creating a Voltage array of magnitude 1 size of # of buses
# creating a delta angle array of zeros size of # of buses
V = np.ones(len(Y_dict))
delta = np.zeros(len(Y_dict))

# ------- 3a) Construct J1 - P-delta -------------------------------------
# dP/ddelta - P equations for every non-slack bus. Delta unknowns for every non-slack bus
# J1 size is N_nonslack x N_nonslack
J1 = np.zeros((len(non_slack_dict), len(non_slack_dict)))
#print(J1)

# loop through every possible combination of k and n and use appropriate equation for on vs off diagonals
for bus_k, k in non_slack_dict.items():     # Size N_nonslack
    for bus_n, n in non_slack_dict.items(): # Size N_nonslack
        if bus_k != bus_n: # off diagonal
            yk = Y_dict[bus_k] # Index to Y_dict same indexing scheme as V and delta arrays. 
            yn = Y_dict[bus_n] # Index to Y_dict same indexing scheme as V and delta arrays. 

            V_k = V[yk] # lookup V for yk
            V_n = V[yn] # lookup V for yn
            delta_k = delta[yk] # lookup delta for yk
            delta_n = delta[yn] # lookup delta for yn

            Y_kn = Ybus_magnitude[yk][yn]
            theta_kn = Ybus_angle[yk][yn]
            # compute J1[k][n]
            J1_kn = V_k*Y_kn*V_n*np.sin(delta_k - delta_n - theta_kn)
            J1[k][n] = J1_kn # Store the value in the correct NONSLACK index since we only calc nonslack voltages and angles.

            # compute J1[k][k] On Diagonal
            # each off diagonal pair contributes to the on diagonal calc. So it's the same calc, just summing the offdiagonal values to the on diagonals values for each pair.
            J1[k][k] -= J1_kn # Store the value in the correct NONSLACK index since we only calc nonslack voltages and angles.
print("Jacobian 1 matrix:\n", J1)

# ------- 3b) Construct J2 - P-V -----------------------------------------
# dP/dvoltage - P equations for every non-slack bus. Voltage unknowns for all PQ buses since PV buses have known voltage.
# J2 size is N-nonslack x N_PQ
J2 = np.zeros((len(non_slack_dict), len(PQ_dict)))
#print(J2)

# loop through every possible combination of k and n and use appropriate equation for on vs off diagonals
for bus_k, k in non_slack_dict.items(): # Size N_nonslack
    for bus_n, n in PQ_column_dict.items():    # Size N_PQ
        if bus_k != bus_n: # off diagonal
            yk = Y_dict[bus_k] # Index to Y_dict same indexing scheme as V and delta arrays. 
            yn = Y_dict[bus_n] # Index to Y_dict same indexing scheme as V and delta arrays. 

            V_k = V[yk] # lookup V for yk
            V_n = V[yn] # lookup V for yn
            delta_k = delta[yk] # lookup delta for yk
            delta_n = delta[yn] # lookup delta for yn

            Y_kn = Ybus_magnitude[yk][yn]
            theta_kn = Ybus_angle[yk][yn]
            # compute J1[k][n]
            J2_kn = V_k*Y_kn*V_n*np.cos(delta_k - delta_n - theta_kn)
            J2[k][n] = J2_kn # Store the value in the correct NONSLACK index since we only calc nonslack voltages and angles.

            # compute J1[k][k] On Diagonal
            # each off diagonal pair contributes to the on diagonal calc. So it's the same calc, just summing the offdiagonal values to the on diagonals values for each pair.
            # Index the columns correctly for the 'diagonal' calc. No longer a square matrix. N_nonslack x N_PQ
            if bus_k in PQ_column_dict:
                n_col = PQ_column_dict[bus_k]
                J2[k][n_col] += J2_kn # Store the value in the correct NONSLACK index since we only calc nonslack voltages and angles.
    
    ## for J2, we have an extra term out front of the summation, so we need to add that only once for the entire diagonal term
    if bus_k in PQ_column_dict: # only PQ buses have a diagonal element
        yk = Y_dict[bus_k]
        n_col = PQ_column_dict[bus_k]
        J2[k][n_col] += V[yk] * Ybus_magnitude[yk][yk] * np.cos(Ybus_angle[yk][yk])
print("Jacobian 2 matrix:\n", J2)

print(Ybus.real)

# ------- 3c) Construct J3 - Q-delta -------------------------------------
# dQ/ddelta - Q equations for all PQ buses. PV buses have fixed voltage so don't write a Q mismatch for them. Delta unknowns for all non-slack buses.
# J3 size is N_PQ x N_nonslack

# ------- 3d) Construct J4 - Q-V -----------------------------------------
# dQ/dvoltage - Q equations for all PQ buses. Voltage unknowns for all PQ buses
# J4 size is N_PQ x N_PQ


