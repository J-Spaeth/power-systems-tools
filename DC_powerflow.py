"""
Creating a DC powerflow script. This work is based on examples in the Power Systems Analysis and Design textbook by Glover. Section 6.10 dealing in DC Powerflow.
This is a linearized version of the AC powerflow computed via the Newton-Raphson method with the Jacobian.
"""

import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ------- 1. READ IN EXCEL VIA DATAFRAME -------------------------------------
# Start by reading in an excel file contaning the network data
excel_input = pd.read_excel(r"C:\Users\223106456\Desktop\PyPSA\network_data.xlsx", sheet_name=None)

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


# ------- 2. BUILD THE DC POWERFLOW EQUATIONS -------------------------------------

"""
Two primary equations are needed to build a DC powerflow

1. Reduced power balance equation: -B*(delta) = P
    - where B is the imaginary component of the Ybus, calculated neglecting line resistance and excepting the slack bus row and column.
    - delta is the voltage angle of each bus
    - P is the power (in MW typically) produced or consumed at each bus. 
        - B and P are known from the topology of the system.
        - delta is being solved for use in the following equation

2. Simplified power flow equation: P_jk = ((delta_j)-(delta_k))/X_jk
    - where P_jk is the powerflow between two buses
    - delta_j is the voltage angle of bus j. delta_k is the voltage angle of bus k.
    - X_jk is the reactance between bus j and bus k. No resistance.
        - Assuming voltage magnitudes are constant 1.0 per unit, the 4th Jacobian quadrant - the QV equation - can be completely eliminated.
        - Of the original 4 Jacobians, only J1 is left. The Power delta equation.
"""

# --------- 2a) Construct the B matrix. This will be an NxN matrix where N = N - 1 to remove the slack bus. -----------------

n = 0 #variable to hold the N dimensions of the B matrix
B_dict = {}
for index, row in buses_df.iterrows():

    if row['Bus Type'] == "Slack": # if the bus is a slack bus, skip it. Don't need to include in B matrix.
        continue
    
    B_dict[row['Bus Name']] = n
    n = n + 1

print("This is the bus dictionary: ",B_dict) # this is the dictionary that will be a guide to storing values in the actual B matrix later on.

# Constructing a B matrix with NxN zeros
B = np.zeros((n,n), dtype=float) 
print("Here is the size of the matrix prior to filling it: \n", B)

# Fill the B matrix with susceptance values from the topology.
for index, row in lines_df.iterrows():

    # 3 Potential Cases:

    # CASE 1: if both buses are not in B_dict -> skip        (the only way this would happen if both buses are slack bus which would be an input error)
    if row['Start Bus'] not in B_dict and row['End Bus'] not in B_dict:
         print("ERROR: A line is connecting two slack buses. Check input data.")
         continue
    
    # CASE 2: if both buses ARE in B_dict -> update all 4 matrix elements
    if row['Start Bus'] in B_dict and row['End Bus'] in B_dict:

        # assign the start bus and end bus
        bus_j = row['Start Bus']
        bus_k = row['End Bus']
        #print(bus_j, bus_k)

        b_jk = 1/row['Impedance (x)'] # calc the line susceptance from the reactance

        # identify the position of each bus on either side of the line in the B_dict
        j = B_dict[bus_j]
        k = B_dict[bus_k]

        # use the j and k buses to fill the correct matrix element with the correct susceptance value.
        # Diagonal Elements: sum of all susceptances connected to bus
        # Off-Diagonal Elements: negative sum of susceptances between bus j and bus k
        # Because each loop defines all combinations of Diagonal and Off-Diagonal elements, this allows different lines to add to the same diagonal bus.
        #   - example: Line_1_2 and Line_1_3 will both have a B[1][1] position because they're both connected to Bus_1, but the off diagonals will be different.
        #              Because of the += and matrix B is stored outside of the loop, each loop iteration adds to the previous B matrix value. 
        B[j][j] += b_jk
        B[k][k] += b_jk
        B[j][k] -= b_jk     # logic is -= instead of just -b_jk in case there are parallel lines between buses.
        B[k][j] -= b_jk

    # CASE 3: if one of the buses is not in B_dict -> update only the diagonal of the non slack bus. 
    # The slack bus has been removed from the calc and is excluded from the B matrix, but the susceptance on the line between the slack bus and non-slack bus must be included in the diagonal B matrix calc.
    # Without this section, at least one of the diagonals will be incorrect. For any bus connected to the slack bus, the diagonal value will be incorrect if this code is excluded.
    else:

        b_jk = 1/row['Impedance (x)'] # calc the line susceptance from the reactance

        #identify the non-slack bus and add the line susceptance to the non-slack bus diagonal. The non-slack bus can be in the j or k position. The Start or End Bus position.
        if row['Start Bus'] in B_dict:
             j = B_dict[row['Start Bus']]
             B[j][j] += b_jk
        else:
             k = B_dict[row['End Bus']]
             B[k][k] += b_jk



print("Here is the B matrix filled with susceptance values: \n", B)

# --------- 2b) Construct the P vector -----------------

# create the P matrix which has n rows and just 1 column. vertical vector.
P = np.zeros((n,1), dtype=float)
print("Here is the P matrix filled with zeros: \n", P)

# P_net = P_gen - P_load. So at each bus the load will be subtracted from the generation to get a net injection or load

for index, row in generators_df.iterrows():
    if row['Bus'] not in B_dict:
        continue
    
    i = B_dict[row['Bus']]
    p_gen = row['Capacity (MW)']
    P[i] += p_gen # assign the max gen power to this bus. This is the total gen this bus will produce at an instant


for index, row in loads_df.iterrows():
    if row['Bus'] not in B_dict:
            continue

    i = B_dict[row['Bus']]
    p_load = row['Load Value (MW)']
    P[i] -= p_load # subtract the max load from this bus.

# Convert MW values to per unit values using the system MVA base
mva_base = system_base_df['System Base (MVA)'].iloc[0]
P = P / mva_base

print("Here is the P matrix filled with total power values: \n",P)

# --------- 2c) Invert the B matrix and solve for delta -----------------

# Solving Bδ = P where B follows Ybus sign convention (positive diagonal)
# Equivalent to textbook's -Bδ = P where textbook defines B with negative diagonals
Delta = np.linalg.solve(B,P)

# Add back in slack bus with it's assumed 0 degree voltage angle
Delta = np.insert(Delta, 0, 0)
Delta_deg = np.degrees(Delta)
print("Here is the Delta matrix filled with voltage angles in radians: \n", Delta)
print("Here is the Delta matrix filled with voltage angles in degrees: \n", Delta_deg)


# --------- 2d) Calculate Line Flows -----------------

# create new dictionary with all buses included because we want to solve for all line flows including those connected to slack bus
Delta_dict = {}
d = 0
for index, row in buses_df.iterrows():
     Delta_dict[row['Bus Name']] = d
     d = d + 1

P_flow_results = []
for index, row in lines_df.iterrows():
    
    bus_j = row['Start Bus']
    bus_k = row['End Bus']

    delta_j = Delta_dict[row['Start Bus']]
    delta_k = Delta_dict[row['End Bus']]

    X_jk = row['Impedance (x)']
    
    P_flow = (Delta[delta_j] - Delta[delta_k]) / X_jk
    P_flow = P_flow * mva_base
    P_flow_results.append(P_flow)
    print(f"{P_flow:.2f}")

#print(f"{P_flow_results:.2f}")
