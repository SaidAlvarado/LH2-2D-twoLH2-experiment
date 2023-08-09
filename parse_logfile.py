import pandas as pd
from datetime import datetime
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2

#############################################################################
###                                Options                                ###
#############################################################################

robot_colors = { '5037ba3e25bbda90': 'R',
                 'c407deb8205167a4': 'B',
                 'defe4df80cb89ccd': 'G'
}


#############################################################################
###                                Code                                   ###
#############################################################################

## Read the struct log with the information
# Define a regular expression pattern to extract timestamp and source from log lines
log_pattern = re.compile(r'timestamp=(?P<timestamp>.*?) .*? source=(?P<source>\S+) .*? sweep_0_poly=(?P<sweep_0_poly>\d+) sweep_0_off=(?P<sweep_0_off>\d+) sweep_0_bits=(?P<sweep_0_bits>\d+) sweep_1_poly=(?P<sweep_1_poly>\d+) sweep_1_off=(?P<sweep_1_off>\d+) sweep_1_bits=(?P<sweep_1_bits>\d+) sweep_2_poly=(?P<sweep_2_poly>\d+) sweep_2_off=(?P<sweep_2_off>\d+) sweep_2_bits=(?P<sweep_2_bits>\d+) sweep_3_poly=(?P<sweep_3_poly>\d+) sweep_3_off=(?P<sweep_3_off>\d+) sweep_3_bits=(?P<sweep_3_bits>\d+)')

# Create an empty list to store the extracted data
data = []

# Open the log file and iterate over each line
with open("./pydotbot.log", "r") as log_file:
    for line in log_file:
        # Extract timestamp and source from the line
        match = log_pattern.search(line)
        if match and "lh2-4" in line:
            # Append the extracted data to the list
            data.append({
                "timestamp": datetime.strptime(match.group("timestamp"), "%Y-%m-%dT%H:%M:%S.%fZ"),
                "source": match.group("source"),
                "poly_0": int(match.group("sweep_0_poly")),
                "off_0":  int(match.group("sweep_0_off")),
                "bits_0": int(match.group("sweep_0_bits")),
                "poly_1": int(match.group("sweep_1_poly")),
                "off_1":  int(match.group("sweep_1_off")),
                "bits_1": int(match.group("sweep_1_bits")),
                "poly_2": int(match.group("sweep_2_poly")),
                "off_2":  int(match.group("sweep_2_off")),
                "bits_2": int(match.group("sweep_2_bits")),
                "poly_3": int(match.group("sweep_3_poly")),
                "off_3":  int(match.group("sweep_3_off")),
                "bits_3": int(match.group("sweep_3_bits")),
            })
# Create a pandas DataFrame from the extracted data
df = pd.DataFrame(data)

## Remove lines that don't have the data from both lighthouses
# Define the conditions
cond1 = df[['poly_0', 'poly_1', 'poly_2', 'poly_3']].isin([0, 1]).sum(axis=1) == 2
cond2 = df[['poly_0', 'poly_1', 'poly_2', 'poly_3']].isin([2, 3]).sum(axis=1) == 2
cond = cond1 & cond2
# Filter the rows that meet the condition
df = df.loc[cond].reset_index(drop=True)
# Change the MAC to a color tag
df['source'] = df['source'].replace(robot_colors)

## Convert the data to a numpy a array and sort them to make them compatible with Cristobal's code
poly_array = df[["bits_0", "bits_1", "bits_2", "bits_3"]].to_numpy()
sorted_indices = np.argsort(df[['poly_0','poly_1','poly_2','poly_3']].values,axis=1)
bits_df = df[['bits_0','bits_1','bits_2','bits_3']]
sorted_bits = np.empty_like(bits_df)
for i, row in enumerate(sorted_indices):
    sorted_bits[i] = bits_df.values[i, row]


## Sort the columns for LH2-A and LH2-B separatedly.
c01 = np.sort(sorted_bits[:,0:2], axis=1).astype(int)
c23 = np.sort(sorted_bits[:,2:4], axis=1).astype(int)
# Re-join the columns and separate them into the variables used by cristobals code.
c0123 = np.hstack([c01, c23])
c0123 = np.sort(sorted_bits, axis=1).astype(int)
# This weird order to asign the columns is because there was an issue with the dataset, and the data order got jumbled.
c1A = c0123[:,0] 
c2A = c0123[:,2]
c1B = c0123[:,1]
c2B = c0123[:,3]


#############################################################################
###                           Save reordered data                         ###
#############################################################################

sorted_df = pd.DataFrame({
                          'timestamp' : df['timestamp'],

                          'source'    : df["source"],

                          'LHA_count_1': c0123[:,0],

                          'LHA_count_2': c0123[:,2],

                          'LHB_count_1': c0123[:,1],

                          'LHB_count_2': c0123[:,3]},
                          index = df.index
                          )


# Change the format of the timestamp column
sorted_df['timestamp'] = sorted_df['timestamp'].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))


#############################################################################
###                           Clear Outliers                         ###
#############################################################################
# This goes grid point by grid point and removes datapoints who are too far away from mean.

# # Find outliers by lookingvery strong sudden jumps the measurements of each gridpoints.
# prev_diff_df = abs(sorted_df.loc[ (sorted_df['timestamp'] >= start) & (sorted_df['timestamp'] <= end),['LHA_count_1', 'LHA_count_2', 'LHB_count_1', 'LHB_count_2']].diff().fillna(0).shift(1))
# next_diff_df = abs(sorted_df.loc[ (sorted_df['timestamp'] >= start) & (sorted_df['timestamp'] <= end),['LHA_count_1', 'LHA_count_2', 'LHB_count_1', 'LHB_count_2']].diff().fillna(0).shift(-1))
# # Get a boolean dataframe with indexes of the good measurement-s
# filter_df = pd.concat([filter_df, (prev_diff_df['LHA_count_1'] <= 20 ) & (next_diff_df['LHA_count_1'] <= 20 ) & (prev_diff_df['LHA_count_2'] <= 20 ) & (next_diff_df['LHA_count_2'] <= 20 ) & (prev_diff_df['LHB_count_1'] <= 20 ) & (next_diff_df['LHB_count_1'] <= 20 ) & (prev_diff_df['LHB_count_2'] <= 20 ) & (next_diff_df['LHB_count_2'] <= 20 )])
# # filter_df = pd.concat([filter_df, diff_df.le(20).all(axis=1)])

# # Apply the filter that removes the outliers
# sorted_df_bak = sorted_df
# sorted_df = sorted_df.iloc[filter_df.index[filter_df[0] == True]].reset_index(drop=True)
# Get the cleaned values back on the variables needed for the next part of the code.

# Clear all point for which you don't know the corresponding 3D coordinate, and print.
sorted_df.to_csv('./data.csv', index=True)


a=0
