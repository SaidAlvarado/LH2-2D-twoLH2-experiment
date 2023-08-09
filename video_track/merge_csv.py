# Blender exports the tracked csv file individually per marker.
# This is a simple script to combine them into a single file and sort the data by taimestamp
#
import pandas as pd

# choose which experiment to process
exp_num = 5

# Read the CSV files
red = pd.read_csv(f'video_track/{exp_num}_blender_R.csv', parse_dates=['timestamp'])
green = pd.read_csv(f'video_track/{exp_num}_blender_G.csv', parse_dates=['timestamp'])
# blue = pd.read_csv(f'video_track/{exp_num}_blender_B.csv', parse_dates=['timestamp'])

# Concatenate the dataframes
# df = pd.concat([red, green, blue])
df = pd.concat([red, green])

# Sort the dataframe by 'timestamp'
df = df.sort_values(by='timestamp')
df['timestamp'] = df['timestamp'].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))

# Write the sorted dataframe to a new CSV file
df.to_csv(f'video_track/{exp_num}_blender.csv', index=False)