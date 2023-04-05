# Author: eoin@flaresafety.com
# Date: 2023-04-05
# Description: Extract raw data from the Real World Dataset and resturn single csv with Flare ML features.

import numpy as np
import pandas as pd
import os
from scipy.stats import skew, kurtosis


# Calculates the max, min, range and moments of samples in time series.

# Tries to calculate the skew of a set of values, returns 0 if skew(x) = NaN. We need this because sometimes sigma = 0, which gives a division error.
def skew_no_nans(x: float) -> float:
    if np.std(x) == 0:
        return 0
    else:
        return skew(x)

# Same again, but with kurt.
def kurt_no_nans(x: float) -> float:
    if np.std(x) == 0:
        return 0
    else:
        return kurtosis(x)

def column_names(dataSeries: pd.DataFrame) -> list:
    metrics = ['min','max','diff','std','var','skew','kurt']
    return [metric + dataSeries.name for metric in metrics]

def maxMinDiff(dataSeries: pd.DataFrame, sampleSize=100) -> np.array:
    sampleMax = []
    sampleMin = []
    sampleRange = []
    sampleStd = []
    sampleVar = []
    sampleSkew = []
    sampleKurt = []
    for i in range(0,len(dataSeries),sampleSize): 
        sampleMax.append(dataSeries[i:(i+sampleSize)].max())
        sampleMin.append(dataSeries[i:(i+sampleSize)].min())
        sampleRange.append(dataSeries[i:(i+sampleSize)].max()-dataSeries[i:(i+sampleSize)].min())
        sampleStd.append(np.std(dataSeries[i:(i+sampleSize)]))
        sampleVar.append(np.var(dataSeries[i:(i+sampleSize)]))
        sampleSkew.append((skew_no_nans(dataSeries[i:(i+sampleSize)])))
        sampleKurt.append((kurt_no_nans(dataSeries[i:(i+sampleSize)])))

    output = np.array([sampleMax,sampleMin, sampleRange,sampleStd,sampleVar, sampleSkew, sampleKurt,]).T

    return output
    
# This processes all columns at once and sticks them together.
def Process(df: pd.DataFrame, columns = ['attr_x','attr_y','attr_z']) -> pd.DataFrame:
    # The [:-1] gets rid of the time column.
    data = [maxMinDiff(df[column]) for column in columns]
    column_labels = np.hstack([np.array(column_names(df[column])) for column in columns])
    return pd.DataFrame(np.hstack(data),columns=column_labels)



'''Data wrangling'''
activity_types = ['lying','running','walking','sitting','standing']
instrument = ['acc','Gyroscope']

'''Helper Functions'''
def extractLabel(filename: str) -> str:
    for activity in activity_types:
        if activity in filename:
            return activity
        
def extractSubject(filename: str) -> str:
    for subject in ['Subject_1', 'Subject_4', 'Subject_5', 'Subject_6']:
        if subject in filename:
            return subject

# Extracts data from files and output into a list of dataframes.
def main() -> None:
    filenames = []
    for subject in ['Subject_1', 'Subject_4', 'Subject_5', 'Subject_6']:
        for activity in os.listdir(f'C:\\Users\\EoinB\\Dev\\Data\\Real-world-data\\{subject}'):
            filenames.append(f'C:\\Users\\EoinB\\Dev\\Data\\Real-world-data\\{subject}\\{activity}')

    acc_list = [filename for filename in filenames if 'acc' in filename]
    gyr_list = [filename for filename in filenames if 'Gyroscope' in filename]

    dfs_acc = [] 
    for filename in acc_list:
        dataframe = pd.read_csv(filename).drop(columns=['id','attr_time'])
        dataframe.columns = ['accelX','accelY','accelZ']
        dataframe_ = Process(dataframe,dataframe.columns)
        # dataframe_['subject'] = [extractSubject(filename)]*len(dataframe_)
        # dataframe_['label'] = [extractLabel(filename)]*len(dataframe_)
        dfs_acc.append(dataframe_)

    dfs_gyr = [] 
    for filename in gyr_list:
        dataframe = pd.read_csv(filename).drop(columns=['id','attr_time'])
        dataframe.columns = ['gyroX','gyroY','gyroZ']
        dataframe_ = Process(dataframe,dataframe.columns)
        dataframe_['subject'] = [extractSubject(filename)]*len(dataframe_)
        dataframe_['label'] = [extractLabel(filename)]*len(dataframe_)
        dfs_gyr.append(dataframe_)

    # Merges the lists of acceleration and gyroscope datasets and puts the test subject and label on the end.
    dfs = [pd.concat([dfs_acc[i],dfs_gyr[i]],axis=1) for i in range(len(dfs_acc))]

    # Concatenate the list of datasets.
    df = pd.concat(dfs,ignore_index=True)

    # Filter out the times the test subject was actually standing still in the entries labelled 'running'.
    df_run = df[df['label'] == 'running']
    indices_to_drop = []
    for i in df_run.index:
        if df_run['stdaccelY'][i] < 5:
            indices_to_drop.append(i)

    df_ = df.drop(indices_to_drop).reset_index(drop=True)

    df_.to_csv('C:\\Users\\EoinB\\dev\\data\\Real-World-Data\\Dataset.csv')
    print('Done.')

if __name__ == '__main__':
    main()

