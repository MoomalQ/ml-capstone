import glob
import pandas as pd
import numpy as np

def get_paths():

    '''
    Creates dictionary of house folder names and data file paths.
    '''

    # Create dictionary of house names based on folder names with empty lists as values
    paths_dict = {'house_' + str(i):[] for i in range(1,7)}

    for house in paths_dict.keys():

        paths_dict[house] = glob.glob('./REDD/low_freq/' + house + '/*.dat')

        # Drop labels file path
        paths_dict[house] = paths_dict[house][:-1]

        # Append 0 to single digit channels for sorting purposes
        for i in range(len(paths_dict[house])):

            if len(paths_dict[house][i]) == 37:
                paths_dict[house][i] = paths_dict[house][i][:32] + '0' + paths_dict[house][i][32:]

        paths_dict[house].sort()

        for i in range(9):
            paths_dict[house][i] = paths_dict[house][i][:32] + paths_dict[house][i][33:]

    return paths_dict


def get_labels():

    '''
    Creates dictionary of house folder names and appliance labels
    '''

    labels_dict = {'house_' + str(i):[] for i in range(1,7)}

    for house in paths_dict.keys():

        df_temp = pd.read_csv('./REDD/low_freq/'+ house +'/labels.dat', names = ['Labels'])
        labels_dict[house] = df_temp['Labels'].tolist()

        # Add a 0 to single digit labels for sorting purposes
        if len(labels_dict[house]) >= 10:
            for j in range(9):
                labels_dict[house][j] = '0' + labels_dict[house][j]

    return labels_dict

def create_dataframes(paths_dict, labels_dict):

    house_dict = {'house_' + str(i):{} for i in range(1,7)}

    column_names = ['Date','Power']

    for house in house_dict.keys():

        temp_df_list = [pd.read_csv(path,sep = ' ',names = column_names) for path in paths_dict[house]]

        temp_df_dict = dict(zip(labels_dict[house], temp_df_list))

        for appliance in temp_df_dict.keys():
            temp_df_dict[appliance]['Date'] = pd.to_datetime(temp_df_dict[appliance]['Date'], unit = 's')
            temp_df_dict[appliance] = temp_df_dict[appliance].set_index(['Date'])
            temp_df_dict[appliance].columns = [appliance]

        house_dict[house] = pd.concat([temp_df_dict[appliance] for appliance in labels_dict[house]],
                                    axis = 1,
                                    join = 'inner')

    return house_dict