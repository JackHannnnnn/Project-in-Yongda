# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 14:19:45 2016

@author: Chaofan
"""

'''
A revised version of this file is going to be integrated into a large system with a friendly user interface
which our customer will use to do the business analysis. 
'''

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing

def main():
    #test if the function works well
    return generate_user_profile([1, 2, 3, 4, 5], 3)

def generate_user_profile(feature_ID_list, num_clusters):
    '''    
    The first column in input_data is CustomerCode. The remaining columns are the features we may use to do the clustering.
    '1,2,3,...' in feature_ID_list correspond to the 1st, 2nd, 3rd etc. feaure in our feature pool.
    '''
   #check if two input parameters are legal and valid
    if not isinstance(feature_ID_list, list):
        raise TypeError('feature_ID_list must be a list')
    
    if len(feature_ID_list) == 0:
        raise ValueError('feature_ID_list can not be empty')    

    for i in feature_ID_list:
        if i not in range(1, 21):
            raise IndexError('The index in the feature_ID_list is out of bounds')
    
    if not isinstance(int(num_clusters), int):
        raise TypeError('The num_clusters must be a int')
    
    if num_clusters < 2 or num_clusters > 10:
        raise ValueError('The num_clusters can not be less than 2 or greater than 10')
    
    #read the input data used for generating user profiles
    input_data = pd.read_csv('user_profile_features_test_file.csv')
    
    #slicing the input data with selected features    
    used_input_data = input_data.ix[:, feature_ID_list]
    
    #normalize the input data and each feature has a mean 0 and std 1     
    scaled_used_input_data = preprocessing.scale(used_input_data)
    scaled_used_input_data = pd.DataFrame(scaled_used_input_data, index=used_input_data.index, columns=used_input_data.columns)
    
    #train the model    
    y_predict = KMeans(n_clusters=num_clusters, random_state=6, verbose=2).fit_predict(scaled_used_input_data)
    
    #integrate the group labels
    name_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'}
    y_predict = pd.Series(y_predict, index=used_input_data.index, name='group_id').map(name_mapping)    
    scaled_used_input_data_with_y = pd.concat([scaled_used_input_data, y_predict], axis=1)
    origin_used_input_data_with_y = pd.concat([used_input_data, y_predict], axis=1)
    
    #output three tables whose data are in the form of JSON
    '''The first table contains CustomerCode corresponding to each group.'''
    customer_group_classification = {}    
    num_of_customer_by_group = {}
    customer_group_id = input_data[['CustomerCode']].join(scaled_used_input_data_with_y[['group_id']])
    for group_id, group_data in customer_group_id.groupby('group_id'):
        customer_group_classification[group_id] = list(group_data['CustomerCode'])
        num_of_customer_by_group[group_id] = len(group_data['CustomerCode'])
    
    scaled_feature_mean_value_by_group = scaled_used_input_data_with_y.groupby('group_id').mean()
    #transform the data to have a range [0, 5] used for understandable visualization in business application 
    max_abs_value = np.max(np.max(np.abs(scaled_feature_mean_value_by_group)))
    scaled_feature_mean_value_by_group = (scaled_feature_mean_value_by_group + max_abs_value) * (2.5 / max_abs_value)
    
    '''The second table contains means in selected scaled feature values and the number of customers for each group.
       These feature values have been scaled before and this table can be used to generate a radar graph.'''
    scaled_group_data = {}
    scaled_group_data['data'] = []
    scaled_group_data['key'] = list(scaled_feature_mean_value_by_group.columns)
    for group_id in scaled_feature_mean_value_by_group.index:
        group_data = {}
        group_data['group_id'] = group_id
        group_data['values'] = list(scaled_feature_mean_value_by_group.ix[group_id])
        group_data['customer_num'] = num_of_customer_by_group[group_id]      
        scaled_group_data['data'].append(group_data)
    
    '''The third table contains means in selected original feature values and the number of customers for each group.'''
    origin_feature_mean_value_by_group = origin_used_input_data_with_y.groupby('group_id').mean()
    origin_group_data = {}
    origin_group_data['data'] = []
    origin_group_data['key'] = list(origin_feature_mean_value_by_group.columns)
    for group_id in origin_feature_mean_value_by_group.index:
        group_data = {}
        group_data['group_id'] = group_id
        group_data['values'] = list(origin_feature_mean_value_by_group.ix[group_id])
        group_data['customer_num'] = num_of_customer_by_group[group_id]
        origin_group_data['data'].append(group_data)
    
    return customer_group_classification, scaled_group_data, origin_group_data
  
if __name__ == '__main__':
    result = main()