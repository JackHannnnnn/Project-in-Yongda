# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 10:45:02 2016

@author: Chaofan
"""


import pandas as pd
import numpy as np
import datetime
import copy

import sys
reload(sys)
sys.setdefaultencoding("utf8")

import pymssql
import pandas.io.sql as sql


#=================Import data from the Server=====================
conn = pymssql.connect(host='YDBI001', database='CRM_CentralDB', charset="utf8")

voucher = sql.read_sql(
    'SELECT VoucherCode, VoucherTypeID, BusinessVoucherID, IssueDate, TotalAccountRecieveAmount, CompanyCode, ParentVoucherCode FROM TBIZ_BusinessVoucher',
    conn)
car_sales = sql.read_sql(
    'SELECT BusinessVoucherID, VehicleSeriesRaw, VehicleColorCode, LicenseNumber, VehicleBrandCode, AccountRecieveAmount, ChassisNumber, IssueDate FROM TBIZ_VehicleSales',
    conn)
maintain = sql.read_sql(
    'SELECT BusinessVoucherID, MaintainTypeCode, LicenseNumber, UsedMileage, VehicleBrandRaw, VehicleColorCode, VehicleBrandCode, ChassisNumber FROM TBIZ_Maintain',
    conn)
customer = sql.read_sql(
    'SELECT CustomerCode, BusinessVoucherID, Birthday, AgeRange, Gender, CHK_CustomerType, CHK_RelationCompany, AttributeCode, OriginCustomerKey FROM TBIZ_VoucherCustomer',
    conn)
insurance = sql.read_sql(
    'SELECT BusinessVoucherID, InsuranceCompanyCode, BusinessInsuranceEndDate, BusinessInsuranceStartDate FROM TBIZ_Insurance',
    conn)
capital_leases_order = sql.read_sql(
    'SELECT BusinessVoucherID, TotalAmount, RentalNumber, PoundageAmount FROM TBIZ_CapitalLeasesOrder', conn)
insurance_item = sql.read_sql(
    'SELECT BusinessVoucherID, InsuranceItemTypeCode, DiscountInsuranceAmount FROM TBIZ_InsuranceItem', conn)
accessories_sales = sql.read_sql('SELECT BusinessVoucherID, Amount FROM TBIZ_AccessorieSales', conn)
agent_service = sql.read_sql('SELECT BusinessVoucherID, ItemCode, Amount FROM TBIZ_AgentService', conn)
payment = sql.read_sql('SELECT BusinessVoucherID, PaymentMethodCode FROM TBIZ_Receipts_Payment', conn)
city = sql.read_sql('SELECT VoucherCustomerKey, CityCode FROM TBIZ_VoucherCustomerAddress', conn)
customer_t = sql.read_sql('SELECT CustomerCode,Gender FROM TCUS_Customer', conn)
second_hand = sql.read_sql(
    'SELECT BusinessVoucherID, LicenseNumber, SecondHandCarDate, SecondHandCarType, TransactionNumber, TransactionAmount FROM TBIZ_SecondHandOrder',
    conn)

conn.close()

conn_p = pymssql.connect(host='YDBI001', database='princetechs', charset="utf8")
city = sql.read_sql('SELECT * FROM CustomerCityAll', conn_p)
conn_p.close()


#=================Helper function=====================

#Check if the key contains Chinese
import re
zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
def contain_chinese(key):    
    #When the CustomerKey contains Chinese, it means this person is just a contact rather than a real customer
    match = zhPattern.search(key)
    if match:
        return 1
    return 0

#Get useful characters for classifying cities to which the license numbers belong
def change_character(number, m=0, n=2):
    try:
        num = number[m:n]
    except:
        num = number
    return num

def change_character2(number, m=0, n=1):
    try:
        num = number[m:n]
    except:
        num = number
    return num
    
#Conversion function and Create dummy variables
def license_num(lis_num):
    lis_num['LicenseNumber'] = lis_num['LicenseNumber'].map(change_character)
    lis_num['license_number_2'] = lis_num['LicenseNumber'].map(change_character2)
    lis_num['license_num'] = np.zeros(len(lis_num)) + 3
    lis_num.ix[lis_num.license_number_2.isin([u'沪', u'京']), 'license_num'] = 1
    lis_num.ix[lis_num.LicenseNumber.isin([u'粤A', u'津A', u'浙A']), 'license_num'] = 1
    lis_num.ix[lis_num.LicenseNumber.isin(
        [u'冀A', u'豫A', u'云A', u'辽A', u'黑A', u'湘A', u'皖A', u'鲁A', u'新A', u'苏A', u'苏E', u'浙B', u'赣A', u'鄂A', u'桂A',
         u'甘A',
         u'晋A', u'蒙A', u'陕A', u'吉A', u'吉B', u'闽A', u'贵A', u'川A', u'青A', u'藏A', u'琼A', u'宁A', u'渝A', u'渝B',
         u'沪C']), 'license_num'] = 2
    lis_num = lis_num.drop(['LicenseNumber'], axis=1)
    license_n = pd.get_dummies(lis_num['license_num'], prefix='License')
    lis_num = pd.merge(pd.DataFrame(lis_num['CustomerCode']), license_n, left_index=True, right_index=True,
                        how='left')
    return lis_num


#=================Feature extraction & Feature engineering=====================

#Remove invalid and duplicate CustomerCode
customer = customer[customer['CustomerCode'].notnull()]
CustomerID_duplicate = customer[customer['BusinessVoucherID'].duplicated()]
customer_duplicate = customer[customer['BusinessVoucherID'].isin(CustomerID_duplicate['BusinessVoucherID'])]
customer_difference = list(
    set(customer['BusinessVoucherID']).difference(set(customer_duplicate['BusinessVoucherID'])))
customer_unique = customer[customer['BusinessVoucherID'].isin(customer_difference)]
customer_duplicate['OriginCustomerKey'] = customer_duplicate['OriginCustomerKey'].fillna(u'ori')
customer_duplicate['OriginCustomerKey'] = customer_duplicate['OriginCustomerKey'].map(contain_chinese)
customer_duplicate = customer_duplicate[customer_duplicate['OriginCustomerKey'] == 0]
customer = pd.concat([customer_duplicate, customer_unique])
customer = customer.drop(['OriginCustomerKey'], axis=1)

    
#Choose transactions related to the audi brand since its corresponding data is relatively complete
voucher = voucher[voucher['CompanyCode'].isin(['BU0000353', 'BU0000354', 'BU0000355', 'BU0000356', 'BU0000357',
                                               'BU0000358', 'BU0000359', 'BU0000360', 'BU0000461'])]

car_sales = pd.merge(car_sales, voucher[['BusinessVoucherID', 'CompanyCode']], on='BusinessVoucherID', how='left')

car_sales = car_sales[car_sales['CompanyCode'].isin(['BU0000353', 'BU0000354', 'BU0000355', 'BU0000356', 'BU0000357',
                                         'BU0000358', 'BU0000359', 'BU0000360', 'BU0000461'])]

maintain = pd.merge(maintain, voucher[['BusinessVoucherID', 'CompanyCode']], on='BusinessVoucherID', how='left')

maintain = maintain[maintain['CompanyCode'].isin(['BU0000353', 'BU0000354', 'BU0000355', 'BU0000356', 'BU0000357',
                                                  'BU0000358', 'BU0000359', 'BU0000360', 'BU0000461'])]
del maintain['CompanyCode'], car_sales['CompanyCode']


#Define key dates
d_2016 = datetime.datetime(2016, 7, 1)
d_2013 = datetime.datetime(2013, 7, 1)

#Choose transactions between 2013 and 2016
voucher = voucher[voucher['IssueDate'].notnull()]
voucher = voucher[(voucher['IssueDate'] <= d_2016) & (voucher['IssueDate'] >= d_2013)]

car_sales = car_sales[(car_sales['IssueDate'] <= d_2016) & (car_sales['IssueDate'] >= d_2013)]
car_sales['IssueDate'] = pd.to_datetime(car_sales['IssueDate'])

maintain = maintain[(maintain['IssueDate'] <= d_2016) & (maintain['IssueDate'] >= d_2013)]
maintain['IssueDate'] = pd.to_datetime(maintain['IssueDate'])


#Choose observations between 2013 and 2016
customer = customer[customer['BusinessVoucherID'].isin(voucher['BusinessVoucherID'])]
car_sales = car_sales[car_sales['BusinessVoucherID'].isin(voucher['BusinessVoucherID'])]
maintain = maintain[maintain['BusinessVoucherID'].isin(voucher['BusinessVoucherID'])]


#=====Features relevant to Customer basic info====
customer_profile = customer[['CustomerCode', 'Gender', 'AgeRange', 'AttributeCode']]
customer_profile = customer_profile.drop_duplicates()   
customer_profile = customer_profile.rename(columns={'AgeRange': 'Age'})
customer_profile['Gender'][customer_profile['Gender'] == True] = 'Male'  
customer_profile['Gender'][customer_profile['Gender'] == False] = 'Female'

customer_gender = pd.get_dummies(customer_profile['Gender'])
customer_profile = pd.merge(customer_profile, customer_gender, left_index=True, right_index=True)
del customer_profile['Gender']

customer_attribute = pd.get_dummies(customer_profile['AttributeCode'], prefix='CustomerType')
customer_profile = pd.merge(customer_profile, customer_attribute, left_index=True, right_index=True)
del customer_profile['AttributeCode']
del customer_profile['CustomerType_CPR0000002'], customer_profile['CustomerType_CPR0000003'], customer_profile['CustomerType_CPR0000004']
customer_profile = customer_profile.drop_duplicates(['CustomerCode'])  

#Used for joining other tables
customer_join_keys = customer[['CustomerCode', 'BusinessVoucherID']]


#Joined to other tables
voucher = pd.merge(voucher, customer_join_keys, on='BusinessVoucherID', how='inner')
car_sales = pd.merge(car_sales, customer_join_keys, on='BusinessVoucherID', how='inner')
maintain = pd.merge(maintain, customer_join_keys, on='BusinessVoucherID', how='inner')

#FirstDealDuration: the duration from the first deal to now
first_deal = voucher[['CustomerCode', 'IssueDate']].groupby('CustomerCode').min()
customer_profile = pd.merge(customer_profile, first_deal, left_on='CustomerCode', right_index=True, how='left')
customer_profile['FirstDealDuration'] = (d_2016 - customer_profile['IssueDate']).map(lambda x: x.days) / 30
del customer_profile['IssueDate']

#License_1.0: if a customer is in a city where license numbers are restricted to reduce congestion
maintain_license = license_num(maintain)
maintain_license = maintain_license.groupby('CustomerCode').max()  #Someone may have multiple cars
car_sales_license = license_num(car_sales)
car_sales_license = car_sales_license.groupby('CustomerCode').max()
all_license = pd.concat([maintain_license, car_sales_license])
all_license = pd.merge(customer_join_keys, all_license, left_on='CustomerCode', right_index=True).groupby('CustomerCode').max()
del all_license['BusinessVoucherID']
customer_profile = pd.merge(customer_profile, all_license, left_on='CustomerCode', right_index=True, how='left')
del customer_profile['License_2.0'], customer_profile['License_3.0']  #Only care if customers are in first-tier cities


#====Final feature output table====
customer_all = copy.deepcopy(customer_profile)


#=====Features relevant to Vehicle Sales====
customer_all['BuyNewCar'] = 0
customer_all['BuyNewCar'][customer_all['CustomerCode'].isin(car_sales['CustomerCode'])] = 1

car_sales_by_customer = car_sales.groupby('CustomerCode')

num_cars_bought = car_sales_by_customer.count()[['BusinessVoucherID']]
num_cars_bought = num_cars_bought.rename(columns={'BusinessVoucherID': 'NumCarsBought'})

avg_car_price = car_sales_by_customer.mean()[['AccountRecieveAmount']]
avg_car_price = avg_car_price.rename(columns={'AccountRecieveAmount': 'AvgCarPrice'})

car_sales_info = pd.concat([num_cars_bought, avg_car_price], axis=1)

#Joined to the final output table
customer_all = pd.merge(customer_all, car_sales_info, left_on='CustomerCode', right_index=True, how='left')


#=====Features relevant to NormalMaintain & Repair====
customer_all['WhetherMaintainRepair'] = 0
customer_all['WhetherMaintainRepair'][customer_all['CustomerCode'].isin(maintain['CustomerCode'])] = 1
customer_all['FirstNormalMaintain'] = 0     #First Normal Maintain is free when you buy a new car in Yongda distributor
customer_all['FirstNormalMaintain'][customer_all['CustomerCode'].isin(maintain[maintain['MaintainTypeCode'] == 'WO0000003']['CustomerCode'])] = 1

maintain = pd.merge(maintain, voucher[['BusinessVoucherID', 'TotalAccountRecieveAmount', 'IssueDate']], on='BusinessVoucherID')


general_repair = maintain[maintain['MaintainTypeCode'].isin(
    ['WO0000001', 'WO0000002', 'WO0000011', 'WO0000012',
     'WO0000015', 'WO0000018', 'WO0000019', 'WO0000021', 'WO0000032', 'WO0000038'])]
incident_repair = maintain[maintain['MaintainTypeCode'].isin(['WO0000006', 'WO0000026', 'WO0000029', 'WO0000039'])]
all_repair = pd.concat([general_repair, incident_repair])

normal_maintain = maintain[maintain['MaintainTypeCode'].isin(
    ['WO0000003', 'WO0000004', 'WO0000017', 'WO0000022', 'WO0000028', 'WO0000040'])]

all_repair_by_customer = all_repair[['BusinessVoucherID', 'CustomerCode', 'TotalAccountRecieveAmount']].groupby('CustomerCode')
total_num_repair = all_repair_by_customer.count()['BusinessVoucherID']
total_num_repair = pd.DataFrame(total_num_repair).rename(columns={'BusinessVoucherID': 'TotalNumRepair'})
total_amount_repair = all_repair_by_customer.sum()['TotalAccountRecieveAmount']
total_amount_repair = pd.DataFrame(total_amount_repair).rename(columns={'TotalAccountRecieveAmount': 'TotalAmountRepair'})
avg_amount_repair = all_repair_by_customer.mean()['TotalAccountRecieveAmount']
avg_amount_repair = pd.DataFrame(avg_amount_repair).rename(columns={'TotalAccountRecieveAmount': 'AvgAmountRepair'})

normal_maintain_by_customer = normal_maintain[['BusinessVoucherID', 'CustomerCode', 'TotalAccountRecieveAmount', 'IssueDate', 'UsedMileage']].groupby('CustomerCode')
total_num_normal_maintain = normal_maintain_by_customer.count()['BusinessVoucherID']
total_num_normal_maintain = pd.DataFrame(total_num_normal_maintain).rename(columns={'BusinessVoucherID': 'TotalNumNormalMaintain'})
total_amount_normal_maintain = normal_maintain_by_customer.sum()['TotalAccountRecieveAmount']
total_amount_normal_maintain = pd.DataFrame(total_amount_normal_maintain).rename(columns={'TotalAccountRecieveAmount': 'TotalAmountNormalMaintain'})
avg_amount_normal_maintain = normal_maintain_by_customer.mean()['TotalAccountRecieveAmount']
avg_amount_normal_maintain = pd.DataFrame(avg_amount_normal_maintain).rename(columns={'TotalAccountRecieveAmount': 'AvgAmountNormalMaintain'})

normal_maintain_date_min = normal_maintain[['CustomerCode', 'IssueDate']].groupby('CustomerCode').min()
normal_maintain_date_max = normal_maintain[['CustomerCode', 'IssueDate']].groupby('CustomerCode').max()
normal_maintain_duration = (normal_maintain_date_max - normal_maintain_date_min).applymap(lambda x: x.days) / 30
normal_maintain_duration = pd.DataFrame(normal_maintain_duration).rename(columns={'IssueDate': 'NormalMaintainDuration'})
normal_maintain_duration[normal_maintain_duration['NormalMaintainDuration'] == 0] = np.nan
normal_maintain_interval = normal_maintain_duration['NormalMaintainDuration'] / (total_num_normal_maintain['TotalNumNormalMaintain'] - 1)
normal_maintain_interval.name = 'NormalMaintainInterval'
normal_maintain_interval = pd.DataFrame(normal_maintain_interval)

maintain_features = pd.concat([total_num_repair, total_amount_repair, avg_amount_repair, total_num_normal_maintain, total_amount_normal_maintain, avg_amount_normal_maintain, normal_maintain_interval], axis=1)

#Joined to the final output table
customer_all = pd.merge(customer_all, maintain_features, left_on='CustomerCode', right_index=True, how='left')

#Save the original final table
customer_all.to_csv('user_profile_features_unfilled.csv', index=False)
user_profile_features_unfilled = copy.deepcopy(customer_all)

customer_all.isnull().sum()

#Handling missing values
customer_all['Age'].fillna(customer_all['Age'].median(), inplace=True)
customer_all['Age'] = customer_all['Age'].astype(int)
customer_all['NumCarsBought'].fillna(0, inplace=True)
customer_all['AvgCarPrice'].fillna(customer_all['AvgCarPrice'].median(), inplace=True)
customer_all['TotalNumRepair'].fillna(0, inplace=True)
customer_all['TotalAmountRepair'].fillna(0, inplace=True)
customer_all['AvgAmountRepair'].fillna(0, inplace=True)
customer_all['TotalNumNormalMaintain'].fillna(0, inplace=True)
customer_all['TotalAmountNormalMaintain'].fillna(0, inplace=True)
customer_all['AvgAmountNormalMaintain'].fillna(0, inplace=True)
customer_all['License_1.0'].fillna(0, inplace=True)
customer_all['NormalMaintainInterval'].fillna(12, inplace=True)
customer_all['AvgNormalMaintainOverCarPrice'] = customer_all['AvgAmountNormalMaintain'] / customer_all['AvgCarPrice']

customer_all.dropna(inplace=True)
customer_all.to_csv('user_profile_features_filled.csv', index=False)


#Adding new feature: Potential value, Loyalty index, Dropout probability
customer_chassis_num = pd.concat([car_sales[['CustomerCode', 'ChassisNumber']], maintain[['CustomerCode', 'ChassisNumber']]], ignore_index=True)
customer_chassis_num = customer_chassis_num.drop_duplicates()
potential_value_by_chassis = pd.read_csv('y_pred_1416.csv')
potential_value_by_chassis = potential_value_by_chassis.rename(columns={'YPred': 'PotentialValue'})
del potential_value_by_chassis['Unnamed: 0']
loyalty_index = pd.read_csv('ChassisNumberMaintainYDRate.csv')
loyalty_index = loyalty_index.rename(columns={'MaintainYDRate': 'LoyaltyIndex'})
new_feature = pd.merge(customer_chassis_num, potential_value_by_chassis, on='ChassisNumber', how='left')
new_feature = pd.merge(new_feature, loyalty_index, on='ChassisNumber', how='left')
potential_value_by_customer = new_feature.groupby('CustomerCode')['PotentialValue'].sum()
loyalty_index_by_customer = new_feature.groupby('CustomerCode')['LoyaltyIndex'].mean()

customer_all = pd.merge(customer_all, pd.concat([potential_value_by_customer, loyalty_index_by_customer], axis=1), left_on='CustomerCode', right_index=True, how='left')
customer_all['PotentialValue'].fillna(0, inplace=True)
customer_all['LoyaltyIndex'].fillna(0, inplace=True)
del customer_all['PotentialValue'], customer_all['LoyaltyIndex']

dropout_rate = pd.read_csv('rf_result.csv')
dropout_probability = dropout_rate[['CustomerCode', 'y20167']]
dropout_probability = dropout_probability.rename(columns={'y20167': 'DropoutProbability'})

customer_all = pd.merge(customer_all, dropout_probability, on='CustomerCode', how='left')
customer_all['DropoutProbability'].fillna(1, inplace=True)



'''
Run KMeans model and Generate the data used for the radar graph analysis.
Through analysing radar graphs, some insights can be obtained into consumption characteristics of different clusterings/consumer groups.
Finally, we visualize interesting business findings and integrate them into the final report. 
'''
   
from sklearn.cluster import KMeans
from sklearn import preprocessing

def generate_user_profile(input_data, feature_ID_list, num_clusters, random_state=None):
    '''    
    The first column in input_data is CustomerCode. The remaining columns are the features we may use to do the clustering.
    '1,2,3,...' in feature_ID_list correspond to the 1st, 2nd, 3rd etc. feaure in our feature pool.
    '''
    #slicing the input data with selected features    
    used_input_data = input_data.ix[:, feature_ID_list]
    
    #normalize the input data and each feature has a mean 0 and std 1     
    scaled_used_input_data = preprocessing.scale(used_input_data)
    scaled_used_input_data = pd.DataFrame(scaled_used_input_data, index=used_input_data.index, columns=used_input_data.columns)
    
    #train the model  
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=random_state, verbose=2).fit(scaled_used_input_data)
    y_predict = kmeans_model.predict(scaled_used_input_data)
    scaled_used_input_data_with_y = pd.concat([scaled_used_input_data, pd.Series(y_predict, index=used_input_data.index, name='group_ID')], axis=1)
    origin_used_input_data_with_y = pd.concat([used_input_data, pd.Series(y_predict, index=used_input_data.index, name='group_ID')], axis=1)
    
    #output three tables
    customer_group_ID = input_data[['CustomerCode']].join(scaled_used_input_data_with_y[['group_ID']])
    scaled_feature_mean_value_by_group = scaled_used_input_data_with_y.groupby('group_ID').mean()
    origin_feature_mean_value_by_group = origin_used_input_data_with_y.groupby('group_ID').mean()
    
    return customer_group_ID, scaled_feature_mean_value_by_group, origin_feature_mean_value_by_group, kmeans_model, \
            scaled_used_input_data_with_y

def output_radar_graph_data(input_data, feature_ID_list, clustering_name, random_state=17):
    results =[]
    for i in range(5):
        results.append(generate_user_profile(input_data, feature_ID_list, i+2, random_state))

    for i in range(5):
        results[i][1].to_csv('%s_scaled_feature_mean_value_by_group_%s_clusters.csv' % (clustering_name, str(i+2)))
        results[i][2].to_csv('%s_origin_feature_mean_value_by_group_%s_clusters.csv' % (clustering_name, str(i+2)))

   
output_radar_graph_data(customer_all, range(1, 20), 'all_features')

new_car_data = customer_all[customer_all['BuyNewCar'] == 1]
del new_car_data['BuyNewCar']
output_radar_graph_data(new_car_data, range(1, 19), 'new_car')

only_maintain_repair = customer_all[customer_all['WhetherMaintainRepair'] == 1]
output_radar_graph_data(only_maintain_repair, [5, 12, 13, 14, 15, 16, 17, 18], 'only_maintain_repair')
