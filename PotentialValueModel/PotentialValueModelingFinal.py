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


#Payment method of buying new cars in yongda
new_car_chassis = voucher[voucher['VoucherTypeID'] == 1][['BusinessVoucherID', 'VoucherCode']]
new_car_chassis = pd.merge(new_car_chassis, car_sales[['BusinessVoucherID', 'ChassisNumber']], on='BusinessVoucherID')
del new_car_chassis['BusinessVoucherID']
voucher_type_5 = voucher[voucher['VoucherTypeID'] == 5][['ParentVoucherCode', 'BusinessVoucherID']]
voucher_payment = pd.merge(voucher_type_5, payment, on='BusinessVoucherID', how='left')
new_car_payment = pd.merge(new_car_chassis, voucher_payment, left_on='VoucherCode', right_on='ParentVoucherCode', how='left')
new_car_payment['WhetherMortgage'] = new_car_payment['PaymentMethodCode'] == 'PT0000022'
new_car_payment['WhetherMortgage'] = new_car_payment['WhetherMortgage'].map({True: 1, False: 0})
new_car_payment_final = new_car_payment[['ChassisNumber', 'WhetherMortgage']].groupby('ChassisNumber').max()


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
    license_dummies = pd.get_dummies(lis_num['license_num'], prefix='License')
    output = pd.merge(pd.DataFrame(lis_num['ChassisNumber']), license_dummies, left_index=True, right_index=True,
                        how='left')
    return output


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


#Choose transctions related to the audi brand since its corresponding data is relatively complete
voucher = voucher[voucher['CompanyCode'].isin(['BU0000353', 'BU0000354', 'BU0000355', 'BU0000356', 'BU0000357',
                                               'BU0000358', 'BU0000359', 'BU0000360', 'BU0000461'])]

car_sales = pd.merge(car_sales, voucher[['BusinessVoucherID', 'CompanyCode']], on='BusinessVoucherID', how='left')

car_sales = car_sales[car_sales['CompanyCode'].isin(['BU0000353', 'BU0000354', 'BU0000355', 'BU0000356', 'BU0000357',
                                         'BU0000358', 'BU0000359', 'BU0000360', 'BU0000461'])]

maintain = pd.merge(maintain, voucher[['BusinessVoucherID', 'CompanyCode', 'TotalAccountRecieveAmount', 'IssueDate']], on='BusinessVoucherID', how='left')

maintain = maintain[maintain['CompanyCode'].isin(['BU0000353', 'BU0000354', 'BU0000355', 'BU0000356', 'BU0000357',
                                                  'BU0000358', 'BU0000359', 'BU0000360', 'BU0000461'])]

maintain = maintain.drop([1393743])           #drop abnormal values


#Define key dates
d_2016 = datetime.datetime(2016, 7, 1)

d_2015 = datetime.datetime(2015, 7, 1)

d_2014 = datetime.datetime(2014, 7, 1)

d_2013 = datetime.datetime(2013, 7, 1)


#==============================================================================
# ##Test data distribution
# voucher_test = voucher[(voucher['IssueDate'] < d_2016) & (voucher['IssueDate'] >= d_2013)]
# maintain_test = maintain[(maintain['IssueDate'] < d_2016) & (maintain['IssueDate'] >= d_2013)]
# 
# all_repair_test = maintain_test[maintain_test['MaintainTypeCode'].isin(
#     ['WO0000001', 'WO0000002', 'WO0000011', 'WO0000012',
#      'WO0000015', 'WO0000018', 'WO0000019', 'WO0000021', 'WO0000032', 'WO0000038',   
#      'WO0000006', 'WO0000026', 'WO0000029', 'WO0000039'])]  
# all_repair_test.shape
# all_repair_test['TotalAccountRecieveAmount'].hist(bins=30)
# 
# 
# all_repair_test[all_repair_test['TotalAccountRecieveAmount'] == 0].shape
# all_repair_test[all_repair_test['TotalAccountRecieveAmount'] == 0]['MaintainTypeCode'].value_counts()
# 
# all_repair_test[all_repair_test['TotalAccountRecieveAmount'] > 40000].shape
# all_repair_test[all_repair_test['TotalAccountRecieveAmount'] > 40000]['MaintainTypeCode'].value_counts()
# 
# 
# all_repair_test['MaintainTypeCode'].value_counts()
# all_repair_test[all_repair_test['MaintainTypeCode'] == u'WO0000029']['TotalAccountRecieveAmount'].hist(bins=30)
# (all_repair_test[all_repair_test['MaintainTypeCode'] == u'WO0000029']['TotalAccountRecieveAmount'] > 60000).sum()
# 
# 
# all_repair_test[all_repair_test['MaintainTypeCode'] == u'WO0000032']['TotalAccountRecieveAmount'].hist(bins=100)
# (all_repair_test[all_repair_test['MaintainTypeCode'] == u'WO0000032']['TotalAccountRecieveAmount'] > 25000).sum()
# 
# 
# all_repair_test[all_repair_test['MaintainTypeCode'] == u'WO0000026']['TotalAccountRecieveAmount'].hist(bins=30)
# (all_repair_test[all_repair_test['MaintainTypeCode'] == u'WO0000026']['TotalAccountRecieveAmount'] > 70000).sum()
# 
# all_repair_test[all_repair_test['MaintainTypeCode'] == u'WO0000001']['TotalAccountRecieveAmount'].hist(bins=30)


# maintain_test = maintain[(maintain['IssueDate'] < d_2016) & (maintain['IssueDate'] >= d_2013)]
# maintain_test['MaintainTypeCode'].value_counts()
# normal_maintain_test = maintain_test[maintain_test['MaintainTypeCode'].isin(
#     ['WO0000003', 'WO0000004', 'WO0000017', 'WO0000022', 'WO0000028', 'WO0000040'])]
# normal_maintain_test['TotalAccountRecieveAmount'].hist(bins=300)
# normal_maintain_test['MaintainTypeCode'].value_counts()
# 
# normal_maintain_test[normal_maintain_test['TotalAccountRecieveAmount'] == 0]['MaintainTypeCode'].value_counts()
#==============================================================================


#Calculating y
voucher_16 = voucher[(voucher['IssueDate'] < d_2016) & (voucher['IssueDate'] >= d_2015)]
maintain_16 = maintain[(maintain['IssueDate'] < d_2016) & (maintain['IssueDate'] >= d_2015)]
                         
normal_maintain_16 = maintain_16[maintain_16['MaintainTypeCode'].isin(
    ['WO0000003', 'WO0000004', 'WO0000017', 'WO0000022', 'WO0000028', 'WO0000040'])]

#Handling outliers or wrong data inputs
normal_maintain_16 = normal_maintain_16[normal_maintain_16['TotalAccountRecieveAmount'] <= 12000]
#normal_maintain_16['TotalAccountRecieveAmount'].hist(bins=100)

y_normal_maintain_16 = normal_maintain_16[['ChassisNumber', 'TotalAccountRecieveAmount']].groupby('ChassisNumber').sum()
y_normal_maintain_16 = y_normal_maintain_16.rename(columns={'TotalAccountRecieveAmount': 'y_normal_maintain'})


#Handling outliers or wrong data inputs
y_normal_maintain_16 = y_normal_maintain_16[y_normal_maintain_16['y_normal_maintain'] <= 18000]
#y_normal_maintain_16.shape
#(y_normal_maintain_16['y_normal_maintain'] > 18000).sum()
#y_normal_maintain_16['y_normal_maintain'].hist(bins=100)


date_start, date_end = d_2013, d_2015
#Choose transactions between 2013 and 2016
voucher = voucher[voucher['IssueDate'].notnull()]
voucher = voucher[(voucher['IssueDate'] < date_end) & (voucher['IssueDate'] >= date_start)]

car_sales = car_sales[(car_sales['IssueDate'] < date_end) & (car_sales['IssueDate'] >= date_start)]
maintain = maintain[(maintain['IssueDate'] < date_end) & (maintain['IssueDate'] >= date_start)]


#=====Features relevant to Customer basic info====
customer = customer[['BusinessVoucherID', 'Gender', 'AgeRange', 'AttributeCode', 'CHK_RelationCompany']]
customer['CHK_RelationCompany'][customer['CHK_RelationCompany'].notnull()] = 1
customer['CHK_RelationCompany'].fillna(0, inplace=True)
customer = customer.rename(columns={'CHK_RelationCompany': 'RelationCompany'})

chassis_number_base = pd.concat([maintain[['BusinessVoucherID', 'ChassisNumber']], car_sales[['BusinessVoucherID', 'ChassisNumber']]], ignore_index=True)


chassis_profile = pd.merge(chassis_number_base, customer, on='BusinessVoucherID', how='left')
del chassis_profile['BusinessVoucherID']
chassis_profile['Gender'].fillna('UnknownGender', inplace=True)
chassis_profile = chassis_profile.drop_duplicates()
chassis_profile = chassis_profile.groupby('ChassisNumber').max()
chassis_profile = chassis_profile.rename(columns={'AgeRange': 'Age'})

chassis_profile['Gender'][chassis_profile['Gender'] == True] = 'Male'  
chassis_profile['Gender'][chassis_profile['Gender'] == False] = 'Female'
chassis_gender = pd.get_dummies(chassis_profile['Gender'])
chassis_profile = pd.merge(chassis_profile, chassis_gender, left_index=True, right_index=True)
del chassis_profile['Gender']

#==============================================================================
# chassis_profile['Birthday'] = chassis_profile['Birthday'].map(lambda x: x.month)
# chassis_profile['Birthday'].fillna('UnknownBirth', inplace=True)
# chassis_birth = pd.get_dummies(chassis_profile['Birthday'], prefix='Birth')
# chassis_profile = pd.merge(chassis_profile, chassis_birth, left_index=True, right_index=True)
# chassis_profile = chassis_profile.rename(columns={'Birth_UnknownBirth': 'UnknownBirth'})
# del chassis_profile['Birthday']
#==============================================================================

chassis_attribute = pd.get_dummies(chassis_profile['AttributeCode'], prefix='CustomerType')
chassis_profile = pd.merge(chassis_profile, chassis_attribute, left_index=True, right_index=True)
del chassis_profile['AttributeCode'], chassis_profile['CustomerType_CPR0000002'], chassis_profile['CustomerType_CPR0000003'], chassis_profile['CustomerType_CPR0000004']


voucher = pd.merge(voucher, chassis_number_base, on='BusinessVoucherID')  # remove other transctions except for maintain and car_sales

#FirstDealDuration: the duration from the first deal to now
first_deal = voucher[['ChassisNumber', 'IssueDate']].groupby('ChassisNumber').min()  
chassis_profile = pd.merge(chassis_profile, first_deal, left_index=True, right_index=True, how='left')
chassis_profile['FirstDealDuration'] = (date_end - chassis_profile['IssueDate']).map(lambda x: x.days) / 30
del chassis_profile['IssueDate']

#LastDealDuration: the duration from the last deal to now
last_deal = voucher[['ChassisNumber', 'IssueDate']].groupby('ChassisNumber').max()
chassis_profile = pd.merge(chassis_profile, last_deal, left_index=True, right_index=True, how='left')
chassis_profile['LastDealDuration'] = (date_end - chassis_profile['IssueDate']).map(lambda x: x.days) / 30
del chassis_profile['IssueDate']

#License_1.0: if a customer is in a city where the number of licenses is restricted to reduce congestion and pollution
maintain_license = license_num(maintain[['ChassisNumber', 'LicenseNumber']])
maintain_license = maintain_license.groupby('ChassisNumber').max()  #Someone may have multiple cars
car_sales_license = license_num(car_sales[['ChassisNumber', 'LicenseNumber']])
car_sales_license = car_sales_license.groupby('ChassisNumber').max()
all_license = pd.concat([maintain_license, car_sales_license])
all_license['ChassisNumber'] = all_license.index
all_license.index = np.arange(len(all_license))
all_license = all_license.drop_duplicates()
all_license = all_license.groupby('ChassisNumber').max()
chassis_profile = pd.merge(chassis_profile, all_license, left_index=True, right_index=True, how='left')
del chassis_profile['License_2.0'], chassis_profile['License_3.0']  #Only care if customers are in first-tier cities


#Series info
def series_sales(x):
    if len(str(x)) > 4:
        return x[3:5]
    return x[0:2]

car_sales['VehicleSeriesRaw'] = car_sales['VehicleSeriesRaw'].map(series_sales)


def change_series(data):
    data['VehicleSeriesRaw'][data['VehicleSeriesRaw'].isin(
        ['B6', 'B7', 'B8'])] = 'A4'
    data['VehicleSeriesRaw'][data['VehicleSeriesRaw'].isin(
        ['C5', 'C6', 'C7'])] = 'A6'
    data['VehicleSeriesRaw'][data['VehicleSeriesRaw'].isin(
        ['S3', 'S5', 'S6', 'S7', 'S8'])] = u'S'
    # data['VehicleSeriesRaw'][data['VehicleSeriesRaw'] == 'RS'] = u'其他'
    data = data.rename(columns={'VehicleSeriesRaw': 'VehicleSeriesName'})
    return data

car_sales = change_series(car_sales)


maintain['VehicleBrandRaw'][maintain['VehicleBrandRaw'] == u'奥迪老TT'] = u'奥迪TT'
maintain['VehicleBrandRaw'][maintain['VehicleBrandRaw'] == u'奥迪新A3 原'] = u'奥迪A3'
maintain['VehicleBrandRaw'][maintain['VehicleBrandRaw'] == u'奥迪A4 allro'] = u'奥迪A4'
maintain['VehicleBrandRaw'][maintain['VehicleBrandRaw'] == u'AUDI C6'] = u'奥迪C6'
maintain['VehicleBrandRaw'][maintain['VehicleBrandRaw'] == u'AUDI Q5'] = u'奥迪Q5'
maintain['VehicleBrandRaw'][maintain['VehicleBrandRaw'] == u'奥迪TTC'] = u'奥迪TT'
maintain['VehicleBrandRaw'][maintain['VehicleBrandRaw'] == u'奥迪敞篷'] = u'奥迪RS'
maintain['VehicleBrandRaw'][maintain['VehicleBrandRaw'] == '0'] = u'奥迪RS'
maintain['VehicleBrandRaw'][maintain['VehicleBrandRaw'] == u'AUDI new A3'] = u'奥迪A3'


def change_series_maintain(data):
    data['VehicleBrandRaw'][data['VehicleBrandRaw'].isin(
        ['B6', 'B7', 'B8'])] = 'A4'
    data['VehicleBrandRaw'][data['VehicleBrandRaw'].isin(
        ['C5', 'C6', 'C7'])] = 'A6'
    data['VehicleBrandRaw'][data['VehicleBrandRaw'].isin(
        ['S3', 'S5', 'S6', 'S7', 'S8'])] = u'S'
    # data['VehicleSeriesRaw'][data['VehicleSeriesRaw'] == 'RS'] = u'其他'
    data = data.rename(columns={'VehicleBrandRaw': 'VehicleSeriesName'})
    return data

maintain['VehicleBrandRaw'] = maintain['VehicleBrandRaw'].apply(lambda x: x[2:4])
maintain = change_series_maintain(maintain)


vehicle_series = pd.concat([maintain[['ChassisNumber', 'VehicleSeriesName']], car_sales[['ChassisNumber', 'VehicleSeriesName']]])
vehicle_series = vehicle_series.drop_duplicates()
vehicle_series.dropna(inplace=True)  #a car belongs to different series???
vehicle_series.drop_duplicates(['ChassisNumber'], inplace=True)
chassis_profile = pd.merge(chassis_profile, vehicle_series, left_index=True, right_on='ChassisNumber', how='left')
chassis_profile.index = np.arange(len(chassis_profile))
series_dummies = pd.get_dummies(chassis_profile['VehicleSeriesName'], prefix='Series')
chassis_profile = chassis_profile.join(series_dummies)
del chassis_profile['VehicleSeriesName']


#Car price
car_sales_amount_mean_by_series = car_sales[['VehicleSeriesName', 'AccountRecieveAmount']].groupby('VehicleSeriesName').mean()
series_to_price = dict(zip(car_sales_amount_mean_by_series.index, car_sales_amount_mean_by_series['AccountRecieveAmount']))
vehicle_series_car_price = pd.merge(vehicle_series, car_sales[['ChassisNumber', 'AccountRecieveAmount']], on='ChassisNumber', how='left')
vehicle_series_car_price['VehicleSeriesName'] = vehicle_series_car_price['VehicleSeriesName'].map(series_to_price)
vehicle_series_car_price = vehicle_series_car_price.rename(columns={'AccountRecieveAmount': 'CarPrice'})
vehicle_series_car_price['CarPrice'][vehicle_series_car_price['CarPrice'].isnull()] = vehicle_series_car_price['VehicleSeriesName'][vehicle_series_car_price['CarPrice'].isnull()]
vehicle_series_car_price.drop_duplicates(['ChassisNumber'], inplace=True)
del vehicle_series_car_price['VehicleSeriesName']
chassis_profile = pd.merge(chassis_profile, vehicle_series_car_price, on='ChassisNumber', how='left')

#Whether a customer buys a new car in Yongda
chassis_profile['WhetherBuyCarInYongda'] = 0
chassis_profile['WhetherBuyCarInYongda'][chassis_profile['ChassisNumber'].isin(new_car_payment_final.index)] = 1
chassis_profile = pd.merge(chassis_profile, new_car_payment_final, left_on='ChassisNumber', right_index=True, how='left')


#====Final feature output table====
chassis_all = copy.deepcopy(chassis_profile)



#=====Features relevant to NormalMaintain & Repair====
chassis_all['FirstNormalMaintain'] = 0     #First Normal Maintain is free when you buy a new car in Yongda distributor
chassis_all['FirstNormalMaintain'][chassis_all['ChassisNumber'].isin(maintain[maintain['MaintainTypeCode'] == 'WO0000003']['ChassisNumber'])] = 1

general_repair = maintain[maintain['MaintainTypeCode'].isin(
    ['WO0000001', 'WO0000002', 'WO0000011', 'WO0000012',
     'WO0000015', 'WO0000018', 'WO0000019', 'WO0000021', 'WO0000032', 'WO0000038'])]
incident_repair = maintain[maintain['MaintainTypeCode'].isin(['WO0000006', 'WO0000026', 'WO0000029', 'WO0000039'])]
all_repair = pd.concat([general_repair, incident_repair])

normal_maintain = maintain[maintain['MaintainTypeCode'].isin(
    ['WO0000003', 'WO0000004', 'WO0000017', 'WO0000022', 'WO0000028', 'WO0000040'])]

#Handling outliers or wrong data inputs
normal_maintain = normal_maintain[normal_maintain['TotalAccountRecieveAmount'] <= 12000] 


all_repair_by_chassis = all_repair[['BusinessVoucherID', 'ChassisNumber', 'TotalAccountRecieveAmount']].groupby('ChassisNumber')
total_num_repair = all_repair_by_chassis.count()['BusinessVoucherID']
total_num_repair = pd.DataFrame(total_num_repair).rename(columns={'BusinessVoucherID': 'TotalNumRepair'})
total_amount_repair = all_repair_by_chassis.sum()['TotalAccountRecieveAmount']
total_amount_repair = pd.DataFrame(total_amount_repair).rename(columns={'TotalAccountRecieveAmount': 'TotalAmountRepair'})
avg_amount_repair = all_repair_by_chassis.mean()['TotalAccountRecieveAmount']
avg_amount_repair = pd.DataFrame(avg_amount_repair).rename(columns={'TotalAccountRecieveAmount': 'AvgAmountRepair'})

#Define key date splitting points
d_1st_half = datetime.datetime(2014, 1, 1)
d_2nd_half = datetime.datetime(2015, 1, 1)

normal_maintain_1 = normal_maintain[(normal_maintain['IssueDate'] >= d_2013) & (normal_maintain['IssueDate'] < d_1st_half)]
normal_maintain_2 = normal_maintain[(normal_maintain['IssueDate'] >= d_1st_half) & (normal_maintain['IssueDate'] < d_2014)]
normal_maintain_3 = normal_maintain[(normal_maintain['IssueDate'] >= d_2014) & (normal_maintain['IssueDate'] < d_2nd_half)]
normal_maintain_4 = normal_maintain[(normal_maintain['IssueDate'] >= d_2nd_half) & (normal_maintain['IssueDate'] < d_2015)]


last_2_year_normal_maintain_by_chassis = normal_maintain[['BusinessVoucherID', 'ChassisNumber', 'TotalAccountRecieveAmount', 'IssueDate', 'UsedMileage']].groupby('ChassisNumber')
last_2_year_total_num_normal_maintain = last_2_year_normal_maintain_by_chassis.count()['BusinessVoucherID']
last_2_year_total_num_normal_maintain = pd.DataFrame(last_2_year_total_num_normal_maintain).rename(columns={'BusinessVoucherID': 'Last2YearTotalNumNormalMaintain'})
last_2_year_total_amount_normal_maintain = last_2_year_normal_maintain_by_chassis.sum()['TotalAccountRecieveAmount']
last_2_year_total_amount_normal_maintain = pd.DataFrame(last_2_year_total_amount_normal_maintain).rename(columns={'TotalAccountRecieveAmount': 'Last2YearTotalAmountNormalMaintain'})
last_2_year_avg_amount_normal_maintain = last_2_year_normal_maintain_by_chassis.mean()['TotalAccountRecieveAmount']
last_2_year_avg_amount_normal_maintain = pd.DataFrame(last_2_year_avg_amount_normal_maintain).rename(columns={'TotalAccountRecieveAmount': 'AvgAmountNormalMaintain'})

last_1_year_normal_maintain = pd.concat([normal_maintain_3, normal_maintain_4])
last_1_year_normal_maintain_by_chassis = last_1_year_normal_maintain[['BusinessVoucherID', 'ChassisNumber', 'TotalAccountRecieveAmount', 'IssueDate', 'UsedMileage']].groupby('ChassisNumber')
last_1_year_total_num_normal_maintain = last_1_year_normal_maintain_by_chassis.count()['BusinessVoucherID']
last_1_year_total_num_normal_maintain = pd.DataFrame(last_1_year_total_num_normal_maintain).rename(columns={'BusinessVoucherID': 'Last1YearTotalNumNormalMaintain'})
last_1_year_total_amount_normal_maintain = last_1_year_normal_maintain_by_chassis.sum()['TotalAccountRecieveAmount']
last_1_year_total_amount_normal_maintain = pd.DataFrame(last_1_year_total_amount_normal_maintain).rename(columns={'TotalAccountRecieveAmount': 'Last1YearTotalAmountNormalMaintain'})

last_half_year_normal_maintain_by_chassis = normal_maintain_4[['BusinessVoucherID', 'ChassisNumber', 'TotalAccountRecieveAmount', 'IssueDate', 'UsedMileage']].groupby('ChassisNumber')
last_half_year_total_num_normal_maintain = last_half_year_normal_maintain_by_chassis.count()['BusinessVoucherID']
last_half_year_total_num_normal_maintain = pd.DataFrame(last_half_year_total_num_normal_maintain).rename(columns={'BusinessVoucherID': 'LastHalfYearTotalNumNormalMaintain'})
last_half_year_total_amount_normal_maintain = last_half_year_normal_maintain_by_chassis.sum()['TotalAccountRecieveAmount']
last_half_year_total_amount_normal_maintain = pd.DataFrame(last_half_year_total_amount_normal_maintain).rename(columns={'TotalAccountRecieveAmount': 'LastHalfYearTotalAmountNormalMaintain'})

#Calculating the growth in the last one year
first_year_normal_maintain = pd.concat([normal_maintain_1, normal_maintain_2])
first_year_normal_maintain_by_chassis = first_year_normal_maintain[['BusinessVoucherID', 'ChassisNumber', 'TotalAccountRecieveAmount', 'IssueDate', 'UsedMileage']].groupby('ChassisNumber')
first_year_total_num_normal_maintain = first_year_normal_maintain_by_chassis.count()['BusinessVoucherID']
first_year_total_num_normal_maintain = pd.DataFrame(first_year_total_num_normal_maintain).rename(columns={'BusinessVoucherID': 'FirstYearTotalNumNormalMaintain'})
first_year_total_amount_normal_maintain = first_year_normal_maintain_by_chassis.sum()['TotalAccountRecieveAmount']
first_year_total_amount_normal_maintain = pd.DataFrame(first_year_total_amount_normal_maintain).rename(columns={'TotalAccountRecieveAmount': 'FirstYearTotalAmountNormalMaintain'})

last_1_year_growth_total_amount_normal_maintain = pd.concat([first_year_total_amount_normal_maintain, last_1_year_total_amount_normal_maintain], axis=1)
last_1_year_growth_total_amount_normal_maintain.fillna(0, inplace=True)
last_1_year_growth_total_amount_normal_maintain['Last1YearGrowthTotalAmountNormalMaintain'] = (
            last_1_year_growth_total_amount_normal_maintain['Last1YearTotalAmountNormalMaintain'] - last_1_year_growth_total_amount_normal_maintain['FirstYearTotalAmountNormalMaintain'])
del last_1_year_growth_total_amount_normal_maintain['FirstYearTotalAmountNormalMaintain'], last_1_year_growth_total_amount_normal_maintain['Last1YearTotalAmountNormalMaintain']

#Calculating the growth in the last half year
half_3rd_normal_maintain_by_chassis = normal_maintain_3[['BusinessVoucherID', 'ChassisNumber', 'TotalAccountRecieveAmount', 'IssueDate', 'UsedMileage']].groupby('ChassisNumber')
half_3rd_total_num_normal_maintain = half_3rd_normal_maintain_by_chassis.count()['BusinessVoucherID']
half_3rd_total_num_normal_maintain = pd.DataFrame(half_3rd_total_num_normal_maintain).rename(columns={'BusinessVoucherID': '3rdHalfTotalNumNormalMaintain'})
half_3rd_total_amount_normal_maintain = half_3rd_normal_maintain_by_chassis.sum()['TotalAccountRecieveAmount']
half_3rd_total_amount_normal_maintain = pd.DataFrame(half_3rd_total_amount_normal_maintain).rename(columns={'TotalAccountRecieveAmount': '3rdHalfTotalAmountNormalMaintain'})

last_half_year_growth_total_amount_normal_maintain = pd.concat([half_3rd_total_amount_normal_maintain, last_half_year_total_amount_normal_maintain], axis=1)
last_half_year_growth_total_amount_normal_maintain.fillna(0, inplace=True)
last_half_year_growth_total_amount_normal_maintain['LastHalfYearGrowthTotalAmountNormalMaintain'] = (
            last_half_year_growth_total_amount_normal_maintain['LastHalfYearTotalAmountNormalMaintain'] - last_half_year_growth_total_amount_normal_maintain['3rdHalfTotalAmountNormalMaintain'])
del last_half_year_growth_total_amount_normal_maintain['3rdHalfTotalAmountNormalMaintain'], last_half_year_growth_total_amount_normal_maintain['LastHalfYearTotalAmountNormalMaintain']

#Calculating the normal maintenance interval
normal_maintain_date_min = normal_maintain[['ChassisNumber', 'IssueDate']].groupby('ChassisNumber').min()
normal_maintain_date_max = normal_maintain[['ChassisNumber', 'IssueDate']].groupby('ChassisNumber').max()
normal_maintain_duration = (normal_maintain_date_max - normal_maintain_date_min).applymap(lambda x: x.days) / 30
normal_maintain_duration = pd.DataFrame(normal_maintain_duration).rename(columns={'IssueDate': 'NormalMaintainDuration'})
normal_maintain_duration[normal_maintain_duration['NormalMaintainDuration'] == 0] = np.nan
normal_maintain_interval = normal_maintain_duration['NormalMaintainDuration'] / (last_2_year_total_num_normal_maintain['Last2YearTotalNumNormalMaintain'] - 1)
normal_maintain_interval.name = 'NormalMaintainInterval'
normal_maintain_interval = pd.DataFrame(normal_maintain_interval)


maintain_features = pd.concat([total_num_repair, total_amount_repair, avg_amount_repair,
                               last_2_year_total_num_normal_maintain, last_2_year_total_amount_normal_maintain, last_2_year_avg_amount_normal_maintain,
                               last_1_year_total_num_normal_maintain, last_1_year_total_amount_normal_maintain,
                               last_half_year_total_num_normal_maintain, last_half_year_total_amount_normal_maintain,
                               last_1_year_growth_total_amount_normal_maintain, last_half_year_growth_total_amount_normal_maintain,
                               normal_maintain_interval], axis=1)


#Joined to the final output table
chassis_all = pd.merge(chassis_all, maintain_features, left_on='ChassisNumber', right_index=True, how='left')

chassis_all['AvgAmountNormalMaintainOverCarPrice'] = chassis_all['AvgAmountNormalMaintain'] / chassis_all['CarPrice']



#Calcuating features relevant to the mileages of a car
first_deal_mileage = maintain[['ChassisNumber', 'UsedMileage']].groupby('ChassisNumber').min()
first_deal_mileage = first_deal_mileage.rename(columns={'UsedMileage': 'FirstDealMileage'})
last_deal_mileage = maintain[['ChassisNumber', 'UsedMileage']].groupby('ChassisNumber').max()
last_deal_mileage = last_deal_mileage.rename(columns={'UsedMileage': 'LastDealMileage'})
avg_deal_mileage = maintain[['ChassisNumber', 'UsedMileage']].groupby('ChassisNumber').mean()
avg_deal_mileage = avg_deal_mileage.rename(columns={'UsedMileage': 'AvgDealMileage'})

mileage_features = pd.concat([first_deal_mileage, last_deal_mileage, avg_deal_mileage], axis=1)
chassis_all = pd.merge(chassis_all, mileage_features, left_on='ChassisNumber', right_index=True, how='left')


#Save the original unfilled final table
chassis_all.to_csv('chassis_all_features_1315_unfilled.csv', index=False)


#===========Handling missing values===========
age_value_filled = chassis_all[chassis_all['Age'].notnull()]['Age'].map(int).mean()
chassis_all['Age'].fillna(age_value_filled, inplace=True)
chassis_all['Age'] = chassis_all['Age'].astype(int)       

chassis_all['Last2YearTotalNumNormalMaintain'].fillna(0, inplace=True)
chassis_all['Last2YearTotalAmountNormalMaintain'].fillna(0, inplace=True)
chassis_all['AvgAmountNormalMaintain'].fillna(0, inplace=True)
chassis_all['AvgAmountNormalMaintainOverCarPrice'].fillna(0, inplace=True)
chassis_all['Last1YearTotalNumNormalMaintain'].fillna(0, inplace=True)
chassis_all['Last1YearTotalAmountNormalMaintain'].fillna(0, inplace=True)
chassis_all['LastHalfYearTotalNumNormalMaintain'].fillna(0, inplace=True)
chassis_all['LastHalfYearTotalAmountNormalMaintain'].fillna(0, inplace=True)

chassis_all['Last1YearGrowthTotalAmountNormalMaintain'].fillna(0, inplace=True)
chassis_all['LastHalfYearGrowthTotalAmountNormalMaintain'].fillna(0, inplace=True)

chassis_all['TotalNumRepair'].fillna(0, inplace=True)
chassis_all['TotalAmountRepair'].fillna(0, inplace=True)
chassis_all['AvgAmountRepair'].fillna(0, inplace=True)

chassis_all['LastDealMileage'].fillna(0, inplace=True)
chassis_all['FirstDealMileage'].fillna(0, inplace=True)
chassis_all['AvgDealMileage'].fillna(0, inplace=True)

#Handling NormalMaintainInterval null values
chassis_all['NormalMaintainInterval'][
    (chassis_all['FirstDealDuration'] <= 6) & 
    (chassis_all['Last2YearTotalNumNormalMaintain'] == 0)] = chassis_all['NormalMaintainInterval'].median()
chassis_all['NormalMaintainInterval'][
    (chassis_all['FirstDealDuration'] > 6) & 
    (chassis_all['FirstDealDuration'] <= 12) & 
    (chassis_all['Last2YearTotalNumNormalMaintain'] == 0)] = 9
chassis_all['NormalMaintainInterval'][(chassis_all['FirstDealDuration'] > 12) & (chassis_all['Last2YearTotalNumNormalMaintain'] == 0)] = 24
chassis_all['NormalMaintainInterval'][(chassis_all['Last2YearTotalNumNormalMaintain'] == 1) & (chassis_all['FirstDealDuration'] > 12)] = 24
chassis_all['NormalMaintainInterval'][(chassis_all['Last2YearTotalNumNormalMaintain'] == 1) & (chassis_all['FirstDealDuration'] <= 12)] = 8
chassis_all['NormalMaintainInterval'].fillna(chassis_all['NormalMaintainInterval'].median(), inplace=True) #several abnormal values


#Adding y
chassis_all = pd.merge(chassis_all, y_normal_maintain_16, left_on='ChassisNumber', right_index=True, how='left')

chassis_all['WhetherLostIn2016'] = np.zeros(len(chassis_all))                   #About half people are lost
chassis_all['WhetherLostIn2016'][chassis_all['y_normal_maintain'].isnull()] = 1

chassis_all['y_normal_maintain'].fillna(0, inplace=True)

#Output the final feature & y table
chassis_all.to_csv('chassis_all.csv', index=False)
