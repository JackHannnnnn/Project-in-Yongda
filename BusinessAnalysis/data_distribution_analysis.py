# -*- coding: utf-8 -*-
"""
Created on Mon Sep 05 16:30:25 2016

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


#====================All brand analysis=======================

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
def license_num(lis_num):   #need changes
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
    del lis_num['license_number_2']



#Define key dates
d_2016 = datetime.datetime(2016, 7, 1)
d_2013 = datetime.datetime(2013, 7, 1)

#Choose transactions between 2013 and 2016
voucher = voucher[voucher['IssueDate'].notnull()]
voucher = voucher[(voucher['IssueDate'] <= d_2016) & (voucher['IssueDate'] >= d_2013)]

#Choose observations between 2013 and 2016
#customer = customer[customer['BusinessVoucherID'].isin(voucher['BusinessVoucherID'])]
car_sales = car_sales[car_sales['BusinessVoucherID'].isin(voucher['BusinessVoucherID'])]
maintain = maintain[maintain['BusinessVoucherID'].isin(voucher['BusinessVoucherID'])]


company_to_brand = pd.read_csv('companybrand_v2.csv', encoding='utf-8')
company_to_brand.columns = ['CompanyCode', 'BrandName']
company_to_brand[company_to_brand['BrandName'] == u'奥迪']['CompanyCode']
voucher['BrandName'] = np.nan
voucher['BrandName'][voucher['CompanyCode'].isin(company_to_brand[company_to_brand['BrandName'] == u'奥迪']['CompanyCode'])] = u'奥迪'
voucher['BrandName'][voucher['CompanyCode'].isin(company_to_brand[company_to_brand['BrandName'] == u'一汽大众']['CompanyCode'])] = u'一汽大众'
voucher['BrandName'][voucher['CompanyCode'].isin(company_to_brand[company_to_brand['BrandName'] == u'宝马']['CompanyCode'])] = u'宝马'
voucher['BrandName'][voucher['CompanyCode'].isin(company_to_brand[company_to_brand['BrandName'] == u'别克']['CompanyCode'])] = u'别克'
voucher['BrandName'][voucher['CompanyCode'].isin(company_to_brand[company_to_brand['BrandName'] == u'MINI']['CompanyCode'])] = u'MINI'
voucher['BrandName'][voucher['CompanyCode'].isin(company_to_brand[company_to_brand['BrandName'] == u'雪佛兰']['CompanyCode'])] = u'雪佛兰'
voucher['BrandName'][voucher['CompanyCode'].isin(company_to_brand[company_to_brand['BrandName'] == u'上海大众']['CompanyCode'])] = u'上海大众'
voucher['BrandName'][voucher['CompanyCode'].isin(company_to_brand[company_to_brand['BrandName'] == u'斯柯达']['CompanyCode'])] = u'斯柯达'
voucher['BrandName'][voucher['CompanyCode'].isin(company_to_brand[company_to_brand['BrandName'] == u'沃尔沃']['CompanyCode'])] = u'沃尔沃'
voucher['BrandName'][voucher['CompanyCode'].isin(company_to_brand[company_to_brand['BrandName'] == u'荣威']['CompanyCode'])] = u'荣威'
voucher['BrandName'][voucher['CompanyCode'].isin(company_to_brand[company_to_brand['BrandName'] == u'英菲尼迪']['CompanyCode'])] = u'英菲尼迪'
voucher['BrandName'][voucher['CompanyCode'].isin(company_to_brand[company_to_brand['BrandName'] == u'捷豹路虎']['CompanyCode'])] = u'捷豹路虎'
voucher['BrandName'][voucher['CompanyCode'].isin(company_to_brand[company_to_brand['BrandName'] == u'凯迪拉克']['CompanyCode'])] = u'凯迪拉克'
voucher['BrandName'][voucher['CompanyCode'].isin(company_to_brand[company_to_brand['BrandName'] == u'福特']['CompanyCode'])] = u'福特'
voucher.dropna(subset=['BrandName'], inplace=True)


car_sales = pd.merge(car_sales, voucher[['BusinessVoucherID', 'BrandName']], on='BusinessVoucherID', how='left')
maintain = pd.merge(maintain, voucher[['BusinessVoucherID', 'BrandName']], on='BusinessVoucherID', how='left')
car_sales.dropna(subset=['BrandName'], inplace=True)
maintain.dropna(subset=['BrandName'], inplace=True)

car_sales.dropna(subset=['ChassisNumber'], inplace=True)
car_sales.drop_duplicates(['ChassisNumber'], inplace=True)
new_car_chassis = car_sales['ChassisNumber']

#Output the new car sales distribution by brand
car_sales['BrandName'].value_counts().to_csv('new_car_brand_distribution.csv')

no_new_car_maintain = maintain[~maintain['ChassisNumber'].isin(new_car_chassis)]
no_new_car_chassis = no_new_car_maintain['ChassisNumber'].unique()
no_new_car_brand = no_new_car_maintain[['ChassisNumber', 'BrandName']].drop_duplicates(['ChassisNumber'])

#Output the sales distribution of customers not buying new cars in Yongda by brand
no_new_car_brand['BrandName'].value_counts().to_csv('no_new_car_brand_distribution.csv')


general_repair = maintain[maintain['MaintainTypeCode'].isin(
    ['WO0000001', 'WO0000002', 'WO0000011', 'WO0000012',
     'WO0000015', 'WO0000018', 'WO0000019', 'WO0000021', 'WO0000032', 'WO0000038'])]
incident_repair = maintain[maintain['MaintainTypeCode'].isin(['WO0000006', 'WO0000026', 'WO0000029', 'WO0000039'])]
all_repair = pd.concat([general_repair, incident_repair])

normal_maintain = maintain[maintain['MaintainTypeCode'].isin(
    ['WO0000003', 'WO0000004', 'WO0000017', 'WO0000022', 'WO0000028', 'WO0000040'])]      
    
#Output the distribution of the number of normal maintenance & repair of buying new cars in Yongda
new_car_num_normal_maintain = new_car_chassis.isin(normal_maintain['ChassisNumber'].unique()).sum()
new_car_num_all_repair = new_car_chassis.isin(all_repair['ChassisNumber'].unique()).sum()
new_car_num_normal_maintain_repair = (new_car_chassis.isin(normal_maintain['ChassisNumber'].unique()) | new_car_chassis.isin(all_repair['ChassisNumber'].unique())).sum()
new_car_num_no_normal_maintain_repair = new_car_chassis.shape[0] - new_car_num_normal_maintain_repair

#Output the distribution of the number of normal maintenance & repair of not buying new cars in Yongda
no_new_car_num_normal_maintain = normal_maintain['ChassisNumber'].unique().shape[0] - new_car_num_normal_maintain
no_new_car_num_all_repair = all_repair['ChassisNumber'].unique().shape[0] - new_car_num_all_repair
no_new_car_num_normal_maintain_repair = pd.concat([all_repair, normal_maintain])['ChassisNumber'].unique().shape[0] - new_car_num_normal_maintain_repair
no_new_car_no_normal_maintain_repair = no_new_car_chassis.shape[0] - no_new_car_num_normal_maintain_repair

#Output the  distribution of license city tiers of buying new cars & not buying new cars
license_num(maintain)
license_num(no_new_car_maintain)
license_num(car_sales)
car_sales['license_num'].value_counts()
no_new_car_maintain[['ChassisNumber', 'license_num']].drop_duplicates(['ChassisNumber'])['license_num'].value_counts()


#====================Audi analysis=======================
audi_voucher = voucher[voucher['BrandName'] == u'奥迪']
audi_maintain = maintain[maintain['BrandName'] ==  u'奥迪']
audi_car_sales = car_sales[car_sales['BrandName'] ==  u'奥迪']


##Series info
def series_sales(x):
    if len(str(x)) > 4:
        return x[3:5]
    return x[0:2]


audi_car_sales['VehicleSeriesRaw'] = audi_car_sales['VehicleSeriesRaw'].map(series_sales)

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

audi_car_sales = change_series(audi_car_sales)


audi_maintain['VehicleBrandRaw'][audi_maintain['VehicleBrandRaw'] == u'奥迪老TT'] = u'奥迪TT'
audi_maintain['VehicleBrandRaw'][audi_maintain['VehicleBrandRaw'] == u'奥迪新A3 原'] = u'奥迪A3'
audi_maintain['VehicleBrandRaw'][audi_maintain['VehicleBrandRaw'] == u'奥迪A4 allro'] = u'奥迪A4'
audi_maintain['VehicleBrandRaw'][audi_maintain['VehicleBrandRaw'] == u'AUDI C6'] = u'奥迪C6'
audi_maintain['VehicleBrandRaw'][audi_maintain['VehicleBrandRaw'] == u'AUDI Q5'] = u'奥迪Q5'
audi_maintain['VehicleBrandRaw'][audi_maintain['VehicleBrandRaw'] == u'奥迪TTC'] = u'奥迪TT'
audi_maintain['VehicleBrandRaw'][audi_maintain['VehicleBrandRaw'] == u'奥迪敞篷'] = u'奥迪RS'
audi_maintain['VehicleBrandRaw'][audi_maintain['VehicleBrandRaw'] == '0'] = u'奥迪RS'
audi_maintain['VehicleBrandRaw'][audi_maintain['VehicleBrandRaw'] == u'AUDI new A3'] = u'奥迪A3'


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


audi_maintain['VehicleBrandRaw'] = audi_maintain['VehicleBrandRaw'].apply(lambda x: x[2:4])
audi_maintain = change_series_maintain(audi_maintain)

#Output the audi sales distribution by Series
audi_car_sales['VehicleSeriesName'].value_counts()


#Output the distribution of the number of normal maintenance & repair of buying new cars & not buying new cars in Yongda
audi_new_car_chassis = audi_car_sales['ChassisNumber']
audi_no_new_car_maintain = audi_maintain[~audi_maintain['ChassisNumber'].isin(audi_new_car_chassis)]
audi_no_new_car_chassis = audi_no_new_car_maintain['ChassisNumber'].unique()
audi_no_new_car_maintain[['ChassisNumber', 'VehicleSeriesName']].groupby('VehicleSeriesName').agg(lambda x: x.unique().shape[0])

audi_all_repair = all_repair[all_repair['BrandName'] == u'奥迪']  
audi_normal_maintain = normal_maintain[normal_maintain['BrandName'] == u'奥迪']

audi_new_car_num_normal_maintain = audi_new_car_chassis.isin(audi_normal_maintain['ChassisNumber'].unique()).sum()
audi_new_car_num_all_repair = audi_new_car_chassis.isin(audi_all_repair['ChassisNumber'].unique()).sum()
audi_new_car_num_normal_maintain_repair = (audi_new_car_chassis.isin(audi_normal_maintain['ChassisNumber'].unique()) | audi_new_car_chassis.isin(audi_all_repair['ChassisNumber'].unique())).sum()
audi_new_car_num_no_normal_maintain_repair = audi_new_car_chassis.shape[0] - audi_new_car_num_normal_maintain_repair

audi_no_new_car_num_normal_maintain = audi_normal_maintain['ChassisNumber'].unique().shape[0] - audi_new_car_num_normal_maintain
audi_no_new_car_num_all_repair = audi_all_repair['ChassisNumber'].unique().shape[0] - audi_new_car_num_all_repair
audi_no_new_car_num_normal_maintain_repair = pd.concat([audi_all_repair, audi_normal_maintain])['ChassisNumber'].unique().shape[0] - audi_new_car_num_normal_maintain_repair
audi_no_new_car_no_normal_maintain_repair = audi_no_new_car_chassis.shape[0] - audi_no_new_car_num_normal_maintain_repair

#Output the audi sales distribution by store
audi_car_sales = pd.merge(audi_car_sales, audi_voucher[['BusinessVoucherID', 'CompanyCode']], on='BusinessVoucherID', how='left')
audi_car_sales['CompanyCode'].value_counts().sum()

#Output the audi sales distribution of license city tiers of buying new cars & not buying new cars in Yongda
license_num(audi_maintain)
license_num(audi_no_new_car_maintain)
license_num(audi_car_sales)
audi_car_sales['license_num'].value_counts()
audi_no_new_car_maintain[['ChassisNumber', 'license_num']].drop_duplicates(['ChassisNumber'])['license_num'].value_counts()
