'''
Analyze if a long holiday like Spring Festival or National Day has an impact on the maintenance interval of a customer.
'''
import pandas as pd
import numpy as np
import datetime

normal_maintain = pd.read_csv('normal_maintain.csv')

national_13_start = datetime.datetime(2013, 10, 1)
national_13_end = datetime.datetime(2013, 10, 8)
national_14_start = datetime.datetime(2014, 10, 1)
national_14_end = datetime.datetime(2014, 10, 8)
national_15_start = datetime.datetime(2015, 10, 1)
national_15_end = datetime.datetime(2015, 10, 8)
SF_13_start = datetime.datetime(2013, 2, 9)
SF_13_end = datetime.datetime(2013, 2, 16)
SF_14_start = datetime.datetime(2014, 1, 31)
SF_14_end = datetime.datetime(2014, 2, 7)
SF_15_start = datetime.datetime(2015, 2, 18)
SF_15_end = datetime.datetime(2015, 2, 25)
SF_16_start = datetime.datetime(2016, 2, 7)
SF_16_end = datetime.datetime(2016, 2, 14)


def check_date(record):
    #check if the maintain duration covers at least 5 days of a long holiday like National Day or Spring Festival
    if (record['StartDate'] <= (national_13_start + datetime.timedelta(2)) and record['EndDate'] > national_13_end) \
        or (record['StartDate'] <= national_13_start and record['EndDate'] > (national_13_end - datetime.timedelta(2))):
        return 1
    elif (record['StartDate'] <= (national_14_start + datetime.timedelta(2)) and record['EndDate'] > national_14_end) \
        or (record['StartDate'] <= national_14_start and record['EndDate'] > (national_14_end - datetime.timedelta(2))):
        return 1
    elif (record['StartDate'] <= (national_15_start + datetime.timedelta(2)) and record['EndDate'] > national_15_end) \
        or (record['StartDate'] <= national_15_start and record['EndDate'] > (national_15_end - datetime.timedelta(2))):
        return 1
    elif (record['StartDate'] <= (SF_14_start + datetime.timedelta(2)) and record['EndDate'] > SF_14_end) \
        or (record['StartDate'] <= SF_14_start and record['EndDate'] > (SF_14_end - datetime.timedelta(2))):
        return 1
    elif (record['StartDate'] <= (SF_15_start + datetime.timedelta(2)) and record['EndDate'] > SF_15_end) \
        or (record['StartDate'] <= SF_15_start and record['EndDate'] > (SF_15_end - datetime.timedelta(2))):
        return 1
    elif (record['StartDate'] <= (SF_16_start + datetime.timedelta(2)) and record['EndDate'] > SF_16_end) \
        or (record['StartDate'] <= SF_16_start and record['EndDate'] > (SF_16_end - datetime.timedelta(2))):
        return 1
    else:
        return 0

def check_national_date(record):
    #check if the maintain duration covers at least 5 days of National Day
    if (record['StartDate'] <= (national_13_start + datetime.timedelta(2)) and record['EndDate'] > national_13_end) \
        or (record['StartDate'] <= national_13_start and record['EndDate'] > (national_13_end - datetime.timedelta(2))):
        return 1
    elif (record['StartDate'] <= (national_14_start + datetime.timedelta(2)) and record['EndDate'] > national_14_end) \
        or (record['StartDate'] <= national_14_start and record['EndDate'] > (national_14_end - datetime.timedelta(2))):
        return 1
    elif (record['StartDate'] <= (national_15_start + datetime.timedelta(2)) and record['EndDate'] > national_15_end) \
        or (record['StartDate'] <= national_15_start and record['EndDate'] > (national_15_end - datetime.timedelta(2))):
        return 1
    else:
        return 0
        
def check_SF_date(record):
    #check if the maintain duration covers at least 5 days of Spring Festival
    if (record['StartDate'] <= (SF_14_start + datetime.timedelta(2)) and record['EndDate'] > SF_14_end) \
        or (record['StartDate'] <= SF_14_start and record['EndDate'] > (SF_14_end - datetime.timedelta(2))):
        return 1
    elif (record['StartDate'] <= (SF_15_start + datetime.timedelta(2)) and record['EndDate'] > SF_15_end) \
        or (record['StartDate'] <= SF_15_start and record['EndDate'] > (SF_15_end - datetime.timedelta(2))):
        return 1
    elif (record['StartDate'] <= (SF_16_start + datetime.timedelta(2)) and record['EndDate'] > SF_16_end) \
        or (record['StartDate'] <= SF_16_start and record['EndDate'] > (SF_16_end - datetime.timedelta(2))):
        return 1
    else:
        return 0
    

holiday_analysis = normal_maintain[['ChassisNumber', 'UsedMileage', 'IssueDate']]
def holiday_normal_maintain_analysis(check_date_func):
    customer_data = []
    for chassis_num, data in holiday_analysis.groupby('ChassisNumber'):
        size = data.shape[0]
        if size > 2:
            data = data.sort_index(by='IssueDate')
            new_data = pd.DataFrame(np.array(data['ChassisNumber'][:(size - 1)]), columns=['ChassisNumber'])
            new_data['MileageDiff'] = np.array(data['UsedMileage'][1:size]) - np.array(data['UsedMileage'][:(size - 1)])
            new_data['StartDate'] = np.array(data['IssueDate'][:(size - 1)])
            new_data['EndDate'] = np.array(data['IssueDate'][1:size])
            new_data['Duration'] = (new_data['EndDate'] - new_data['StartDate']).map(lambda x: x.days)
            new_data['MileageEachDay'] = new_data['MileageDiff'] / new_data['Duration']
            is_covered = []     
            for i in xrange(len(new_data)):
                is_covered.append(check_date_func(new_data.ix[i]))
            new_data['IsCovered'] = is_covered
            if len(new_data['IsCovered'].unique()) > 1:
                customer_data.append(new_data)
                                            
    holiday_data = pd.concat(customer_data, ignore_index=True)  
    holiday_data.dropna(inplace=True)
    
    holiday_data = holiday_data[holiday_data['MileageEachDay'] > 0]
    holiday_data = holiday_data[holiday_data['Duration'] > 0]
    
    normal_maintain_interval_by_car = holiday_data.groupby('ChassisNumber')['Duration'].mean()
    normal_maintain_interval_by_car = normal_maintain_interval_by_car[normal_maintain_interval_by_car <= 270]
    holiday_data = holiday_data[holiday_data['ChassisNumber'].isin(normal_maintain_interval_by_car.index)]
    
    num_normal_maintan_interval_by_car = holiday_data.groupby('ChassisNumber').size()
    num_normal_maintan_interval_by_car = num_normal_maintan_interval_by_car[num_normal_maintan_interval_by_car > 3]
    holiday_data = holiday_data[holiday_data['ChassisNumber'].isin(num_normal_maintan_interval_by_car.index)]
    
    holiday_covered = holiday_data[holiday_data['IsCovered'] == 1]
    holiday_not_covered = holiday_data[holiday_data['IsCovered'] == 0]
    holiday_covered_by_customer = holiday_covered.groupby('ChassisNumber')[['MileageEachDay']].mean()
    holiday_covered_by_customer = holiday_covered_by_customer.rename(columns={'MileageEachDay': 'MileageEachDayCovered' })
    holiday_not_covered_by_customer = holiday_not_covered.groupby('ChassisNumber')[['MileageEachDay']].mean()
    holiday_not_covered_by_customer = holiday_not_covered_by_customer.rename(columns={'MileageEachDay': 'MileageEachDayNotCovered' })
    
    holiday_mileage_growth = pd.merge(holiday_covered_by_customer, holiday_not_covered_by_customer, left_index=True, right_index=True)
    holiday_mileage_growth['GrowthRate'] = (holiday_mileage_growth['MileageEachDayCovered'] - holiday_mileage_growth['MileageEachDayNotCovered']) \
                                            / holiday_mileage_growth['MileageEachDayNotCovered']
    
    mean_growth_rate = holiday_mileage_growth['GrowthRate'].mean()
    positive_growth_rate_mean = holiday_mileage_growth['GrowthRate'][holiday_mileage_growth['GrowthRate'] > 0].mean()
    negative_growth_rate_mean = holiday_mileage_growth['GrowthRate'][holiday_mileage_growth['GrowthRate'] < 0].mean()
    ratio_positive_growth_to_total = (holiday_mileage_growth['GrowthRate'] > 0).sum() / float(holiday_mileage_growth.shape[0])
    
    output = {}
    output['data'] = holiday_data
    output['holiday_mileage_growth'] = holiday_mileage_growth
    output['mean_growth_rate'] = mean_growth_rate
    output['positive_growth_rate_mean'] = positive_growth_rate_mean
    output['negative_growth_rate_mean'] = negative_growth_rate_mean
    output['ratio_positive_growth_to_total'] = ratio_positive_growth_to_total
    
    return output

total_holiday_result = holiday_normal_maintain_analysis(check_date)
national_holiday_result = holiday_normal_maintain_analysis(check_national_date)
SF_holiday_result = holiday_normal_maintain_analysis(check_SF_date)

del total_holiday_result['data'], national_holiday_result['data'], SF_holiday_result['data']
del total_holiday_result['holiday_mileage_growth'], national_holiday_result['holiday_mileage_growth'], SF_holiday_result['holiday_mileage_growth']
all_result = pd.DataFrame([total_holiday_result, national_holiday_result, SF_holiday_result],  
                            index=['total_holiday_result', 'national_holiday_result', 'SF_holiday_result'])
all_result.to_csv('all_result.csv')
