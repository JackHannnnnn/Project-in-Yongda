# Project in Yongda
This is a project I completed in Yongda group when doing 2016 summer internship in PrinceTechs . 
##  Project Backgroud
Yongda group is famous for its complete  industry chain of car sales  service . In recent years, its core business, after-sales maintenance service, has encountered  serious challenges. First, membership points in different stores can not be exchanged generally which negatively affects customers' experience. Second, there is a high user attrition rate. Finally, it can not  effectively manage its customer information and do the precision marketing from the level of the whole group.

PrinceTechs  plans to develop an integrated business management & application system which  can provide a  one-stop service to solve all above problems. The system primarily involves three models: **_Potential Value Model, User Attrition Early  Warning  Model, User Portrait Model_**. My **_role_** in this project is as follows:
  - Building User Portrait Model
  - Optimizing Potential Value Model 
  - Doing some business analysis

## Building User Portrait Model
The data of the audi brand between 2013 and 2016 in Yongda's database is relatively complete. I used this data to figure out what features were appropriate for generating the user portrait. The final features mainly consist of three kinds of information: _customer basic personal info, customer info of buying new cars in Yongda, customer maintenance consumption behavior_. The code of generating those features is in the file `user_profile_features_final.py` in the folder UserPortraitModel. The core algorithm I used to do the clustering of user features was **_KMeans_**. I also did one brief sample analysis of the radar graphs generated from user portrait features which contained the characteristics of each user group and corresponding marketing measures towards each user group. This analysis can be found in the file `user_profile_radar_graph_sample_analysis.pptx` and its original data of these radar graphs are saved in the folder RadarGraphData.

In the end, I wrote a complete and systematic function of generating user profiles  which needed to be embedded into our system delivered. The code is saved in the file `generate_user_profile.py` and  a test file called `user_profile_features_test_file.csv` can be used to test the function.

##  Optimizing Potential Value Model
The Potential Value  refers to  the value a customer creates for a company in a future period. In Yongda's case, after exploratory analysis of its database, we decided to use 2013-2015 data to generate features X and 2015-2016 data to generate the target y since these 3 years of data had better quality and values of analysis. The **_goal_** is to predict the total amount of normal maintenance of a customer in the next year based on his consumption behavior in the last two years, which is a typical regression problem.

### My contribution:
The performance of the model has been improved from a MSE(Mean Squared Error) of about _8,000,000_ to a MSE of about _6,000,000_ through creative data preprocessing & feature engineering which are as follows:
- Handling incorrect data entries 
- Adding new features like consumption behavior features related to the last half year
- More careful way of handling missing data

**25%** improvement in the model performance is not a small progress. The specifics of genearating features of Potential Value Model, data preprocessing and feature engineering can be found in the file `potential_value_model_features.py`.

### Model training:
I trained the model using Linear Regression like Lasso & Ridge Regression, Polynomial Regression, Random Forest, Gradient Boosting Tree. I used Grid Search  & Cross Validation to tune the parameters primarily. The details of the model training are included in the file `PotentialValueModelTraining.ipynb`.

## Doing Business Analysis
Sometimes I helped Program Manager and Business Analyst do certain business analysis. I list two samples in the folder BusinessAnalysis which includes two files `data_distribution_analysis.py` and `holiday_analysis.py`.
