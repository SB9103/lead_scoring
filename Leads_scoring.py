#!/usr/bin/env python
# coding: utf-8

# # Lead Scoring

# ### Problem Statement
# An education company named X Education sells online courses to industry professionals. On any given day, many professionals who are interested in the courses land on their website and browse for courses.
# 
# The company markets its courses on several websites and search engines like Google. Once these people land on the website, they might browse the courses or fill up a form for the course or watch some videos. When these people fill up a form providing their email address or phone number, they are classified to be a lead. Moreover, the company also gets leads through past referrals. Once these leads are acquired, employees from the sales team start making calls, writing emails, etc. Through this process, some of the leads get converted while most do not. The typical lead conversion rate at X education is around 30%.
# 
# Now, although X Education gets a lot of leads, its lead conversion rate is very poor. For example, if, say, they acquire 100 leads in a day, only about 30 of them are converted. To make this process more efficient, the company wishes to identify the most potential leads, also known as ‘Hot Leads’. If they successfully identify this set of leads, the lead conversion rate should go up as the sales team will now be focusing more on communicating with the potential leads rather than making calls to everyone. A typical lead conversion process can be represented using the following funnel:
# 
# ![image.png](attachment:image.png)
# 
#                            Lead Conversion Process - Demonstrated as a funnel
# As you can see, there are a lot of leads generated in the initial stage (top) but only a few of them come out as paying customers from the bottom. In the middle stage, you need to nurture the potential leads well (i.e. educating the leads about the product, constantly communicating etc. ) in order to get a higher lead conversion.
# 
# X Education has appointed you to help them select the most promising leads, i.e. the leads that are most likely to convert into paying customers.The company requires you to build a model wherein you need to assign a lead score to each of the leads such that the customers with higher lead score have a higher conversion chance and the customers with lower lead score have a lower conversion chance.The CEO, in particular, has given a ballpark of the target lead conversion rate to be around 80%.
# 
# Data
# You have been provided with a leads dataset from the past with around 9000 data points. This dataset consists of various attributes such as Lead Source, Total Time Spent on Website, Total Visits, Last Activity, etc. which may or may not be useful in ultimately deciding whether a lead will be converted or not. The target variable, in this case, is the column ‘Converted’ which tells whether a past lead was converted or not wherein 1 means it was converted and 0 means it wasn’t converted.
# 
# Another thing that you also need to check out for are the levels present in the categorical variables.
# 
# Many of the categorical variables have a level called 'Select' which needs to be handled because it is as good as a null value.
# 
# Goal
# There are quite a few goals for this case study.
# 
# Build a logistic regression model to assign a lead score between 0 and 100 to each of the leads which can be used by the company to target potential leads. A higher score would mean that the lead is hot, i.e. is most likely to convert whereas a lower score would mean that the lead is cold and will mostly not get converted.

# In[1]:


##Importing Pandas and NumPy
import pandas as pd, numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')


# In[7]:


df = pd.read_csv("Leads.csv")
df.head()


# ## Data Inspection

# In[9]:


# checking the shape of the data 
df.shape


# In[10]:


# checking non null count and datatype of the variables
df.info()


# #### All the dataypes of the variables are in correct format.

# In[11]:


# Describing data
df.describe()


# ##### From above description about counts, we can see that there are missing values present in our data.

# ### Data Cleaning

# In[13]:


# Converting 'Select' values to NaN.
df = df.replace('Select', np.nan)


# In[14]:


# checking the columns for null values
df.isnull().sum()


# In[15]:


# Finding the null percentages across columns
round(df.isnull().sum()/len(df.index),2)*100


# We see that for some columns we have high percentage of missing values. We can drop the columns with missing values greater than 40% and those whoes are less importent for our model.

# In[17]:


# dropping the columns with missing values greater than or equal to 40% .
df=df.drop(columns=['How did you hear about X Education','Lead Quality','Lead Profile',
                                  'Asymmetrique Activity Index','Asymmetrique Profile Index','Asymmetrique Activity Score',
                                 'Asymmetrique Profile Score'])


# In[18]:


# Finding the null percentages across columns after removing the above columns
round(df.isnull().sum()/len(df.index),2)*100


# 1) Column: 'Specialization'
# This column has 37% missing values

# In[21]:


plt.figure(figsize=(10,5))
sns.countplot(df['Specialization'])
plt.xticks(rotation=90)


# There is 37% missing values present in the Specialization column .It may be possible that the lead may leave this column blank if he may be a student or not having any specialization or his specialization is not there in the options given. So we can create a another category 'Others' for this.

# In[22]:


# Creating a separate category called 'Others' for this 
df['Specialization'] = df['Specialization'].replace(np.nan, 'Others')


# 2) Tags column
# 'Tags' column has 36% missing values

# In[24]:


# Visualizing Tags column
plt.figure(figsize=(10,5))
sns.countplot(df['Tags'])
plt.xticks(rotation=90)


# Since most values are 'Will revert after reading the email' , we can impute missing values in this column with this value.

# In[25]:


# Imputing the missing data in the tags column with 'Will revert after reading the email'
df['Tags']=df['Tags'].replace(np.nan,'Will revert after reading the email')


# 3) Column: 'What matters most to you in choosing a course'
# this column has 29% missing values

# In[27]:


# Visualizing this column
sns.countplot(df['What matters most to you in choosing a course'])
plt.xticks(rotation=90)


# In[28]:


# Finding the percentage of the different categories of this column:
round(df['What matters most to you in choosing a course'].value_counts(normalize=True),2)*100


# We can see that this is highly skewed column so we can remove this column.

# In[29]:


# Dropping this column 
df=df.drop('What matters most to you in choosing a course',axis=1)


# 4) Column: 'What is your current occupation'
# 
# this column has 29% missing values

# In[30]:


sns.countplot(df['What is your current occupation'])
plt.xticks(rotation=20)


# In[32]:


# Finding the percentage of the different categories of this column:
round(df['What is your current occupation'].value_counts(normalize=True),2)*100


# Since the most values are 'Unemployed' , we can impute missing values in this column with this value.

# In[33]:


# Imputing the missing data in the 'What is your current occupation' column with 'Unemployed'
df['What is your current occupation']=df['What is your current occupation'].replace(np.nan,'Unemployed')


# 5) Column: 'Country'
# 
# This column has 27% missing values

# In[34]:


plt.figure(figsize=(10,5))
sns.countplot(df['Country'])
plt.xticks(rotation=90)


# We can see that this is highly skewed column but it is an important information w.r.t. to the lead. Since most values are 'India' , we can impute missing values in this column with this value.

# In[35]:


# Imputing the missing data in the 'Country' column with 'India'
df['Country']=df['Country'].replace(np.nan,'India')


# 6) Column: 'City'
#     
# This column has 40% missing values

# In[40]:


plt.figure(figsize=(5,5))
sns.countplot(df['City'])
plt.xticks(rotation=45
          )


# In[41]:


# Finding the percentage of the different categories of this column:
round(df['City'].value_counts(normalize=True),2)*100


# Since most values are 'Mumbai' , we can impute missing values in this column with this value.

# In[42]:


# Imputing the missing data in the 'City' column with 'Mumbai'
df['City']=df['City'].replace(np.nan,'Mumbai')


# In[43]:


# Finding the null percentages across columns after removing the above columns
round(df.isnull().sum()/len(df.index),2)*100


# Rest missing values are under 2% so we can drop these rows.

# In[44]:


# Dropping the rows with null values
df.dropna(inplace = True)


# In[45]:


# Finding the null percentages across columns after removing the above columns
round(df.isnull().sum()/len(df.index),2)*100


# Now we don't have any missing value in the dataset.

# #### We can find the percentage of rows retained.

# In[46]:


# Percentage of rows retained 
(len(df.index)/9240)*100


# #### We have retained 98% of the rows after cleaning the data 

# ### Exploratory Data Anaysis

# #### Checking for duplicates:

# In[48]:


df[df.duplicated()]


# We see there are no duplicate records in our lead dataset.

# ### Univariate Analysis and Bivariate Analysis

# #### 1) Converted

# Converted is the target variable, Indicates whether a lead has been successfully converted (1) or not (0)

# In[49]:


Converted = (sum(df['Converted'])/len(df['Converted'].index))*100
Converted


# The lead conversion rate is 38%

# #### 2) Lead Origin

# In[51]:


plt.figure(figsize=(10,5))
sns.countplot(x = "Lead Origin", hue = "Converted", data = df,palette='Set1')
plt.xticks(rotation = 0)


# #### Inference :
# 1.API and Landing Page Submission have 30-35% conversion rate but count of lead originated from them are considerable.
# 
# 2.Lead Add Form has more than 90% conversion rate but count of lead are not very high.
# 
# 3.Lead Import are very less in count.
# 
# To improve overall lead conversion rate, we need to focus more on improving lead converion of API and Landing Page Submission origin and generate more leads from Lead Add Form.

# #### 3) Lead Source

# In[52]:


plt.figure(figsize=(10,5))
sns.countplot(x = "Lead Source", hue = "Converted", data = df, palette='Set1')
plt.xticks(rotation = 90)


# In[53]:


# Need to replace 'google' with 'Google'
df['Lead Source'] = df['Lead Source'].replace(['google'], 'Google')


# In[54]:


# Creating a new category 'Others' for some of the Lead Sources which do not have much values.
df['Lead Source'] = df['Lead Source'].replace(['Click2call', 'Live Chat', 'NC_EDM', 'Pay per Click Ads', 'Press_Release',
  'Social Media', 'WeLearn', 'bing', 'blog', 'testone', 'welearnblog_Home', 'youtubechannel'], 'Others')


# In[55]:


# Visualizing again
plt.figure(figsize=(10,5))
sns.countplot(x = "Lead Source", hue = "Converted", data = df,palette='Set1')
plt.xticks(rotation = 90)


# #### Inference
# 1.Google and Direct traffic generates maximum number of leads.
# 
# 2.Conversion Rate of reference leads and leads through welingak website is high.
# 
# To improve overall lead conversion rate, focus should be on improving lead converion of olark chat, organic search, direct traffic, and google leads and generate more leads from reference and welingak website.

# #### 4) Do not Emai

# In[59]:


plt.figure(figsize=(5,5))
sns.countplot(x = "Do Not Email", hue = "Converted", data = df,palette='Set1')
plt.xticks(rotation = 0)


# #### Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.

# #### 5) Do not call

# In[61]:


plt.figure(figsize=(5,5))
sns.countplot(x = "Do Not Call", hue = "Converted", data = df,palette='Set1')
plt.xticks(rotation = 0)


# #### Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.

# #### 6) TotalVisits

# In[62]:


df['TotalVisits'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])


# In[63]:


sns.boxplot(df['TotalVisits'],orient='vert',palette='Set1')


# As we can see there are a number of outliers in the data. We will cap the outliers to 95% value for analysis.

# In[64]:


percentiles = df['TotalVisits'].quantile([0.05,0.95]).values
df['TotalVisits'][df['TotalVisits'] <= percentiles[0]] = percentiles[0]
df['TotalVisits'][df['TotalVisits'] >= percentiles[1]] = percentiles[1]


# In[65]:


# Visualizing again
sns.boxplot(df['TotalVisits'],orient='vert',palette='Set1')


# In[66]:


sns.boxplot(y = 'TotalVisits', x = 'Converted', data = df
            ,palette='Set1')


# #### Inference
#  1.Median for converted and not converted leads are the same.
# 
# Nothing can be concluded on the basis of Total Visits.
# 
# ### 7) Total Time Spent on Website

# In[67]:


df['Total Time Spent on Website'].describe()


# In[68]:


sns.boxplot(df['Total Time Spent on Website'],orient='vert',palette='Set1')


# In[70]:


sns.boxplot(y = 'Total Time Spent on Website', x = 'Converted', data = df,palette='Set1')


# #### Inference
# Leads spending more time on the weblise are more likely to be converted.
# Website should be made more engaging to make leads spend more time.
# 
# #### 8) Page Views Per Visit

# In[71]:


df['Page Views Per Visit'].describe()


# In[72]:


sns.boxplot(df['Page Views Per Visit'],orient='vert',palette='Set1')


# As we can see there are a number of outliers in the data. We will cap the outliers to 95% value for analysis.

# In[73]:


percentiles = df['Page Views Per Visit'].quantile([0.05,0.95]).values
df['Page Views Per Visit'][df['Page Views Per Visit'] <= percentiles[0]] = percentiles[0]
df['Page Views Per Visit'][df['Page Views Per Visit'] >= percentiles[1]] = percentiles[1]


# In[74]:


# Visualizing again
sns.boxplot(df['Page Views Per Visit'],palette='Set1',orient='vert')


# In[75]:


sns.boxplot(y = 'Page Views Per Visit', x = 'Converted', data =df,palette='Set1')


# #### Inference
# Median for converted and unconverted leads is the same.
# 
# Nothing can be said specifically for lead conversion from Page Views Per Visit

# #### 9) Last Activity

# In[76]:


df['Last Activity'].describe()


# In[79]:


plt.figure(figsize=(10,6))
sns.countplot(x = "Last Activity", hue = "Converted", data = df,palette='Set1')
plt.xticks(rotation = 90)


# In[80]:


# We can club the last activities to "Other_Activity" which are having less data.
df['Last Activity'] = df['Last Activity'].replace(['Had a Phone Conversation', 'View in browser link Clicked', 
                                                       'Visited Booth in Tradeshow', 'Approached upfront',
                                                       'Resubscribed to emails','Email Received', 'Email Marked Spam'], 'Other_Activity')


# In[82]:


# Visualizing again
plt.figure(figsize=(10,6))
sns.countplot(x = "Last Activity", hue = "Converted", data = df,palette='Set1')
plt.xticks(rotation = 90)


# #### Inference
# Most of the lead have their Email opened as their last activity.
# 
# Conversion rate for leads with last activity as SMS Sent is almost 60%.
# 
# #### 10) Country

# In[83]:


plt.figure(figsize=(10,6))
sns.countplot(x = "Country", hue = "Converted", data = df,palette='Set1')
plt.xticks(rotation = 90)


# #### Inference
# Most values are 'India' no such inference can be drawn
# 
# #### 11) Specialization

# In[84]:


plt.figure(figsize=(10,6))
sns.countplot(x = "Specialization", hue = "Converted", data = df,palette='Set1')
plt.xticks(rotation = 90)


# #### Inference
# Focus should be more on the Specialization with high conversion rate.
# 
# #### 12) What is your current occupation

# In[89]:


plt.figure(figsize=(10,5))
sns.countplot(x = "What is your current occupation", hue = "Converted", data = df,palette='Set1')
plt.xticks(rotation = 90)


# #### Inference
# Working Professionals going for the course have high chances of joining it.
# Unemployed leads are the most in numbers but has around 30-35% conversion rate.
# #### 13) Search

# In[102]:


plt.figure(figsize=(10,5))
sns.countplot(x = "Search", hue = "Converted", data = df,palette='Set1')
plt.xticks(rotation = 0)


# #### Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.
# 
# #### 14) Magazine

# In[101]:


plt.figure(figsize=(10,5))
sns.countplot(x = "Magazine", hue = "Converted", data = df,palette='Set1')
plt.xticks(rotation = 0)


# #### Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.
# 
# #### 15) Newspaper Article

# In[100]:


plt.figure(figsize=(10,5))
sns.countplot(x = "Newspaper Article", hue = "Converted", data = df,palette='Set1')
plt.xticks(rotation = 0)


# #### Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.
# 
# #### 16) X Education Forums

# In[99]:


plt.figure(figsize=(10,5))
sns.countplot(x = "X Education Forums", hue = "Converted", data = df,palette='Set1')
plt.xticks(rotation = 0)


# #### Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.
# 
# #### 17) Newspaper

# In[105]:


plt.figure(figsize=(10,5))
sns.countplot(x = "Newspaper", hue = "Converted", data = df,palette='Set1')
plt.xticks(rotation = 0)


# #### Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.
# 
# #### 18) Digital Advertisement

# In[106]:


plt.figure(figsize=(10,5))
sns.countplot(x = "Digital Advertisement", hue = "Converted", data = df,palette='Set1')
plt.xticks(rotation = 90)


# #### Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.
# 
# #### 19) Through Recommendations

# In[108]:


plt.figure(figsize=(10,5))
sns.countplot(x = "Through Recommendations", hue = "Converted", data = df,palette='Set1')
plt.xticks(rotation =0)


# #### Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.
# 
# #### 20) Receive More Updates About Our Courses
# 

# In[110]:



sns.countplot(x = "Receive More Updates About Our Courses", hue = "Converted", data =df,palette='Set1')
plt.xticks(rotation = 0)


# #### Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.
# 
# #### 21) Tags

# In[112]:


plt.figure(figsize=(10,5))
sns.countplot(x = "Tags", hue = "Converted", data = df,palette='Set1')
plt.xticks(rotation = 90)


# #### Inference
# Since this is a column which is generated by the sales team for their analysis , so this is not available for model building . So we will need to remove this column before building the model.
# 
# #### 22) Update me on Supply Chain Content
# 

# In[114]:


sns.countplot(x = "Update me on Supply Chain Content", hue = "Converted", data = df,palette='Set1')
plt.xticks(rotation = 0)


# #### Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.
# 
# #### 23) Get updates on DM Content
# 

# In[117]:


sns.countplot(x = "Get updates on DM Content", hue = "Converted", data = df,palette='Set1')


# #### Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.
# 
# #### 24) City

# In[121]:



plt.figure(figsize=(12,5))
sns.countplot(x = "City", hue = "Converted", data = df,palette='Set1')


# #### Inference
# Most leads are from mumbai with around 50% conversion rate.
# 
# #### 25) I agree to pay the amount through cheque

# In[122]:



sns.countplot(x = "I agree to pay the amount through cheque", hue = "Converted", data = df,palette='Set1')


# #### Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.
# 
# #### 26) A free copy of Mastering The Interview

# In[123]:



sns.countplot(x = "A free copy of Mastering The Interview", hue = "Converted", data = df,palette='Set1')


# #### Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.
# 
# #### 27) Last Notable Activity

# In[125]:



plt.figure(figsize=(15,5))
sns.countplot(x = "Last Notable Activity", hue = "Converted", data = df,palette='Set1')
plt.xticks(rotation = 90)


# #### Results
# Based on the univariate analysis we have seen that many columns are not adding any information to the model, hence we can drop them for further analysis
# 
# 

# In[128]:


df = df.drop(['Lead Number','Tags','Country','Search','Magazine','Newspaper Article','X Education Forums',
                            'Newspaper','Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses',
                            'Update me on Supply Chain Content','Get updates on DM Content','I agree to pay the amount through cheque',
                            'A free copy of Mastering The Interview'],1)


# In[127]:


df.shape


# In[130]:


df.info()


# #### Data Preparation
# ##### 1) Converting some binary variables (Yes/No) to 1/0
# 

# In[131]:


vars =  ['Do Not Email', 'Do Not Call']

def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

df[vars] = df[vars].apply(binary_map)


# #### 2) Creating Dummy variables for the categorical features:
# 'Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation','City','Last Notable Activity'
# 
# 
# 

# In[132]:


# Creating a dummy variable for the categorical variables and dropping the first one.
dummy_df = pd.get_dummies(df[['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation',
                             'City','Last Notable Activity']], drop_first=True)
dummy_df.head()


# In[133]:


# Concatenating the dummy_data to the lead_data dataframe
df = pd.concat([df, dummy_df], axis=1)
df.head()


# #### Dropping the columns for which dummies were created
# 

# In[134]:


df = df.drop(['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation',
                             'City','Last Notable Activity'], axis = 1)


# In[135]:


df.head()


# #### 3) Splitting the data into train and test set.
# 

# In[136]:


from sklearn.model_selection import train_test_split

# Putting feature variable to X
X = df.drop(['Prospect ID','Converted'], axis=1)
X.head()


# In[137]:


# Putting target variable to y
y = df['Converted']

y.head()


# In[138]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# #### 4) Scaling the features
# 

# In[139]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])

X_train.head()


# In[140]:


# Checking the Lead Conversion rate
Converted = (sum(df['Converted'])/len(df['Converted'].index))*100
Converted


# We have almost 38% lead conversion rate.

# #### Feature Selection Using RFE
# 

# In[166]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.feature_selection import RFE
rfe = RFE(logreg,n_features_to_select=20)             
rfe = rfe.fit(X_train, y_train)


# In[168]:


rfe.support_


# In[169]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[170]:


# Viewing columns selected by RFE
cols = X_train.columns[rfe.support_]
cols


# ## Model Building
# 
# ### Assessing the model with StatsModels
# 
# #### Model-1

# In[171]:


import statsmodels.api as sm
X_train_sm = sm.add_constant(X_train[cols])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
result = logm1.fit()
result.summary()


# Since Pvalue of 'What is your current occupation_Housewife' is very high, we can drop this column.
# 

# In[172]:


# Dropping the column 'What is your current occupation_Housewife'
col1 = cols.drop('What is your current occupation_Housewife')


# ### Model-2

# In[173]:



X_train_sm = sm.add_constant(X_train[col1])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# Since Pvalue of 'Last Notable Activity_Had a Phone Conversation' is very high, we can drop this column.
# 
# 

# In[174]:


col1 = col1.drop('Last Notable Activity_Had a Phone Conversation')


# ### Model-3
# 

# In[175]:


X_train_sm = sm.add_constant(X_train[col1])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# Since Pvalue of 'What is your current occupation_Student' is very high, we can drop this column.
# 
# 

# In[176]:


col1 = col1.drop('What is your current occupation_Student')


# ### Model-4
# 

# In[177]:


X_train_sm = sm.add_constant(X_train[col1])
logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
res.summary()


# 
# Since Pvalue of 'Lead Origin_Lead Add Form' is very high, we can drop this column.
# 

# In[178]:


col1 = col1.drop('Lead Origin_Lead Add Form')


# #### Model-5

# In[179]:


X_train_sm = sm.add_constant(X_train[col1])
logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm5.fit()
res.summary()


# #### Checking for VIF values:
# 

# In[180]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col1].columns
vif['VIF'] = [variance_inflation_factor(X_train[col1].values, i) for i in range(X_train[col1].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[181]:


# Dropping the column  'What is your current occupation_Unemployed' because it has high VIF
col1 = col1.drop('What is your current occupation_Unemployed')


# #### Model-6

# In[182]:



X_train_sm = sm.add_constant(X_train[col1])
logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm5.fit()
res.summary()


# In[183]:


# Dropping the column  'Lead Origin_Lead Import' because it has high Pvalue
col1 = col1.drop('Lead Origin_Lead Import')


# #### Model-7

# In[184]:



X_train_sm = sm.add_constant(X_train[col1])
logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm5.fit()
res.summary()


# #### Checking for VIF values:

# In[185]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col1].columns
vif['VIF'] = [variance_inflation_factor(X_train[col1].values, i) for i in range(X_train[col1].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[186]:


# Dropping the column  'Last Activity_Unsubscribed' to reduce the variables
col1 = col1.drop('Last Activity_Unsubscribed')


# #### Model-8

# In[188]:



X_train_sm = sm.add_constant(X_train[col1])
logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm5.fit()
res.summary()


# #### Checking for VIF values:
# 

# In[189]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col1].columns
vif['VIF'] = [variance_inflation_factor(X_train[col1].values, i) for i in range(X_train[col1].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[190]:


# Dropping the column  'Last Notable Activity_Unreachable' to reduce the variables
col1 = col1.drop('Last Notable Activity_Unreachable')


# ### Model-9

# In[191]:


X_train_sm = sm.add_constant(X_train[col1])
logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm5.fit()
res.summary()


# #### Checking for VIF values:
# 

# In[192]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col1].columns
vif['VIF'] = [variance_inflation_factor(X_train[col1].values, i) for i in range(X_train[col1].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ###### Since the Pvalues of all variables is 0 and VIF values are low for all the variables, model-9 is our final model. We have 12 variables in our final model.

# #### Making Prediction on the Train set

# In[193]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[194]:


# Reshaping into an array
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# ###### Creating a dataframe with the actual Converted flag and the predicted probabilities
# 

# In[195]:



y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()


# ###### Choosing an arbitrary cut-off probability point of 0.5 to find the predicted labels
# 
# ###### Creating new column 'predicted' with 1 if Converted_Prob > 0.5 else 0
# 

# In[196]:


y_train_pred_final['predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# ##### Making the Confusion matrix

# In[197]:



from sklearn import metrics

# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
print(confusion)


# In[199]:


# The confusion matrix indicates as below
# Predicted     not_converted    converted
# Actual
# not_converted         3461      444
# converted             719       1727 


# In[200]:


# Let's check the overall accuracy.
print('Accuracy :',metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))


# ##### Metrics beyond simply accuracy

# In[201]:



TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[202]:



# Sensitivity of our logistic regression model
print("Sensitivity : ",TP / float(TP+FN))


# In[203]:


# Let us calculate specificity
print("Specificity : ",TN / float(TN+FP))


# In[204]:


# Calculate false postive rate - predicting converted lead when the lead actually was not converted
print("False Positive Rate :",FP/ float(TN+FP))


# In[205]:


# Negative predictive value
print ("Negative predictive value :",TN / float(TN+ FN))


# In[206]:


# positive predictive value 
print("Positive Predictive Value :",TP / float(TP+FP))


# We found out that our specificity was good (~88%) but our sensitivity was only 70%. Hence, this needed to be taken care of.
# 
# 
# We have got sensitivity of 70% and this was mainly because of the cut-off point of 0.5 that we had arbitrarily chosen. Now, this cut-off point had to be optimised in order to get a decent value of sensitivity and for this we will use the ROC curve.
# 

# ### Plotting the ROC Curve

# ##### An ROC curve demonstrates several things:
# 
# It shows the tradeoff between sensitivity and specificity (any increase in sensitivity will be accompanied by a decrease in specificity).
# 
# The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test.
# 
# The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test.

# In[207]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[208]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_prob, drop_intermediate = False )


# In[209]:


draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# ###### Since we have higher (0.89) area under the ROC curve , therefore our model is a good one.
# 
# Finding Optimal Cutoff Point
# Above we had chosen an arbitrary cut-off value of 0.5. We need to determine the best cut-off value and the below section deals with that. Optimal cutoff probability is that prob where we get balanced sensitivity and specificity

# In[210]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[211]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[212]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# ##### `From the curve above, 0.34 is the optimum point to take it as a cutoff probability.

# In[213]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.34 else 0)

y_train_pred_final.head()


# ##### Assigning Lead Score to the Training data
# 

# In[214]:


y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))

y_train_pred_final.head()


# ##### Model Evaluation
# 

# In[215]:


# Let's check the overall accuracy.
print("Accuracy :",metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted))


# In[216]:


# Confusion matrix
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[217]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[218]:


# Let's see the sensitivity of our logistic regression model
print("Sensitivity : ",TP / float(TP+FN))


# In[219]:


# Let us calculate specificity
print("Specificity :",TN / float(TN+FP))


# In[220]:


# Calculate false postive rate - predicting converted lead when the lead was actually not have converted
print("False Positive rate : ",FP/ float(TN+FP))


# In[221]:


# Negative predictive value
print("Negative Predictive Value : ",TN / float(TN+ FN))


# In[222]:


# Positive predictive value 
print("Positive Predictive Value :",TP / float(TP+FP))


# #### Precision and Recall
# 
# 1.Precision = Also known as Positive Predictive Value, it refers to the percentage of the results which are relevant.
# 
# 2.Recall = Also known as Sensitivity , it refers to the percentage of total relevant results correctly classified by the algorithm.

# In[223]:


#Looking at the confusion matrix again

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
confusion


# In[224]:


# Precision
TP / TP + FP

print("Precision : ",confusion[1,1]/(confusion[0,1]+confusion[1,1]))


# In[225]:


# Recall
TP / TP + FN

print("Recall :",confusion[1,1]/(confusion[1,0]+confusion[1,1]))


# In[226]:


from sklearn.metrics import precision_score, recall_score
print("Precision :",precision_score(y_train_pred_final.Converted , y_train_pred_final.predicted))


# In[227]:


print("Recall :",recall_score(y_train_pred_final.Converted, y_train_pred_final.predicted))


# #### Precision and recall tradeoff

# In[228]:



from sklearn.metrics import precision_recall_curve

y_train_pred_final.Converted, y_train_pred_final.predicted


# In[229]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# In[230]:


# plotting a trade-off curve between precision and recall
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# **The above graph shows the trade-off between the Precision and Recall .

# #### Making predictions on the test set
# 

# Scaling the test data

# In[231]:



X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.transform(X_test[['TotalVisits',
                                                                                                        'Total Time Spent on Website',
                                                                                                        'Page Views Per Visit']])


# In[232]:


# Assigning the columns selected by the final model to the X_test 
X_test = X_test[col1]
X_test.head()


# In[233]:


# Adding a const
X_test_sm = sm.add_constant(X_test)

# Making predictions on the test set
y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]


# In[234]:


# Converting y_test_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)
# Let's see the head
y_pred_1.head()


# In[236]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
# Putting Prospect ID to index
y_test_df['Prospect ID'] = y_test_df.index
# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()


# In[237]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_prob'})
# Rearranging the columns
y_pred_final = y_pred_final.reindex(columns=['Prospect ID','Converted','Converted_prob'])
# Let's see the head of y_pred_final
y_pred_final.head()


# In[238]:


y_pred_final['final_predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.34 else 0)
y_pred_final.head()


# In[239]:


# Let's check the overall accuracy.
print("Accuracy :",metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted))


# In[240]:


# Making the confusion matrix
confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted )
confusion2


# In[241]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[242]:


# Let's see the sensitivity of our logistic regression model
print("Sensitivity :",TP / float(TP+FN))


# In[243]:


# Let us calculate specificity
print("Specificity :",TN / float(TN+FP))


# #### Assigning Lead Score to the Testing data

# In[244]:



y_pred_final['Lead_Score'] = y_pred_final.Converted_prob.map( lambda x: round(x*100))

y_pred_final.head()


# ### Observations:
# After running the model on the Test Data , we obtain:
# 
# Accuracy : 80.4 %
# Sensitivity : 80.4 %
# Specificity : 80.5 %
# Results :
# ### 1) Comparing the values obtained for Train & Test:
# ##### Train Data:
# Accuracy : 81.0 %
# Sensitivity : 81.7 %
# Specificity : 80.6 %
# Test Data:
# Accuracy : 80.4 %
# Sensitivity : 80.4 %
# Specificity : 80.5 %
# Thus we have achieved our goal of getting a ballpark of the target lead conversion rate to be around 80% . The Model seems to predict the Conversion Rate very well and we should be able to give the CEO confidence in making good calls based on this model to get a higher lead conversion rate of 80%.
# 
# #### 2) Finding out the leads which should be contacted:
# The customers which should be contacted are the customers whose "Lead Score" is equal to or greater than 85. They can be termed as 'Hot Leads'.
# 

# In[245]:


hot_leads=y_pred_final.loc[y_pred_final["Lead_Score"]>=85]
hot_leads


# #### So there are 368 leads which can be contacted and have a high chance of getting converted. The Prospect ID of the customers to be contacted are :

# In[246]:


print("The Prospect ID of the customers which should be contacted are :")

hot_leads_ids = hot_leads["Prospect ID"].values.reshape(-1)
hot_leads_ids


# #### 3) Finding out the Important Features from our final model:

# In[247]:


res.params.sort_values(ascending=False)


# ## Recommendations:
# ###### The company should make calls to the leads coming from the lead sources "Welingak Websites" and "Reference" as these are more likely to get converted.
# ###### The company should make calls to the leads who are the "working professionals" as they are more likely to get converted.
# ###### The company should make calls to the leads who spent "more time on the websites" as these are more likely to get converted.
# ###### The company should make calls to the leads coming from the lead sources "Olark Chat" as these are more likely to get converted.
# ###### The company should make calls to the leads whose last activity was SMS Sent as they are more likely to get converted.
# ###### The company should not make calls to the leads whose last activity was "Olark Chat Conversation" as they are not likely to get converted.
# ###### The company should not make calls to the leads whose lead origin is "Landing Page Submission" as they are not likely to get converted.
# ###### The company should not make calls to the leads whose Specialization was "Others" as they are not likely to get converted.
# ###### The company should not make calls to the leads who chose the option of "Do not Email" as "yes" as they are not likely to get converted.

# In[ ]:




