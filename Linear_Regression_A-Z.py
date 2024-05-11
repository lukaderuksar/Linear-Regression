#!/usr/bin/env python
# coding: utf-8

# # Linear Regression

# In[1]:


import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[3]:


data = C:\Users\lukad\OneDrive\Documents\Projects\used_cars_data.csv"
data1 = pd.read_csv(data) 
df = data1.copy() #making a copy to avoid changes to the data
df.head()


# In[4]:


print(f'There are {df.shape[0]} rows and {df.shape[1]} columns in this dataset.')  # f-string to get the shape of dataset


# In[5]:


np.random.seed(68) #let's take a look at 10 random rows in the dataset
df.sample(n=10)


# * We can drop the S.No column as its repetitive of the index and not required in further analysis.
# * The Mileage, Engine and Power column are represented as strings when they should be in numerical.
# * Mileage has a 0.0 value that should be replace by Nan
# * Power has string 'null bhp' which should be replaced by Nan
# * The above random sample shows that some columns have a lot of missingness, so that needs to be analysed later and New_price particularly has a lot missing values.
# * Price is the dependent variable

# In[6]:


df.drop(['S.No.'],axis=1,inplace=True)


# In[7]:


df.info() 


# **Observations**
# * There are 7253 entries across 13 columns
# * Mileage, Engine, Power and New_Price columns are in Object datatype.These columns need to be converted to Numerical. 
# * The rest of the object datatype need to be converted to Category.
#     `coverting "objects" to "category" reduces the space required to store the dataframe. It also helps in analysis`
# * We can also see that New_Price column has only 1006 entries. 
# * The Price column also has significant missing values.
# * Power, Mileage, Seats and Engine have comparatively lesser missing values than the above two.
# * From the above details, we see that New_Price column has almost 80% of data missing. This may impact the performance of the model to caluculate the price. Therefore we will drop this column for further analysis.

# In[8]:


df.drop(["New_Price"],axis=1,inplace=True)


# Let's replace any possible corrupt values like, 0.0 to Nan before proceeding.

# In[9]:


import numpy as np
import pandas as pd

num_col = df.select_dtypes(include=np.number).columns.tolist()

for col in num_col:
    df[col] = df[col].replace(0.0, np.nan)


# In[10]:


num_col


# # **FIXING DATATYPES**:
# - Before getting the summary statistics of the data to analyse the distribution, we must convert them to numerical columns.

# In[11]:


num_values = []
#the loop will add all the columns we want to convert form object to numerical into a list
# we can then use this list for conversion
for colname in df.columns[df.dtypes == 'object']:
    if df[colname].str.endswith(('pl', 'kg', 'CC', 'bhp', 'Lakh')).any():
        num_values.append(colname)
print(num_values)


# In[12]:


#Writing a function that will help split the string from the numerical values in the columns
#This function will also drop the string and convert to float datatype.
#This function will ensure a clean and faster code
def obj_to_num(n):
    if isinstance(n,str): #checks if the columns are string datatype
        if n.endswith('kmpl'):
            return float(n.split('kmpl')[0])     
        elif n.endswith('km/kg'):                   
            return float(n.split('km/kg')[0])
        elif n.endswith('CC'):
            return float(n.split('CC')[0])
        elif n.startswith('null'):     #replaces values that have string 'null bhp' to Nan
            return(np.nan)          
        elif n.endswith('bhp'):
             return float(n.split('bhp')[0])
    else: 
        return np.nan

for colname in num_values:
    df[colname] = df[colname].apply(obj_to_num)#applying above function to the column list    
    df[colname]=df[colname].replace(0.0,np.nan)


# ## Fixing Datatypes

# In[13]:


df["Name"]=df["Name"].astype("category")
df["Location"]=df["Location"].astype("category")
df["Fuel_Type"]=df["Fuel_Type"].astype("category")
df["Transmission"]=df["Transmission"].astype("category")
df["Owner_Type"]=df["Owner_Type"].astype("category") 


# In[14]:


np.random.seed(68) 
df.sample(n=10)


# In[15]:


df.info()  


# * All datatypes are now fixed and the memory useage has reduced.
# * We noticed that the number of missing values has also increased

# ### Summary of Categorical Variables

# In[16]:


df.describe(include=["category"]).T


# **Observations:**
# - We see that ther are 2041 total unique Cars
# - More cars are sold in Mumbai and Diesel is the preffered Fuel Type
# - Most of the cars sold are Manual Transmission and have only had one previous owner.

# - For further processing we have to make the data more manageable.
# - Let's group the cars by Brand and Model

# In[17]:


df[['Car_Brand','Model']] = df.Name.str.split(n=1,expand=True) #splitting the Brand and the car model


# In[18]:


Brand_name=df['Car_Brand'].unique()
Model=df['Model'].unique() # Model names are unique to the Car Brands. 


# In[19]:


Brand_name #Checking car brand names 


# **Observations**:
# - We see that Land Rover is mentioned as Land and the Brand Isuzu is mentioned twice

# In[20]:


df['Car_Brand']=df['Car_Brand'].replace('Land','Land_Rover') 
df['Car_Brand']=df['Car_Brand'].replace('ISUZU','Isuzu')  #correcting the brands
df['Car_Brand'].value_counts()


# **Observations**:
# - Maruti and Hyundai are the most popular cars brands
# - Honda and Toyota are the next most popular brands
# - We also see that the expensive luxury car brands are very few

# # **MISSING VALUE:**
# - We will replace the missing values in Power,Engine,Mileage and Seats with its median value.

# In[21]:


numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
numeric_columns


# In[22]:


numeric_columns.remove('Price') #It's the dependent variable
medianFiller = lambda x: x.fillna(x.median())
df[numeric_columns] = df[numeric_columns].apply(medianFiller,axis=0)


# * The Price Column also has missing values(1234) that needs to be treated. 
# * Hence we will calculate Median Price per Brand and per Brand's model and replace the missing values in Price column

# In[23]:


Median1=[] #creating an empty list to add the median Price of Cars per Brand
for i in range(len(Brand_name)):
    x=df['Price'][df['Car_Brand']==Brand_name[i]].median()
    Median1.append(x)


# In[24]:


Median2=[] #Creating an empty list to add the median price of cars per Car model
for i in range(len(Model)):
    x=df['Price'][df['Model']==Model[i]].median()
    Median2.append(x)


# In[25]:


df['Price']= df['Price'].fillna(0.0)


# In[26]:


for i in range(len(df)):  #running a loop to check every row in df dataset
    if df.Price[i]==0.00:
        for j in range(len(Model)):  
            if df.Model[i]==Model[j]:  #Comparing the Car model  names in both datasets
                df.Price[i]=Median2[j]  #replacing the Price of the car with the median price of its subsequent model


# In[27]:


df.info()


# In[28]:


df[df['Price'].isna()]


# * The above mentioned cars appear only once in the dataset. Hence we dont have a median price value per its model.
# * Therefore we will replace the missing Price of these cars with the median Price of its corresponding Brand, that was calculated earlier. 

# In[29]:


df['Price']= df['Price'].fillna(0.0) #replacing the missing values with float 0.0
for i in range(len(df)):  #running a loop to check every row in df dataset
    if df.Price[i]==0.00:
        for j in range(len(Brand_name)):  
            if df.Car_Brand[i]==Brand_name[j]:  #Comparing the brand names in both datasets
                df.Price[i]=Median1[j]     #replacing with corresponding missing values


# In[30]:


df[df['Price'].isna()]


# In[31]:


#Dropping the above two cars as there are only one of each per brand
#Also we do not have any further information to calculate its price
df.dropna(axis=0,inplace=True)
df.shape  #we now have 7251 rows and 14 columns


# ### Summary of Numerical Columns

# In[32]:


pd.set_option('display.float_format', lambda x: '%.3f' % x) # to display numbers in digits
df.describe().T


# **Observations**:
# * Year:
#     - Mean Year of car's is 2013 which is one year short of median. Year starts from 1998 till 2019 implying older to latest car models
# * Kilometers_Driven: 
#     - The Mean is slightly higher than the median, but the max value is very hight, suggesting outliers
# * Mileage: 
#     - The Mean and Median of Mileage are fairly close
# * Engine and Power & Price
#     - The Mean value is significantly higher than the median for all three variables. 
#     - Average Price is at 9.33 Lakhs. The Variance is greater than the Mean which suggests wide distribution(skewness) of data. 

# # **EXPLORATORY DATA ANALYSIS**

# In[33]:


#Performing Univariate Analysis to study the central tendency and dispersion
#Plotting histogram to study distribution
from scipy.stats import norm
Uni_num = df.select_dtypes(include=np.number).columns.tolist()
plt.figure(figsize=(17,75))
for i in range(len(Uni_num)):     #creating a loop that will show the plots for the columns in one plot
    plt.subplot(18,3,i+1)
    sns.histplot(df[Uni_num[i]],kde=False)
    plt.tight_layout()
    plt.title(Uni_num[i],fontsize=25)

plt.show()


# In[34]:


#Plotting a box plot to study central tendency
plt.figure(figsize=(15,35))
for i in range(len(Uni_num)):
    plt.subplot(10,3,i+1)
    sns.boxplot(df[Uni_num[i]],showmeans=True, color='yellow')
    plt.tight_layout()
    plt.title(Uni_num[i],fontsize=25)

plt.show()


# **Observations:** From Both Histogram and Box plots :
# 
# * Only Mileage has a somewhat normal distribution
# * Year is left-skewed and has comparatively less outliers in the lower end.
# * Engine & Power:
#     - Both columns are right-skwed with a moderate Inter-Quartile Range and several outliers at the higher scale.
#       Power has more outliers comapred to Engine. 
# * Kilometer_Driven and Price:
#     - Both these columns are heavily right-skewed, with Kilometers_Driven having a very small IQR and one large outlier in the max end. Price column also has several outliers in the higher end. 
#     We will treat these outliers as they might have adverse effect in the accuracy of the prediction. But sometimes outliers might have independent significance to the data.
#     So, We will also the building model to decide on the outlier treatment

# ## Feature Engineering:
# 
# ### Grouping Location by Regions

# In[35]:


regions ={'Delhi':'North','Jaipur':'North',
          'Chennai':'South','Coimbatore':'South','Hyderabad':'South','Bangalore':'South','Kochi':'South',
        'Kolkata':'East',
         'Mumbai':'West','Pune':'West','Ahmedabad':'West'}
df['Region']=df['Location'].replace(regions)


# ### Binning the Car Names by different Price Levels:
# - We have 33 car brands and even higher individual models. 
# - To manage the data subesequently, we will bin them according to their Price Ranges; from lower/economic cars to luxury/expensive cars
# - This will reduce total categories of Cars to just six.

# In[36]:


df.drop(["Car_Brand","Model"],axis=1,inplace=True) # no longer needed for Analysis
df['Car_Type'] = pd.cut(df['Price'],[-np.inf,5.5,10.5,20.5,45.0,75.0,np.inf],
                       labels=["Tier1","Tier2","Tier3","Tier4","Tier5","Tier6"])

df['Car_Type'].value_counts()


# In[37]:


df.sample()


# In[38]:


#Univariate Analysis on Categorical Variables
categorical_val = df.select_dtypes(exclude=np.number).columns.tolist()
categorical_val.remove('Name')
categorical_val.remove('Location')


# In[39]:


plt.figure(figsize=(17, 75))
for i in range(len(categorical_val)):
    plt.subplot(18, 3, i + 1)
    df[categorical_val[i]] = df[categorical_val[i]].astype("category")  # Convert to category data type
    ax = sns.countplot(data=df, x=categorical_val[i], palette='Dark2')
    plt.tight_layout()
    plt.title(categorical_val[i], fontsize=25)
    if df[categorical_val[i]].dtype != "category":
        total = len(df[categorical_val[i]])
        for p in ax.patches:
            percentage = '{:.1f}%'.format(100 * p.get_height() / total)
            x = p.get_x() + (p.get_width() / 2) - 0.1
            y = p.get_height()
            ax.annotate(percentage, (x, y), size=13.5, color='black')

plt.show()


# **Observations**:
# * 82.1% of all cars only have One previous owner.
# * 71% of the cars are of Manual Transmission and 47.6% of cars sold in South region.
# * We also see that about 49.6% of cars are in Tier1 i.e at Price below 5.5 Lakhs INR.
# * Diesel is the most Preferred Fuel_type at 53.1% followed by Petrol 45.8%.CNG and LPG(Gas-reliant) together make 1% of all cars. We are see that the Electric Fuel_Type is at 0.0%, Let's check that

# In[40]:


df[df['Fuel_Type']=='Electric']


# - There are only two cars running in Electric Fuel in this dataset

# ### Coorelation Matrix:

# In[41]:


corr= df.corr().sort_values(by=['Price'],ascending=False) #coorelation matrix with respect to dependent variable Price
plt.figure(figsize=(10,7))
sns.heatmap(corr,annot= True,vmin=0,vmax=1, cmap='coolwarm',linewidths=0.75)
plt.show()


# **Insights**:
# * Price has high positive correlation with Engine and Power and a lower positive correlation with Year.
# * Price has a lower negative correlation with Mileage 
# * Engine and Power have a very high positive correlation.
# * Mileage has a high negative correlation with Engine and Power

# ### Bivariate and Multivariate Analysis:

# In[42]:


#Analysis of variables that have high correlation with Price
#Price Vs Engine Vs Region
plt.figure(figsize=(15,7))
sns.scatterplot(data=df,y='Price',x='Engine',hue='Region')
plt.show()


# **Observations**:
# * We see that as Engine capacity increases Price of cars also increase.
# * We also notice several exceptions to the above case

# In[43]:


#Price Vs Power Vs Region
plt.figure(figsize=(15,7))
sns.scatterplot(data=df,y='Price',x='Power',hue='Region')
plt.show()


# **Observations**:
# - Price does increase with Power, but we can also see several exceptions.

# In[44]:


plt.figure(figsize=(15,7))
sns.scatterplot(data=df,y='Engine',x='Power',hue='Car_Type')
plt.show()


# **Observations**:
# - From the plot, we see that all three variables have a positive correlation.
# - This also suggest multicolinearity between Engine and Power, which must be addressed later

# In[45]:


#Price Vs Mileage Vs Region
plt.figure(figsize=(15,7))
sns.scatterplot(data=df,y='Price',x='Mileage',hue='Region')
plt.show()


# **Observations**:
# * Price and Mileage have a negative correlation with a few exceptions.

# In[46]:


#How does Manufacture Year affect Price?
plt.figure(figsize=(15,7))
sns.lineplot(x='Year', y='Price',
             data=df);


# **Observations**:
# * Overall as Manufacture Year rises, Price of Car also increases.

# In[47]:


#Kilometers_Driven Vs Year
#Since The range for Kilometers is very wide, we will log transform to a manageable scale
plt.figure(figsize=(15,7))
sns.lineplot(x='Year', y=np.log(df['Kilometers_Driven']),
             data=df)


# **Observations**:
# * Year and Kilomertes_driven have a negative correlation
# * This is to be expected as lastest model used cars probably have less useage before being sold.

# In[48]:


#Engine Vs Mileage Vs Car_Type
plt.figure(figsize=(15,7))
sns.scatterplot(x='Engine', y='Mileage',hue='Car_Type',
             data=df)


# **Observations**:
# - Most cars in Tier1 have less Engine CC and therefore Higher Mileage

# In[49]:


#Does type of ownership affect Car price?
df_hm =df.pivot_table(index = 'Region',columns ='Owner_Type',values ="Price",aggfunc=np.median)
# Draw a heatmap 
plt.subplots(figsize=(10,7))
sns.heatmap(df_hm,cmap='copper',linewidths=.5, annot=True);


# **Observations**:
# * Mean Price of cars decrease as number of ownership of cars increases across all regions
# * The South region also has the highest Median Price for Cars with only one previous owner, followed by West then North and East
# * We also see that in East there are only two Owner_types
# * This suggests that type of ownership does affects overall car price.

# In[50]:


#Does type of Fuel affect car price?
plt.figure(figsize=(7,5))
sns.barplot(data=df,x='Fuel_Type',y='Price')


# **Observations**: 
# * Electric Car's have an equal Price range compared to Diesel.
# * We know that there are only two cars with Electric Fuel_Type in this data, which is a very small sample size.
# * Hence we will drop Fuel_Type while building the ML model, as it might affect the accuracy.

# # **OUTLIER TREATMENT**

# In[51]:


# Lets treat outliers by flooring and capping
def treat_outliers(df,col):
   
    Q1=df[col].quantile(0.25) # 25th quantile
    Q3=df[col].quantile(0.75)  # 75th quantile
    IQR=Q3-Q1
    Lower_Whisker = Q1 - 1.5*IQR 
    Upper_Whisker = Q3 + 1.5*IQR
    df[col] = np.clip(df[col], Lower_Whisker, Upper_Whisker) # all the values samller than Lower_Whisker will be assigned value of Lower_whisker 
                                                            # and all the values above upper_whishker will be assigned value of upper_Whisker 
    return df

def treat_outliers_all(df, col_list):
    
    for c in col_list:
        df = treat_outliers(df,c)
        
        
    return df    


# # **MODEL BUILDING**
# 
# ### We will build a Predictive model with and without treating the Outliers and compare it's performances to decide if the outliers have any adverse impact to the linear model.

# In[52]:


df2=df.copy() #making the first copy
numerical_col = df2.select_dtypes(include=np.number).columns.tolist()
numerical_col.remove('Year')
numerical_col.remove('Mileage')
numerical_col.remove('Seats')  #Dropping Year,Mileage and Seats as they dont have very high outliers
numerical_col


# In[53]:


df2 = treat_outliers_all(df2,numerical_col) #treating outliers 


# In[54]:


#checking if the outliers are treated
plt.figure(figsize=(15,35))
for i in range(len(numerical_col)):
    plt.subplot(10,3,i+1)
    sns.boxplot(df2[numerical_col[i]],showmeans=True, color='yellow')
    plt.tight_layout()
    plt.title(numerical_col[i],fontsize=25)

plt.show()


# **Observations:**
# - The Outliers for Engine, Price, Power and Kilometers_driven is treated
# - We will build a model with this treated dataset to analyse the Price

# ## Model Building 1 - With Treated Outliers

# In[55]:


df2.head()


# In[56]:


X = df2.drop(['Name','Fuel_Type','Location','Price'], axis=1)
#dropping Name as we bins via Car_Type
#dropping Fuel_Type to not affect the accuracy of the model
y = df2[['Price']]

print(X.shape)
print(y.shape)


# In[57]:


#Creating Dummy Variabls for the Categorical Columns
#Dummy variable will be used as independent variables and will not impose any ranking
X = pd.get_dummies(X, columns=['Transmission','Owner_Type','Region','Car_Type'], drop_first=True)
X.head()


# In[58]:


#split the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=56) # keeping random_state =56 ensuring datasplit remains consistent
X_train.head()


# In[59]:


#Fitting linear model

from sklearn.linear_model import LinearRegression
linearregression = LinearRegression()                                    
linearregression.fit(X_train, y_train)
print("Intercept of the linear equation:", linearregression.intercept_) 
for idx, col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {}".format(col_name, linearregression.coef_[0][idx]))


# In[60]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pred = linearregression.predict(X_test) 


# ### Model Performances:

# In[61]:


# Mean Absolute Error on test
mean_absolute_error(y_test, pred)


# In[62]:


# RMSE on test data
mean_squared_error(y_test, pred)**0.5


# In[63]:


# R-squared on test
r2_score(y_test, pred)


# In[64]:


# Training Score

linearregression.score(X_train, y_train)  # 70 % data 


# In[65]:


# Testing score

linearregression.score(X_test, y_test) # unseen data


# **Observations**:
# - From the above model we see that the  $R^2$ is 0.953, that explains 95.3% of total variation in dataset. This model is a good fit.

# ## Model Building 2 - Without Treating Outliers

# In[66]:


df3=df.copy() #making the second copy
df3.head()


# In[67]:


X1 = df3.drop(['Name','Fuel_Type','Location','Price'], axis=1)
#dropping Name as we bins via Car_Type
#dropping Fuel_Type to not affect the accuracy of the model
y1 = df3[['Price']]

print(X1.shape)
print(y1.shape)


# In[68]:


#Creating Dummy Variabls for the Categorical Columns
#Dummy variable will be used as independent variables and will not impose any ranking
X1 = pd.get_dummies(X1, columns=['Transmission','Owner_Type','Region','Car_Type'], drop_first=True)
X1.head()


# In[69]:


#split the data into train and test
from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=56)
X1_train.head()


# In[70]:


#Fitting linear model
from sklearn.linear_model import LinearRegression
linearregression = LinearRegression()                                    
linearregression.fit(X1_train, y1_train)                                  
print("Intercept of the linear equation:", linearregression.intercept_) 
for idx, col_name in enumerate(X1_train.columns):
    print("The coefficient for {} is {}".format(col_name, linearregression.coef_[0][idx]))  


# In[71]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pred1 = linearregression.predict(X1_test)


# In[72]:


# Mean Absolute Error on test
mean_absolute_error(y1_test, pred1)


# In[73]:


# RMSE on test data
mean_squared_error(y1_test, pred1)**0.5


# In[74]:


# R-squared on test
r2_score(y1_test, pred1)


# In[75]:


# Training Score

linearregression.score(X1_train, y1_train)  # 70 % data


# In[76]:


# Testing score

linearregression.score(X1_test, y1_test) # unseen data


# **Observations**:
# - From the above model we see that the  $R^2$ is 0.921, that explains 92.1% of total variation in dataset. Though this model is a decent fit, its less than the $R^2$ value from Model 1. 
# - Also we see that the Training and Testing Scores for this model are 94.5% and 92.1% which has a marginal difference.
# -  Hence we shall proceed with the Model 1 for further analysis and  Stats model.

# # Stats Model:
# - Using Stats Model in Python, we will get an list of statistical results for each estimator.
# - Stats Model is also used to further conduct tests and statistical data exploration

# In[77]:


# Lets us build linear regression model using statsmodel 
import statsmodels.api as sm
X = sm.add_constant(X)
X_train1, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=56)

olsmod0 = sm.OLS(y_train, X_train1) #y_train remains same 
olsres0 = olsmod0.fit()
print(olsres0.summary())


# **Insights**:
# - The P-Value of the variable indicates if the its significant or not.
# - The level of significance is 0.05 and any p-value less than 0.05 , then that variable would be considered significant.

# ## Interpreting the Regression Results:
# 
# 1. **Adjusted. R-squared**: It reflects the fit of the model and ranges from 0 to 1
#     - A high Adjusted R-Squared values indicated a good fit. In this model, the Adj. R-squared is **0.953**, which is good!
# 2. **const coefficient** is the Y-intercept.
#     - If all the independent variables are zero, then the expected output will be equal to const coefficient, which in this case is **-40.48**
# 3. **std err**: It reflects the level of accuracy of the coefficients.
#       - The lower it is, the higher is the level of accuracy.
# 5. **P >|t|**: It is p-value.
#     -  This shows that for each independent feature there is a null hypothesis and alternate hypothesis 
# 
#     Ho : Independent variable is not significant 
# 
#     Ha : Independent variable is significant
#     - If p-value is less than 0.05 , then the variable is considered to be statistically significant.
#   
# 6. **Confidence Interval**: It represents the range in which our coefficients are likely to fall.
#     - The current confidence interval is at 95% 

# # **LINEAR REGRESSION ASSUMPTIONS**:
# -  No Multicollinearity
# -  Mean of residuals should be 0
# -  No Heteroscedacity
# -  Linearity of variables
# -  Normality of error terms

# ### Checking for Multicollinearity using VIF Scores:
# - Multicollinearity occurs when there is correlation between the predictor variables.
# - Since the variables are required to be independent,having a correlation will lead to inaccuracy in the model.
# - VIF(Variance Inflation Factor) scores measures how much the variance of an estimated regression coefficient is increased by collinearity. VIF scores quantify the severity of multi-collinearity in OLS stats model.
# - If VIF value exceeds or is close to 5 then we there is moderate correlation.
# - IF VIF value exceed or is close to 10 then it shows high multi-collinearity

# In[78]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_series1 = pd.Series([variance_inflation_factor(X_train1.values,i) for i in range(X_train1.shape[1])],index=X_train1.columns)
print('Series before feature selection: \n\n{}\n'.format(vif_series1))


# **Observations**:
# * Engine and Power have VIF scores greater than 5. This suggests that there is moderate to high collinearity suggesting the 2 variables are correlated to each other.
# * This makes sense as High Engine efficiency leads to high Power in a vehicle.
# * Hence to remove multi-collinearity, we will drop Power column first as it has a higher score.

# In[79]:


X_train2 = X_train1.drop('Power', axis=1)
vif_series2 = pd.Series([variance_inflation_factor(X_train2.values,i) for i in range(X_train2.shape[1])],index=X_train2.columns)
print('Series before feature selection: \n\n{}\n'.format(vif_series2))


# - The VIF scores have reduced for Engine and there is no more collinearity in the model

# In[80]:


olsmod1 = sm.OLS(y_train, X_train2)
olsres1 = olsmod1.fit()
print(olsres1.summary())


# **Observations**:
# - The Adj.$R^2$ has reduced from 0.952 to 0.949 - which is still good.

# In[81]:


X_train3 = X_train1.drop('Engine', axis=1)
vif_series3 = pd.Series([variance_inflation_factor(X_train3.values,i) for i in range(X_train3.shape[1])],index=X_train3.columns)
print('Series before feature selection: \n\n{}\n'.format(vif_series3))


# - The VIF scores have reduced for Power and there is no more collinearity in the model

# In[82]:


olsmod2 = sm.OLS(y_train, X_train3)
olsres2 = olsmod2.fit()
print(olsres2.summary())


# **Observations**:
# - The Adj. $R^2$ is 0.951 for olsres2, which is better than the olsres1 at 0.949
# - Hence we will proceed further with olsres2 for further analysis.
# - Now that there is no multi-collinerity, we check the p-values of the predictor variables for insignificance
# 
# 
# **Observations**:
# * Kilometers_Driven, Owner_Type_Fourth & Above and Owner_Type_Second p-value greater than 0.05 and therefore is not significant.
# * We will only be dropping Kilometers_Driven and not the other two despite high p-values. 
# * Owner_Type_Fourth & Above and Owner_Type_Second are part of the categorical variable Owner_Type and there are other significant values in this category.

# In[83]:


X_train4 = X_train3.drop('Kilometers_Driven', axis=1)
olsmod3 = sm.OLS(y_train, X_train4)
olsres3 = olsmod3.fit()
print(olsres3.summary())


# #### Since there are no more p-values greater than 0.05, olsres 3 is the final model and X_train4 is the final data.
# 
# **Observations**:
# - The Adjusted R-Squared for the model is 0.951. This shows that the model is able to explain 95.1% of the variance.
# - The Adjusted R-Squared in OLSres0 was 95.2%.This shows that the dropped variables did not affect the model very much.
# - Hence this model is a good fit.

# #### Checking if Mean of residuals should be 0 for OLSres3
# * Residual is the difference between the observed x-value and the fitted x-value to the best fit line.

# In[84]:


residual= olsres3.resid
np.mean(residual)


# * Mean of Residuals is very close to 0.

# ### Test for Linearity:
# * To check if there is a linear (Straight-line) relationship between the dependent and independent variables.
# * To check, we will plot between Fitted values Vs Residuals
# * Fitted or Predicted value describes where the particular x-value fits in the best fit line.

# In[85]:


residual=olsres3.resid
fitted=olsres3.fittedvalues #predicted values


# In[86]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
sns.residplot(x=fitted, y=residual, color="olive", lowess=True)
plt.xlabel("Fitted Values")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.show()


# * The Scatterplot shows that the distribution between Residuals(errors) and Fitted values has no pattern.
# * Hence Linearity assumption is satisfied

# ### TEST FOR NORMALITY:
# * The Residuals should be normally distributed.
# * We will perform the test for Normality in the following steps:
#     - Histogram of Residuals
#     - Q-Q plot 
# * Further analysis of data will be performed if any the above tests fail.

# In[87]:


#Histogram of Residuals
sns.distplot(residual) 
plt.title('Normality of residuals')
plt.show()


# In[88]:


# Q-Q plot to check the normal probability of residuals.
# It should approximately follow a straight line
import pylab
import scipy.stats as stats
stats.probplot(residual,dist="norm",plot=pylab)
plt.show()


# * The Q-Q plot is approximately straight line
# * Hence the Test for Linearity is satisfied

# ### TEST FOR HOMOSCEDASTICITY:
# * The assumption is that the variance of the residuals  is equal/same across all values of independent variabels for the final data. i.e The Error term doesnt vary too much when the Indipendent(Predictor) variable changes, Homoscedastic
# * If the variance is not equal then the data is Hetroscedastic
#     - Null Hypothesis : Residuals are equal across independent variables
#     - Alternate Hypothesis : Residuals are not equal across independent variables

# In[89]:


import statsmodels.stats.api as sms
from statsmodels.compat import lzip

name = ['F statistic', 'p-value']
test = sms.het_goldfeldquandt(residual, X_train4)
lzip(name, test) #returns a list of values


# * The p-value 0.202 is greater than level of confidence. i.e p-value>0.05.
# * Hence we fail to reject the Null Hypothesis. Thus the Residuals are equal (Homoscedastic) across all independent variables.
# 
# #### All Linear Regression Assumptions have been satisfied.
# 
# 
# ### Predicting on Test Data:

# In[90]:


X_train4.columns


# In[91]:


X_test_final = X_test[X_train4.columns]
X_test_final.head()


# In[92]:


y_pred = olsres3.predict(X_test_final)


# ### Checking the performace of Train and test data using RMSE metric
# * Root Mean Squared Error (RMSE) is the Standard Deviation (S.D) of residuals. 
# * Lower RMSE values indicate a good model fit

# In[93]:


#Checking root mean squared error on both train and test set  

from sklearn.metrics import mean_squared_error
rms = np.sqrt(mean_squared_error(y_train, fitted))
print('Train error:',rms)

rms1 = np.sqrt(mean_squared_error(y_test, y_pred))
print('Test error:',rms1)


# **Observations**:
# * The Train and Test Errors are comparable and quite low.
# * This suggests that the model does not suffer from either over-fitting(noise + information) or under-fitting(less information)

# In[94]:


olsmodtest = sm.OLS(y_test, X_test_final)
olsrestest = olsmodtest.fit()
print(olsrestest.summary())


# * Ajd.$R^2$ is 0.952 which is close to the traing data Ajd.$R^2$ 0.951

# In[95]:


print(olsres3.summary())


# ## Ridge Regression

# In[96]:


from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
ridge = Ridge()


# In[97]:


parameters = {'alpha' : [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor = GridSearchCV(ridge, parameters, scoring="neg_mean_squared_error",cv=5)
ridge_regressor.fit(X_train,y_train)


# In[98]:


print("Best Parameters:", ridge_regressor.best_params_)
print("Best Score (negative mean squared error):", ridge_regressor.best_score_)


# ## Lasso Regression

# In[99]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

lasso = Lasso()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}
lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)

# Remove constant term from training data if it exists
if 'const' in X_train.columns:
    X_train = X_train.drop('const', axis=1)

lasso_regressor.fit(X_train, y_train)

print("Best Parameters:", lasso_regressor.best_params_)
print("Best Score (negative mean squared error):", lasso_regressor.best_score_)

# Ensure the constant term is also removed from the test data if it exists
if 'const' in X_test.columns:
    X_test = X_test.drop('const', axis=1)

prediction_lasso = lasso_regressor.predict(X_test)
prediction_ridge = ridge_regressor.predict(X_test)


# In[100]:


import seaborn as sns
Y_test_array = y_test.to_numpy()
sns.distplot(Y_test_array-prediction_lasso)


# ## Conclusion:
# * We conclude that olsres3 is a good model for prediction and inference at 95.1% Ajd.$R^2$.
# * Only Transmission and Onwer_type have a negative correlation to Price; ie. As Manual Transmissions lower the overall Pricing of used Cars than Automatic.
# * As ownership level increases, the Pricing of used cars drop. 
# * Year, Mileage, Power and Seats have positive assosiation with Pricing. 
# * The above variables are the main features that impact the Price of a Used car
