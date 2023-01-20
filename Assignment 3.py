import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as skmet
from scipy.optimize import curve_fit
import scipy.optimize as opt
from sklearn.cluster import KMeans
import warnings
from matplotlib import style
import statsmodels.api as sm
sns.set()

warnings.filterwarnings('ignore')

a_df = pd.read_csv('Climate_data.csv', skiprows=4)

a_df.dropna(how='all')

n_cluster = 3
kmeans = KMeans(n_clusters=n_cluster, random_state=42)

#---------------------------------------------------------------------------------------------------------------
# 1 st indicator: Renewable electricity output (% of total electricity output)
a_ind1 = a_df[a_df['Indicator Code'] == 'EG.ELC.RNEW.ZS'] 
print(a_ind1)
a_ind1.fillna(0, inplace=True)

a_ind_1a = pd.DataFrame()

# creating clusters for the years 1997 and 2015
a_ind_1a['1997'] = a_ind1['1997'].copy()
a_ind_1a['2015'] = a_ind1['2015'].copy()

# resetting the column index
a_ind_1a.reset_index(drop=True)

a_ind_1a.info()


a_ind_1a_colnam = a_ind_1a.columns.values.tolist()

# create scaled DataFrame where each variable has mean of 0 and standard dev of 1
scaled_ind_1 = StandardScaler().fit_transform(a_ind_1a.to_numpy())

# creating the dataframe
scaled_ind_1 = pd.DataFrame(scaled_ind_1, columns=[a_ind_1a_colnam])

# kmeans=kmeans.fit(df_1.drop(['Country Name','Country Code','Indicator Name','Indicator Code'],axis=1))

scaled_ind_1.head()

scaled_ind_1.describe()

# changing the datatype
scaled_ind_1 = scaled_ind_1.astype(float)

# running k means clustering
kmeans = kmeans.fit(scaled_ind_1)
scaled_ind_1a = scaled_ind_1

# creating cluster ids
scaled_ind_1a['clust_id'] = kmeans.predict(scaled_ind_1)

scaled_ind_1a.head()

t1 = pd.DataFrame()
t1['clust_id'] = scaled_ind_1a['clust_id'].copy()

t1.reset_index(drop=True, inplace=True)

a_ind_1a.reset_index(drop=True, inplace=True)

a_ind_1a_final = pd.concat([a_ind_1a, t1], axis=1)
a_ind_1a_final.head()

# deterining the labels
labels = kmeans.labels_

# finding the centers of the clusters
centers = kmeans.cluster_centers_
print(centers)


# calculate the silhoutte score
print(skmet.silhouette_score(scaled_ind_1a, labels))

# plot using the labels to select colour
plt.figure(figsize=(10.0, 10.0),dpi=720)

for l in range(n_cluster):  
    plt.scatter(scaled_ind_1a[labels == l][a_ind_1a_colnam[0]],
                scaled_ind_1a[labels == l][a_ind_1a_colnam[1]])
   


for ix in range(n_cluster):
    xc, yc = centers[ix, :]
    plt.plot(xc, yc, "dk", markersize=8)

plt.xlabel(a_ind_1a_colnam[0])
plt.ylabel(a_ind_1a_colnam[1])
plt.title('Clustering for 1997 and 2015 for the Renewable electricity output (% of total electricity output)')
plt.show()

#--------------------------------------------------------------------------------------------------------
# 2nd indicator Renewable energy consumption (% of total final energy consumption)
b_ind1 = a_df[a_df['Indicator Code'] == 'EG.FEC.RNEW.ZS']

b_ind1.fillna(0, inplace=True)

b_ind_1a = pd.DataFrame()

# creating clusters for the years 1997 and 2015.
b_ind_1a['1997'] = b_ind1['1997'].copy()
b_ind_1a['2015'] = b_ind1['2015'].copy()

# resetting the column index
b_ind_1a.reset_index(drop=True)

b_ind_1a.info()


b_ind_1a_colnam = b_ind_1a.columns.values.tolist()

# create scaled DataFrame where each variable has mean of 0 and standard dev of 1
scaled_ind_2 = StandardScaler().fit_transform(b_ind_1a.to_numpy())

# creating the dataframe
scaled_ind_2 = pd.DataFrame(scaled_ind_2, columns=[b_ind_1a_colnam])

scaled_ind_2.head()

scaled_ind_2.describe()

# changing the datatype
scaled_ind_2 = scaled_ind_2.astype(float)

# running k means clustering
kmeans = kmeans.fit(scaled_ind_2)
scaled_ind_2a = scaled_ind_2
scaled_ind_2a['clust_id'] = kmeans.predict(scaled_ind_2)
scaled_ind_2a.head()

t2 = pd.DataFrame()
t2['clust_id'] = scaled_ind_2a['clust_id'].copy()

t2.reset_index(drop=True, inplace=True)

b_ind_1a.reset_index(drop=True, inplace=True)

b_ind_1a_final = pd.concat([b_ind_1a, t2], axis=1)
b_ind_1a_final.head()

# deterining the labels
labels = kmeans.labels_

# finding the centers of the clusters
centers = kmeans.cluster_centers_
print(centers)


# calculate the silhoutte score
print(skmet.silhouette_score(scaled_ind_2a, labels))

# plot using the labels to select colour
plt.figure(figsize=(10.0, 10.0),dpi=720)

for l in range(n_cluster):  # loop over the different labels
    plt.scatter(scaled_ind_2a[labels == l][b_ind_1a_colnam[0]],
                scaled_ind_2a[labels == l][b_ind_1a_colnam[1]])
    
for ix in range(n_cluster):
    xc, yc = centers[ix, :]
    plt.plot(xc, yc, "dk", markersize=8)

plt.xlabel(b_ind_1a_colnam[0])
plt.ylabel(b_ind_1a_colnam[1])
plt.title('Clustering for 1997 and 2015 for Renewable energy consumption (% of total final energy consumption)')
plt.show()

#--------------------------------------------------------------------------------------------------------
# 3 rd indicator Electricity production from coal sources (% of total)
c_ind1 = a_df[a_df['Indicator Code'] == 'EG.ELC.COAL.ZS']

c_ind1.fillna(0, inplace=True)

c_ind_1a = pd.DataFrame()

# creating clusters for the years 1997 and 2015.
c_ind_1a['1997'] = c_ind1['1997'].copy()
c_ind_1a['2015'] = c_ind1['2015'].copy()

# resetting the column index
c_ind_1a.reset_index(drop=True)

c_ind_1a.info()


c_ind_1a_colnam = c_ind_1a.columns.values.tolist()

# create scaled DataFrame where each variable has mean of 0 and standard dev of 1
scaled_ind_3 = StandardScaler().fit_transform(c_ind_1a.to_numpy())

# creating the dataframe
scaled_ind_3 = pd.DataFrame(scaled_ind_3, columns=[c_ind_1a_colnam])
scaled_ind_3.head()
scaled_ind_3.describe()

# changing the datatype
scaled_ind_3 = scaled_ind_3.astype(float)

# running k means clustering
kmeans = kmeans.fit(scaled_ind_3)

# creating a duplicate dataframe
scaled_ind_3a = scaled_ind_3

scaled_ind_3a['clust_id'] = kmeans.predict(scaled_ind_3)
scaled_ind_3a.head()
t3 = pd.DataFrame()
t3['clust_id'] = scaled_ind_3a['clust_id'].copy()

t3.reset_index(drop=True, inplace=True)

c_ind_1a.reset_index(drop=True, inplace=True)

c_ind_1a_final = pd.concat([c_ind_1a, t3], axis=1)
c_ind_1a_final.head()

# deterining the labels
labels = kmeans.labels_

# finding the centers of the clusters
centers = kmeans.cluster_centers_
print(centers)


# calculate the silhoutte score
print(skmet.silhouette_score(scaled_ind_3a, labels))

# plot using the labels to select colour
plt.figure(figsize=(10.0, 10.0),dpi=720)

for l in range(n_cluster):  # loop over the different labels
    plt.scatter(scaled_ind_3a[labels == l][c_ind_1a_colnam[0]],
                scaled_ind_3a[labels == l][c_ind_1a_colnam[1]])
    
for ix in range(n_cluster):
    xc, yc = centers[ix, :]
    plt.plot(xc, yc, "dk", markersize=8)

plt.xlabel(c_ind_1a_colnam[0])
plt.ylabel(c_ind_1a_colnam[1])
plt.title('Clustering for years 1997 and 2015 for the Electricity production from coal sources (% of total)')
plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------
"""# Choosing India and Brazil for Reneawable energy consumption"""

x1=pd.DataFrame()
x1[0]=a_ind1.iloc[29]
x1[1]=a_ind1.iloc[109]
print('x1=', x1)

x1.info()
x1.head()
x1t=x1.T
x1t.head()

x1_c=x1.T

x1['year']=list(x1_c.columns)
x1.reset_index(drop=True,inplace=True)
 
x1a=x1.T
x1a.reset_index()
x1a.head()

pp_b=x1.iloc[:,2].iloc[33:66]  
print (pp_b)
pp_b=pp_b.astype(float)

# Function to calculate the linear with constants a and b
def linear(x, a, b):
    
    return a*x+b

y_dummy = linear(np.array(pp_b), 5,-0.25)
def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    
    f = n0 / (1 + np.exp(-g*(t - t0)))
    
    return f



# Fit the dummy exponential data

pars, cov = opt.curve_fit(f=linear, xdata=pp_b, ydata=y_dummy,p0=[4e8, 0.5])


# Plot the noisy exponential data
plt.figure(figsize=(10.0, 10.0),dpi=720)
plt.xlabel("Years")
plt.ylabel("Renewable Energy Consumption")
plt.scatter(pp_b, y_dummy, s=20, color='#00b3b3', label='Data')
plt.plot(pp_b, linear(pp_b, *pars), linestyle='--', linewidth=3, color='red',label='Fit')
plt.title('Curve-Fit for the Renewable enrgy consumption')
plt.legend()
plt.show()
















