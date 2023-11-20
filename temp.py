import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

excel_file_path = r'C:\Users\berra_sevinc\Desktop\mydata\ccgeneral.csv'  
data = pd.read_csv(excel_file_path)
print(data.head(8950))

data.info()

print(data.isnull().sum())

data['CREDIT_LIMIT'].fillna(data['CREDIT_LIMIT'].mean(), inplace=True)
data['MINIMUM_PAYMENTS'].fillna(data['MINIMUM_PAYMENTS'].mean(), inplace=True)
data.info()

print(data.describe())


data.set_index(data.CUST_ID, inplace= True)
print(data.head())

del data['CUST_ID']
print(data.head())


data.hist(figsize=(15, 12), bins=18)
plt.suptitle("BEHAVIOURS", y=1.02)
plt.show()

correlation_matrix = data[['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
                            'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
                            'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
                            'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
                            'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT',
                            'TENURE']].corr()


plt.figure(figsize=(20, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Behaviors')
plt.show()


lower_threshold = 0.65
upper_threshold = 1.0
strong_correlations = correlation_matrix[((correlation_matrix > lower_threshold) & (correlation_matrix < upper_threshold))].stack().reset_index()
strong_correlations.columns = ['Feature 1', 'Feature 2', 'Correlation']


correlation_table_np = strong_correlations.to_numpy()


print("Strong Correlation")
print(correlation_table_np)
df = pd.DataFrame(strong_correlations, columns=['Feature 1', 'Feature 2', 'Correlation'])

print(df)

lower_threshold_ = -0.4
upper_threshold_ = 0.1
low_correlations = correlation_matrix[((correlation_matrix > lower_threshold_) & (correlation_matrix < upper_threshold_))].stack().reset_index()
low_correlations.columns = ['Feature 1', 'Feature 2', 'Correlation']


correlation_table__np = low_correlations.to_numpy()


print("Low Correlation")
print(correlation_table__np)
df = pd.DataFrame(low_correlations, columns=['Feature 1', 'Feature 2', 'Correlation'])

print(df)


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

data_scaled_df = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)

print(data_scaled_df.head(8949))

outliers = (data_scaled_df > 2) | (data_scaled_df < -2)

for column in data.columns:
    data_scaled_df.loc[outliers[column], column] = data[column].mean() 

print(data_scaled_df.head(8949))


inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(data)
    inertia.append(kmeans.inertia_)
plt.plot(range(1, 11), inertia,marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()


kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)


sns.scatterplot(data=data, x='PURCHASES', y='PAYMENTS', hue='Cluster', palette='viridis')
plt.title('KMeans Clustering Results')
plt.show()



pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

data_pca_df = pd.DataFrame(data_pca, columns=['PCA1', 'PCA2'], index=data.index)


plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue=data['Cluster'], palette='viridis', data=data_pca_df)
plt.title('PCA Clustering Results')
plt.xlabel('PURCHASES')
plt.ylabel('PAYMENTS')
plt.show()



