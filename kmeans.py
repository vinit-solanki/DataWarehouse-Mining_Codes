import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()
df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)

print("Feature columns:", df.columns)
X = df 
y = cancer.target

scalar = StandardScaler()
X_scaled = scalar.fit_transform(df)

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

kmeans = KMeans(n_clusters=2, random_state=42)

kmeans.fit(X_scaled)

silhouette_avg = silhouette_score(X_scaled,kmeans.labels_)
print("Silhouette Score:", silhouette_avg)


plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis', marker='o', edgecolor='k', alpha=0.6)
plt.title('KMeans Clustering of Breast Cancer Dataset')
plt.xlabel(cancer.feature_names[0])  # Feature 1
plt.ylabel(cancer.feature_names[1])  # Feature 2
# Overlay the actual labels for comparison
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='coolwarm', marker='x', alpha=0.3, label='Actual Labels')
plt.legend(['Cluster Labels', 'Actual Labels'])
plt.grid()
plt.show()