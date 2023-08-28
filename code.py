# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
data = pd.read_csv('creditrisk_dataset')

# Handle missing values (you can choose different strategies)
data.fillna(data.mean(), inplace=True)

# Convert categorical columns to numerical using Label Encoding
label_encoder = LabelEncoder()
categorical_columns = ['Customer Name', 'Credit Risk', 'Occupation', 'Education Level', 'Marital Status', 'Property']
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])
# Here, we import the required libraries and load the dataset. The missing values are filled using their column mean. Categorical columns are converted to numerical values using Label Encoding.

# Perform Hierarchical Clustering and Initial Dendrogram
data_scaled = StandardScaler().fit_transform(data)
linkage_matrix = linkage(data_scaled, method='average', metric='euclidean')
plt.figure(figsize=(8, 6))
dendrogram(linkage_matrix, color_threshold=np.inf)
plt.title('Initial Hierarchical Dendrogram')
plt.xlabel('Sample Indices')
plt.ylabel('Euclidean Distance')
plt.show()
# Performing Hierarchical Clustering on the scaled data and plotting the initial dendrogram.

# Determine the number of clusters (K) using the cut_distance
cut_distance = 4  # Choose an appropriate cut distance from the dendrogram
hierarchical_k = len(np.where(linkage_matrix[:, 2] > cut_distance)[0]) + 1

# Assign samples to clusters based on the chosen cut
hierarchical_clusters = fcluster(linkage_matrix, hierarchical_k, criterion='maxclust')

# Perform Hierarchical Clustering and Dendrogram after the cut
plt.figure(figsize=(8, 6))
dendrogram(linkage_matrix, color_threshold=cut_distance)
plt.title('Hierarchical Dendrogram with Cut')
plt.xlabel('Sample Indices')
plt.ylabel('Euclidean Distance')
plt.show()
# Determining the number of clusters using a cut distance, and then assigning samples to clusters accordingly. Finally, displaying the hierarchical dendrogram with the cut.

# Add the hierarchical cluster labels to the dataset
data['Hierarchical_Cluster'] = hierarchical_clusters

# Display the dataset divided into hierarchical clusters
for cluster_id in np.unique(hierarchical_clusters):
    cluster_data = data[data['Hierarchical_Cluster'] == cluster_id]
    print(f'Cluster {cluster_id}:')
    print(cluster_data)
    print()
# Adding hierarchical cluster labels to the dataset and displaying data divided into hierarchical clusters.

# Perform K-Means Clustering and determine the optimal K using the Elbow Method and Silhouette Score
inertia = []
silhouette_scores = []
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(data_scaled, kmeans.labels_))

optimal_k_kmeans = int((np.argmin(inertia) + np.argmax(silhouette_scores)) / 2) + 2

print(f'Optimal K calculated as the average of Elbow Method and Silhouette Score: {optimal_k_kmeans}')

# Divide the dataset into K-Means clusters using the optimal K
kmeans_optimal = KMeans(n_clusters=optimal_k_kmeans, random_state=0)
data['KMeans_Cluster'] = kmeans_optimal.fit_predict(data_scaled)

# Display the dataset divided into K-Means clusters
for cluster_id in np.unique(data['KMeans_Cluster']):
    cluster_data = data[data['KMeans_Cluster'] == cluster_id]
    print(f'Cluster {cluster_id}:')
    print(cluster_data)
    print()
# Performing K-Means Clustering, determining the optimal K, and displaying data divided into K-Means clusters.

# One-hot encode categorical variables
data_encoded = pd.get_dummies(data, columns=categorical_columns)

# Random Forest
X = data.drop(['Credit Risk'], axis=1)  # 'Credit Risk' is the target variable
y = data['Credit Risk']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create the Random Forest model
model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')  # You can choose 'micro', 'macro', or 'weighted'
recall = recall_score(y_test, y_pred, average='macro')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Make predictions on the entire dataset
data['Predicted_Credit_Risk'] = model.predict(X)

# Display the dataset with predicted credit risk
print(data)
# One-hot encoding categorical variables, then training a Random Forest model, making predictions, calculating evaluation metrics, and displaying a confusion matrix. Finally, making predictions on the entire dataset and displaying the dataset with predicted credit risk.
