# Credit-Risk-Analysis
This repository contains code and data for performing credit risk analysis using clustering techniques and random forest classification. The aim of this project is to uncover patterns within a credit risk dataset and make accurate predictions for risk assessment and management.

**Introduction**
Credit risk assessment is an important aspect of the financial industry. This project explores the application of clustering techniques, such as hierarchical clustering and K-Means clustering, and random forest classification for credit risk analysis. By identifying patterns in the dataset, we can gain insights into risk profiles and make informed decisions.

**Data**
The credit risk dataset used in this project consists of customer information, financial attributes, and risk classifications. The dataset has been preprocessed to handle missing values, encode categorical variables, and scale features for compatibility with the analysis techniques.

**Methods**
In this project, we employ hierarchical clustering and K-Means clustering techniques to uncover inherent structures within the credit risk dataset.

* **Hierarchical Clustering**: We apply hierarchical clustering to the preprocessed dataset. This technique creates a hierarchy of clusters by iteratively merging or splitting clusters based on a distance metric. The resulting dendrogram provides a visual representation of the clustering structure, allowing us to identify distinct groups of customers with similar risk profiles. We determine the optimal number of clusters by cutting the dendrogram at a threshold distance and assign samples to clusters using the fcluster function.

* **K-Means Clustering**: We also utilize K-Means clustering to partition the preprocessed dataset into K clusters. This algorithm iteratively assigns data points to the nearest cluster centroid and updates the centroids based on the mean value of the assigned points. By leveraging the within-cluster sum of squares and the silhouette score, we determine the optimal number of clusters. This approach enables us to identify groups of customers with similar risk profiles, providing valuable insights for risk assessment and management.

Additionally, we employ random forest classification for predictive modeling to make accurate credit risk predictions.

**Random Forest Classification**: The random forest algorithm is an ensemble learning technique that combines multiple decision trees to make predictions. In this project, we utilize random forest classification to predict credit risk. We divide the dataset into training and testing sets, train the random forest model on the training data, and evaluate its performance on the testing data. We measure various performance metrics such as accuracy, precision, recall, and F1-score to assess the model's predictive capabilities. By analyzing the feature importance provided by the random forest model, we can identify the key factors influencing credit risk and gain insights into the risk assessment process.

**Results**
The application of clustering techniques and random forest classification yields valuable results for credit risk analysis.

* **Clustering Analysis**: The hierarchical clustering analysis reveals distinct groups of customers with similar risk profiles. By examining the dendrogram, we can visually observe the clustering structure and identify meaningful clusters. These clusters provide insights into the underlying patterns within the credit risk dataset, enabling more targeted risk assessment and management strategies. The K-Means clustering analysis further reinforces these findings, allowing us to validate and refine the identified clusters.
<img width="797" alt="image" src="https://github.com/Simidio/Credit-Risk-Analysis/assets/117855290/7c7a1ee0-d25a-430e-861c-358b81a51966">

* **Random Forest Classification**: The random forest classification model achieves high predictive accuracy for credit risk assessment. By training the model on the dataset and evaluating its performance on the testing data, we can make accurate predictions about credit risk. The model's performance metrics, such as accuracy, precision, recall, and F1-score, demonstrate its effectiveness in classifying customers into risk categories. Additionally, by analyzing the feature importance provided by the random forest model, we can identify the key factors that significantly influence credit risk, providing valuable insights for risk assessment and decision-making.
<img width="743" alt="image" src="https://github.com/Simidio/Credit-Risk-Analysis/assets/117855290/1ca7366c-b461-48da-8e48-7f508a4ae5e4">

The results obtained from the clustering analysis and random forest classification contribute to the understanding of credit risk analysis and enable more informed decision-making in the financial industry. These insights can aid in developing effective risk management strategies and improving the overall credit risk assessment process.
