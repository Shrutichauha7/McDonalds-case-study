# McDonalds-case-study
Objective
The goal of this study is to analyze customer perceptions of McDonald's using survey responses. The study leverages Principal Component Analysis (PCA) and Clustering (k-Means) to uncover key insights about how customers perceive different attributes of McDonald's food.

1. Data Description
The dataset contains 1,453 survey responses on 15 attributes describing McDonald's food. The first 11 columns contain responses to attributes like:
✅ Yummy (Tasty or not)
✅ Convenient (Easy to access)
✅ Spicy (Perception of spiciness)
✅ Fattening (Health impact)
✅ Greasy (Oily or not)
✅ Fast (Speed of service)
✅ Cheap (Affordability)
✅ Tasty (General taste perception)
✅ Expensive (Price perception)
✅ Healthy (Nutritional value)
✅ Disgusting (Negative sentiment)

2. Data Preprocessing
Converting Responses to Binary Format
The dataset contains "Yes" or "No" responses, which we converted to binary values:
🔹 "Yes" → 1
🔹 "No" → 0
This transformation allows us to apply numerical techniques like PCA and clustering.

python
Copy
Edit
MD_x = (mcdonalds.iloc[:, :11] == "Yes").astype(int)
3. Principal Component Analysis (PCA)
Why PCA?
🔹 To reduce dimensionality while preserving maximum variance.
🔹 To identify the most important factors driving customer perception.

PCA Results
python
Copy
Edit
pca = PCA()
MD_pca = pca.fit_transform(MD_x)

# Explained variance and cumulative proportion
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
📌 The first 5 principal components explain ~76.79% of the variance, meaning they capture most of the dataset’s information.

4. Interpreting PCA Components
python
Copy
Edit
print("PC  Std Dev  Prop. of Var  Cumulative Prop.")
for i, (std_dev, prop_var, cum_var) in enumerate(zip(np.sqrt(pca.explained_variance_), explained_variance, cumulative_variance), start=1):
    print(f"PC{i}: {std_dev:.4f}  {prop_var:.4f}  {cum_var:.4f}")
Findings from PCA:
1️⃣ PC1 (29.94%) – Represents "Convenience & Speed", as "Convenient," "Fast," and "Cheap" contribute highly.
2️⃣ PC2 (19.28%) – Represents "Health vs. Taste", where "Healthy" and "Tasty" are in contrast.
3️⃣ PC3 (13.31%) – Highlights "Fattening & Greasiness" perceptions.
4️⃣ PC4 & PC5 (~8% each) – Capture "Expensive" vs. "Disgusting" sentiment.

📊 Visualization of PCA Projection

python
Copy
Edit
plt.scatter(MD_pca[:, 0], MD_pca[:, 1], color='grey', alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid()
plt.show()
📌 Most customers see McDonald's as fast & cheap, but opinions on health and taste vary.

5. Clustering Analysis (K-Means)
Why Clustering?
🔹 To group customers based on perception similarities.
🔹 To identify market segments for targeted marketing.

Clustering Execution
python
Copy
Edit
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=1234, n_init=10)
MD_clusters = kmeans.fit_predict(MD_x)
📌 We segmented customers into 4 groups based on their opinions.

Choosing the Optimal Number of Clusters
We use stepFlexclust (in R) / Elbow Method (in Python) to determine the ideal number of clusters (2 to 8).

python
Copy
Edit
inertia = []
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=1234, n_init=10)
    kmeans.fit(MD_x)
    inertia.append(kmeans.inertia_)

plt.plot(range(2, 9), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Clusters')
plt.show()
📌 4 clusters provide a balance between variance explained and overfitting.

6. Interpretation of Customer Segments
Cluster	Characteristics	Marketing Implications
Segment 1: Fast-Food Lovers	Love McDonald's for speed, convenience, and affordability.	Promote value meals & quick service.
Segment 2: Health-Conscious Consumers	View McDonald's as unhealthy but tasty.	Promote salads, healthy options, and transparency.
Segment 3: Premium Customers	Believe McDonald's is expensive but tasty.	Focus on premium burgers & better dining experience.
Segment 4: Critics	Find McDonald's greasy, fattening, and disgusting.	Improve brand perception & introduce healthier alternatives.
7. Conclusion & Business Recommendations
📌 Key Takeaways ✅ Most customers appreciate speed & affordability.
✅ Health concerns are a major issue – more than half see McDonald’s as fattening.
✅ Some customers perceive McDonald's as expensive, suggesting a need for better pricing strategies.

📢 Strategic Recommendations 🎯 For Segment 1 (Fast-Food Lovers): Offer loyalty programs & faster service.
🥗 For Segment 2 (Health-Conscious): Promote low-calorie & organic menu items.
🍔 For Segment 3 (Premium Customers): Introduce high-end gourmet burgers.
🚀 For Segment 4 (Critics): Improve brand image & food quality transparency.

Final Thoughts
This analysis provides data-driven insights into customer perceptions of McDonald's. Using PCA for dimensionality reduction and clustering for segmentation, we can tailor marketing strategies to different customer groups.







