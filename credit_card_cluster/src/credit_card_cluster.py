from preprocess_data import prepare_data
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from decision_tree import cluster_report

colors = ['green','orange','pink','dodgerblue','red']

def find_optimal_K(data):
    #Elbow method
    distortions = []
    for i in range(1, 11):
        km = KMeans(
            n_clusters=i, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        km.fit(data)
        distortions.append(km.inertia_)

    # plot
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()

def show_eda_information(data):
    # Visualize box plot after data preprocess(there shouldn't be outliers)
    chart = sns.catplot(kind="box", data=data)
    chart.set_xticklabels(rotation=45, horizontalalignment='right')
    plt.show()

    #Show correlation between data variables
    cut_data = data[["PURCHASES", "BALANCE", "PAYMENTS", "PURCHASES_FREQUENCY","CASH_ADVANCE"]]
    correlation = cut_data.corr()
    sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns, annot=True)
    plt.show()

if __name__ == '__main__':
    file_path = "../data/credit_card_data.csv"
    data = prepare_data(file_path)
    show_eda_information(data)

    #Transform data to lower dimensions so it can be represented to 2D, choose number of components so that they explain 90% of the variance
    pca = PCA(n_components=0.9)
    pca.fit(data)
    principalComponents = pca.transform(data)
    #print(len(principalComponents[0])) There are 8 dimension now for each row

    find_optimal_K(principalComponents)

    #We use 5 for cluster number based on elbow method for finding optimal K number
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(principalComponents)
    labels = kmeans.predict(principalComponents)

    #Define cluster labels using decision tree
    cluster_rules = cluster_report(data, labels)

    #Visualize the results of clastering in 2D
    legendLabels = []
    for i in np.unique(labels):
        legendLabels.append('Cluster ' + str(i) + ' : ' + cluster_rules[i])

    fig, ax = plt.subplots()
    ax.set_title("K-means Clustering with 2 dimensions'")
    ax.set_xlabel("PC 0")
    ax.set_ylabel("PC 1")
    # visualize clustering according to first two prinicple components
    scatter = plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=labels.astype(float))
    legend = ax.legend(scatter.legend_elements()[0], legendLabels, loc="lower left", title="Clusters")
    legend.set_bbox_to_anchor([-0.1, -0.4])
    ax.add_artist(legend)
    plt.show()