from preprocess_data import prepare_data
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
from decision_tree import cluster_report

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

    #Show correlation between data variables
    cut_data = data[["PURCHASES", "BALANCE", "PAYMENTS", "PURCHASES_FREQUENCY","CASH_ADVANCE"]]
    correlation = cut_data.corr()
    sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns, annot=True)
    plt.show()

if __name__ == '__main__':
    file_path = "../data/credit_card_data.csv"
    data = prepare_data(file_path)
    show_eda_information(data)

    #Transform data to lower dimensions so it can be represented to 2D
    pca = PCA(n_components=0.95)
    pca.fit(data)
    principalComponents = pca.transform(data)

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
    ax.set_title("K-means Clustering with 2 dimensions.")
    ax.set_xlabel("PC 0")
    ax.set_ylabel("PC 1")



    g = plt.scatter(principalComponents[:, 0], principalComponents[:, 1],
                c=labels.astype(float), label=legendLabels[0])  # visualize clustering according to first two prinicple components



    ax.legend()
    #
    # plt.title('K-means Clustering with 2 dimensions')
    # reduced_data = pd.DataFrame()
    # reduced_data['PC 0'] = principalComponents[:, 0]
    # reduced_data['PC 1'] = principalComponents[:, 1]
    # reduced_data['cluster'] = labels
    # g = sns.scatterplot(x='PC 0', y='PC 1', hue='cluster', data=reduced_data, edgecolor='k')
    # leg = g.legend_
    # # g.fig.suptitle("K-means Clustering with 2 dimensions")
    # leg.get_texts()[0].set_text('Clusters')
    # # for t, l in zip(leg.texts, legend): t.set_text(l)
    # print(legendLabels)
    # for i, label in enumerate(legendLabels):
    #     # i+1 because i=0 is the title, and i starts at 0
    #     print( i + 1)
    #     # print(leg.get_texts()[i + 1])
    #     # print(i + 1)
    #     # leg.get_texts()[i + 1].set_text("aaa")
    #
    #     # # leg.set_bbox_to_anchor([0.7, 1.3])
    # #
    # # leg._loc = 2



    plt.show()



