from preprocess_data import prepare_data
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

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



if __name__ == '__main__':
    file_path = "../data/credit_card_data.csv"
    data = prepare_data(file_path)

    sns.catplot(kind="box", data=data)
    plt.show()

    #transform data so it can be represented to 2D
    pca = PCA(n_components=0.95)
    pca.fit(data)
    principalComponents = pca.transform(data)

    find_optimal_K(principalComponents)

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(principalComponents)
    labels = kmeans.predict(principalComponents)

    fig, ax = plt.subplots()
    ax.set_xlabel("PC 0")
    ax.set_ylabel("PC 1")
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1],
                c=labels.astype(float))
    plt.show()




