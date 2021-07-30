!pip install japanize-matplotlib
import japanize_matplotlib
from sklearn.decomposition import PCA


# CBoW
pca = PCA(n_components=2)
pca.fit(cbow_weight)
data_pca = pca.transform(cbow_weight)
for i in range(n_vocab):
    plt.plot(data_pca[i][0], data_pca[i][1], ms=5.0, zorder=2, marker="x")
    plt.annotate(id2w[i], (data_pca[i][0], data_pca[i][1]))
plt.title("CBoW")
plt.show()

# skip-gram
pca = PCA(n_components=2)
pca.fit(sg_weight)
data_pca = pca.transform(sg_weight)
for i in range(n_vocab):
    plt.plot(data_pca[i][0], data_pca[i][1], ms=5.0, zorder=2, marker="x")
    plt.annotate(id2w[i], (data_pca[i][0], data_pca[i][1]))
plt.title("skip-gram")
plt.show()