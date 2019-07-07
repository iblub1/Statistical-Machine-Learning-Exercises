# do PCA
def PCA(data_normalized):
    cov = np.cov(data_normalized,rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    explained_var = eigenvalues / sum(eigenvalues)
    return eigenvalues, eigenvectors, explained_var

eigenvalues = PCA(data_normalized)[0]
eigenvectors = PCA(data_normalized)[1]
explained_var = PCA(data_normalized)[2]

# plot explained cumulative variance over number of principal components
fig, ax = plt.subplots()
explained_var_cumulative = np.cumsum(explained_var)
objects = ('1', '2', '3', '4')
y_pos = np.arange(len(objects))
plt.axhline(y=0.95, xmin=0, xmax=4)
plt.xticks(y_pos, objects)
plt.yticks(np.arange(0, 1.05, step=0.05))
plt.ylabel('cumulative explained variance of dataset')
plt.xlabel('number of principal components')
plt.title('cumulative explained variance of dataset over number of components')
rects = ax.bar(y_pos, np.around(explained_var_cumulative, decimals=3))
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rects)
plt.savefig('explained_var_cumulative.png',dpi=300)
plt.show()
print(explained_var_cumulative)