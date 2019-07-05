#project data into lower dimensional space
def projection(data_normalized, eigenvectors, num_components):
    eigenvectors = eigenvectors[:,:num_components]
    data_normalized = np.matmul(np.transpose(eigenvectors),np.transpose(data_normalized))
    return data_normalized

#plot lower dimensional space
num_components = 2
projected_data_2comp = projection(data_normalized,eigenvectors,num_components)
x = projected_data_2comp[0,:]
y = projected_data_2comp[1,:]
classes = data[:,4]
colors = ['red','green','blue']
unique = list(set(classes))
plant = None
for i, u in enumerate(unique):
    xi = [x[j] for j  in range(len(x)) if classes[j] == u]
    yi = [y[j] for j  in range(len(x)) if classes[j] == u]
    if colors[i] == 'red':
        plant = 'Setosa'
    elif colors[i] == 'green':
        plant = 'Versicolour'
    elif colors[i] == 'blue':
        plant = 'Virginica'
    plt.scatter(xi, yi, c=colors[i], label=plant)
plt.legend()
plt.title("2-dimensional projection of the data")
plt.savefig('PCA_scatter.png',dpi=300)
plt.show()