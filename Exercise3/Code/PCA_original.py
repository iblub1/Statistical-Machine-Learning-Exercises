#project data back to orgiginal space
def reconstruct_data(data_normalized, eigenvectors, num_components, mean, std_deviation):
    projected_data = projection(data_normalized, eigenvectors, num_components)
    data_reconstructed = np.transpose((np.matmul(eigenvectors[:,:num_components],projected_data)))
    data_reconstructed = ( data_reconstructed *std_deviation ) + mean
    return data_reconstructed

def nrmse(data_reconstructed,data):
    y_max = np.amax(data,axis=0)
    y_min = np.amin(data,axis=0)
    error = data - data_reconstructed
    error_squared = error**2
    error_squared_sum = np.sum(error_squared,axis=0)
    nrmse = np.sqrt( 1/data.shape[0] * ( error_squared_sum ) ) / ( y_max - y_min ) 
    return nrmse

num_components = 0
for i in range(4):
    num_components += 1
    data_reconstructed = reconstruct_data(data_normalized, eigenvectors, num_components, mean, std_deviation)    
    print(nrmse(data_reconstructed,data[:,:4]))