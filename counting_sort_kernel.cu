/* Write GPU code to perform the step(s) involved in counting sort. 
 Add additional kernels and device functions as needed. */
__global__ void counting_sort_kernel(int *input_array, int *sorted_array, int *histogram, int *scan, int num_elements, int range)
{
    extern __shared__ int temp[];

    int threadID =  blockIdx.x * blockDim.x + threadIdx.x;  
    int blockID = threadIdx.x;

    int pout = 0, pin = 1;
    int n = range + 1;
    
    temp[blockID] = (blockID > 0) ? histogram[blockID - 1] : 0;
    
    int offset;
    for (offset = 1; offset < n; offset *= 2) {
        pout = 1 - pout;
        pin = 1 - pout;
        __syncthreads();
    
        temp[pout * n + blockID] = temp[pin * n + blockID];
       
   	    if (blockID >= offset)
            temp[pout * n + blockID] += temp[pin * n + blockID - offset];
    }
       
    __syncthreads(); 
    
    scan[blockID] = temp[pout * n + blockID];
    
    int j;
    int start_idx = scan[threadID];
    if (histogram[threadID] != 0) 
      for (j = 0; j < histogram[threadID]; j++)
	     sorted_array[start_idx + j] = threadID;

    return;
}

__global__ void histogram_kernel_fast(int *input_data, int *histogram, int num_elements, int histogram_size)
{
    extern __shared__ unsigned int s[];
    
    if(threadIdx.x < histogram_size)
        s[threadIdx.x] = 0;
    
    __syncthreads();

    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    while (offset < num_elements) {
        atomicAdd(&s[input_data[offset]], 1);
        offset += stride;
    }

    __syncthreads();

    if(threadIdx.x < histogram_size)
        atomicAdd(&(histogram[threadIdx.x]), s[threadIdx.x]);
}