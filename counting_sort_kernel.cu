/* Write GPU code to perform the step(s) involved in counting sort. 
 Add additional kernels and device functions as needed. */


__global__ void counting_sort_kernel(int *input_array, int *sorted_array, int num_elements, int range, int *hist, int *scan)
{
    extern __shared__ int temp[];

    int threadID =  blockIdx.x * blockDim.x + threadIdx.x;  
    int tid = threadIdx.x;

    int pout = 0, pin = 1;
    int n = range + 1;
    
    temp[tid] = (tid > 0) ? hist[tid - 1] : 0;
    
    int offset;
    for (offset = 1; offset < n; offset *= 2) {
        pout = 1 - pout;
        pin = 1 - pout;
        __syncthreads();
    
        temp[pout * n + tid] = temp[pin * n + tid];
       
   	    if (tid >= offset)
            temp[pout * n + tid] += temp[pin * n + tid - offset];
    }
       
    __syncthreads(); 
    
    scan[tid] = temp[pout * n + tid];
    
    int j;
    int start_idx = scan[threadID];
    if (hist[threadID] != 0) 
      for (j = 0; j < hist[threadID]; j++)
	     sorted_array[start_idx + j] = threadID;

    return;
}

__global__ void hist_kernel(int *input_array, int num_elements, int *hist)
{
    extern __shared__ int temp[];
    
    temp[threadIdx.x] = 0;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while (i < num_elements) {
        atomicAdd(&temp[input_array[i]], 1);
        i += offset;
    }
     __syncthreads();

    atomicAdd(&(hist[threadIdx.x]), temp[threadIdx.x]);
}