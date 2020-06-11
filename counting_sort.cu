/* Host-side code to perform counting sort 
 * Author: Naga Kandasamy
 * Date modified: May 27, 2020
 * 
 * Compile as follows: make clean && make
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#include "counting_sort_kernel.cu"

struct timeval start, stop;

/* Do not change the range value */
#define MIN_VALUE 0 
#define MAX_VALUE 255

/* Uncomment to spit out debug info */
// #define DEBUG

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

extern "C" int counting_sort_gold(int *, int *, int, int);
int rand_int(int, int);
void print_array(int *, int);
void print_min_and_max_in_array(int *, int);
void compute_on_device(int *, int *, int, int);
int check_if_sorted(int *, int);
int compare_results(int *, int *, int);

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("Usage: %s num-elements\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int num_elements = atoi(argv[1]);
    int range = MAX_VALUE - MIN_VALUE;
    int *input_array, *sorted_array_reference, *sorted_array_d;

    /* Populate input array with random integers between [0, RANGE] */
    printf("Generating input array with %d elements in the range 0 to %d\n", num_elements, range);
    input_array = (int *)malloc(num_elements * sizeof(int));
    if (input_array == NULL) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    
    srand(time(NULL));
    int i;
    for (i = 0; i < num_elements; i++)
        input_array[i] = rand_int (MIN_VALUE, MAX_VALUE);

#ifdef DEBUG
    print_array(input_array, num_elements);
    print_min_and_max_in_array(input_array, num_elements);
#endif

    /* Sort elements in input array using reference implementation. 
     * The result is placed in sorted_array_reference. */
    printf("\nSorting array on CPU\n");
    gettimeofday(&start, NULL);
    
    int status;
    sorted_array_reference = (int *)malloc(num_elements * sizeof(int));
    if (sorted_array_reference == NULL) {
        perror("malloc"); 
        exit(EXIT_FAILURE);
    }
    memset(sorted_array_reference, 0, num_elements);
    
    status = counting_sort_gold(input_array, sorted_array_reference, num_elements, range);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "Gold Execution time = %fs\n", (float) (stop.tv_sec - start.tv_sec\
                    + (stop.tv_usec - start.tv_usec)/(float)1000000));

    if (status == -1) {
        exit(EXIT_FAILURE);
    }

    status = check_if_sorted(sorted_array_reference, num_elements);
    if (status == -1) {
        printf("Error sorting the input array using the reference code\n");
        exit(EXIT_FAILURE);
    }

    printf("Counting sort was successful on the CPU\n");

#ifdef DEBUG
    print_array(sorted_array_reference, num_elements);
#endif

    /* FIXME: Write function to sort elements in the array in parallel fashion. 
     * The result should be placed in sorted_array_mt. */
    printf("\nSorting array on GPU\n");
    sorted_array_d = (int *)malloc(num_elements * sizeof(int));
    if (sorted_array_d == NULL) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    memset(sorted_array_d, 0, num_elements);

    gettimeofday(&start, NULL);
    compute_on_device(input_array, sorted_array_d, num_elements, range);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "GPU Execution time = %fs\n", (float) (stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));

#ifdef DEBUG
    print_array(sorted_array_d, num_elements);
#endif

    /* Check the two results for correctness */
    printf("\nComparing CPU and GPU results\n");
    status = compare_results(sorted_array_reference, sorted_array_d, num_elements);
    if (status == 0)
        printf("Test passed\n");
    else
        printf("Test failed\n");

    exit(EXIT_SUCCESS);
}


/* FIXME: Write the GPU implementation of counting sort */
void compute_on_device(int *input_array, int *sorted_array, int num_elements, int range)
{
    int *d_input_array;
    int size = num_elements * sizeof(int);
    cudaMalloc((void **)&d_input_array, size);
    cudaMemcpy(d_input_array, input_array, size, cudaMemcpyHostToDevice);

    int *d_sorted_array;
    size = num_elements * sizeof(int);
    cudaMalloc((void **)&d_sorted_array, size);

    int* d_histogram;
    size = (range + 1) * sizeof(int);
    cudaMalloc((void **)&d_histogram, size);
    cudaMemset(d_histogram, 0, size);

    int* d_scan;
    size = (range + 1) * sizeof(int);
    cudaMalloc((void **)&d_scan, (range + 1) * sizeof(int));
    cudaMemset(d_scan, 0, size);

    dim3 threads(range + 1);
    dim3 grid(40, 1);
    histogram_kernel_fast<<<grid, threads, size>>>(d_input_array, d_histogram, num_elements, (range + 1));
    cudaDeviceSynchronize();
    grid.x = 1;
    counting_sort_kernel<<<grid, threads, 2 * size>>>(d_input_array, d_sorted_array, d_histogram, d_scan, num_elements, range);
    cudaMemcpy(sorted_array, d_sorted_array, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input_array);
    cudaFree(d_sorted_array);
    cudaFree(d_histogram);
    cudaFree(d_scan);
    return;
}

/* Check if array is sorted */
int check_if_sorted(int *array, int num_elements)
{
    int status = 0;
    int i;
    for (i = 1; i < num_elements; i++) {
        if (array[i - 1] > array[i]) {
            status = -1;
            break;
        }
    }

    return status;
}

/* Check if the arrays elements are identical */ 
int compare_results(int *array_1, int *array_2, int num_elements)
{
    int status = 0;
    int i;
    for (i = 0; i < num_elements; i++) {
        if (array_1[i] != array_2[i]) {
            status = -1;
            break;
        }
    }

    return status;
}

/* Return random integer between [min, max] */ 
int rand_int(int min, int max)
{
    float r = rand()/(float)RAND_MAX;
    return (int)floorf(min + (max - min) * r);
}

/* Print given array */
void print_array(int *this_array, int num_elements)
{
    printf("Array: ");
    int i;
    for (i = 0; i < num_elements; i++)
        printf("%d ", this_array[i]);
    
    printf("\n");
    return;
}

/* Return min and max values in given array */
void print_min_and_max_in_array(int *this_array, int num_elements)
{
    int i;

    int current_min = INT_MAX;
    for (i = 0; i < num_elements; i++)
        if (this_array[i] < current_min)
            current_min = this_array[i];

    int current_max = INT_MIN;
    for (i = 0; i < num_elements; i++)
        if (this_array[i] > current_max)
            current_max = this_array[i];

    printf("Minimum value in the array = %d\n", current_min);
    printf("Maximum value in the array = %d\n", current_max);
    return;
}


