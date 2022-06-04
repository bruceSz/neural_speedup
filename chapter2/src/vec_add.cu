#include <iostream>
#include <cstdlib>
#include <sys/time.h>

using namespace std;

void vecAdd(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

__global__
void vecAddKernel(float* A_d, float* B_d, float* C_d, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) C_d[i] = A_d[i] + B_d[i];
}


void cpu_add(int n) {
    cout << "cpu add: " << n << endl;

    size_t size = n * sizeof(float);

    // host memery
    float *a = (float *)malloc(size);
    float *b = (float *)malloc(size);
    float *c = (float *)malloc(size);

    for (int i = 0; i < n; i++) {
        float af = rand() / double(RAND_MAX);
        float bf = rand() / double(RAND_MAX);
        a[i] = af;
        b[i] = bf;
        c[i] = a[i] + b[i];
    }
    printf("add one.\n");

   struct timeval t1, t2;

    gettimeofday(&t1, NULL);

    vecAdd(a, b, c, n);

    gettimeofday(&t2, NULL);

    //for (int i = 0; i < 10; i++) 
    //    cout << vecA[i] << " " << vecB[i] << " " << vecC[i] << endl;
    double timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
    cout  << timeuse << endl;

    free(a);
    free(b);
    free(c);
}


void gpu_add(int n) {
    cout  << "gpu add: " << n << endl;

    size_t size = n * sizeof(float);

    // host memery
    float *a = (float *)malloc(size);
    float *b = (float *)malloc(size);
    float *c = (float *)malloc(size);

    for (int i = 0; i < n; i++) {
        float af = rand() / double(RAND_MAX);
        float bf = rand() / double(RAND_MAX);
        a[i] = af;
        b[i] = bf;
    }

    float *da = NULL;
    float *db = NULL;
    float *dc = NULL;

    cudaMalloc((void **)&da, size);
    cudaMalloc((void **)&db, size);
    cudaMalloc((void **)&dc, size);

    cudaMemcpy(da,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(db,b,size,cudaMemcpyHostToDevice);
    cudaMemcpy(dc,c,size,cudaMemcpyHostToDevice);

    struct timeval t1, t2;

    int threadPerBlock = 256;
    int blockPerGrid = (n + threadPerBlock - 1)/threadPerBlock;
    printf("threadPerBlock: %d \nblockPerGrid: %d \n",threadPerBlock,blockPerGrid);

    gettimeofday(&t1, NULL);

    vecAddKernel <<< blockPerGrid, threadPerBlock >>> (da, db, dc, n);

    gettimeofday(&t2, NULL);

    cudaMemcpy(c,dc,size,cudaMemcpyDeviceToHost);

    //for (int i = 0; i < 10; i++) 
    //    cout << vecA[i] << " " << vecB[i] << " " << vecC[i] << endl;
    double timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
    cout << timeuse << endl;

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    free(a);
    free(b);
    free(c);
}


int main(int argc, char *argv[]) {

    cout << "argc: " << argc << endl;
    if (argc < 2) {
        return 0;
    }
    int n = atoi(argv[1]);
    cpu_add(n);
    gpu_add(n);
    return 0;

}