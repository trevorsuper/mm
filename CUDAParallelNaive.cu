#include <stdlib.h>
#include <stdio.h>

const int matrixSize = 512;

__global__ void kernel_mm(int* matA, int* matB, int* product, int matrixSize){
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int row = threadIndex / matrixSize;
    int column = threadIndex % matrixSize;
    for (int k = 0; k < matrixSize; k++){
        product[row * matrixSize + column] += matA[row * matrixSize + k] * matB[k * matrixSize + column];
    }
}

int* allocateMatrix(int n){
    return (int*)malloc(n * n * 4);
}

void fillMatrix(int n, int* mat){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            mat[i * n + j] = rand() % 5;
        }
    }
}

void printMatrix(int n, int* mat){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            printf("%d ", mat[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int* hostMat1 = allocateMatrix(matrixSize);
    fillMatrix(matrixSize, hostMat1);
    int* hostMat2 = allocateMatrix(matrixSize);
    fillMatrix(matrixSize, hostMat2);
    int* hostMat3 = allocateMatrix(matrixSize);
    for (int i = 0; i < matrixSize; i++){
        for (int j = 0; j < matrixSize; j++){
            hostMat3[i * matrixSize + j] = 0;
        }
    }

    //printMatrix(matrixSize, hostMat1);
    //printMatrix(matrixSize, hostMat2);
    //printMatrix(matrixSize, hostMat3);

    size_t bytes = matrixSize * matrixSize * sizeof(int);
    int *deviceMat1, *deviceMat2, *deviceMat3;
    cudaMalloc(&deviceMat1, bytes);
    cudaMalloc(&deviceMat2, bytes);
    cudaMalloc(&deviceMat3, bytes);
    cudaMemcpy(deviceMat1, hostMat1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMat2, hostMat2, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMat3, hostMat3, bytes, cudaMemcpyHostToDevice);

    const int threads = 512;
    const int blocks = (matrixSize * matrixSize) / threads;
    dim3 gridSize(blocks, 1, 1);
    dim3 blockSize(threads, 1, 1);

    cudaEventRecord(start);
    kernel_mm<<<gridSize, blockSize>>>(deviceMat1, deviceMat2, deviceMat3, matrixSize);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaMemcpy(hostMat3, deviceMat3, bytes, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel duration: %3.6fms\n", milliseconds);
    //printMatrix(matrixSize, hostMat3);

    cudaFree(deviceMat1);
    cudaFree(deviceMat2);
    cudaFree(deviceMat3);
    free(hostMat1);
    free(hostMat2);
    free(hostMat3);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
