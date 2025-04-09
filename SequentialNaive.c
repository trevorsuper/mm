// test and record the time to multiply matrices of these sizes
// 512x512 by 512x512
// 1024x1024 by 1024x1024
// 2048x2048 by 2048x2048
// 4096x4096 by 4096x4096
// 8192x8192 by 8192x8192
// 16384x16384 by 16384x16384
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// this creates a matrix with random float values
void fillMatrix(float** matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = (float)(rand() % 100); // Random numbers 0–99
        }
    }
}

// memory allocation for a matrix
float** allocateMatrix(int size) {
    float** matrix = (float**)malloc(size * sizeof(float*));
    for (int i = 0; i < size; i++) {
        matrix[i] = (float*)malloc(size * sizeof(float));
    }
    return matrix;
}

// Free the allocated matrix
void freeMatrix(float** matrix, int size) {
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int main() {
    int n;
    printf("Enter the size of the square matrix (e.g., 500): ");
    scanf("%d", &n);

    srand(time(NULL)); // Seed for randomness

    // Allocate and fill matrices A and B with random values
    float** A = allocateMatrix(n);
    float** B = allocateMatrix(n);
    float** C = allocateMatrix(n);

    fillMatrix(A, n);
    fillMatrix(B, n);

    clock_t start = clock();
    // Naive matrix multiplication: C = A * B
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time taken for matrix multiplication: %.6f seconds\n", time_taken);

    freeMatrix(A, n);
    freeMatrix(B, n);
    freeMatrix(C, n);

    return 0;
}
