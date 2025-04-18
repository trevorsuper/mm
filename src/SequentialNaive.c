#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void fillMatrix(int** matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = rand() % 100;
        }
    }
}

int** allocateMatrix(int size) {
    int** matrix = malloc(size * sizeof(int*));
    for (int i = 0; i < size; i++) {
        matrix[i] = malloc(size * sizeof(int));
    }
    return matrix;
}

void freeMatrix(int** matrix, int size) {
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int main() {
    int n = 512;

    srand(time(NULL)); // Seed for randomness

    int** A = allocateMatrix(n);
    int** B = allocateMatrix(n);
    int** C = allocateMatrix(n);

    fillMatrix(A, n);
    fillMatrix(B, n);

    clock_t start = clock();
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
