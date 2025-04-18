#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1024  // Can go higher now!

void multiply_matrices(int **A, int **B, int **C) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void initialize_matrix(int **matrix) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            matrix[i][j] = rand() % 10;
}

int main() {
    // Dynamically allocate memory
    int **A = malloc(N * sizeof(int *));
    int **B = malloc(N * sizeof(int *));
    int **C = malloc(N * sizeof(int *));
    for (int i = 0; i < N; i++) {
        A[i] = malloc(N * sizeof(int));
        B[i] = malloc(N * sizeof(int));
        C[i] = malloc(N * sizeof(int));
    }

    // Initialize matrices once
    initialize_matrix(A);
    initialize_matrix(B);

    // Try different thread counts
    for (int t = 1; t <= 16; t *= 2) {
        omp_set_num_threads(t);

        // Clear matrix C before each test
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                C[i][j] = 0;

        double start = omp_get_wtime();
        multiply_matrices(A, B, C);
        double end = omp_get_wtime();

        printf("Threads: %2d | Time taken: %f seconds\n", t, end - start);
    }

    // Free memory
    for (int i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A); free(B); free(C);

    return 0;
}