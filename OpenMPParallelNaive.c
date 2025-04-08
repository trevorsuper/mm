// test and record the time to multiply matrices of these sizes
// 512x512 by 512x512
Threads:  1 | Time taken: 0.365000 seconds
Threads:  2 | Time taken: 0.185000 seconds
Threads:  4 | Time taken: 0.096000 seconds
Threads:  8 | Time taken: 0.065000 seconds
Threads: 16 | Time taken: 0.052000 seconds
// 1024x1024 by 1024x1024
Threads:  1 | Time taken: 3.494000 seconds
Threads:  2 | Time taken: 1.853000 seconds
Threads:  4 | Time taken: 1.066000 seconds
Threads:  8 | Time taken: 0.669000 seconds
Threads: 16 | Time taken: 0.506000 seconds
// 2048x2048 by 2048x2048
Threads:  1 | Time taken: 36.756000 seconds
Threads:  2 | Time taken: 19.491000 seconds
Threads:  4 | Time taken: 11.863000 seconds
Threads:  8 | Time taken: 7.386000 seconds
Threads: 16 | Time taken: 6.678000 seconds
// 4096x4096 by 4096x4096
Threads:  1 | Time taken: 402.261000 seconds
Threads:  2 | Time taken: 201.072000 seconds
Threads:  4 | Time taken: 123.896000 seconds
Threads:  8 | Time taken: 81.206000 seconds
Threads: 16 | Time taken: 69.284000 seconds
// 8192x8192 by 8192x8192
Threads:  1 | Time taken: 3545.138000 seconds
Threads:  2 | Time taken: 1774.221000 seconds
Threads:  4 | Time taken: 1020.191000 seconds
Threads:  8 | Time taken: 646.426000 seconds
Threads: 16 | Time taken: 641.261000 seconds
    
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
