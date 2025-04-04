// test and record the time to multiply matrices of these sizes
// 512x512 by 512x512
// 1024x1024 by 1024x1024
// 2048x2048 by 2048x2048
// 4096x4096 by 4096x4096
// 8192x8192 by 8192x8192
// 16384x16384 by 16384x16384
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 512  // Matrix size (adjust as needed)

void multiply_matrices(int A[N][N], int B[N][N], int C[N][N]) {
    int i, j, k;

    // Parallelizing outer loop with OpenMP
    #pragma omp parallel for private(i, j, k) shared(A, B, C)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            C[i][j] = 0;
            for (k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void initialize_matrix(int matrix[N][N]) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            matrix[i][j] = rand() % 10;  // Fill with small random numbers for testing
}

void print_matrix(int matrix[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%d ", matrix[i][j]);
        printf("\n");
    }
}

int main() {
    omp_set_num_threads(4);  

    int A[N][N], B[N][N], C[N][N];

    initialize_matrix(A);
    initialize_matrix(B);

    double start_time = omp_get_wtime();
    multiply_matrices(A, B, C);
    double end_time = omp_get_wtime();

    printf("Time taken: %f seconds\n", end_time - start_time);

    return 0;
}
