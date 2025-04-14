#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 512  // Matrix size (must be a power of 2)
#define THRESHOLD 64  // Switch to naive multiplication below this size

int** allocate_matrix(int n) {
    int **matrix = malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        matrix[i] = malloc(n * sizeof(int));
    }
    return matrix;
}

void free_matrix(int **matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void add_matrices(int size, int** A, int** B, int** C){
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            C[i][j] = A[i][j] + B[i][j];
}

void subtract_matrices(int size, int** A, int** B, int** C) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            C[i][j] = A[i][j] - B[i][j];
}

void naive_multiplication(int size, int** A, int** B, int** C) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++) {
            C[i][j] = 0;
            for (int k = 0; k < size; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
}

void strassen(int n, int **A, int **B, int **C) {
    if (n <= THRESHOLD) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                C[i][j] = 0;
                for (int k = 0; k < n; k++)
                    C[i][j] += A[i][k] * B[k][j];
            }
        return;
    }

    int new_size = n / 2;

    // Allocate submatrices
    int **A11 = allocate_matrix(new_size);
    int **A12 = allocate_matrix(new_size);
    int **A21 = allocate_matrix(new_size);
    int **A22 = allocate_matrix(new_size);
    int **B11 = allocate_matrix(new_size);
    int **B12 = allocate_matrix(new_size);
    int **B21 = allocate_matrix(new_size);
    int **B22 = allocate_matrix(new_size);
    int **C11 = allocate_matrix(new_size);
    int **C12 = allocate_matrix(new_size);
    int **C21 = allocate_matrix(new_size);
    int **C22 = allocate_matrix(new_size);
    int **M1 = allocate_matrix(new_size);
    int **M2 = allocate_matrix(new_size);
    int **M3 = allocate_matrix(new_size);
    int **M4 = allocate_matrix(new_size);
    int **M5 = allocate_matrix(new_size);
    int **M6 = allocate_matrix(new_size);
    int **M7 = allocate_matrix(new_size);
    int **temp1 = allocate_matrix(new_size);
    int **temp2 = allocate_matrix(new_size);

    // Divide matrices into submatrices
    for (int i = 0; i < new_size; i++)
        for (int j = 0; j < new_size; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + new_size];
            A21[i][j] = A[i + new_size][j];
            A22[i][j] = A[i + new_size][j + new_size];
            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + new_size];
            B21[i][j] = B[i + new_size][j];
            B22[i][j] = B[i + new_size][j + new_size];
        }

    // Compute M1 to M7
            add_matrices(new_size, A11, A22, temp1);
            add_matrices(new_size, B11, B22, temp2);
            strassen(new_size, temp1, temp2, M1);

            add_matrices(new_size, A21, A22, temp1);
            strassen(new_size, temp1, B11, M2);

            subtract_matrices(new_size, B12, B22, temp1);
            strassen(new_size, A11, temp1, M3);

            subtract_matrices(new_size, B21, B11, temp1);
            strassen(new_size, A22, temp1, M4);

            add_matrices(new_size, A11, A12, temp1);
            strassen(new_size, temp1, B22, M5);

            subtract_matrices(new_size, A21, A11, temp1);
            add_matrices(new_size, B11, B12, temp2);
            strassen(new_size, temp1, temp2, M6);

            subtract_matrices(new_size, A12, A22, temp1);
            add_matrices(new_size, B21, B22, temp2);
            strassen(new_size, temp1, temp2, M7);

    // Calculate C11, C12, C21, C22
            add_matrices(new_size, M1, M4, temp1);
            subtract_matrices(new_size, temp1, M5, temp2);
            add_matrices(new_size, temp2, M7, C11);

            add_matrices(new_size, M3, M5, C12);

            add_matrices(new_size, M2, M4, C21);

            add_matrices(new_size, M1, M3, temp1);
            subtract_matrices(new_size, temp1, M2, temp2);
            add_matrices(new_size, temp2, M6, C22);


    // Merge back into result matrix C
    for (int i = 0; i < new_size; i++)
        for (int j = 0; j < new_size; j++) {
            C[i][j] = C11[i][j];
            C[i][j + new_size] = C12[i][j];
            C[i + new_size][j] = C21[i][j];
            C[i + new_size][j + new_size] = C22[i][j];
        }

    // Free memory
    free_matrix(A11, new_size); free_matrix(A12, new_size);
    free_matrix(A21, new_size); free_matrix(A22, new_size);
    free_matrix(B11, new_size); free_matrix(B12, new_size);
    free_matrix(B21, new_size); free_matrix(B22, new_size);
    free_matrix(C11, new_size); free_matrix(C12, new_size);
    free_matrix(C21, new_size); free_matrix(C22, new_size);
    free_matrix(M1, new_size); free_matrix(M2, new_size);
    free_matrix(M3, new_size); free_matrix(M4, new_size);
    free_matrix(M5, new_size); free_matrix(M6, new_size); free_matrix(M7, new_size);
    free_matrix(temp1, new_size); free_matrix(temp2, new_size);
}

void initialize_matrix(int n, int **matrix) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            matrix[i][j] = rand() % 10;
}
int main() {
    int **A = allocate_matrix(N);
    int **B = allocate_matrix(N);
    int **C = allocate_matrix(N);

    initialize_matrix(N, A);
    initialize_matrix(N, B);

    clock_t start = clock();
    strassen(N, A, B, C);
    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time taken for matrix multiplication: %.6f seconds\n", time_taken);

    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C, N);
    
    return 0;
}