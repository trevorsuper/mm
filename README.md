# A comparison of O(N^3) and O(N^2.81) parallel matrix multiplication algorithms
Parallel implementations written with OpenMP and CUDA. Tested on Intel(R) Xeon(R) 6148, 80 Cores @ 2.40GHz and Nvidia Tesla K80, 2496 CUDA Cores. GCC Version 8.5.0, NVCC Version 10.1.243<br>
## Benchmarks
![Sequential](img/Sequential.png)<br>
![1](img/O1.png)<br>
![2](img/O2.png)<br>
![4](img/O4.png)<br>
![8](img/O8.png)<br>
![16](img/O16.png)<br>
![32](img/O32.png)<br>
![64](img/O64.png)<br>
![80](img/O80.png)<br>
![512](img/CUDA512.png)<br>
![2496](img/CUDA2496.png)<br>
![seq_vs_para](img/Sequential_Vs_Parallel.png)<br>
## Findings
For CUDA implementations, if the number of threads exceeds the matrix size, an error occurs resulting in zeroes to fill the product matrix.<br>
If available computing power and number of threads are small, strassen is preferred over naive.<br>
If available computing power and number of threads are large, such as a many core CPU or a GPU, naive is preferred over strassen. The number of workers far overpowers the smarter, less time complex approach.<br>
## Dumber, but faster