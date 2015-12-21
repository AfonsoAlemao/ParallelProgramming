# ParallelProgramming

Gain computing performance: 1) game of life problem with OpenMP and MPI; 2) exclusive scan and find repeats with CUDA; 3) newton method with ISPC and AVX.

## Game of life problem

The [game of life problem](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) is written in [OpenMP](http://openmp.org/wp/) to run on multi-cores of a computer and in [MPI](http://www.open-mpi.org) to run on a computer cluster.

> Speed-up

OpenMP on a dual-core computer with 3000 x 3000 cells for 10 iterations.

| Number of threads | Execution time (s) | Speed up |
|-------------------|--------------------|----------|
| 1                 | 11.69              | 1.00     |
| 2                 | 5.98               | 1.96     |
| 4                 | 6.00               | 1.95     |
| 8                 | 5.97               | 1.96     |

MPI on a cluster (more than 9 nodes) with 3000 x 3000 cells for 10 iterations.

| Number of workers | Execution time (s) | Speed up |
|-------------------|--------------------|----------|
| 1                 | 17.26              | 1.00     |
| 2                 | 10.25              | 1.68     |
| 4                 | 6.84               | 2.52     |
| 8                 | 6.34               | 2.72     |

> Installation

```
// Compile
cd game-of-life-openmp-mpi
sh make.sh
```

```
/*** OpenMP ***/
// Command
// ./game_of_life_omp [rows] [cols] [iters] [debug]

// Example
export OMP_NUM_THREADS=2
./game_of_life_omp 4 12 2 1
export OMP_NUM_THREADS=1
./game_of_life_omp 1500 1500 15 0
export OMP_NUM_THREADS=2
./game_of_life_omp 1500 1500 15 0
export OMP_NUM_THREADS=4
./game_of_life_omp 1500 1500 15 0
```

```
/*** MPI ***/
// Command
// mpirun -np [number of slaves + 1 master] -hostfile hostfile.txt game_of_life_mpi [rows] [cols] [iters] [debug]

// Example
mpirun -np 5 -hostfile hostfile.txt game_of_life_mpi 4 12 2 1
mpirun -np 2 -hostfile hostfile.txt game_of_life_mpi 1500 1500 15 0
mpirun -np 3 -hostfile hostfile.txt game_of_life_mpi 1500 1500 15 0
mpirun -np 5 -hostfile hostfile.txt game_of_life_mpi 1500 1500 15 0
mpirun -np 9 -hostfile hostfile.txt game_of_life_mpi 1500 1500 15 0
```

## Exclusive scan and find repeats

The parallel [exclusive scan](http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html) algorithm written in [CUDA](http://www.nvidia.com/object/cuda_home_new.html) is running on NVIDIA GPU. 

> Speed-up

CUDA GPU with 1000000 random integers between 0 and 100.

|            | Exclusive scan | Repeat indices | Remain entries |
|------------|----------------|----------------|----------------|
| Sequential | 10.18 ms       | 16.22 ms       | 17.18 ms       |
| Parallel   | 2.49 ms        | 4.08 ms        | 4.35 ms        |
| Speedup    | 4.09           | 3.98           | 3.95           |

> Installation

Install the CUDA with the [Guide](http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Quick_Start_Guide.pdf).

```
cd prefix-sum-cuda
nvcc main.cu -o main
// ./main [number of inputs]
// Or .main [file_name]
// Example
./main 1000000
./main sample_input.txt
```
## Newton method for root

The [Newton method](https://en.wikipedia.org/wiki/Newton%27s_method) is written in [ISPC](https://ispc.github.io) and [AVX](https://software.intel.com/en-us/articles/introduction-to-intel-advanced-vector-extensions) to run on a multi-core computer.

> Speed-up

20 million random numbers between 0 and 3.

|                     | Sequential | AVX    | ISPC (single-core) | ISPC (quad-core) |
|---------------------|------------|--------|--------------------|------------------|
| Execution time (ms) | 905.90     | 204.16 | 165.71             | 38.67            |
| Speedup             | 1.00       | 4.44   | 5.47               | 23.42            |

> Installation

```
cd iterative-sort-avx-ispc
sh make.sh
./iterative_sort
```


