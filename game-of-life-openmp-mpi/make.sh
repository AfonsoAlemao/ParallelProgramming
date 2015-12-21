mpicc -std=gnu99 game_of_life_mpi.c game_of_life.h timer.h -o game_of_life_mpi
gcc -std=gnu99 -fopenmp game_of_life_omp.c game_of_life.h timer.h -o game_of_life_omp
