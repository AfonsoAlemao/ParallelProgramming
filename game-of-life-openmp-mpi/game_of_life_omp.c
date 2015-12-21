/********************************************//**
 * Run program in a multi-core computer
 * Assign works in a SIMD pattern 
 *
 * Written by:
 * Dongyang Yao (dongyang.yao@rutgers.edu)
 ***********************************************/


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "timer.h"
#include "game_of_life.h"

int main(int argc, char **argv) {

  double total_start = get_time();
  
  // Parse the command line
  int row = atoi(argv[1]);
  int col = atoi(argv[2]);
  int iter = atoi(argv[3]);
  DEBUG = atoi(argv[4]);

  // Add the boundary
  int row_num = row + 2;
  int col_num = col + 2;

  // Allocate contiguous 2d array
  int *data = (int *) malloc(row_num * col_num * sizeof(int));
  int **cells = (int **) malloc(row_num * sizeof(int *));
  for (int i = 0; i < row_num; i++)
    cells[i] = &(data[i * col_num]);

  int *tmp = (int *) malloc(row_num * col_num * sizeof(int));
  
  int iteration = 0;
  
  get_random_cells(cells, row_num, col_num);

  print_cells(cells, row_num, col_num, iteration);

  int nthreads = 0;
  int ncores = 0;
  int i = 0, j = 0;

#pragma omp parallel
  {
#pragma omp master
    {
      ncores = omp_get_num_procs();
      nthreads = omp_get_num_threads();
    }
  }
  
  double start = get_time();
  
  while (iteration < iter) {
    
    // Update by game of life rules
#pragma omp parallel for private(i, j)
    for (i = 0; i < row_num; i++) {
      for (j = 0; j < col_num; j++) {
	tmp[i * col_num + j] = (i == 0 || i == row_num - 1 || j == 0 || j == col_num - 1) ? BOUNDARY : update_cell(i, j, cells);
      }
    }
    
#pragma omp parallel for private(i, j)
    for (i = 0; i < row_num; i++) {
      for (j = 0; j < col_num; j++) {
	cells[i][j] = tmp[i * col_num + j];
      }
    }
    
    iteration++;
    print_cells(cells, row_num, col_num, iteration);
  }
  
  double end = get_time();
  
  printf("****measurement****\n");
  printf("number of cores: %d\n", ncores); 
  printf("number of threads: %d\n", nthreads);
  printf("total time: %gs\n", end - total_start);
  printf("parallel time: %gs\n", end - start); 
  printf("****end****\n");

  free(tmp);
  free(cells[0]);
  free(cells);

  return 0;
}


