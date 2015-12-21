/********************************************//**
 * Run program in a computer cluster by MPI
 * Use 2-d block assignment in row
 *
 * Written by:
 * Dongyang Yao (dongyang.yao@rutgers.edu)
 ***********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "timer.h"
#include "game_of_life.h"

int main(int argc, char **argv) {

  double total_start = get_time();
  
  int proc_num = 0;
  int proc_id = 0;;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
  

  // Parse the command line
  int row = atoi(argv[1]);
  int col = atoi(argv[2]);
  int iter = atoi(argv[3]);
  DEBUG = atoi(argv[4]);

  //printf("row: %d, col: %d, iter: %d\n", row, col, iter);

  // Add the boundary
  int row_num = row + 2;
  int col_num = col + 2;

  // Allocate contiguous 2d array
  int *data = (int *) malloc(row_num * col_num * sizeof(int));
  int **cells = (int **) malloc(row_num * sizeof(int *));
  for (int i = 0; i < row_num; i++)
    cells[i] = &(data[i * col_num]);

  int worker_num = proc_num - 1;

  if (row < worker_num) {
    printf("number of rows should be larger than number of workers\n");
    return 1;
  }

  int iteration = 0;

  /*** Master process ***/
  if (proc_id == MASTER) {

    get_random_cells(cells, row_num, col_num);

    double start = get_time();
    
    print_cells(cells, row_num, col_num, iteration);

    while (iteration < iter) {

      // Decomposition
      int averow = row / worker_num;
      int extra = row % worker_num;
      int offset = 1;
      int mtype = FROM_MASTER;

      for (int dest = 1; dest <= worker_num; dest++) {
	int rows = (dest <= extra) ? averow + 1 : averow;

	//printf("id: %d, offset: %d, rows: %d, dest: %d\n", proc_id, offset, rows, dest);

	MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
	MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
	int count = rows * col_num;

	// Put additional upper and lower row
	MPI_Send(&(cells[offset - 1][0]), count + col_num * 2, MPI_INT, dest, mtype, MPI_COMM_WORLD);
	offset += rows;
      }

      mtype = FROM_WORKER;
      MPI_Status status;

      for (int i = 1; i <= worker_num; i++) {
	int source = i;
	int rows = 0;
	MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
	MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
	int count = rows * col_num;

	//printf("id: %d, offset: %d, rows: %d, source: %d\n", proc_id, offset, rows, source);

	MPI_Recv(&(cells[offset][0]), count, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
      }

      iteration++;

      print_cells(cells, row_num, col_num, iteration);
    }

    double end = get_time();

    printf("****measurement****\n");
    printf("number of workers: %d\n", worker_num);
    printf("total time: %gs\n", end - total_start); 
    printf("parallel time: %gs\n", end - start);
    printf("****end****\n");

  } /*** End of master ***/

  /*** Worker process ***/
  if (proc_id > MASTER) {

    while (iteration < iter) {

      int mtype = FROM_MASTER;
      int source = MASTER;
      int offset = 0;
      MPI_Status status;
      int rows = 0;
      
      MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
      int count = rows * col_num;
      
      // Recv additional two rows
      MPI_Recv(&(cells[offset - 1][0]), count + col_num * 2, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);

      //printf("id: %d, offset: %d, rows: %d, source: %d\n", proc_id, offset, rows, source);
      
      // Test case
      /*
      for (int i = 0; i < rows; i++) {
	for (int j = 1; j <= col; j++) {
	  cells[offset + i][j] = proc_id;
	}
      }
      */

      // Result matrix
      int *result_data = (int *) malloc(rows * col_num * sizeof(int));
      int **result_cells = (int **) malloc(rows * sizeof(int *));
      for (int i = 0; i < rows; i++)
	result_cells[i] = &(result_data[i * col_num]);
      
      // Update by game of life rules
      for (int i = 0; i < rows; i++) {
	for (int j = 0; j < col_num; j++) {
	  if (j == 0 || j == col_num - 1) result_cells[i][j] = BOUNDARY;
	  else result_cells[i][j] = update_cell(offset + i, j, cells);
	}
      }

      mtype = FROM_WORKER;
      MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&(result_cells[0][0]), count, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
      
      free(result_cells[0]);
      free(result_cells);

      iteration++;
    }
    
  } /*** End of worker ***/
  
  MPI_Finalize();
  
  free(cells[0]);
  free(cells);
  
  return 0;
}
