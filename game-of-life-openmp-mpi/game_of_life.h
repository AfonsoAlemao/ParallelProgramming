/********************************************//**
 * The basic methods
 * For processing cell matrix
 *
 * Written by:
 * Dongyang Yao (dongyang.yao@rutgers.edu)
 ***********************************************/

#ifndef GAME_OF_LIFE_H__
#define GAME_OF_LIFE_H__

#include <stdio.h>
#include <stdlib.h>
#include <time.h> 

#define ALIVE 1
#define DEAD 0
#define BOUNDARY 8
#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

int DEBUG = 1;

/* Init the cells with random status */
void get_random_cells(int **cells, int row_num, int col_num) {

  srand(time(NULL));

  for (int i = 0; i < row_num; i++) {
    for (int j = 0; j < col_num; j++) {
      if ((i == 0) || (i == row_num - 1) ||
	  (j == 0) || (j == col_num - 1)) {
	cells[i][j] = BOUNDARY;
      }
      else {
	cells[i][j] = rand() % 2;
      }
    }
  }
}

/* Print out the cells' status */
void print_cells(int **cells, int row_num, int col_num, int iter) {
  if (DEBUG) {
    printf("round: %d\n", iter);
    printf("****cell****\n");
    for (int i = 0; i < row_num; i++) {
      for (int j = 0; j < col_num; j++) {
	// Do not print the boundary out
	if (cells[i][j] == BOUNDARY) printf(" ");
	else printf("%d ", cells[i][j]);
      }
      printf("\n");
    }
    printf("****ends****\n");
  }
}

/* Update the cell by game of life rules */
int update_cell(int i, int j, int **cells) {
  int status = cells[i][j];
  int count = 0;

  for (int m = -1; m <= 1; m++) {
    for (int n = -1; n <= 1; n++) {
      if (m == 0 && n == 0) continue;
      if (cells[i + m][j + n] == ALIVE) count++;
    }
  }

  if (status == ALIVE) {
    if (count < 2 || count > 3) return DEAD;
  } else {
    if (count > 1 && count < 4) return ALIVE;
  }

  return status;
}  

#endif
