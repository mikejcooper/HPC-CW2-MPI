/*
 ** Code to implement a d2q9-bgk lattice boltzmann scheme.
 ** 'd2' inidates a 2-dimensional grid, and
 ** 'q9' indicates 9 velocities per grid cell.
 ** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
 **
 ** The 'speeds' in each cell are numbered as follows:
 **
 ** 6 2 5
 **  \|/
 ** 3-0-1
 **  /|\
 ** 7 4 8
 **
 ** A 2D grid:
 **
 **           cols
 **       --- --- ---
 **      | D | E | F |
 ** rows  --- --- ---
 **      | A | B | C |
 **       --- --- ---
 **
 ** 'unwrapped' in row major order to give a 1D array:
 **
 **  --- --- --- --- --- ---
 ** | A | B | C | D | E | F |
 **  --- --- --- --- --- ---
 **
 ** Grid indicies are:
 **
 **          ny
 **          ^       cols(jj)
 **          |  ----- ----- -----
 **          | | ... | ... | etc |
 **          |  ----- ----- -----
 ** rows(ii) | | 1,0 | 1,1 | 1,2 |
 **          |  ----- ----- -----
 **          | | 0,0 | 0,1 | 0,2 |
 **          |  ----- ----- -----
 **          ----------------------> nx
 **
 ** Note the names of the input parameter and obstacle files
 ** are passed on the command line, e.g.:
 **
 **   d2q9-bgk.exe input.params obstacles.dat
 **
 ** Be sure to adjust the grid dimensions in the parameter file
 ** if you choose a different obstacle file.
 */

#include<stdio.h>
#include<stdlib.h>
#include <string.h>
#include<math.h>
#include<time.h>
#include<sys/time.h>
#include<sys/resource.h>

#include<omp.h>
#include "mpi.h"



/* struct to hold the parameter values */
typedef struct
{
    int    nx;            /* no. of cells in x-direction */
    int    ny;            /* no. of cells in y-direction */
    int    maxIters;      /* no. of iterations */
    int    reynolds_dim;  /* dimension for Reynolds number */
    float density;       /* density per link */
    float accel;         /* density redistribution */
    float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
    float speeds[9];
} t_speed;

typedef struct
{
    float speeds[3];
} t_buffer;

typedef struct
{
    int mpi_rank;
    int top_y;
    int bottom_y;
    int left_x;
    int right_x;
    // neighbours
    int nb_top_y;
    int nb_bottom_y;
    int nb_left_x;
    int nb_right_x;
    
    int nx;
    int ny;
} mpi_index;

typedef struct
{
    t_buffer* top_y;
    t_buffer* bottom_y;
    t_buffer* left_x;
    t_buffer* right_x;
} mpi_halo;


/*
 ** function prototypes
 */

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise_params(const char* paramfile, t_param* params);

/*
 ** The main calculation methods.
 ** timestep calls, in order, the functions:
 ** accelerate_flow(), propagate(), rebound() & collision()
 */
void accelerate_flow(const t_param params, t_speed* cells, int* obstacles, mpi_index params_mpi);
void collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, float* av_vels, int* tt, const int* tot_cells, mpi_index params_mpi);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels, FILE* fp_fs, mpi_index params_mpi);


/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
 ** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles, float* av_vels, int tt);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

void mpi(int argc, char* argv[], const t_param params, mpi_index* params_mpi, int y_split, int x_split);
void halo_exchange(const t_param params, t_speed* cells, mpi_index params_mpi, mpi_halo mpi_halo_snd, mpi_halo mpi_halo_rcv);
int initialise_mpi(const char* obstaclefile,
                   t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
                   int** obstacles_ptr, float** av_vels_ptr, mpi_index mpi_indexA);
void initialise_mpi_halos(mpi_index mpi_indexA, mpi_halo* mpi_halo_snd, mpi_halo* mpi_halo_rcv);

void halo_left_x(const t_param params, t_speed* cells, t_speed* tmp_cells,  mpi_index mpi_params, float* tot_u, t_buffer* left_x_buffer, int* obstacles, int y_n, int y_s, int x_e, int x_w, int ii, int jj);
void halo_right_x(const t_param params, t_speed* cells, t_speed* tmp_cells,  mpi_index mpi_params, float* tot_u, t_buffer* left_x_buffer, int* obstacles, int y_n, int y_s, int x_e, int x_w, int ii, int jj);
void halo_top_y(const t_param params, t_speed* cells, t_speed* tmp_cells,  mpi_index mpi_params, float* tot_u, t_buffer* top_y_buffer, mpi_halo snd_buffer, int* obstacles, int y_n, int y_s, int ii);
void halo_bottom_y(const t_param params, t_speed* cells, t_speed* tmp_cells,  mpi_index mpi_params, float* tot_u, t_buffer* top_y_buffer, mpi_halo snd_buffer, int* obstacles, int y_n, int y_s, int ii);


float collision_mpi(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, const int* tot_cells, mpi_index mpi_params, mpi_halo mpi_halos, mpi_halo mpi_halos_snd);
void print_cells(const t_param params, t_speed* cells);
void mpi_final_state(const t_param params, t_speed* cells, int* obstacles, float* av_vels, mpi_index params_mpi, int num_threads, int x_split);
void distribute_indexes(const t_param params, int world_size, int y_split, int x_split);
void get_neighbours(mpi_index* params_mpi, const t_param params, int y_split, int x_split);