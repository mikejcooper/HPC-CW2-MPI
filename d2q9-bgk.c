#include "d2q9-bgk.h"
int main(int argc, char* argv[])
{
    char*    paramfile = NULL;    /* name of the input parameter file */
    char*    obstaclefile = NULL; /* name of a the input obstacle file */
    t_param  params;              /* struct to hold parameter values */
    t_speed* cells     = NULL;    /* grid containing fluid densities */
    t_speed* tmp_cells = NULL;    /* scratch space */
    int*     obstacles = NULL;    /* grid indicating which cells are blocked */
    float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
    struct timeval timstr;        /* structure to hold elapsed time */
    struct rusage ru;             /* structure to hold CPU time--system and user */
    double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
    double usrtim;                /* floating point number to record elapsed user CPU time */
    double systim;                /* floating point number to record elapsed system CPU time */
    int tot_cells;
    mpi_halo mpi_halos_snd;
    mpi_halo mpi_halos_rcv;
    mpi_index params_mpi;
    int y_split = 32;
    int x_split = 1;
      
    /* parse the command line */
    if (argc != 3)
    {
        usage(argv[0]);
    }
    else
    {
        paramfile = argv[1];
        obstaclefile = argv[2];
    }
    
    /* initialise our data structures and load values from file */
    initialise_params(paramfile, &params);

    mpi(argc, argv, params, &params_mpi, y_split,x_split);
    
    //printf("INDEX: t: %d b: %d l: %d r: %d rk: %d \n", params_mpi.top_y, params_mpi.bottom_y, params_mpi.left_x, params_mpi.right_x, params_mpi.mpi_rank);
    //printf("t: %d b: %d l: %d r: %d rk: %d \n", params_mpi.nb_top_y, params_mpi.nb_bottom_y, params_mpi.nb_left_x, params_mpi.nb_right_x, params_mpi.mpi_rank);

    
    tot_cells = initialise_mpi(obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, params_mpi);
    initialise_mpi_halos(params_mpi,&mpi_halos_snd, &mpi_halos_rcv, cells);
     
    /* iterate for maxIters timesteps */
    gettimeofday(&timstr, NULL);
    tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    
    for (int tt = 0; tt < params.maxIters; tt++)
    {
        halo_exchange(params, cells, params_mpi, mpi_halos_snd, mpi_halos_rcv, y_split);
        if (params_mpi.top_y == params.ny)
            accelerate_flow(params, cells, obstacles, params_mpi);
        av_vels[tt] = collision_mpi(params, cells, tmp_cells, obstacles, &tot_cells, params_mpi, mpi_halos_rcv, mpi_halos_snd);
        t_speed* temp = cells;
        cells = tmp_cells;
        tmp_cells = temp;
#ifdef DEBUG
        printf("==timestep: %d==\n", tt);
        printf("av velocity: %.12E\n", av_vels[tt]);
        printf("tot density: %.12E\n", total_density(params, cells));
#endif
    }
    gettimeofday(&timstr, NULL);
    toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    getrusage(RUSAGE_SELF, &ru);
    timstr = ru.ru_utime;
    usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    timstr = ru.ru_stime;
    systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    
    /* write final values and free memory */
    // Synchronise nodes here?
    if(params_mpi.mpi_rank == 0){
      printf("==done==\n");
      printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
      printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
      printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
    }
    
    mpi_final_state(params, cells, obstacles, av_vels, params_mpi, x_split * y_split, x_split);
    // Finalize the MPI environment.
   MPI_Finalize();
    // write_values(params, cells, obstacles, av_vels);
    finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);
    
    return EXIT_SUCCESS;
}


void mpi(int argc, char* argv[], const t_param params, mpi_index* params_mpi_arg, int y_split, int x_split)
{
    int mpi_rank;
   int size;
   int dest;              /* destination rank for message */
   int source;            /* source rank of a message */
   int tag = 0;           /* scope for adding extra information to a message */
   MPI_Status status;     /* struct used by MPI_Recv */
   char message[BUFSIZ];
   mpi_index params_mpi;
   
   
   
   // Initialize the MPI environment
   MPI_Init(&argc, &argv);
   
   // Get the number of processes
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   
   // Get the rank of the  process
   MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
   
   // Get the name of the processor
   char processor_name[MPI_MAX_PROCESSOR_NAME];
   int name_len;
   MPI_Get_processor_name(processor_name, &name_len);
   
   if(mpi_rank == 0){
       int y_top = params.ny - (y_split - 1) * (params.ny / y_split);
       int y_bottom = params.ny - y_split * (params.ny / y_split);
       int x_right = params.nx - (x_split - 1) * (params.nx / x_split);
       int x_left = params.nx - x_split * (params.nx / x_split);
       
       params_mpi = (mpi_index) {mpi_rank, y_top, y_bottom, x_left, x_right, 0, 0, 0, 0, x_right - x_left, y_top - y_bottom};
       get_neighbours(&params_mpi, params, y_split, x_split);
       distribute_indexes(params, size, y_split, x_split);
   } else {
       MPI_Recv(&params_mpi, 11, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
       params_mpi.mpi_rank = mpi_rank;
       get_neighbours(&params_mpi, params, y_split, x_split);
   }
   
   *params_mpi_arg = params_mpi;
}


void accelerate_flow(const t_param params, t_speed* cells, int* obstacles, mpi_index mpi_index_current)
{
    /* compute weighting factors */
  float w2 = params.density * params.accel / 36.0f;
  float w1 = 4 * w2;


    /* modify the 2nd row of the grid */
    int ii = mpi_index_current.ny - 2;
    // array indexing: 16128 -> 16255

    for (int jj = 0; jj < mpi_index_current.nx; jj++)
    {
        int index = ii * mpi_index_current.nx + jj;
        /* if the cell is not occupied and
         ** we don't send a negative density */
        if (!obstacles[index]
            && (cells[index].speeds[3] - w1) > 0.0f
            && (cells[index].speeds[6] - w2) > 0.0f
            && (cells[index].speeds[7] - w2) > 0.0f)
        {
            /* increase 'east-side' densities */
            cells[index].speeds[1] += w1;
            cells[index].speeds[5] += w2;
            cells[index].speeds[8] += w2;
            /* decrease 'west-side' densities */
            cells[index].speeds[3] -= w1;
            cells[index].speeds[6] -= w2;
            cells[index].speeds[7] -= w2;
        }
        
    }
}


void halo_exchange(const t_param params, t_speed* cells, mpi_index params_mpi, mpi_halo mpi_halo_snd, mpi_halo mpi_halo_rcv, int y_split)
{
   MPI_Status status;     /* struct used by MPI_Recv */
   MPI_Request req;

  //  MPI_Win win1, win2;
  //   printf("here \n");
  //   float a = 1;
  //   float b = 0;



  // if (params_mpi.mpi_rank == 0) {
  //     MPI_Win_create(&a,sizeof(float), 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win1);

  // }
  // if(params_mpi.mpi_rank == 1){
  //     MPI_Win_create(MPI_BOTTOM, 0, 1, MPI_INFO_NULL,MPI_COMM_WORLD, &win1);
  // }

  //  MPI_Win_fence(0,win1);  
  //  if (params_mpi.mpi_rank == 1){
      
  //     int source = 0;
  //     MPI_Get(&b, 1, MPI_FLOAT, source, 0, 1, MPI_FLOAT, win1);
  //  }
  //   MPI_Win_fence(0,win1);  
    
  //   printf("rank: %d rcv: %f\n",params_mpi.mpi_rank, b);
  //   MPI_Win_free(&win1);
  //   MPI_Win_free(&win2);



   // if ((params_mpi.mpi_rank % 2) != 0) {  /* this is _NOT_ the master process */
       
   //     // // printf("recieved top, sent bottom at rank :%d\n", params_mpi.mpi_rank);
       
   //    if(params_mpi.mpi_rank != (y_split - 1)){
   //      MPI_Recv(mpi_halo_rcv.bottom_y, params_mpi.nx*3, MPI_FLOAT, params_mpi.nb_bottom_y, 1, MPI_COMM_WORLD, &status);
   //      MPI_Ssend(mpi_halo_snd.top_y, params_mpi.nx*3, MPI_FLOAT, params_mpi.nb_top_y, 1, MPI_COMM_WORLD);
   //     // printf("recieved bottom, sent top at rank :%d\n", params_mpi.mpi_rank);
   //      MPI_Recv(mpi_halo_rcv.top_y, params_mpi.nx*3, MPI_FLOAT, params_mpi.nb_top_y, 1, MPI_COMM_WORLD, &status);
   //      MPI_Ssend(mpi_halo_snd.bottom_y, params_mpi.nx*3, MPI_FLOAT, params_mpi.nb_bottom_y, 1, MPI_COMM_WORLD);
   //     } else {
   //        MPI_Recv(mpi_halo_rcv.bottom_y, params_mpi.nx*3, MPI_FLOAT, params_mpi.nb_bottom_y, 1, MPI_COMM_WORLD, &status);
   //        MPI_Ssend(mpi_halo_snd.bottom_y, params_mpi.nx*3, MPI_FLOAT, params_mpi.nb_bottom_y, 1, MPI_COMM_WORLD);
   //     }
       
   //     // printf("recieved top, sent bottom at rank :%d\n", params_mpi.mpi_rank);

       
       
   // }
   // else {             /* i.e. this is the master process */
   //    if(params_mpi.mpi_rank != 0){
   //     MPI_Ssend(mpi_halo_snd.top_y, params_mpi.nx*3, MPI_FLOAT, params_mpi.nb_top_y, 1, MPI_COMM_WORLD);
   //     MPI_Recv(mpi_halo_rcv.bottom_y, params_mpi.nx*3, MPI_FLOAT, params_mpi.nb_bottom_y, 1, MPI_COMM_WORLD, &status);
   //     // printf("sent top, recieved bottom at rank :%d\n", params_mpi.mpi_rank);
       
   //      MPI_Ssend(mpi_halo_snd.bottom_y, params_mpi.nx*3, MPI_FLOAT, params_mpi.nb_bottom_y, 1, MPI_COMM_WORLD);
   //      MPI_Recv(mpi_halo_rcv.top_y, params_mpi.nx*3, MPI_FLOAT, params_mpi.nb_top_y, 1, MPI_COMM_WORLD, &status);
   //     } else {
   //      MPI_Ssend(mpi_halo_snd.top_y, params_mpi.nx*3, MPI_FLOAT, params_mpi.nb_top_y, 1, MPI_COMM_WORLD);
   //      MPI_Recv(mpi_halo_rcv.top_y, params_mpi.nx*3, MPI_FLOAT, params_mpi.nb_top_y, 1, MPI_COMM_WORLD, &status);
   //     }
   // }




   if ((params_mpi.mpi_rank % 2) != 0) {  /* this is _NOT_ the master process */
       
       // // printf("recieved top, sent bottom at rank :%d\n", params_mpi.mpi_rank);
       
        MPI_Recv(mpi_halo_rcv.bottom_y, params_mpi.nx*3, MPI_FLOAT, params_mpi.nb_bottom_y, 1, MPI_COMM_WORLD, &status);
        MPI_Ssend(mpi_halo_snd.top_y, params_mpi.nx*3, MPI_FLOAT, params_mpi.nb_top_y, 1, MPI_COMM_WORLD);
       // printf("recieved bottom, sent top at rank :%d\n", params_mpi.mpi_rank);
        MPI_Recv(mpi_halo_rcv.top_y, params_mpi.nx*3, MPI_FLOAT, params_mpi.nb_top_y, 1, MPI_COMM_WORLD, &status);
        MPI_Ssend(mpi_halo_snd.bottom_y, params_mpi.nx*3, MPI_FLOAT, params_mpi.nb_bottom_y, 1, MPI_COMM_WORLD);
        

       
       
   }
   else {             /* i.e. this is the master process */
       MPI_Ssend(mpi_halo_snd.top_y, params_mpi.nx*3, MPI_FLOAT, params_mpi.nb_top_y, 1, MPI_COMM_WORLD);
       MPI_Recv(mpi_halo_rcv.bottom_y, params_mpi.nx*3, MPI_FLOAT, params_mpi.nb_bottom_y, 1, MPI_COMM_WORLD, &status);
       // printf("sent top, recieved bottom at rank :%d\n", params_mpi.mpi_rank);
       
        MPI_Ssend(mpi_halo_snd.bottom_y, params_mpi.nx*3, MPI_FLOAT, params_mpi.nb_bottom_y, 1, MPI_COMM_WORLD);
        MPI_Recv(mpi_halo_rcv.top_y, params_mpi.nx*3, MPI_FLOAT, params_mpi.nb_top_y, 1, MPI_COMM_WORLD, &status);
        
        // MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        //         int dest, int sendtag,
        //         void *recvbuf, int recvcount, MPI_Datatype recvtype,
        //         int source, int recvtag,
        //         MPI_Comm comm, MPI_Status *status)

        // MPI_Sendrecv(mpi_halo_snd.top_y, params_mpi.nx*3, MPI_FLOAT, params_mpi.nb_top_y, 0, 
        //              mpi_halo_rcv.top_y, params_mpi.nx*3, MPI_FLOAT, params_mpi.nb_top_y, 0, 
        //              MPI_COMM_WORLD, &status);

        // MPI_Sendrecv(mpi_halo_snd.bottom_y, params_mpi.nx*3, MPI_FLOAT, params_mpi.nb_bottom_y, 0, 
        //              mpi_halo_rcv.bottom_y, params_mpi.nx*3, MPI_FLOAT, params_mpi.nb_bottom_y, 0, 
        //              MPI_COMM_WORLD, &status);
   }
}

void initialise_mpi_halos(mpi_index mpi_params, mpi_halo* mpi_halo_snd, mpi_halo* mpi_halo_rcv, t_speed* cells){
    
    mpi_halo_snd->top_y = malloc(sizeof(t_buffer) * (mpi_params.right_x - mpi_params.left_x));
    mpi_halo_snd->bottom_y = malloc(sizeof(t_buffer) * (mpi_params.right_x - mpi_params.left_x));
    // mpi_halo_snd->right_x = malloc(sizeof(t_buffer) * (mpi_params.top_y - mpi_params.bottom_y));
    // mpi_halo_snd->left_x = malloc(sizeof(t_buffer) * (mpi_params.top_y - mpi_params.bottom_y));
    
    mpi_halo_rcv->top_y = malloc(sizeof(t_buffer) * (mpi_params.right_x - mpi_params.left_x));
    mpi_halo_rcv->bottom_y = malloc(sizeof(t_buffer) * (mpi_params.right_x - mpi_params.left_x));
    // mpi_halo_rcv->right_x = malloc(sizeof(t_speed) * (mpi_params.top_y - mpi_params.bottom_y));
    // mpi_halo_rcv->left_x = malloc(sizeof(t_speed) * (mpi_params.top_y - mpi_params.bottom_y));

    for (int ii = 0; ii < mpi_params.ny; ii++)
    {
        int y_s = (ii == 0) ? (ii + mpi_params.ny - 1) : (ii - 1); // could move top
        int y_n = (ii + 1) % mpi_params.ny; // Could move top
        
        for (int jj = 0; jj < mpi_params.nx; jj++)
        {
            int index = ii * mpi_params.nx + jj;
            if(ii == 0)
            {
                mpi_halo_snd->bottom_y[jj].speeds[0] = cells[index].speeds[4];
                mpi_halo_snd->bottom_y[jj].speeds[1] = cells[index].speeds[7];
                mpi_halo_snd->bottom_y[jj].speeds[2] = cells[index].speeds[8];
            }
            else if(ii == mpi_params.ny - 1)
            {
                mpi_halo_snd->top_y[jj].speeds[0] = cells[index].speeds[2];
                mpi_halo_snd->top_y[jj].speeds[1] = cells[index].speeds[5];
                mpi_halo_snd->top_y[jj].speeds[2] = cells[index].speeds[6];
            }
        }
    }
}


int initialise_mpi(const char* obstaclefile,
                   t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
                   int** obstacles_ptr, float** av_vels_ptr, mpi_index mpi_indexA)
{
    char   message[1024];  /* message buffer */
    FILE*   fp;            /* file pointer */
    int    xx, yy;         /* generic array indices */
    int    blocked;        /* indicates whether a cell is blocked by an obstacle */
    int    retval;         /* to hold return value for checking */
    
    
    /* main grid */
    *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (mpi_indexA.nx * mpi_indexA.ny));
    
    /* 'helper' grid, used as scratch space */
    *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (mpi_indexA.nx * mpi_indexA.ny));
    
    /* the map of obstacles */
    *obstacles_ptr = malloc(sizeof(int) * (mpi_indexA.nx * mpi_indexA.ny));
    
    /* initialise densities */
    float w0 = params->density * 4.0f / 9.0f;
    float w1 = params->density      / 9.0f;
    float w2 = params->density      / 36.0f;
    
    // ny, nx = 128
    for (int ii = 0; ii < mpi_indexA.ny; ii++)
    {
        for (int jj = 0; jj < mpi_indexA.nx; jj++)
        {
            /* centre */
            (*cells_ptr)[ii * mpi_indexA.nx + jj].speeds[0] = w0;
            /* axis directions */
            (*cells_ptr)[ii * mpi_indexA.nx + jj].speeds[1] = w1;
            (*cells_ptr)[ii * mpi_indexA.nx + jj].speeds[2] = w1;
            (*cells_ptr)[ii * mpi_indexA.nx + jj].speeds[3] = w1;
            (*cells_ptr)[ii * mpi_indexA.nx + jj].speeds[4] = w1;
            /* diagonals */
            (*cells_ptr)[ii * mpi_indexA.nx + jj].speeds[5] = w2;
            (*cells_ptr)[ii * mpi_indexA.nx + jj].speeds[6] = w2;
            (*cells_ptr)[ii * mpi_indexA.nx + jj].speeds[7] = w2;
            (*cells_ptr)[ii * mpi_indexA.nx + jj].speeds[8] = w2;
            
            (*obstacles_ptr)[ii * mpi_indexA.nx + jj] = 0;
        }
    }



    /* open the obstacle data file */
    fp = fopen(obstaclefile, "r");
    
    if (fp == NULL)
    {
        sprintf(message, "could not open input obstacles file: %s", obstaclefile);
        die(message, __LINE__, __FILE__);
    }
    
    int tot_cells_blocked = 0;
    
    /* read-in the blocked cells list */
    while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
    {
        /* some checks */
        if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);
        
        if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);
        
        if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);
        
        if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);
        
        /* assign to array */
        
        if((mpi_indexA.bottom_y <= yy) && (yy <= mpi_indexA.top_y)){
            if ((mpi_indexA.left_x <= xx) && (xx <= mpi_indexA.right_x)){
                int y = yy - mpi_indexA.bottom_y;
                int x = xx - mpi_indexA.left_x;
                int ind = y * mpi_indexA.nx + x;
                (*obstacles_ptr)[ind] = blocked;
            }
        }
        tot_cells_blocked++;
        
    }
    

    
    /* and close the file */
    fclose(fp);
    
    /*
     ** allocate space to hold a record of the avarage velocities computed
     ** at each timestep
     */
    *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);
    
    int tot_cells = params->nx * params->ny - tot_cells_blocked;
    
    return tot_cells;
}


int initialise_params(const char* paramfile, t_param* params)
{
    FILE*   fp;            /* file pointer */
    int retval;
    
    /* open the parameter file */
    fp = fopen(paramfile, "r");
    
    /* read in the parameter values */
    retval = fscanf(fp, "%d\n", &(params->nx));
    retval = fscanf(fp, "%d\n", &(params->ny));
    retval = fscanf(fp, "%d\n", &(params->maxIters));
    retval = fscanf(fp, "%d\n", &(params->reynolds_dim));
    retval = fscanf(fp, "%f\n", &(params->density));
    retval = fscanf(fp, "%f\n", &(params->accel));
    retval = fscanf(fp, "%f\n", &(params->omega));


    /* and close up the file */
    fclose(fp);
    
    return 1;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
    /*
     ** free up allocated memory
     */
    free(*cells_ptr);
    *cells_ptr = NULL;
    
    free(*tmp_cells_ptr);
    *tmp_cells_ptr = NULL;
    
    free(*obstacles_ptr);
    *obstacles_ptr = NULL;
    
    free(*av_vels_ptr);
    *av_vels_ptr = NULL;
    
    return EXIT_SUCCESS;
}

float total_density(const t_param params, t_speed* cells)
{
    float total = 0.0;  /* accumulator */
    
    for (int ii = 0; ii < params.ny; ii++)
    {
        for (int jj = 0; jj < params.nx; jj++)
        {
            for (int kk = 0; kk < 9; kk++)
            {
                total += cells[ii * params.nx + jj].speeds[kk];
            }
        }
    }
    
    return total;
}


void mpi_final_state(const t_param params, t_speed* cells, int* obstacles, float* av_vels, mpi_index params_mpi, int num_threads, int x_split){
  MPI_Status status;
  int tile_size = (params_mpi.nx * params_mpi.ny);
  float av_vels_acc[params.maxIters];
  t_speed* cells_holder = malloc(sizeof(t_speed) * params.nx * (params_mpi.top_y - params_mpi.bottom_y));
  
  if(params_mpi.mpi_rank != 0){
    int dest = 0;      
    MPI_Ssend(cells, tile_size*9, MPI_FLOAT, dest, 1, MPI_COMM_WORLD);
    MPI_Ssend(obstacles, tile_size, MPI_INT, dest, 1, MPI_COMM_WORLD);
    MPI_Ssend(av_vels, params.maxIters, MPI_FLOAT, dest, 1, MPI_COMM_WORLD);
    MPI_Ssend(&params_mpi, 11, MPI_INT, dest, 1, MPI_COMM_WORLD);

  }
  else 
  {
    // Master thread 

    FILE* fp_fs = fopen("final_state.dat", "w");

    for (int ii = 0; ii < params.maxIters; ii++)
    {
        av_vels_acc[ii] = av_vels[ii];
    }

    if (x_split > 1){
            for (int ii = 0; ii < params_mpi.ny; ii++)
            {
                for (int jj = 0; jj < params_mpi.nx; jj++)
                {
                    int index_new = ii * params.nx + (jj + params_mpi.left_x);
                    int index= ii * params.nx + jj;
                    for(int k = 0; k<9; k++)
                    {
                        cells_holder[index_new].speeds[k] = cells[index].speeds[k];
                    }
                }
            }
    }
    else 
    {
      write_values(params, cells, obstacles, av_vels, fp_fs, params_mpi);
    }


    for(int source = 1; source < num_threads; source++){

        MPI_Recv(cells, tile_size*9, MPI_FLOAT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(obstacles, tile_size, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(av_vels, params.maxIters, MPI_FLOAT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&params_mpi, 11, MPI_INT, source, 1, MPI_COMM_WORLD, &status);

      // if x segment, store in temp. 
        if (x_split > 1)
        {
            for (int ii = 0; ii < params_mpi.ny; ii++)
            {
                for (int jj = 0; jj < params_mpi.nx; jj++)
                {
                    int index_new = ii * params.nx + (jj + params_mpi.left_x);
                    int index= ii * params.nx + jj;
                    for(int k = 0; k<9; k++)
                    {
                        cells_holder[index_new].speeds[k] = cells[index].speeds[k];
                    }
                }
            }

            if((params_mpi.mpi_rank % x_split) == (x_split - 1)){
                params_mpi.left_x = 0;
                write_values(params, cells_holder, obstacles, av_vels, fp_fs, params_mpi);
            }
        }
        else {
          write_values(params, cells, obstacles, av_vels, fp_fs, params_mpi);
        }

      for (int ii = 0; ii < params.maxIters; ii++)
        {
          av_vels_acc[ii] += av_vels[ii];
        }
    }
    fclose(fp_fs);


    FILE* fp_av = fopen("av_vels.dat", "w");
    for (int ii = 0; ii < params.maxIters; ii++)
    {
        fprintf(fp_av, "%d:\t%.12E\n", ii, av_vels_acc[ii]);
    }
    fclose(fp_av);
  }

}


int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels, FILE* fp_fs, mpi_index params_mpi)
{
    const float c_sq = 1.0 / 3.0; /* sq. of speed of sound */
    float local_density;         /* per grid cell sum of densities */
    float pressure;              /* fluid pressure in grid cell */
    float u_x;                   /* x-component of velocity in grid cell */
    float u_y;                   /* y-component of velocity in grid cell */
    float u;                     /* norm--root of summed squares--of u_x and u_y */
    
    for (int ii = 0 ;ii < params_mpi.ny; ii++)
    {
        for (int jj = 0; jj < params_mpi.nx; jj++)
        {
            /* an occupied cell */
            if (obstacles[ii * params_mpi.nx + jj])
            {
                u_x = u_y = u = 0.0;
                pressure = params.density * c_sq;
            }
            /* no obstacle */
            else
            {
                local_density = 0.0;
                
                for (int kk = 0; kk < 9; kk++)
                {
                    local_density += cells[ii * params_mpi.nx + jj].speeds[kk];
                }
                
                /* compute x velocity component */
                u_x = (cells[ii * params_mpi.nx + jj].speeds[1]
                       + cells[ii * params_mpi.nx + jj].speeds[5]
                       + cells[ii * params_mpi.nx + jj].speeds[8]
                       - (cells[ii * params_mpi.nx + jj].speeds[3]
                          + cells[ii * params_mpi.nx + jj].speeds[6]
                          + cells[ii * params_mpi.nx + jj].speeds[7]))
                / local_density;
                /* compute y velocity component */
                u_y = (cells[ii * params_mpi.nx + jj].speeds[2]
                       + cells[ii * params_mpi.nx + jj].speeds[5]
                       + cells[ii * params_mpi.nx + jj].speeds[6]
                       - (cells[ii * params_mpi.nx + jj].speeds[4]
                          + cells[ii * params_mpi.nx + jj].speeds[7]
                          + cells[ii * params_mpi.nx + jj].speeds[8]))
                / local_density;
                /* compute norm of velocity */
                u = sqrt((u_x * u_x) + (u_y * u_y));
                /* compute pressure */
                pressure = local_density * c_sq;
            }
            
            int real_ii = ii + params_mpi.bottom_y;
            int real_jj = jj + params_mpi.left_x;
            /* write to file */
            fprintf(fp_fs, "%d %d %.12E %.12E %.12E %.12E %d\n", real_jj, real_ii, u_x, u_y, u, pressure, obstacles[ii * params_mpi.nx + jj]);
        }
    }    
    return EXIT_SUCCESS;
}


void die(const char* message, const int line, const char* file)
{
    fprintf(stderr, "Error at line %d of file %s:\n", line, file);
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
    exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
    fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
    exit(EXIT_FAILURE);
}

void print_cells(const t_param params, t_speed* cells){
    
    for (int ii = 0; ii < params.ny; ii++)
    {
        for (int jj = 0; jj < params.nx; jj++)
        {
            int index = ii * params.nx + jj;
            
            
            printf("%.12f ", cells[index].speeds[5]);
        }
        printf("\n");
        printf("index: %d \n", ii * params.nx + params.nx);
    }
}

void distribute_indexes(const t_param params, int world_size, int y_split, int x_split){
    
    int rank_dest = 1;
    for (int y = y_split; y > 0; y--)
    {
        int y_top = params.ny - (y - 1) * (params.ny / y_split);
        int y_bottom = params.ny - y * (params.ny / y_split);
        for (int x = x_split; x > 0; x--)
        {
            int x_right = params.nx - (x - 1) * (params.nx / x_split);
            int x_left = params.nx - x * (params.nx / x_split);
            
            if(!(y_bottom == 0 && x_left == 0)){
                mpi_index params_mpi = {-1, y_top, y_bottom, x_left, x_right,0,0,0,0, x_right - x_left, y_top - y_bottom};
               MPI_Ssend(&params_mpi, 11, MPI_INT, rank_dest, 1, MPI_COMM_WORLD);
                rank_dest++;
            }
        }
    }
}

void get_neighbours(mpi_index* params_mpi, const t_param params, int y_split, int x_split)
{
    params_mpi->nb_top_y = (params_mpi->top_y == params.ny) ? params_mpi->mpi_rank - (x_split * (y_split - 1)) : params_mpi->mpi_rank + x_split;
    params_mpi->nb_bottom_y = (params_mpi->bottom_y == 0) ? params_mpi->mpi_rank + (x_split * (y_split - 1)) : params_mpi->mpi_rank - x_split;
    params_mpi->nb_left_x = (params_mpi->left_x == 0) ? params_mpi->mpi_rank + (x_split - 1) : params_mpi->mpi_rank - 1;
    params_mpi->nb_right_x = (params_mpi->right_x == params.nx) ? params_mpi->mpi_rank - (x_split - 1) : params_mpi->mpi_rank + 1;
}