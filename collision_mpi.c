#include "d2q9-bgk.h"
float collision_mpi(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, const int* tot_cells, mpi_index mpi_params, mpi_halo mpi_halos, mpi_halo mpi_halo_snd)
{
    float tot_u = 0.0f;    /* accumulated magnitudes of velocity for each cell */
    static const float d1 = 1 / 36.0f;
    
    /* loop over the cells in the grid
     ** NB the collision step is called after
     ** the propagate step and so values of interest
     ** are in the scratch-space grid */

    halo_top_y(params, cells, tmp_cells, mpi_params, &tot_u, mpi_halos.top_y,mpi_halo_snd,obstacles, 0, mpi_params.ny - 2,mpi_params.ny - 1);
    halo_bottom_y(params, cells, tmp_cells, mpi_params, &tot_u, mpi_halos.bottom_y, mpi_halo_snd, obstacles, 1, mpi_params.ny - 1,0);

    #pragma omp parallel for simd reduction(+:tot_u) schedule(static) num_threads(4)
    for (int ii = 1; ii < mpi_params.ny - 1; ii++)
    {
        int y_s = ii - 1; // could move up
        int y_n = ii + 1; // Could move up
        
        
            for (int jj = 0; jj < mpi_params.nx; jj++)
            {
                int index = ii * mpi_params.nx + jj;
                
                int x_e = (jj + 1) % mpi_params.nx;
                int x_w = (jj == 0) ? (jj + mpi_params.nx - 1) : (jj - 1);
               
               // ** Remove if x_split == 1 ** 
               // if(jj == mpi_params.left_x){
               //     halo_left_x(params, cells, tmp_cells, mpi_params, &tot_u, mpi_halos.left_x, obstacles, y_n, y_s, x_e, x_w, ii, jj);
               // }
               // else if (jj == mpi_params.right_x - 1){
               //     halo_right_x(params, cells, tmp_cells, mpi_params, &tot_u, mpi_halos.right_x, obstacles, y_n, y_s, x_e, x_w, ii, jj);
               // }
                // -------------rebound--------------------------------
                /* don't consider occupied cells */
                if (obstacles[index])
                {
                    /* called after propagate, so taking values from scratch space
                     ** mirroring, and writing into main grid */
                    tmp_cells[index].speeds[0] = cells[ii * params.nx + jj].speeds[0]; /* central cell, no movement */
                    tmp_cells[index].speeds[1] = cells[ii * mpi_params.nx + x_e].speeds[3];
                    tmp_cells[index].speeds[2] = cells[y_n * mpi_params.nx + jj].speeds[4];
                    tmp_cells[index].speeds[3] = cells[ii * mpi_params.nx + x_w].speeds[1];
                    tmp_cells[index].speeds[4] = cells[y_s * mpi_params.nx + jj].speeds[2];
                    tmp_cells[index].speeds[5] = cells[y_n * mpi_params.nx + x_e].speeds[7];
                    tmp_cells[index].speeds[6] = cells[y_n * mpi_params.nx + x_w].speeds[8];
                    tmp_cells[index].speeds[7] = cells[y_s * mpi_params.nx + x_w].speeds[5];
                    tmp_cells[index].speeds[8] = cells[y_s * mpi_params.nx + x_e].speeds[6];
                }
                // ----------------END--------------------------------------------
                else
                {
                    
                    /* compute local density total */
                    float local_density = 0.0f;
                    local_density += cells[ii * mpi_params.nx + jj].speeds[0];
                    local_density += cells[ii * mpi_params.nx + x_e].speeds[3];
                    local_density += cells[y_n * mpi_params.nx + jj].speeds[4];
                    local_density += cells[ii * mpi_params.nx + x_w].speeds[1];
                    local_density += cells[y_s * mpi_params.nx + jj].speeds[2];
                    local_density += cells[y_n * mpi_params.nx + x_e].speeds[7];
                    local_density += cells[y_n * mpi_params.nx + x_w].speeds[8];
                    local_density += cells[y_s * mpi_params.nx + x_w].speeds[5];
                    local_density += cells[y_s * mpi_params.nx + x_e].speeds[6];
                    
                    
                    float local_density_invert = 1 / local_density;
                    /* compute x velocity component */
                    float u_x = (cells[ii * mpi_params.nx + x_w].speeds[1]
                                 + cells[y_s * mpi_params.nx + x_w].speeds[5]
                                 + cells[y_n * mpi_params.nx + x_w].speeds[8]
                                 - (cells[ii * mpi_params.nx + x_e].speeds[3]
                                    + cells[y_s * mpi_params.nx + x_e].speeds[6]
                                    + cells[y_n * mpi_params.nx + x_e].speeds[7]))
                    * local_density_invert;
                    /* compute y velocity component */
                    float u_y = (cells[y_s * mpi_params.nx + jj].speeds[2]
                                 + cells[y_s * mpi_params.nx + x_w].speeds[5]
                                 + cells[y_s * mpi_params.nx + x_e].speeds[6]
                                 - (cells[y_n * mpi_params.nx + jj].speeds[4]
                                    + cells[y_n * mpi_params.nx + x_e].speeds[7]
                                    + cells[y_n * mpi_params.nx + x_w].speeds[8]))
                    * local_density_invert;
                    
                    
                    
                    
                    tmp_cells[index].speeds[0] = cells[ii * mpi_params.nx + jj].speeds[0]
                    + params.omega
                    * (local_density * d1 * (16.0f - (u_x * u_x + u_y * u_y) * 864.0f * d1)
                       - cells[ii * mpi_params.nx + jj].speeds[0]);
                    tmp_cells[index].speeds[1] = cells[ii * mpi_params.nx + x_w].speeds[1]
                    + params.omega
                    * (local_density * d1 * (4.0f + u_x * 12.0f + (u_x * u_x) * 648.0f * d1- (216.0f * d1 * (u_x * u_x + u_y * u_y)))
                       - cells[ii * mpi_params.nx + x_w].speeds[1]);
                    tmp_cells[index].speeds[2] = cells[y_s * mpi_params.nx + jj].speeds[2]
                    + params.omega
                    * (local_density * d1 * (4.0f + u_y * 12.0f + (u_y * u_y) * 648.0f * d1 - (216.0f * d1 * (u_x * u_x + u_y * u_y)))
                       - cells[y_s * mpi_params.nx + jj].speeds[2]);
                    tmp_cells[index].speeds[3] = cells[ii * mpi_params.nx + x_e].speeds[3]
                    + params.omega
                    * (local_density * d1 * (4.0f - u_x * 12.0f + (u_x * u_x) * 648.0f * d1 - (216.0f * d1 * (u_x * u_x + u_y * u_y)))
                       - cells[ii * mpi_params.nx + x_e].speeds[3]);
                    tmp_cells[index].speeds[4] = cells[y_n * mpi_params.nx + jj].speeds[4]
                    + params.omega
                    * (local_density * d1 * (4.0f - u_y * 12.0f + (u_y * u_y) * 648.0f * d1 - (216.0f * d1 * (u_x * u_x + u_y * u_y)))
                       - cells[y_n * mpi_params.nx + jj].speeds[4]);
                    tmp_cells[index].speeds[5] = cells[y_s * mpi_params.nx + x_w].speeds[5]
                    + params.omega
                    * (local_density * d1 * (1.0f + (u_x + u_y) * 3.0f + ((u_x + u_y) * (u_x + u_y)) * 162.0f * d1 - (54.0f * d1 * (u_x * u_x + u_y * u_y)))
                       - cells[y_s * mpi_params.nx + x_w].speeds[5]);
                    tmp_cells[index].speeds[6] = cells[y_s * mpi_params.nx + x_e].speeds[6]
                    + params.omega
                    * (local_density * d1 * (1.0f + (- u_x + u_y) * 3.0f + ((- u_x + u_y) * (- u_x + u_y)) * 162.0f * d1 - (54.0f * d1 * (u_x * u_x + u_y * u_y)))
                       - cells[y_s * mpi_params.nx + x_e].speeds[6]);
                    tmp_cells[index].speeds[7] = cells[y_n * mpi_params.nx + x_e].speeds[7]
                    + params.omega
                    * (local_density * d1 * (1.0f + (- u_x - u_y) * 3.0f + ((- u_x - u_y) * (- u_x - u_y)) * 162.0f * d1 - (54.0f * d1 * (u_x * u_x + u_y * u_y)))
                       - cells[y_n * mpi_params.nx + x_e].speeds[7]);
                    tmp_cells[index].speeds[8] = cells[y_n * mpi_params.nx + x_w].speeds[8]
                    + params.omega
                    * (local_density * d1 * (1.0f + (u_x - u_y) * 3.0f + ((u_x - u_y) * (u_x - u_y)) * 162.0f * d1 - (54.0f * d1 * (u_x * u_x + u_y * u_y)))
                       - cells[y_n * mpi_params.nx + x_w].speeds[8]);
                    
                    
                    // --------------av_velocity-----------------------------------------------
                    
                    /* accumulate the norm of x- and y- velocity components */
                    tot_u += sqrt((u_x * u_x) + (u_y * u_y));
                }
            }
    }
    return tot_u / (float)(*tot_cells);
    
}


void halo_top_y(const t_param params, t_speed* cells, t_speed* tmp_cells,  mpi_index mpi_params, float* tot_u, t_buffer* top_y_buffer, mpi_halo snd_buffer, int* obstacles, int y_n, int y_s, int ii)
{
    /* compute local density total */
    static const float d1 = 1 / 36.0f;

    
    #pragma omp parallel for simd schedule(static) num_threads(4)
    for (int jj = 0; jj < mpi_params.nx; jj++)
    {
        int index = ii * mpi_params.nx + jj;
        
        int x_e = (jj + 1) % mpi_params.nx;
        int x_w = (jj == 0) ? (jj + mpi_params.nx - 1) : (jj - 1);
        
        // -------------rebound--------------------------------
        /* don't consider occupied cells */
        if (obstacles[index])
        {
            /* called after propagate, so taking values from scratch space
             ** mirroring, and writing into main grid */
            tmp_cells[index].speeds[0] = cells[ii * mpi_params.nx + jj].speeds[0]; /* central cell, no movement */
            tmp_cells[index].speeds[1] = cells[ii * mpi_params.nx + x_e].speeds[3];
            snd_buffer.top_y[jj].speeds[0] = tmp_cells[index].speeds[2] = top_y_buffer[jj].speeds[0];
            tmp_cells[index].speeds[3] = cells[ii * mpi_params.nx + x_w].speeds[1];
            tmp_cells[index].speeds[4] = cells[y_s * mpi_params.nx + jj].speeds[2];
            snd_buffer.top_y[jj].speeds[1] = tmp_cells[index].speeds[5] = top_y_buffer[x_e].speeds[1];
            snd_buffer.top_y[jj].speeds[2] = tmp_cells[index].speeds[6] = top_y_buffer[x_w].speeds[2];
            tmp_cells[index].speeds[7] = cells[y_s * mpi_params.nx + x_w].speeds[5];
            tmp_cells[index].speeds[8] = cells[y_s * mpi_params.nx + x_e].speeds[6];

        }
        // ----------------END--------------------------------------------
        else
        {
            float local_density = 0.0f;
            local_density += cells[ii * mpi_params.nx + jj].speeds[0];
            local_density += cells[ii * mpi_params.nx + x_e].speeds[3];
            local_density += top_y_buffer[jj].speeds[0];
            local_density += cells[ii * mpi_params.nx + x_w].speeds[1];
            local_density += cells[y_s * mpi_params.nx + jj].speeds[2];
            local_density += top_y_buffer[x_e].speeds[1];
            local_density += top_y_buffer[x_w].speeds[2];
            local_density += cells[y_s * mpi_params.nx + x_w].speeds[5];
            local_density += cells[y_s * mpi_params.nx + x_e].speeds[6];
            
            
            float local_density_invert = 1 / local_density;
            /* compute x velocity component */
            float u_x = (cells[ii * mpi_params.nx + x_w].speeds[1]
                         + cells[y_s * mpi_params.nx + x_w].speeds[5]
                         + top_y_buffer[x_w].speeds[2]
                         - (cells[ii * mpi_params.nx + x_e].speeds[3]
                            + cells[y_s * mpi_params.nx + x_e].speeds[6]
                            + top_y_buffer[x_e].speeds[1]))
            * local_density_invert;
            /* compute y velocity component */
            float u_y = (cells[y_s * mpi_params.nx + jj].speeds[2]
                         + cells[y_s * mpi_params.nx + x_w].speeds[5]
                         + cells[y_s * mpi_params.nx + x_e].speeds[6]
                         - (top_y_buffer[jj].speeds[0]
                            + top_y_buffer[x_e].speeds[1]
                            + top_y_buffer[x_w].speeds[2]))
            * local_density_invert;
            
            
            
            
            tmp_cells[index].speeds[0] = cells[ii * mpi_params.nx + jj].speeds[0]
            + params.omega
            * (local_density * d1 * (16.0f - (u_x * u_x + u_y * u_y) * 864.0f * d1)
               - cells[ii * mpi_params.nx + jj].speeds[0]);
            
            tmp_cells[index].speeds[1] = cells[ii * mpi_params.nx + x_w].speeds[1]
            + params.omega
            * (local_density * d1 * (4.0f + u_x * 12.0f + (u_x * u_x) * 648.0f * d1- (216.0f * d1 * (u_x * u_x + u_y * u_y)))
               - cells[ii * mpi_params.nx + x_w].speeds[1]);
            
            snd_buffer.top_y[jj].speeds[0] = 
            tmp_cells[index].speeds[2] = cells[y_s * mpi_params.nx + jj].speeds[2]
            + params.omega
            * (local_density * d1 * (4.0f + u_y * 12.0f + (u_y * u_y) * 648.0f * d1 - (216.0f * d1 * (u_x * u_x + u_y * u_y)))
               - cells[y_s * mpi_params.nx + jj].speeds[2]);
            
            tmp_cells[index].speeds[3] = cells[ii * mpi_params.nx + x_e].speeds[3]
            + params.omega
            * (local_density * d1 * (4.0f - u_x * 12.0f + (u_x * u_x) * 648.0f * d1 - (216.0f * d1 * (u_x * u_x + u_y * u_y)))
               - cells[ii * mpi_params.nx + x_e].speeds[3]);
            
            tmp_cells[index].speeds[4] = top_y_buffer[jj].speeds[0]
            + params.omega
            * (local_density * d1 * (4.0f - u_y * 12.0f + (u_y * u_y) * 648.0f * d1 - (216.0f * d1 * (u_x * u_x + u_y * u_y)))
               - top_y_buffer[jj].speeds[0]);
        
            snd_buffer.top_y[jj].speeds[1] =
            tmp_cells[index].speeds[5] = cells[y_s * mpi_params.nx + x_w].speeds[5]
            + params.omega
            * (local_density * d1 * (1.0f + (u_x + u_y) * 3.0f + ((u_x + u_y) * (u_x + u_y)) * 162.0f * d1 - (54.0f * d1 * (u_x * u_x + u_y * u_y)))
               - cells[y_s * mpi_params.nx + x_w].speeds[5]);
            
            snd_buffer.top_y[jj].speeds[2] =
            tmp_cells[index].speeds[6] = cells[y_s * mpi_params.nx + x_e].speeds[6]
            + params.omega
            * (local_density * d1 * (1.0f + (- u_x + u_y) * 3.0f + ((- u_x + u_y) * (- u_x + u_y)) * 162.0f * d1 - (54.0f * d1 * (u_x * u_x + u_y * u_y)))
               - cells[y_s * mpi_params.nx + x_e].speeds[6]);
            
            tmp_cells[index].speeds[7] = top_y_buffer[x_e].speeds[1]
            + params.omega
            * (local_density * d1 * (1.0f + (- u_x - u_y) * 3.0f + ((- u_x - u_y) * (- u_x - u_y)) * 162.0f * d1 - (54.0f * d1 * (u_x * u_x + u_y * u_y)))
               - top_y_buffer[x_e].speeds[1]);
            
            tmp_cells[index].speeds[8] = top_y_buffer[x_w].speeds[2]
            + params.omega
            * (local_density * d1 * (1.0f + (u_x - u_y) * 3.0f + ((u_x - u_y) * (u_x - u_y)) * 162.0f * d1 - (54.0f * d1 * (u_x * u_x + u_y * u_y)))
               - top_y_buffer[x_w].speeds[2]);
            
            


            // --------------av_velocity-----------------------------------------------
            
            /* accumulate the norm of x- and y- velocity components */
            *tot_u += sqrt((u_x * u_x) + (u_y * u_y));
        }
    }
    
    
}

void halo_bottom_y(const t_param params, t_speed* cells, t_speed* tmp_cells, mpi_index mpi_params, float* tot_u, t_buffer* bottom_y_buffer, mpi_halo snd_buffer, int* obstacles, int y_n, int y_s, int ii)
{
    /* compute local density total */
    static const float d1 = 1 / 36.0f;
    
    int buffer_index = mpi_params.nx*(mpi_params.ny - 1);
    

    #pragma omp parallel for simd schedule(static) num_threads(4)
    for (int jj = 0; jj < mpi_params.nx; jj++)
    {
        int index = ii * mpi_params.nx + jj;

        // // bottom left
        // if((mpi_params.nx != params.nx) && (jj == 0)){

        //  }
        // // bottom right
        // if((mpi_params.nx != params.nx) && (jj == mpi_params.nx - 1)){

        //  }        
        int x_e = (jj + 1) % mpi_params.nx;
        int x_w = (jj == 0) ? (jj + mpi_params.nx - 1) : (jj - 1);
        
        // -------------rebound--------------------------------
        /* don't consider occupied cells */
        if (obstacles[index])
        {
            /* called after propagate, so taking values from scratch space
             ** mirroring, and writing into main grid */

            tmp_cells[index].speeds[0] = cells[jj].speeds[0]; /* central cell, no movement */
            tmp_cells[index].speeds[1] = cells[x_e].speeds[3];
            tmp_cells[index].speeds[2] = cells[y_n * mpi_params.nx + jj].speeds[4];
            tmp_cells[index].speeds[3] = cells[x_w].speeds[1];
            snd_buffer.bottom_y[jj].speeds[0] = tmp_cells[index].speeds[4] = bottom_y_buffer[y_s * mpi_params.nx + jj - buffer_index].speeds[0];
            tmp_cells[jj].speeds[5] = cells[y_n * mpi_params.nx + x_e].speeds[7];
            tmp_cells[jj].speeds[6] = cells[y_n * mpi_params.nx + x_w].speeds[8];
            snd_buffer.bottom_y[index].speeds[1] = tmp_cells[index].speeds[7] = bottom_y_buffer[y_s * mpi_params.nx + x_w - buffer_index].speeds[1];
            snd_buffer.bottom_y[index].speeds[2] = tmp_cells[index].speeds[8] = bottom_y_buffer[y_s * mpi_params.nx + x_e - buffer_index].speeds[2];
        }
        // ----------------END--------------------------------------------
        else
        {
            float local_density = 0.0f;
            local_density += cells[jj].speeds[0];
            local_density += cells[x_e].speeds[3];
            local_density += cells[y_n * mpi_params.nx + jj].speeds[4];
            local_density += cells[x_w].speeds[1];
            local_density += bottom_y_buffer[jj].speeds[0];
            local_density += cells[y_n * mpi_params.nx + x_e].speeds[7];
            local_density += cells[y_n * mpi_params.nx + x_w].speeds[8];
            local_density += bottom_y_buffer[x_w].speeds[1];
            local_density += bottom_y_buffer[x_e].speeds[2];
            
            
            float local_density_invert = 1 / local_density;
            /* compute x velocity component */
            float u_x = (cells[x_w].speeds[1]
                         + bottom_y_buffer[x_w].speeds[1]
                         + cells[y_n * mpi_params.nx + x_w].speeds[8]
                         - (cells[x_e].speeds[3]
                            + bottom_y_buffer[x_e].speeds[2]
                            + cells[y_n * mpi_params.nx + x_e].speeds[7]))
            * local_density_invert;
            /* compute y velocity component */
            float u_y = (bottom_y_buffer[jj].speeds[0]
                         + bottom_y_buffer[x_w].speeds[1]
                         + bottom_y_buffer[x_e].speeds[2]
                         - (cells[y_n * mpi_params.nx + jj].speeds[4]
                            + cells[y_n * mpi_params.nx + x_e].speeds[7]
                            + cells[y_n * mpi_params.nx + x_w].speeds[8]))
            * local_density_invert;
            
            
            
            
            tmp_cells[index].speeds[0] = cells[jj].speeds[0]
            + params.omega
            * (local_density * d1 * (16.0f - (u_x * u_x + u_y * u_y) * 864.0f * d1)
               - cells[jj].speeds[0]);
            
            tmp_cells[index].speeds[1] = cells[x_w].speeds[1]
            + params.omega
            * (local_density * d1 * (4.0f + u_x * 12.0f + (u_x * u_x) * 648.0f * d1- (216.0f * d1 * (u_x * u_x + u_y * u_y)))
               - cells[x_w].speeds[1]);
            
            tmp_cells[index].speeds[2] = bottom_y_buffer[jj].speeds[0]
            + params.omega
            * (local_density * d1 * (4.0f + u_y * 12.0f + (u_y * u_y) * 648.0f * d1 - (216.0f * d1 * (u_x * u_x + u_y * u_y)))
               - bottom_y_buffer[jj].speeds[0]);
            
            tmp_cells[index].speeds[3] = cells[x_e].speeds[3]
            + params.omega
            * (local_density * d1 * (4.0f - u_x * 12.0f + (u_x * u_x) * 648.0f * d1 - (216.0f * d1 * (u_x * u_x + u_y * u_y)))
               - cells[x_e].speeds[3]);
            
            snd_buffer.bottom_y[jj].speeds[0] =
            tmp_cells[index].speeds[4] = cells[y_n * mpi_params.nx + jj].speeds[4]
            + params.omega
            * (local_density * d1 * (4.0f - u_y * 12.0f + (u_y * u_y) * 648.0f * d1 - (216.0f * d1 * (u_x * u_x + u_y * u_y)))
               - cells[y_n * mpi_params.nx + jj].speeds[4]);
            
            tmp_cells[index].speeds[5] = bottom_y_buffer[x_w].speeds[1]
            + params.omega
            * (local_density * d1 * (1.0f + (u_x + u_y) * 3.0f + ((u_x + u_y) * (u_x + u_y)) * 162.0f * d1 - (54.0f * d1 * (u_x * u_x + u_y * u_y)))
               - bottom_y_buffer[x_w].speeds[1]);
            
            tmp_cells[index].speeds[6] = bottom_y_buffer[x_e].speeds[2]
            + params.omega
            * (local_density * d1 * (1.0f + (- u_x + u_y) * 3.0f + ((- u_x + u_y) * (- u_x + u_y)) * 162.0f * d1 - (54.0f * d1 * (u_x * u_x + u_y * u_y)))
               - bottom_y_buffer[x_e].speeds[2]);
            
            snd_buffer.bottom_y[jj].speeds[1] =
            tmp_cells[index].speeds[7] = cells[y_n * mpi_params.nx + x_e].speeds[7]
            + params.omega
            * (local_density * d1 * (1.0f + (- u_x - u_y) * 3.0f + ((- u_x - u_y) * (- u_x - u_y)) * 162.0f * d1 - (54.0f * d1 * (u_x * u_x + u_y * u_y)))
               - cells[y_n * mpi_params.nx + x_e].speeds[7]);
            
            snd_buffer.bottom_y[jj].speeds[2] =
            tmp_cells[index].speeds[8] = cells[y_n * mpi_params.nx + x_w].speeds[8]
            + params.omega
            * (local_density * d1 * (1.0f + (u_x - u_y) * 3.0f + ((u_x - u_y) * (u_x - u_y)) * 162.0f * d1 - (54.0f * d1 * (u_x * u_x + u_y * u_y)))
               - cells[y_n * mpi_params.nx + x_w].speeds[8]);
            
            
            // --------------av_velocity-----------------------------------------------
            
            /* accumulate the norm of x- and y- velocity components */
            *tot_u += sqrt((u_x * u_x) + (u_y * u_y));
        }
    }
    
}



void halo_left_x(const t_param params, t_speed* cells, t_speed* tmp_cells,  mpi_index mpi_params, float* tot_u, t_buffer* left_x_buffer, int* obstacles, int y_n, int y_s, int x_e, int x_w, int ii, int jj)
{
    /* compute local density total */
    static const float d1 = 1 / 36.0f;
    int index = ii * mpi_params.nx + jj;
    
    int buffer_index = (mpi_params.left_x == 0) ? (ii + 1) * mpi_params.nx - ii - 1 : ii * mpi_params.nx + jj - ii - 1;
    
    // -------------rebound--------------------------------
    
    /* don't consider occupied cells */
    if (obstacles[index])
    {
        /* called after propagate, so taking values from scratch space
         ** mirroring, and writing into main grid */
        tmp_cells[index].speeds[0] = cells[ii * params.nx + jj].speeds[0]; /* central cell, no movement */
        tmp_cells[index].speeds[1] = cells[ii * mpi_params.nx + x_e].speeds[3];
        tmp_cells[index].speeds[2] = cells[y_n * mpi_params.nx + jj].speeds[4];
        tmp_cells[index].speeds[3] = left_x_buffer[ii].speeds[1];
        tmp_cells[index].speeds[4] = cells[y_s * mpi_params.nx + jj].speeds[2];
        tmp_cells[index].speeds[5] = cells[y_n * mpi_params.nx + x_e].speeds[7];
        tmp_cells[index].speeds[6] = left_x_buffer[y_n].speeds[8];
        tmp_cells[index].speeds[7] = left_x_buffer[y_s].speeds[5];
        tmp_cells[index].speeds[8] = cells[y_s * mpi_params.nx + x_e].speeds[6];
    }
    // ----------------END--------------------------------------------
    else
    {
        
        /* compute local density total */
        float local_density = 0.0f;
        local_density += cells[ii * mpi_params.nx + jj].speeds[0];
        local_density += cells[ii * mpi_params.nx + x_e].speeds[3];
        local_density += cells[y_n * mpi_params.nx + jj].speeds[4];
        local_density += left_x_buffer[ii].speeds[1];
        local_density += cells[y_s * mpi_params.nx + jj].speeds[2];
        local_density += cells[y_n * mpi_params.nx + x_e].speeds[7];
        local_density += left_x_buffer[y_n].speeds[8];
        local_density += left_x_buffer[y_s].speeds[5];
        local_density += cells[y_s * mpi_params.nx + x_e].speeds[6];
        
        
        float local_density_invert = 1 / local_density;
        /* compute x velocity component */
        float u_x = (left_x_buffer[ii].speeds[1]
                     + left_x_buffer[y_s].speeds[5]
                     + left_x_buffer[y_n].speeds[8]
                     - (cells[ii * mpi_params.nx + x_e].speeds[3]
                        + cells[y_s * mpi_params.nx + x_e].speeds[6]
                        + cells[y_n * mpi_params.nx + x_e].speeds[7]))
        * local_density_invert;
        /* compute y velocity component */
        float u_y = (cells[y_s * mpi_params.nx + jj].speeds[2]
                     + left_x_buffer[y_s].speeds[5]
                     + cells[y_s * mpi_params.nx + x_e].speeds[6]
                     - (cells[y_n * mpi_params.nx + jj].speeds[4]
                        + cells[y_n * mpi_params.nx + x_e].speeds[7]
                        + left_x_buffer[y_n].speeds[8]))
        * local_density_invert;
        
        
        
        
        tmp_cells[index].speeds[0] = cells[ii * mpi_params.nx + jj].speeds[0]
        + params.omega
        * (local_density * d1 * (16 - (u_x * u_x + u_y * u_y) * 864 * d1)
           - cells[ii * mpi_params.nx + jj].speeds[0]);
        tmp_cells[index].speeds[1] = left_x_buffer[ii].speeds[1]
        + params.omega
        * (local_density * d1 * (4 + u_x * 12.0f + (u_x * u_x) * 648.0f * d1- (216.0f * d1 * (u_x * u_x + u_y * u_y)))
           - left_x_buffer[ii].speeds[1]);
        tmp_cells[index].speeds[2] = cells[y_s * mpi_params.nx + jj].speeds[2]
        + params.omega
        * (local_density * d1 * (4 + u_y * 12.0f + (u_y * u_y) * 648.0f * d1 - (216.0f * d1 * (u_x * u_x + u_y * u_y)))
           - cells[y_s * mpi_params.nx + jj].speeds[2]);
        tmp_cells[index].speeds[3] = cells[ii * mpi_params.nx + x_e].speeds[3]
        + params.omega
        * (local_density * d1 * (4 - u_x * 12.0f + (u_x * u_x) * 648.0f * d1 - (216.0f * d1 * (u_x * u_x + u_y * u_y)))
           - cells[ii * mpi_params.nx + x_e].speeds[3]);
        tmp_cells[index].speeds[4] = cells[y_n * mpi_params.nx + jj].speeds[4]
        + params.omega
        * (local_density * d1 * (4 - u_y * 12.0f + (u_y * u_y) * 648.0f * d1 - (216.0f * d1 * (u_x * u_x + u_y * u_y)))
           - cells[y_n * mpi_params.nx + jj].speeds[4]);
        tmp_cells[index].speeds[5] = left_x_buffer[y_s].speeds[5]
        + params.omega
        * (local_density * d1 * (1 + (u_x + u_y) * 3 + ((u_x + u_y) * (u_x + u_y)) * 162.0f * d1 - (54 * d1 * (u_x * u_x + u_y * u_y)))
           - left_x_buffer[y_s].speeds[5]);
        tmp_cells[index].speeds[6] = cells[y_s * mpi_params.nx + x_e].speeds[6]
        + params.omega
        * (local_density * d1 * (1 + (- u_x + u_y) * 3 + ((- u_x + u_y) * (- u_x + u_y)) * 162.0f * d1 - (54 * d1 * (u_x * u_x + u_y * u_y)))
           - cells[y_s * mpi_params.nx + x_e].speeds[6]);
        tmp_cells[index].speeds[7] = cells[y_n * mpi_params.nx + x_e].speeds[7]
        + params.omega
        * (local_density * d1 * (1 + (- u_x - u_y) * 3 + ((- u_x - u_y) * (- u_x - u_y)) * 162.0f * d1 - (54 * d1 * (u_x * u_x + u_y * u_y)))
           - cells[y_n * mpi_params.nx + x_e].speeds[7]);
        tmp_cells[index].speeds[8] = left_x_buffer[y_n].speeds[8]
        + params.omega
        * (local_density * d1 * (1 + (u_x - u_y) * 3 + ((u_x - u_y) * (u_x - u_y)) * 162.0f * d1 - (54 * d1 * (u_x * u_x + u_y * u_y)))
           - left_x_buffer[y_n].speeds[8]);
        
        
        // --------------av_velocity-----------------------------------------------
        
        /* accumulate the norm of x- and y- velocity components */
        *tot_u += sqrt((u_x * u_x) + (u_y * u_y));
    }
    
}

void halo_right_x(const t_param params, t_speed* cells, t_speed* tmp_cells,  mpi_index mpi_params, float* tot_u, t_buffer* right_x_buffer, int* obstacles, int y_n, int y_s, int x_e, int x_w, int ii, int jj)
{
    /* compute local density total */
    static const float d1 = 1 / 36.0f;
    int index = ii * mpi_params.nx + jj;
    
    int buffer_index = (mpi_params.right_x == mpi_params.nx) ? ii * mpi_params.nx - ii : ii * mpi_params.nx + jj - ii + 1;
    
    // -------------rebound--------------------------------
    
    /* don't consider occupied cells */
    if (obstacles[index])
    {
        /* called after propagate, so taking values from scratch space
         ** mirroring, and writing into main grid */
        tmp_cells[index].speeds[0] = cells[ii * params.nx + jj].speeds[0]; /* central cell, no movement */
        tmp_cells[index].speeds[1] = right_x_buffer[ii].speeds[3];
        tmp_cells[index].speeds[2] = cells[y_n * mpi_params.nx + jj].speeds[4];
        tmp_cells[index].speeds[3] = cells[ii * mpi_params.nx + x_w].speeds[1];
        tmp_cells[index].speeds[4] = cells[y_s * mpi_params.nx + jj].speeds[2];
        tmp_cells[index].speeds[5] = right_x_buffer[y_n].speeds[7];
        tmp_cells[index].speeds[6] = cells[y_n * mpi_params.nx + x_w].speeds[8];
        tmp_cells[index].speeds[7] = cells[y_s * mpi_params.nx + x_w].speeds[5];
        tmp_cells[index].speeds[8] = right_x_buffer[y_s].speeds[6];
    }
    // ----------------END--------------------------------------------
    else
    {
        
        /* compute local density total */
        float local_density = 0.0f;
        local_density += cells[ii * mpi_params.nx + jj].speeds[0];
        local_density += right_x_buffer[ii].speeds[3];
        local_density += cells[y_n * mpi_params.nx + jj].speeds[4];
        local_density += cells[ii * mpi_params.nx + x_w].speeds[1];
        local_density += cells[y_s * mpi_params.nx + jj].speeds[2];
        local_density += right_x_buffer[y_n].speeds[7];
        local_density += cells[y_n * mpi_params.nx + x_w].speeds[8];
        local_density += cells[y_s * mpi_params.nx + x_w].speeds[5];
        local_density += right_x_buffer[y_s].speeds[6];
        
        
        float local_density_invert = 1 / local_density;
        /* compute x velocity component */
        float u_x = (cells[ii * mpi_params.nx + x_w].speeds[1]
                     + cells[y_s * mpi_params.nx + x_w].speeds[5]
                     + cells[y_n * mpi_params.nx + x_w].speeds[8]
                     - (right_x_buffer[ii].speeds[3]
                        + right_x_buffer[y_s].speeds[6]
                        + right_x_buffer[y_n].speeds[7]))
        * local_density_invert;
        /* compute y velocity component */
        float u_y = (cells[y_s * mpi_params.nx + jj].speeds[2]
                     + cells[y_s * mpi_params.nx + x_w].speeds[5]
                     + right_x_buffer[y_s].speeds[6]
                     - (cells[y_n * mpi_params.nx + jj].speeds[4]
                        + right_x_buffer[y_n].speeds[7]
                        + cells[y_n * mpi_params.nx + x_w].speeds[8]))
        * local_density_invert;
        
        
        
        
        tmp_cells[index].speeds[0] = cells[ii * mpi_params.nx + jj].speeds[0]
        + params.omega
        * (local_density * d1 * (16 - (u_x * u_x + u_y * u_y) * 864 * d1)
           - cells[ii * mpi_params.nx + jj].speeds[0]);
        tmp_cells[index].speeds[1] = cells[ii * mpi_params.nx + x_w].speeds[1]
        + params.omega
        * (local_density * d1 * (4 + u_x * 12.0f + (u_x * u_x) * 648.0f * d1- (216.0f * d1 * (u_x * u_x + u_y * u_y)))
           - cells[ii * mpi_params.nx + x_w].speeds[1]);
        tmp_cells[index].speeds[2] = cells[y_s * mpi_params.nx + jj].speeds[2]
        + params.omega
        * (local_density * d1 * (4 + u_y * 12.0f + (u_y * u_y) * 648.0f * d1 - (216.0f * d1 * (u_x * u_x + u_y * u_y)))
           - cells[y_s * mpi_params.nx + jj].speeds[2]);
        tmp_cells[index].speeds[3] = right_x_buffer[ii].speeds[3]
        + params.omega
        * (local_density * d1 * (4 - u_x * 12.0f + (u_x * u_x) * 648.0f * d1 - (216.0f * d1 * (u_x * u_x + u_y * u_y)))
           - right_x_buffer[ii].speeds[3]);
        tmp_cells[index].speeds[4] = cells[y_n * mpi_params.nx + jj].speeds[4]
        + params.omega
        * (local_density * d1 * (4 - u_y * 12.0f + (u_y * u_y) * 648.0f * d1 - (216.0f * d1 * (u_x * u_x + u_y * u_y)))
           - cells[y_n * mpi_params.nx + jj].speeds[4]);
        tmp_cells[index].speeds[5] = cells[y_s * mpi_params.nx + x_w].speeds[5]
        + params.omega
        * (local_density * d1 * (1 + (u_x + u_y) * 3 + ((u_x + u_y) * (u_x + u_y)) * 162.0f * d1 - (54 * d1 * (u_x * u_x + u_y * u_y)))
           - cells[y_s * mpi_params.nx + x_w].speeds[5]);
        tmp_cells[index].speeds[6] = right_x_buffer[y_s].speeds[6]
        + params.omega
        * (local_density * d1 * (1 + (- u_x + u_y) * 3 + ((- u_x + u_y) * (- u_x + u_y)) * 162.0f * d1 - (54 * d1 * (u_x * u_x + u_y * u_y)))
           - right_x_buffer[y_s].speeds[6]);
        tmp_cells[index].speeds[7] = right_x_buffer[y_n].speeds[7]
        + params.omega
        * (local_density * d1 * (1 + (- u_x - u_y) * 3 + ((- u_x - u_y) * (- u_x - u_y)) * 162.0f * d1 - (54 * d1 * (u_x * u_x + u_y * u_y)))
           - right_x_buffer[y_n].speeds[7]);
        tmp_cells[index].speeds[8] = cells[y_n * mpi_params.nx + x_w].speeds[8]
        + params.omega
        * (local_density * d1 * (1 + (u_x - u_y) * 3 + ((u_x - u_y) * (u_x - u_y)) * 162.0f * d1 - (54 * d1 * (u_x * u_x + u_y * u_y)))
           - cells[y_n * mpi_params.nx + x_w].speeds[8]);
        
        
        // --------------av_velocity-----------------------------------------------
        
        /* accumulate the norm of x- and y- velocity components */
        *tot_u += sqrt((u_x * u_x) + (u_y * u_y));
    }
    
}



