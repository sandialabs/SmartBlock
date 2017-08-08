/*
 * magnitude.c
 *
 * Copyright 2017  National Technology and Engineering Solutions of Sandia,
 * LLC. Under the terms of Contract DE-NA0003525, there is a non-exclusive
 * license for use of this work by or on behalf of the U.S.  Government. Export
 * of this program may require a license from the United States Government.
 *
 *
 * This program is designed to be an intermediary component
 * in a scientific workflow. Given as input a 2D array composed
 * of a quantity broken into its N dimensionsal components over 
 * any number of datapoints and over any number of timesteps, 
 * this program computes and outputs the magnitudes for this quantity 
 * for all data points into a new 1-D array for each timestep, 
 * through a new Flexpath stream.
 * 
 * Input and output is done through ADIOS/FLEXPATH.
 *
 * INPUT: 2D array
 * OUTPUT: 1D array
 *
 * Example input: 
 *    [v1_x, v1_y, v1_z, v2_x, v2_y, v2_z, ... ]
 * Example output:
 *    [ |v1| , |v2| , ... ]
 *
 * Usage:
 *  <exec> input-stream-name input-arr-name \
 *         output-stream-name ouput-arr-name 
 * 
 * Alexis Champsaur, Sandia Labs, June 2015
 */

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>
#include <math.h>
#include "mpi.h"
#include "adios.h"
#include "adios_read.h"
#include "lib_mem_monitor.h"

static int rank, size;
static MPI_Comm comm;
static void get_check_global_range( uint64_t _global_size, 
    uint64_t *_mysize, uint64_t *_mystart, 
    uint64_t *_common_size, int *_mod);
static uint64_t getMyOffset (uint64_t common_size, int numargs, int mod);

int main (int argc, char **argv)
{
  int i,j, arrdims, step, sizeone, mod;
  uint64_t global_size, mycount, mysize, mystart, common_size,
           myoffset, totalsz, tstep, hdr_len;
  int64_t g_handle, out_handle;
  double sum, current_doub, magn;
  char *in_stream, *in_arr_name, *out_stream, *out_arr_name;
  double *data_in, *magnitudes;
  
  comm = MPI_COMM_WORLD;

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (comm, &rank);
  MPI_Comm_size (comm, &size);
  
  if (rank == 0) printf("magnitude start...\n");

  /* Parse arg */
  if (rank == 0 && argc != 5) {
    fprintf(stderr, "\nUsage: <exec> input-array-name "
       "input-stream-name output-array-name output-stream-name\n\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  MPI_Barrier(comm);
  in_stream = argv[1];
  in_arr_name = argv[2];
  //printf("[MAGN%d] in arr name: %s\n", rank, in_arr_name);
  out_stream = argv[3];
  out_arr_name = argv[4];

  /* INPUT */
  adios_read_init_method (ADIOS_READ_METHOD_FLEXPATH, comm, "");
  ADIOS_FILE *f = adios_read_open (in_stream,
      ADIOS_READ_METHOD_FLEXPATH,
      comm,
      ADIOS_LOCKMODE_NONE, 0.0);
  
  /* OUTPUT */
  adios_init_noxml (comm);
  adios_allocate_buffer (ADIOS_BUFFER_ALLOC_NOW, 200);
  adios_declare_group (&g_handle, "magnitudegroup", "", adios_flag_yes);
  adios_select_method (g_handle, "FLEXPATH", "", "");

  step = 0;
  while(adios_errno != err_end_of_stream){
    /* Monitoring library */
#ifdef ENABLE_MONITOR
    lib_mem_init();
    ind_timer_start(0, "whole tstep");
#endif

    ADIOS_VARINFO *arr_info = adios_inq_var(f, in_arr_name);
    ADIOS_VARINFO *hdr_len_info = adios_inq_var(f, "header_len");
    arrdims = arr_info->ndim;
    hdr_len = *(int *)(hdr_len_info->value); 
     
    /* Input array must be 2D */
    if (arrdims != 2){
      fprintf(stderr, "ERROR: input array must be 2-dimensional\n");
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
    
    /* Learn which dimension spans the components, and which spans the "particles"
       The smaller dimension */ 
    uint64_t tempszone;
    global_size = arr_info->dims[0];
    tempszone = arr_info->dims[1];
    if (global_size < tempszone){
      fprintf(stderr, "You have more components per data point than data points themselves."
          " Are you sure input array dimensions are in the correct order?\n");
    }
    if (tempszone > 200){
      fprintf(stderr, "ERROR: Really? More than 200 dimensions?\n");
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
    sizeone = (int) tempszone;
    
    get_check_global_range(global_size, &mysize, &mystart, &common_size, &mod);
     

    /* Allocate space for new data */
    double data_in2D[mysize][sizeone];
    uint64_t bytes_in = (sizeof(double) * mysize * sizeone);
    data_in = (double *)&data_in2D;    

    /* Read data */
    uint64_t starts[] = {mystart, 0};
    uint64_t counts[] = {mysize, sizeone};
    ADIOS_SELECTION *global_select = 
      adios_selection_boundingbox(2, starts, counts);
    adios_schedule_read (f,
                         global_select,
                         in_arr_name,
                         0, 1, data_in);
    //printf("[MAGN%d]: schedule_read ok\n", rank);
#ifdef ENABLE_MONITOR
    nohandler_mem(rank);
#endif
    tstep = 0;
    adios_schedule_read (f, NULL, "ntimestep", 0, 1, &tstep);
    char hdr[80];
    ADIOS_SELECTION *block_sel = adios_selection_writeblock(0);
    adios_schedule_read (f, block_sel, "header_string", 0, 1, hdr);
#ifdef ENABLE_MONITOR
    ind_timer_start(1, "perform_reads");
#endif
    adios_perform_reads (f, 1);
#ifdef ENABLE_MONITOR
    ind_timer_end(1);
#endif
    adios_release_step(f);

    /* Calculate magnitudes and put into new array */
    magnitudes = (double *)malloc(sizeof(double) * mysize);
    memset (magnitudes, 0, sizeof(double) * mysize);
    for (i=0; i<mysize; i++)
    {
      sum = 0;
      for (j=0; j<sizeone; j++){
        current_doub = data_in2D[i][j];
        sum += (current_doub * current_doub);
      }
      magn = sqrt(sum);
      magnitudes[i] = magn;
    }

#ifdef ENABLE_MONITOR
    nohandler_mem(rank);
#endif

    /* OUTPUT */
    myoffset = getMyOffset (common_size, 1, mod);
    
    if (step == 0) {
      adios_define_var (g_handle, "me", "", adios_integer, "", "", "");
      adios_define_var (g_handle, "mpi_size", "", adios_integer, "", "", "");
      adios_define_var (g_handle, "ntimestep", "", adios_unsigned_long, "", "", "");
      adios_define_var (g_handle, "mysize", "", adios_unsigned_long, "", "", "");
      adios_define_var (g_handle, "myoffset", "", adios_unsigned_long, "", "", "");
      adios_define_var (g_handle, "global_size", "", adios_unsigned_long, "", "", "");
      adios_define_var (g_handle, "header_len", "", adios_integer, "", "", "");
      adios_define_var (g_handle, "header_string", "",
          adios_byte, "header_len", "", "");
      adios_define_var (g_handle, out_arr_name, "", adios_double, "mysize", 
          "global_size", "myoffset");
    }
    
    uint64_t ints = sizeof(int) * 3;
    uint64_t strsz = (uint64_t) hdr_len; 
#ifdef ENABLE_MONITOR
    nohandler_mem(rank);
#endif

    if (rank == 0)
    {
        printf("[MAG-%d] opening for writing out_stream: %d\n",
               out_stream);
    }
    
    adios_open  (&out_handle, "magnitudegroup", out_stream, "a", comm);
    adios_group_size (out_handle, ints + strsz + sizeof(uint64_t)*4 + 
        sizeof(double)*mysize, &totalsz);
    adios_write (out_handle, "me", &rank);
    adios_write (out_handle, "mpi_size", &size);
    adios_write (out_handle, "ntimestep", &tstep);
    adios_write (out_handle, "mysize", &mysize);
    adios_write (out_handle, "header_len", &hdr_len);
    adios_write (out_handle, "header_string", hdr); 
    adios_write (out_handle, "myoffset", &myoffset);
    adios_write (out_handle, "global_size", &global_size);
    adios_write (out_handle, out_arr_name, magnitudes);
    adios_close (out_handle);

    /* Monitoring library output */
#ifdef ENABLE_MONITOR
    nohandler_mem(rank);
    ind_timer_end(0);
    char monitor_title[40];
    sprintf(monitor_title, "magnitude");
    monitor_out (rank, size, tstep, bytes_in, comm, monitor_title);
#endif

    if (rank == 0) printf("[MAGNITU%d]: read and wrote data for timestep %" PRIu64 "\n", rank, tstep);
    step++;
    int adv_ret = adios_advance_step(f, 0, -1);
    if (rank == 0) {
        printf("[HIST%d] adios_advance_step returned %d\n", rank, adv_ret);
    }

  }

  printf("[MAGNITU%d]: out of read loop\n", rank);

  adios_read_close(f);
  adios_read_finalize_method(ADIOS_READ_METHOD_FLEXPATH);
  adios_finalize(rank);
  MPI_Finalize();
  return 0;
}


static void
get_check_global_range( uint64_t _global_size, 
                        uint64_t *_mysize,
                        uint64_t *_mystart,
                        uint64_t *_common_size,
                        int *_mod)
{
  /* Error checking: total size must be a multiple of group size */
  int mod;
  //int num_data_points = _global_size / _sizeone; //total data points
  mod = _global_size % size; //where size = mpi size
  *_mod = mod;
  if (mod ==  0) { //every proc receives same number of data points
    *_mysize = _global_size / size;
    *_mystart = *_mysize * rank;
    *_common_size = *_mysize;
  }
  else { //distribute remainder n among first n processes
    *_mysize = _global_size / size;
    *_common_size = *_mysize + 1;
    if (rank < mod){
      (*_mysize)++;
      *_mystart = *_mysize * rank;
    }
    else {
      //no change on _mysize.
      *_mystart = (mod * (*_mysize + 1)) +
                  ((rank - mod) * (*_mysize));
    }
  }
  return;
}


static uint64_t 
getMyOffset (uint64_t _common_size, 
             int _numargs, 
             int _mod)
{
  int offset;
  if (_mod == 0){
    offset = _common_size * rank * _numargs;
  }
  else {
    if (rank < _mod){ //lower ranks have 1 more data point than upper
      offset = _common_size * rank * _numargs;
    }
    else { //rank >= mod (upper ranks have a lower number of elements)
      offset = (_common_size * _numargs * _mod) +
        ((rank - _mod) * (_common_size - 1) * _numargs);
    }
  }
  return offset;
}
