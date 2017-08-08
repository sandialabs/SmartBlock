/* 
 * lammps_analysis.c
 *
 * Copyright 2017  National Technology and Engineering Solutions of Sandia,
 * LLC. Under the terms of Contract DE-NA0003525, there is a non-exclusive
 * license for use of this work by or on behalf of the U.S.  Government. Export
 * of this program may require a license from the United States Government.
 *
 * 
 * This program is a combined analytical
 * component for LAMMPS, representing
 * from-scratch in situ components
 * commonly used in analytical components.
 * Alexis Champsaur, Sandia Labs, June 2015
 */

//#define _GNU_SOURCE
#define __STDC_FORMAT_MACROS
#include <vector>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "mpi.h"
#include "adios.h"
#include "adios_read.h"
#include <math.h>

extern "C" {
  #include <stdio.h>
  #include "lib_mem_monitor.h"
}

static int rank, size; 
static MPI_Comm comm;

/* debug */
static double area;
static struct sigaction act;
static void sighandler(int signum, siginfo_t *siginfo, void *ptr) {
  printf("[SEL%d] received SEGFAULT in area %f\n", rank, area);
  MPI_Abort(comm, -1);
}

static struct rusage ruse;

static void parse( int _argc, char **_argv, 
    char **_in_stream, char **_arr_name, int *_dim_index, 
    char **_out_stream, char **_out_arr_name,
    int *_numargs, char ***_args);
static void get_check_global_range( uint64_t _global_size, 
    uint64_t *_mysize, uint64_t *_mystart, 
    uint64_t *_common_size, int *_mod);
static void fetch_header( ADIOS_FILE *_f, char **_hdr_str, int _hdr_len);
static void parse_header( int **_positions, char **_args,
    char *_hdr_str, int _hdr_len, int _numargs);
static int string_is_member( char *str1, char **str_arr, int arr_len);
static uint64_t getMasterIndex( int _arrdims, uint64_t *_oldDims,
    int _dim_index, int *_positions, uint64_t *_newDims, uint64_t _l );
static void build_output_string( char *_output, int _numargs, char **_args );

static double
wfgettimeofday( void )
{
    double timestamp;
    struct timeval now;
    gettimeofday(&now, NULL);
    timestamp = now.tv_sec + now.tv_usec* 1.0e-6 ;
    return timestamp;
}


int main (int argc, char **argv)
{
  int64_t g_handle, out_handle;
  int header_len, step, arrdims, output_header_len, mod, 
      dim_index, split_index, i, num_dim_vars;
  uint64_t global_size, mysize, mystart, common_size, sel_size, 
      l, total_size, tstep, maxdim, ints, ulongs, doubles, str_size;
  uint64_t sz;
  int *positions;
  double *data_in, *valid_data;
  double *magnitudes;
  double t1;
  double t2;
  char *header_string;
  int bins;
  comm = MPI_COMM_WORLD;

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (comm, &rank);
  MPI_Comm_size (comm, &size);
  /* parsing command line args 
     so far, only 1: number of bins
  */
  if (argc != 2 && rank == 0) {
      fprintf(stderr, "lammps_analysis usage: lammps_analysis [num-bins]\n");
      MPI_Abort(comm, -1);
  }
  bins = atoi(argv[1]);
  if (bins < 2 || bins > 512) {
      fprintf(stderr, "lammps_analysis: bin number out of range\n");
      MPI_Abort(comm, -1);
  }
  
  const char *in_stream = "dump.custom.fp";
  const char *arr_name = "atoms";

  /* ADIOS Input Setup
     Open the input stream for reading */
  adios_read_init_method(ADIOS_READ_METHOD_FLEXPATH, comm, "");
  ADIOS_FILE* f = adios_read_open(in_stream,
      ADIOS_READ_METHOD_FLEXPATH,
      comm,
      ADIOS_LOCKMODE_CURRENT, 0.0);

  /* Sanity check */
  if (rank == 0) printf("[LMPA%d] adios_read_open returned\n", rank);
  step = 0;
  /* This is the main loop used by ADIOS to read a steam until it ends. */
  while(adios_errno != err_end_of_stream) { 
      /* Monitoring library initialization. */
#ifdef ENABLE_MONITOR
    lib_mem_init();
    ind_timer_start(0, "whole tstep");
#endif
    ADIOS_VARINFO * arr_info = adios_inq_var(f, arr_name);//total (all procs) this step
    //ADIOS_VARINFO * tstep_info = adios_inq_var(f, "ntimestep");//total (all procs) this step
    arrdims = arr_info->ndim;
    if (arrdims != 2 && rank == 0) {
        printf("[LMPA] array dimensions wrong, exiting\n");
        MPI_Abort(comm, -1);
    }
    /* Check if the index of the dimension provided 
     * by user is valid considering the input array
     */
    tstep = 0;
    global_size = 0;
    adios_schedule_read (f, NULL, "ntimestep", 0, 1, &tstep);
    adios_perform_reads(f, 1);
    
    nohandler_mem(rank);

    /* SPLIT INDEX: 0 */
    global_size = (uint64_t)(arr_info->dims[0]);
    //nohandler_mem(rank);
    
    /* Print variables received */
    /* Determine what portion of the split dimension I am responsible 
     * for during this timestep */
    get_check_global_range ( global_size, &mysize, 
        &mystart, &common_size, &mod );
    
   //debug
   //parse_header (&positions, args, header_string, 
   //     header_len, numargs); //positions are written to array "positions"
    positions = (int *)malloc(sizeof(int) * 3);
    positions[0] = 2;
    positions[1] = 3;
    positions[2] = 4;
                 
    /* Allocate space for the incoming data */
    uint64_t mydoublesin = 5 * mysize;
    
    uint64_t my_bytes = sizeof(double) * mydoublesin;
    data_in = (double *)malloc(my_bytes);
    memset(data_in, 0, my_bytes);

    uint64_t starts[arrdims];
    uint64_t counts[arrdims];
    
    starts[0] = mystart;
    counts[0] = mysize;
    starts[1] = 0;
    counts[1] = 5;

    //nohandler_mem(rank);

    ADIOS_SELECTION *global_select = 
      adios_selection_boundingbox(arrdims, starts, counts);
    adios_schedule_read (f,
                         global_select,
                         arr_name,
                         0, 1, data_in);
    ind_timer_start(1, "perform_reads");
    adios_perform_reads (f, 1);
    ind_timer_end(1);


    /* Data received : release step */
    adios_release_step(f);

    /* Start magnitudes */
    magnitudes = (double *)malloc(sizeof(double) * mysize);
    double these_squares;
    double current_sum, current_doub;

    
    for (l = 0; l < mysize; l++) {
        current_sum = 0;
        int y,z;
        for (z = 0; z < 3; z++) {
            y = l * 5 + z + 2;
            current_doub = data_in[y];
            current_sum += current_doub * current_doub;
        }
        magnitudes[l] = sqrt(current_sum);
    }

    /* end magnitude */

    /* begin histogram */
    double min = magnitudes[0];
    double max = magnitudes[0];
    for (l = 1; l < mysize; l++) {
        if (magnitudes[l] > max) max = magnitudes[l];
        if (magnitudes[l] < min) min = magnitudes[l];
    }
    double g_min, g_max;
    // Find the global max/min
    MPI_Allreduce (&min, &g_min, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce (&max, &g_max, 1, MPI_DOUBLE, MPI_MAX, comm);
    double width = (g_max - g_min)/bins;
    std::vector<uint64_t> hist(bins);
    for (uint64_t i = 0; i < mysize; i++)
    {
        int idx = int((magnitudes[i] - g_min)/width);
        if (idx == bins)
            --idx;
        ++hist[idx];
    }
    delete[] magnitudes;
    //Global reduce histograms
    std::vector<uint64_t> g_hist(bins);
    MPI_Reduce(&hist[0], &g_hist[0], bins, MPI_UINT64_T, MPI_SUM, 0, comm);

    if (rank == 0)
    {
        FILE *fp;
        const char *log = "histograms.mono_analysis.log";
        fp = fopen(log, "a");
        fprintf(fp, "Histogram for velocities, timestep %" PRIu64"\n", tstep);
        for (int i = 0; i < bins; ++i)
        {
            fprintf(fp, "  %f-%f: %" PRIu64 "\n", g_min + i*width, g_min + (i+1)*width, g_hist[i]);
        }
        fclose(fp);
    }

    if (rank == 0) //print histogram to terminal
    {
        printf("Histogram for velocities, timestep %" PRIu64"\n", tstep);
        for (int i = 0; i < bins; ++i)
            printf("  %f-%f: %" PRIu64 "\n", 
                   g_min + i*width, g_min + (i+1)*width, g_hist[i]);
    }
#ifdef ENABLE_MONITOR
    ind_timer_end(0);
    char monitor_title[40];
    sprintf(monitor_title, "lammps_combined_analysis");
    monitor_out (rank, size, tstep, my_bytes, comm, monitor_title);
#endif
    adios_release_step(f);
    //delete[] data;
    if (rank == 0){ 
        printf("rank 0 in if statement for time.log printing, histogram...\n");
        double t3 = wfgettimeofday();
        FILE *tfp;
        tfp = fopen("time.log", "a");
        fprintf(tfp, "master histogram step %" PRIu64 "end time: %f\n", t3, tstep);
        fclose(tfp);
    }
    
    if (rank == 0) printf("[HIST%d] read and wrote data for timestep %" PRIu64 "\n", rank, tstep);
/* performance measurements */
    if (rank == 0) {
        double ts_end = wfgettimeofday();
        FILE *tsfp;
        tsfp = fopen("lammps-workflow-timestep-times.log", "a");
        fprintf(tsfp, "timestep %d AIO end time: %f\n", step, ts_end);
        fclose(tsfp);
    }
    
    
    step++;
    adios_advance_step(f, 0, -1);
  }//end of timestep read
  if (rank == 0) printf("[SEL%d] out of read loop\n", rank); 
  adios_read_close(f);
  /* performance measurements */
  if (rank == 0){
    double t1 = wfgettimeofday();
    FILE *tfp;
    tfp = fopen("lammps-workflow-start-to-end-time.log", "a");
    fprintf(tfp, "AIO end time: %f\n", t1);
    fclose(tfp);
  }

  adios_read_finalize_method(ADIOS_READ_METHOD_FLEXPATH);
  MPI_Finalize ();
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
