/* histogram.cpp
 *
 * Histogram Component
 *
 * Algorithm between lines 144-179 is by:
 *    Venkatram Vishwanath on 12/17/14.
 * Rest by : Alexis Champsaur, Sandia Labs.
 * Algorithm block Copyright (c) 2016, The Regents of the University of
 * California, through Lawrence Berkeley National Laboratory, UChicago Argonne
 * LLC, as Operator of Argonne National Laboratory, UT-Battelle LLC, through
 * the Oak Ridge National Laboratory,  Georgia Institute of Technology,
 * Intelligent Light,  Kitware Inc.  (subject to receipt of any required
 * approvals from the U.S. Dept. of Energy).  All rights reserved.
 *
 * License: https://gitlab.kitware.com/sensei/sensei/blob/master/LICENSE
 *
 * The rest of the code:
 *
 * Copyright 2017  National Technology and Engineering Solutions of Sandia,
 * LLC. Under the terms of Contract DE-NA0003525, there is a non-exclusive
 * license for use of this work by or on behalf of the U.S.  Government. Export
 * of this program may require a license from the United States Government.
 *
 * This program is designed to be an enpoint component in 
 * a scientific workflow. It reads a 1-D array from a data  
 * stream using ADIOS-FLEXPATH, and computes and writes to a disk file
 * a histogram of the values contained in the array for each timestep 
 * until the end of the stream is reached.
 *
 * Usage:
 *  <exec> input-stream-name input-array-name num-bins
 *   (where num-bins is the number of bins of the final histograms)
 *    
 */


#define __STDC_FORMAT_MACROS
#include <vector>
#include <mpi.h>
#include <stdlib.h>
#include <inttypes.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include "adios.h"
#include "adios_read.h"

extern "C" {
  #include "lib_mem_monitor.h"
  #include <stdio.h>
}

int main(int argc, char ** argv)
{
    int rank, size, varid, numvars;
    int bins, step, mod;
    char *filename, *in_stream, *data_var_name;
    MPI_Comm comm = MPI_COMM_WORLD;
    enum ADIOS_READ_METHOD method = ADIOS_READ_METHOD_FLEXPATH;

    /* */
    ADIOS_SELECTION * global_range_select;

    double *data;
    uint64_t tstep, global_size, mysize, mystart, sz;

    MPI_Init (&argc, &argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    /* debug */ 
    if (rank == 0)
      printf("histogram: started\n");

    /* Command line parsing */
    if (rank == 0 && argc < 4) {
      fprintf(stderr, "\nHistogram usage: <exec> input-stream-name num-bins" 
         " arr1 [arr2] [arr3] [...]\n"
         "\t where arr1, arr2, arr3 ... are the names of the arrays to be analyzed.\n");
      MPI_Abort(comm, -1);
    }
    MPI_Barrier(comm); 
    in_stream = argv[1];

    /* Parse cmd line */
    bins = atoi(argv[2]);
    numvars = argc - 3;
    const char *vars[numvars];
    for (varid=0; varid < numvars; varid++)
    {
      vars[varid] = argv[varid + 3];
    }


    
    /* Adios open and init */
    adios_read_init_method (method, comm, "verbose=1");
    ADIOS_FILE * f = adios_read_open (in_stream, method, comm, ADIOS_LOCKMODE_ALL, -1);

    if (rank == 0) {
        printf("[HIST-%d] adios_read_open returned\n");
    }
    
    step = 0; //not used now
    while (adios_errno != err_end_of_stream) {
      //resource monitor

      /*loop over different arrays inside stream*/
      for (varid = 0; varid < numvars; varid++) {         
#if 0
#ifdef ENABLE_MONITOR
        //double t1 = wfgettimeofday();
        lib_mem_init();
        ind_timer_start(0, "whole timestep");
#endif
#endif
        //Init variables....
        global_size = 0; tstep = 0;
        mod = 0; mysize = 0; mystart = 0;
        adios_schedule_read (f, NULL, "ntimestep", 0, 1, &tstep);
        adios_perform_reads (f, 1);
        ADIOS_VARINFO * glob_info = adios_inq_var (f, vars[varid]);
        global_size = glob_info->dims[0];

        //Array slice computation
        mod = global_size % size;//size = MPI size
        if (mod == 0){
          mysize = global_size / size;
          mystart = mysize * rank;
        }
        else {
          mysize = global_size / (size);
          if (rank < mod){
            mysize++;
            mystart = mysize * rank;
          }
          else {
            mystart = (mod * (mysize + 1)) +
                      ((rank - mod) * mysize);
          }
        }

#ifdef ENABLE_MONITOR
        nohandler_mem(rank);
#endif

        uint64_t starts[] = {mystart};
        uint64_t counts[] = {mysize};
        global_range_select = adios_selection_boundingbox (1, starts, counts);
        
        //Allocate space for arrays
        uint64_t msize = ((uint64_t) sizeof(double) * mysize);
        data = new double[mysize];
        if (data == NULL){
          //printf("DEBUG: malloc returned NULL, size was %d\n", msize);
        }
        else {
          if (rank == 0)
            printf("[HIST0] DEBUG: malloc successful, size was %d\n", mysize);
        }

        //Read data
        adios_schedule_read (f,
                             global_range_select,
                             vars[varid],
                             0, 1, data);
        ind_timer_start(1, "perform_reads");
        adios_perform_reads (f, 1);
        ind_timer_end(1);
#ifdef ENABLE_MONITOR
        nohandler_mem(rank);
#endif

        // find max and min
        sz = 0;
        sz = mysize; 
        double min = data[0];
        double max = data[0];
        for (uint64_t i = 1; i < sz; ++i)
        {
            if (data[i] > max) max = data[i];
            if (data[i] < min) min = data[i];
        }//local max, min found.


        double g_min, g_max;

        // Find the global max/min
        MPI_Allreduce (&min, &g_min, 1, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce (&max, &g_max, 1, MPI_DOUBLE, MPI_MAX, comm);

        

        double width = (g_max - g_min)/bins;
        std::vector<uint64_t>   hist(bins);
        for (uint64_t i = 0; i < sz; ++i)//fill local bins
        {
            int idx = int((data[i] - g_min)/width);//discover index
            if (idx == bins)        // we hit the max
                --idx;
            ++hist[idx];
        }

        delete[] data;
        // Global reduce histograms
        std::vector<uint64_t> g_hist(bins);
        MPI_Reduce(&hist[0], &g_hist[0], bins, MPI_UINT64_T, MPI_SUM, 0, comm);
        

        if (rank == 0) //print histogram to file
        {
          FILE *fp;
          const char *log = "histograms.log";
          fp = fopen(log, "a");
          fprintf(fp, "Histogram for %s, timestep %" PRIu64"\n", vars[varid], tstep);
          for (int i = 0; i < bins; ++i)
            fprintf(fp, "  %f-%f: %" PRIu64 "\n", g_min + i*width, g_min + (i+1)*width, g_hist[i]);
          fclose (fp);
        }
#ifdef ENABLE_MONITOR
        nohandler_mem(rank);
#endif
        

        if (rank == 0) //print histogram to terminal
        {
          printf("Histogram for %s, timestep %" PRIu64"\n", vars[varid], tstep);
          for (int i = 0; i < bins; ++i)
            printf("  %f-%f: %" PRIu64 "\n", 
              g_min + i*width, g_min + (i+1)*width, g_hist[i]);
        }
        
        //resource monitor
#ifdef ENABLE_MONITOR
        ind_timer_end(0);
        char monitor_title[40];
        sprintf(monitor_title, "histogram-%s", vars[varid]);
        monitor_out (rank, size, tstep, msize, comm, monitor_title);
#endif
      }
      //end of read + analysis for 3 variables


      
      adios_release_step(f);

      /* end-of-step timing */
      /*
      if (rank == 0) {
          double ts_end = wfgettimeofday();
          FILE *tsfp;
          tsfp = fopen("lammps-workflow-timestep-times.log", "a");
          fprintf(tsfp, "timestep %d AIO end time: %f\n", step, ts_end);
          fclose(tsfp);
      }
      */

      
      if (rank == 0) printf("[HIST%d] read and wrote data for timestep %" PRIu64 "\n", rank, tstep);

      step++;
      int adv_ret = adios_advance_step(f, 0, -1);
#ifdef ENABLE_MONITOR
      if (rank == 0) {
          printf("[HIST%d] adios_advance_step returned %d\n", rank, adv_ret);
      }
#endif
    } //end of adios stream while loop
    if (rank == 0) printf("[HIST%d] out of read loop\n", rank);
    adios_read_close(f);

    /* end-of-workflow timing */
#ifdef ENABLE_MONITOR
    if (rank == 0) {
        double t1 = wfgettimeofday();
        FILE *tfp;
        tfp = fopen("time.log", "a");
        fprintf(tfp, "histogram out of loop: %f\n", step, t1);
        fclose(tfp);
    }
#endif

    adios_read_finalize_method(method);
    MPI_Finalize();
    return 0;
}
