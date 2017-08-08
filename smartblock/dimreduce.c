/* 
 * dimreduce.c
 *
 * Copyright 2017  National Technology and Engineering Solutions of Sandia, 
 * LLC. Under the terms of Contract DE-NA0003525, there is a non-exclusive
 * license for use of this work by or on behalf of the U.S.  Government. Export
 * of this program may require a license from the United States Government.
 * 
 * This program is designed to be an intermediary component
 * in a scientific workflow. 
 *
 * Usage:
 *  <exec> input-arr-name input-stream-name index-of-select-dimension \
 *         ouput-arr-name output-stream-name \
 *         arg1 [arg2] [arg3] ...
 * 
 * More detailed requirements...
 * Alexis Champsaur, Sandia Labs, June 2015
 */

#define __STDC_FORMAT_MACROS
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>
#include "mpi.h"
#include "adios.h"
#include "adios_read.h"

//#define DEBUG 1 

static int rank, size; 
static MPI_Comm comm;

static void parse( int _argc, char **_argv, 
    char **_in_stream, char **_arr_name, int *_dim_reduce_index, 
    int *_dim_grow_index, char **_out_stream, char **_out_arr_name);
static void get_check_global_range( uint64_t _global_size, 
    uint64_t *_mysize, uint64_t *_mystart, 
    uint64_t *_common_size, int *_mod);

static uint64_t getMasterIndex( int _arrdims, uint64_t *_oldDims,
    uint64_t *_newDims, int _dim_reduce_index, int _dim_grow_index,
    uint64_t _l );

int main (int argc, char **argv)
{
  int step, mod, arrdims, i, maxdim, split_index, num_dim_vars,
      dim_reduce_index, dim_grow_index; 
  uint64_t global_size, tstep, mysize, mystart, common_size,
           mydoublesin, sel_size, l, ints, ulongs, doubles,
           total_size;
  int64_t g_handle, out_handle;
  double *data_in, *valid_data;

  comm = MPI_COMM_WORLD;

  enum ADIOS_READ_METHOD method = ADIOS_READ_METHOD_FLEXPATH;
  
  MPI_Init (&argc, &argv);
  MPI_Comm_rank (comm, &rank);
  MPI_Comm_size (comm, &size);

  //parsing command line args
  char *in_stream, *arr_name, *out_stream, *out_arr_name; 
  parse (argc, argv, &in_stream, &arr_name, &dim_reduce_index,
        &dim_grow_index, &out_stream, &out_arr_name);//fill in the above vars
  if (rank == 0) {
      
      printf ("[DIMRED%d] parse: instr=%s, arr_name=%s, out_strm=%s, out_arr=%s"
              " dim_reduce_index=%d dim_grow_index=%d \n",
              rank, in_stream, arr_name, out_stream, out_arr_name, dim_reduce_index, dim_grow_index);
  }
      
  adios_read_init_method (method, comm, "");
  ADIOS_FILE* f = adios_read_open (in_stream,
      method,
      comm,
      ADIOS_LOCKMODE_CURRENT, 0.0);

#ifdef DEBUG
  printf ("[DIMRED%d] adios_read_open returned\n", rank);
#endif

  adios_init_noxml (comm); //output
  adios_set_max_buffer_size (100);
  //adios_allocate_buffer (ADIOS_BUFFER_ALLOC_NOW, 100);
  adios_declare_group (&g_handle, "dimreduce_group", "", adios_flag_yes);
  adios_select_method (g_handle, "FLEXPATH", "", "");

  step = 0;//this component's understanding of steps
  while(adios_errno != err_end_of_stream){ 
#ifdef ENABLE_MONITOR
    lib_mem_init();
    ind_timer_start(0, "whole tstep");
#endif
    ADIOS_VARINFO * arr_info = adios_inq_var(f, arr_name);//total (all procs) this step
    arrdims = arr_info->ndim;
#ifdef DEBUG
    printf("[DIMRED%d] ndim: %d dims: %"PRIu64" %"PRIu64" %" PRIu64"\n",
       rank, arrdims, arr_info->dims[0], arr_info->dims[1], arr_info->dims[2]);
#endif
    /* Check if the dim indices provided 
     * by user is valid considering the input array
     */
    if 
    (dim_reduce_index > (arr_info->ndim - 1) || dim_grow_index > (arr_info->ndim - 1))
    {
      fprintf(stderr, "ERROR: dimension index provided is larger than input array's\n"
          "number of dimensions.\n");
      MPI_Abort(comm, -1);
    }
    
    global_size = 0;
    adios_schedule_read (f, NULL, "ntimestep", 0, 1, &tstep);
    adios_perform_reads(f, 1);

#ifdef ENABLE_MONITOR
    nohandler_mem();
#endif
    
    /* Figure out the split dimension: the dimension that will be used to split the
     * data among the processes (for particle simulation data, the dimension that spans
     * over the number of particles). This is determined by using the dimension of
     * maximum size. */

    i=0; maxdim=0;
    split_index = 0;
    for ( ; i<arrdims; i++){
      if (maxdim < arr_info->dims[i]){
        maxdim = arr_info->dims[i];
        split_index = i;
      }
    }

#ifdef DEBUG
    printf("[DIMRED%d] split index: %d\n", rank, split_index);
#endif

    // split_index cannot be dim_reduce_index
    if (split_index == dim_reduce_index){
      fprintf(stderr, "ERROR: attempting to reduce the wrong dimension.\n");
      MPI_Abort(comm, -1);
    }
    
    //get size of array in the splitting dimension
    global_size = (uint64_t)(arr_info->dims[split_index]);
    
    /* Print variables received */
    /*
    char info[40];
    sprintf(info, "[DIMRED%d-%d]: global_size=%" PRIu64 " dim_grow_index: %d"
        " split_index: %d\n", 
        rank, step, global_size, dim_grow_index, split_index);
    printf("%s", info);
    */

    /* Determine what portion of the split dimension I am responsible 
     * for during this timestep */
    get_check_global_range ( global_size, &mysize, 
        &mystart, &common_size, &mod );
    
#ifdef DEBUG
    printf("[DIMRED%d] mysize=%" PRIu64 ", mystart=%" PRIu64 "\tcommonsize=%" PRIu64 
        "\tmod=%d\n"
        "dimreducesize: %"PRIu64" dimgrowsize: %" PRIu64"\n", 
        rank, mysize, mystart, common_size, mod,
        arr_info->dims[dim_reduce_index], arr_info->dims[dim_grow_index]);
#endif

    /* Allocate space for the incoming data */
    mydoublesin = 1;
    for (i=0; i<arrdims; i++){
      if (i == split_index)
        mydoublesin *= mysize;
      else
        mydoublesin *= arr_info->dims[i];
    }
    uint64_t bytes_in = sizeof(double) * mydoublesin;
    data_in = (double *)malloc(bytes_in);
    if (data_in == NULL)
    {
      fprintf(stderr, "Malloc failed in dimreduce.\n");
      MPI_Abort(comm, -1);
    }
    memset(data_in, 0, sizeof(double) * mydoublesin);
    
#ifdef ENABLE_MONITOR
    nohandler_mem(rank);
#endif

    // data check
    // printf("[DIMRED%d] mydoublesin: %" PRIu64 "\n", rank, mydoublesin);

    /* Slice up the global array and fetch my portion
     */
    uint64_t starts[arrdims];
    uint64_t counts[arrdims];
    
    
    for (i=0; i<arrdims; i++){
      if (i == split_index){
        starts[i] = mystart;
        counts[i] = mysize;
      }
      else {
        starts[i] = 0;
        counts[i] = arr_info->dims[i];
      }
    }
    
    ADIOS_SELECTION *global_select = 
      adios_selection_boundingbox(arrdims, starts, counts);
    adios_schedule_read (f,
                         global_select,
                         arr_name,
                         0, 1, data_in);
#ifdef ENABLE_MONITOR
    ind_timer_start(1, "perform_reads");
#endif
    adios_perform_reads (f, 1);
#ifdef ENABLE_MONITOR
    ind_timer_end(1);
#endif

#ifdef ENABLE_MONITOR
    nohandler_mem(rank);
#endif
    /* Data Check */
    /*
    if (1) {
      FILE *fp;
      char *log;
      asprintf(&log, "dimreduce-run3-input%d-%d.log", step, rank);
      fp = fopen(log, "w");
      fprintf(fp, "timestep: %" PRIu64 " mysize: %"PRIu64 "\n", 
          tstep, mysize);
      for (i=0; i<(int)mysize; i++){
        fprintf(fp, "%lf\n", data_in[i]);
      }
      fclose(fp);
    }
    sleep(800);
    this is OK
    */

    /* Release */
    adios_release_step(f);
    //printf("[SEL%d]: released step %d\n", rank, step);



    
    /* Create the array of **indices** of fetched data
     * array that will actually be used in the output.
     */
    /* First create dimensions array */
    uint64_t newDims[arrdims - 1];
    uint64_t oldDims[arrdims];
    //calculate size and dimensions of new array
    for (i=0; i<arrdims; i++){
      if (i == split_index){
        oldDims[i] = mysize;
      }
      else 
        oldDims[i] = arr_info->dims[i];
    }

    for (i=0; i<(arrdims); i++){//i : how index of oldDims is handled here, in newDims
      uint64_t tmpdim = 0;
      if (i == dim_grow_index){
        tmpdim = (oldDims[dim_grow_index]) * (oldDims[dim_reduce_index]);
      }
      else 
        tmpdim = oldDims[i];
      //tmpdim holds good value
      if (i == dim_reduce_index)
        continue;
      else if (i < dim_reduce_index){
        newDims[i] = tmpdim;
      }
      else {
        newDims[i-1] = tmpdim;
      }
    }

#ifdef ENABLE_MONITOR
    nohandler_mem(rank);
#endif
    
#ifdef DEBUG
    printf("[DIMRED-%d] olddims: %"PRIu64" %" PRIu64" %" PRIu64" \n",
      rank, oldDims[0], oldDims[1], oldDims[2]);
    printf("[DIMRED-%d] newDims: %"PRIu64" %" PRIu64" \n",
      rank, newDims[0], newDims[1]);
#endif

      
    /* Construct new array */
    sel_size = 1;
    for (i=0; i<arrdims; i++){
      sel_size *= oldDims[i];
    }
    //no change in total size
    uint64_t *selection_inds = (uint64_t *)malloc(sizeof(uint64_t) * sel_size);
    if (selection_inds == NULL){
      fprintf(stderr, "malloc failed in dimreduce selection_inds\n");
      MPI_Abort(comm, -1);
    }


    /* For each index in the old array, calculate the index in the new array */
    for (l=0; l<sel_size; l++){
      uint64_t master_index;
      master_index = getMasterIndex (arrdims, oldDims, newDims, dim_reduce_index, 
          dim_grow_index, l);
      selection_inds[l] = master_index;
    }
    /* Now the values of selection_inds are the 1D indices of the elements in 
     * the output array where the actual index of selection_inds is the 
     * index of the element in the input array.
     */


#ifdef ENABLE_MONITOR
    nohandler_mem(rank);
#endif
    valid_data = (double *)malloc(sizeof(double) * sel_size);
    if (selection_inds == NULL){
      fprintf(stderr, "malloc failed in dimreduce selection_inds\n");
      MPI_Abort(comm, -1);
    }
    
    /* Fill the output data array, using the array of selection indices just created */
    for (l=0; l<sel_size; l++){
      valid_data[selection_inds[l]] = data_in[l];
    }
#ifdef ENABLE_MONITOR
    nohandler_mem(rank);
#endif

    /* Output the data */
    /* Create the strings that will describe to ADIOS the dimensions
     * of the global array and define the variables to ADIOS*/
    if (step == 0) num_dim_vars = 0;
    //the number of "dimension variables" created in the loop below
    char tmpstr[50]; //needed later
    if (step == 0){
      char out_local_dims_str[200];
      char out_global_dims_str[200];
      char out_offsets_str[200];
      memset (out_local_dims_str, 0, 200);
      memset (out_global_dims_str, 0, 200);
      memset (out_offsets_str, 0, 200);
      memset (tmpstr, 0, 50);
      char *comma;
      //TODO: add error checking below
      for (i=0; i<(arrdims-1); i++){
        if (i == 0) comma = "";
        else comma = ",";
        //local dims
        sprintf(tmpstr, "%sldim%d", comma, i);
        if (i == 0) sprintf(out_local_dims_str, "%s", tmpstr);
        else strcat(out_local_dims_str, tmpstr);
        sprintf(tmpstr, "ldim%d", i);
        adios_define_var (g_handle, tmpstr, "", adios_unsigned_long, "", "", "");
        num_dim_vars++;
        memset (tmpstr, 0, 50);
        //global dims
        sprintf(tmpstr, "%sgdim%d", comma, i);
        if (i == 0) sprintf(out_global_dims_str, "%s", tmpstr);
        else strcat(out_global_dims_str, tmpstr);
        sprintf(tmpstr, "gdim%d", i);
        adios_define_var (g_handle, tmpstr, "", adios_unsigned_long, "", "", "");
        num_dim_vars++;
        memset (tmpstr, 0, 50);
        //offsets
        sprintf(tmpstr, "%soff%d", comma, i);
        if (i == 0) sprintf(out_offsets_str, "%s", tmpstr);
        else strcat(out_offsets_str, tmpstr);
        sprintf(tmpstr, "off%d", i);
        adios_define_var (g_handle, tmpstr, "", adios_unsigned_long, "", "", "");
        num_dim_vars++;
        memset (tmpstr, 0, 50);
      }

      adios_define_var (g_handle, "me", "", adios_integer, "", "", "");
      adios_define_var (g_handle, "mpi_size", "", adios_integer, "", "", "");
      adios_define_var (g_handle, "ntimestep", "", adios_unsigned_long, "", "", "");
      adios_define_var (g_handle, out_arr_name, "", adios_double, 
          out_local_dims_str, out_global_dims_str, out_offsets_str);
    }

    /* Check the dimension strings */
    ints = ulongs = doubles = 0;
    ints = (uint64_t) (sizeof(int) * 2);
    ulongs = (uint64_t) (sizeof(uint64_t) * (1 + num_dim_vars));
    doubles = (uint64_t) (sizeof(double) * sel_size);
    adios_open (&out_handle, "dimreduce_group", out_stream, "a", comm);
    adios_group_size (out_handle, ints + ulongs + doubles, &total_size);
    adios_write (out_handle, "me", &rank);
    adios_write (out_handle, "mpi_size", &size);
    adios_write (out_handle, "ntimestep", &tstep);//original tstep num from simluation
    int out_split_index;
    if (split_index < dim_reduce_index)
      out_split_index = split_index;
    else
      out_split_index = split_index - 1;//because dim_reduce_index disappears
    uint64_t global_out_split, offset_split;
    global_out_split = offset_split = 0;
    MPI_Allreduce(&newDims[out_split_index], &global_out_split, 1,
        MPI_UINT64_T, MPI_SUM, comm);
    MPI_Scan(&newDims[out_split_index], &offset_split, 1, 
        MPI_UINT64_T, MPI_SUM, comm);
    offset_split -= newDims[out_split_index];
#ifdef ENABLE_MONITOR
    nohandler_mem(rank);
#endif
    uint64_t zero = 0;
    for (i=0; i<(arrdims-1); i++){
      memset (tmpstr, 0, 50);
      sprintf(tmpstr, "ldim%d", i);
      adios_write (out_handle, tmpstr, &(newDims[i]));
      memset (tmpstr, 0, 50);
      sprintf(tmpstr, "gdim%d", i);
      if (i == out_split_index)
        adios_write (out_handle, tmpstr, &global_out_split);
      else 
        adios_write (out_handle, tmpstr, &newDims[i]);
      memset (tmpstr, 0, 50);
      sprintf(tmpstr, "off%d", i);
      if (i == out_split_index)
        adios_write (out_handle, tmpstr, &offset_split);
      else 
        adios_write (out_handle, tmpstr, &zero);
    }


    /*** WRITE DATA ARRAY ***/      
    adios_write (out_handle, out_arr_name, valid_data);
    adios_close (out_handle);
    if (rank ==0) printf("[DIMRED%d] wrote data for tstep %" PRIu64 "\n", rank, tstep);
    
    free(data_in);
    free(selection_inds);
    free(valid_data);
    step++; 
#ifdef ENABLE_MONITOR
    nohandler_mem(rank);
    ind_timer_end(0);
    char monitor_title[40];
    sprintf(monitor_title, "dimreduce");
    monitor_out (rank, size, tstep, bytes_in, comm, monitor_title);
#endif
    adios_advance_step(f, 0, -1);
  }//end of timestep read
  printf("[DIMR%d] out of read loop\n", rank);
  adios_read_close(f);
  adios_read_finalize_method(method);
  adios_finalize(rank);
  MPI_Finalize ();
  return 0;
}



static void 
parse( int _argc,                   //in
       char **_argv,                //in
       char **_in_stream,           //out
       char **_arr_name,            //out
       int *_dim_reduce_index,      //out
       int *_dim_grow_index,        //out
       char **_out_stream,          //out
       char **_out_arr_name )       //out
{
  if (rank == 0 && _argc != 7) {
    fprintf(stderr, "\nUsage: <exec> input-stream input-array-name"
        "dim-index-to-remove dim-index-to-grow\\\n"
        "output-stream output-array-name\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  *_in_stream = _argv[1];
  *_arr_name = _argv[2];
  *_dim_reduce_index = atoi(_argv[3]);
  *_dim_grow_index = atoi(_argv[4]);
  *_out_stream = _argv[5];
  *_out_arr_name = _argv[6];
  return;
}

static void
get_check_global_range( uint64_t _global_size, 
                        uint64_t *_mysize,
                        uint64_t *_mystart,
                        uint64_t *_common_size,
                        int *_mod )
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



/* getMasterIndex
 *
 * Given the 1D index of an element in the input array we are filling,
 * obtains the 1D index of that element in the output array.
 * l: the 1D index of the element in the input array.
 */
static uint64_t
getMasterIndex( int _arrdims,
                uint64_t *_oldDims,
                uint64_t *_newDims,
                int _dim_reduce_index,
                int _dim_grow_index,
                uint64_t _l )
{
  uint64_t temp_inds_old[_arrdims];
  uint64_t temp_inds_new[_arrdims-1];
  memset(temp_inds_old, 0, sizeof(uint64_t) * _arrdims);
  memset(temp_inds_new, 0, sizeof(uint64_t) * (_arrdims-1));
  uint64_t div, thisMod;
  uint64_t numLeft = _l;
  int ii, ij;
  /* Find multi-dimensional index of the element in the input
   * array
   */
  for (ii=0; ii<(_arrdims); ii++){
    uint64_t sizeofblock = 1;
    for (ij=(ii+1); ij<(_arrdims); ij++){
      sizeofblock *= _oldDims[ij];
    }
    div = numLeft / sizeofblock;
    //printf("div = %d\n", (int) div);
    temp_inds_old[ii] = div;
    thisMod = numLeft % sizeofblock;
    //printf("thisMod = %d\n", (int) thisMod);
    if (thisMod == 0) break;
    numLeft = thisMod;
  }
  /* we now have the multi-D indices of the element in question
   * in the input array. Now, obtain the multi-D indices of the element
   * in the output array.
   */
  for (ii=0; ii<(_arrdims); ii++){
    uint64_t newind;
    if (ii == _dim_grow_index){
      newind = temp_inds_old[_dim_grow_index] + 
        (temp_inds_old[_dim_reduce_index] * _oldDims[ii]);
    }
    else
      newind = temp_inds_old[ii];
    //now fill temp_inds_new
    if (ii == _dim_reduce_index)
      continue;
    else if (ii < _dim_reduce_index){
      temp_inds_new[ii] = newind;
    }
    else {
      temp_inds_new[ii-1] = newind;
    }
  }
  
  /* Now calculate 1D index of the element in the output array */
  uint64_t out = 0;
  for (ii=0; ii<(_arrdims-1); ii++){
    uint64_t prod = 1;
    for (ij = ii+1; ij < (_arrdims-1); ij++){
      prod *= _newDims[ij];
    } 
    out += prod * temp_inds_new[ii];
  }
  return out;
}

