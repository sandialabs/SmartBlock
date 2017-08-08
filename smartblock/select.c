/* 
 * select.c
 *
 * Copyright 2017  National Technology and Engineering Solutions of Sandia,
 * LLC. Under the terms of Contract DE-NA0003525, there is a non-exclusive
 * license for use of this work by or on behalf of the U.S.  Government. Export
 * of this program may require a license from the United States Government.
 *
 * 
 * This program is designed to be an intermediary component
 * in a scientific workflow. It selects certain indices from one 
 * of the dimensions of a multi-dimensional array read as input from
 * a Flexpath stream. The indices to be selected are specified by their
 * names, which are recognized using a header string that is part  of 
 * the input.
 * The output is another multi-dimensional array consisting of the 
 * selected quantities.
 *
 * The names of the input and output streams and arrays, as well as the 
 * quantities to be extracted are passed as command-line arguments.
 * 
 * The input stream must contain a header called "header_string" to 
 * indicate what quantities are contained in the input array, and how they
 * are arranged. The format of the header string is "var1:var2:var3..."
 * where var1, var2, var3... are the names of the quantities in the input
 * array, in the order in which they appear there.
 *
 * Usage:
 *  <exec> input-arr-name input-stream-name index-of-select-dimension \
 *         ouput-arr-name output-stream-name \
 *         arg1 [arg2] [arg3] ...
 * 
 * Alexis Champsaur, Sandia Labs, June 2015
 */

#define _GNU_SOURCE
#define __STDC_FORMAT_MACROS
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "lib_mem_monitor.h"
#include "mpi.h"
#include "adios.h"
#include "adios_read.h"


static int rank, size; 
static MPI_Comm comm;

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


int main (int argc, char **argv)
{
  int64_t g_handle, out_handle;
  int header_len, step, arrdims, output_header_len, mod, 
      dim_index, split_index, i, num_dim_vars;
  uint64_t global_size, mysize, mystart, common_size, sel_size, 
      l, total_size, tstep, maxdim, ints, ulongs, doubles, str_size;
  int *positions;
  double *data_in, *valid_data;
  double t1;
  double t2;
  char *header_string;
  comm = MPI_COMM_WORLD;

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (comm, &rank);
  MPI_Comm_size (comm, &size);
 
  /* parsing command line args */
  char *in_stream, *arr_name, *out_stream, *out_arr_name; 
  int numargs; //number of variables to be extracted from the array
  char **args; //names of the variables to be extracted from the array
  parse(argc, argv, &in_stream, &arr_name, &dim_index,
        &out_stream, &out_arr_name, &numargs, &args);
  if (rank == 0) printf("[SELECT%d] parse: arr_name=%s, out_arr=%s, numargs=%d, args[0]=%s"
         " dim_index=%d\n",
      rank, arr_name, out_arr_name, numargs, args[0], dim_index);

  /* ADIOS Input Setup
     Open the input stream for reading */
  adios_read_init_method(ADIOS_READ_METHOD_FLEXPATH, comm, "");
  ADIOS_FILE* f = adios_read_open(in_stream,
      ADIOS_READ_METHOD_FLEXPATH,
      comm,
      ADIOS_LOCKMODE_CURRENT, 0.0);

  /* Sanity check */
  if (rank == 0) printf("[SEl%d] adios_read_open returned\n", rank);

  /* ADIOS Output Setup 
     Declare the output group and method*/
  adios_init_noxml (comm);
  //adios_allocate_buffer (ADIOS_BUFFER_ALLOC_NOW, 100);
  adios_set_max_buffer_size (100);
  adios_declare_group (&g_handle, "selectiongroup", "", adios_flag_yes);
  adios_select_method (g_handle, "FLEXPATH", "", "");

  /* Read global information from stream 
     For this component, this includes a header with some info.
     Not all components have this.*/
  ADIOS_VARINFO * hdr_len_info = adios_inq_var(f, "header_len");//total (all procs) this step
  header_len = *((int *)(hdr_len_info->value));

  /* This component's internal understanding of steps */
  step = 0;

  /* This is the main loop used by ADIOS to read a steam until it ends. */
  while(adios_errno != err_end_of_stream) {
      /* Monitoring library initialization. */
#ifdef ENABLE_MONITOR
      lib_mem_init();
      ind_timer_start(0, "whole tstep");
#endif

    ADIOS_VARINFO * arr_info = adios_inq_var(f, arr_name);//total (all procs) this step
    arrdims = arr_info->ndim;
    
    /* Check if the index of the dimension provided 
     * by user is valid considering the input array
     */
    if (dim_index > (arr_info->ndim - 1)) {
      fprintf(stderr, "ERROR: dimension index provided is larger than input array's\n"
          "number of dimensions.\n");
      MPI_Abort(comm, -1);
    }
    
    tstep = 0;
    global_size = 0;
    maxdim = 0;
    adios_schedule_read (f, NULL, "ntimestep", 0, 1, &tstep);
    adios_perform_reads(f, 1);
    
    nohandler_mem(rank);

    /* Figure out the split dimension: the dimension that will be used to split the
     * data among the processes (for particle simulation data, the dimension that spans
     * over the number of particles) */
    i=0;
    for ( ; i<arrdims; i++){
      if (maxdim < arr_info->dims[i]){
        maxdim = arr_info->dims[i];
        split_index = i;
      }
    }

    // split_index cannot be dim_index
    if (split_index == dim_index){
      fprintf(stderr, "ERROR: attempting to select in the wrong dimension.\n");
      MPI_Abort(comm, -1);
    }
    
    
    global_size = (uint64_t)(arr_info->dims[split_index]);
#ifdef ENABLE_MONITOR
    nohandler_mem(rank);
#endif
    
    /* Determine what portion of the split dimension I am responsible 
     * for during this timestep */
    get_check_global_range ( global_size, &mysize, 
        &mystart, &common_size, &mod );
    
    /* Fetch the header, and get the local positions 
     * (in their data point group) of the quantities I am reponsible for
     * using the header. */
    fetch_header (f, &header_string, header_len); //written to header_string

    /* Get local positions of desired quantities from header */
    //debug
    parse_header (&positions, args, header_string, 
        header_len, numargs); //positions are written to array "positions"

    
    
    /* Allocate space for the incoming data */
    uint64_t mydoublesin = 1;
    for (i=0; i<arrdims; i++){
      if (i == split_index)
        mydoublesin *= mysize;
      else
        mydoublesin *= arr_info->dims[i];
    }

    int testdoublesin = 5 * mysize;
    
    uint64_t my_bytes = sizeof(double) * mydoublesin;
    data_in = (double *)malloc(my_bytes);
    memset(data_in, 0, my_bytes);
    //printf("[%d] allocated array for  %" PRIu64 " doubles\n", 
    //    rank, mydoublesin); 
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
    
#ifdef ENABLE_MONITOR
    nohandler_mem(rank);
#endif

    ADIOS_SELECTION *global_select = 
      adios_selection_boundingbox(arrdims, starts, counts);
    adios_schedule_read (f,
                         global_select,
                         arr_name,
                         0, 1, data_in);
    ind_timer_start(1, "perform_reads");
    adios_perform_reads (f, 1);
    ind_timer_end(1);



    /* Release */
    adios_release_step(f);

    /* Create the array of **indices** of fetched data
     * array that will actually be used in the output.
     */
    sel_size = 1;
    uint64_t newDims[arrdims];
    uint64_t oldDims[arrdims];
    //calculate size and dimensions of new array
    for (i=0; i<arrdims; i++){
      if (i ==  dim_index){
        oldDims[i] = arr_info->dims[i];
        newDims[i] = numargs;
        sel_size *= numargs;
      }
      else if (i == split_index){
        sel_size *= mysize;
        newDims[i] = mysize;
        oldDims[i] = mysize;
      }
      else{
        sel_size *= arr_info->dims[i];
        newDims[i] = arr_info->dims[i];
        oldDims[i] = arr_info->dims[i];
      }
    }

#ifdef ENABLE_MONITOR    
    nohandler_mem(rank);
#endif

    uint64_t *selection_inds = (uint64_t *)malloc(sizeof(uint64_t) * sel_size);
    memset(selection_inds, 0, sizeof(uint64_t) * sel_size);
    //fill in new dimensions array
    for (l=0; l<sel_size; l++){
      uint64_t master_index;
      master_index = getMasterIndex(arrdims, oldDims, dim_index, positions, newDims, l);
      selection_inds[l] = master_index;
    }

    /* Allocate memory for the new array that will be 
     * made up of the filtered data */
    valid_data = (double *)malloc(sizeof(double) * sel_size);
    memset(valid_data, 0, sizeof(double) * sel_size);
    
    /* Filter data: write selected data into output 1D array */
    for (l=0; l<sel_size; l++){
      uint64_t ix = selection_inds[l];
      valid_data[l] = data_in[ix];
    }

    /* Output the data */
    /* Create the strings that will describe to ADIOS the dimensions
     * of the global array and define the variables to ADIOS*/
    if (step == 0) num_dim_vars = 0; //the number of "dimension variables" created in the loop below
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
      for (i=0; i<arrdims; i++){
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
      adios_define_var (g_handle, "header_len", "", adios_integer, "", "", "");
      adios_define_var (g_handle, "header_string", "", 
          adios_byte, "header_len", "", "");
      adios_define_var (g_handle, out_arr_name, "", adios_double, 
          out_local_dims_str, out_global_dims_str, out_offsets_str);
    }


#ifdef ENABLE_MONITOR
    nohandler_mem(rank);
#endif
    
    /* Check the dimension strings */
    /* Finished dimension strings for ADIOS */ 
    char output_header_string[80];
    memset (output_header_string, 0, 80);
    build_output_string(output_header_string, numargs, args);
    output_header_len = strlen(output_header_string) + 1;
    ints = ulongs = doubles = str_size = 0;
    ints = (uint64_t) (sizeof(int) * 3);
    ulongs = (uint64_t) (sizeof(uint64_t) * (1 + num_dim_vars));
    doubles = (uint64_t) (sizeof(double) * sel_size);
    str_size = (uint64_t) (sizeof(char) * output_header_len);
    adios_open  (&out_handle, "selectiongroup", out_stream, "a", comm);
    adios_group_size (out_handle, ints + ulongs + doubles + str_size, &total_size);
    adios_write (out_handle, "me", &rank);
    adios_write (out_handle, "mpi_size", &size);
    adios_write (out_handle, "ntimestep", &tstep);//original tstep num from simluation
    adios_write (out_handle, "header_len", &output_header_len);
    adios_write (out_handle, "header_string", output_header_string);

    //write the dimension variables
    uint64_t zero = 0;
    for (i=0; i<arrdims; i++){
      memset (tmpstr, 0, 50);
      sprintf(tmpstr, "ldim%d", i);
      adios_write (out_handle, tmpstr, &(newDims[i]));
      memset (tmpstr, 0, 50);
      sprintf(tmpstr, "gdim%d", i);
      if (i == split_index)
        adios_write (out_handle, tmpstr, &global_size);
      else 
        adios_write (out_handle, tmpstr, &newDims[i]);
      memset (tmpstr, 0, 50);
      sprintf(tmpstr, "off%d", i);
      if (i == split_index)
        adios_write (out_handle, tmpstr, &mystart);
      else 
        adios_write (out_handle, tmpstr, &zero);
    }

#ifdef ENABLE_MONITOR    
    nohandler_mem(rank);
#endif

    /*** WRITE DATA ARRAY ***/      
    adios_write (out_handle, out_arr_name, valid_data);
    adios_close (out_handle);
    if (rank == 0) printf("[SELECT%d]: read and wrote data for timestep %" PRIu64 "\n", rank, tstep);
    free(data_in);
    free(selection_inds);
    free(valid_data);
    step++; 
    int ret;
    ind_timer_end(0);
    monitor_out (rank, size, tstep, my_bytes, comm, "select");
    ret = adios_advance_step(f, 0, 40);
  }//end of timestep read
  if (rank == 0) printf("[SEL%d] out of read loop\n", rank); 
  adios_read_close(f);
  adios_read_finalize_method(ADIOS_READ_METHOD_FLEXPATH);
  adios_finalize(rank);
  MPI_Finalize ();
  return 0;
}


static void 
parse( int _argc, 
       char **_argv, 
       char **_in_stream, 
       char **_arr_name, 
       int *_dim_index,
       char **_out_stream, 
       char **_out_arr_name,
       int *_numargs, 
       char ***_args )
{
  if (rank == 0 && _argc < 7) {
    fprintf(stderr, "\nUsage: <exec> input-stream input-array-name dim-index\\\n"
        "output-stream output-array-name arg1 [arg2] [arg3] ...\n"
        "where arg1, arg2, etc. are the names of the variables to be extracted\n"
        "from the global array \"intput-array-name\",\n"
        "and dim-index is the index of the dimension to operate on if the input\n"
        "array is multi-dimensional. If the array is 1D, this field is ignored.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  *_in_stream = _argv[1];
  *_arr_name = _argv[2];
  *_dim_index = atoi(_argv[3]);
  *_out_stream = _argv[4];
  *_out_arr_name = _argv[5];
  *_numargs = _argc - 6;
  int j;
  *_args = (char **) malloc(sizeof(char *) * (*_numargs));
  for (j=0; j<*_numargs; j++){
    (*_args)[j] = _argv[j+6];
  }
  return;
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

static void
 fetch_header( ADIOS_FILE *_f,
                char **_hdr_str,
                int _header_len )
{
  *_hdr_str = malloc(_header_len);
  ADIOS_SELECTION *_block_select;
  _block_select = adios_selection_writeblock(0);//select rank 0's header..
  adios_schedule_read (_f,
                       _block_select,
                       "header_string",
                       0, 1, *_hdr_str);
  adios_perform_reads (_f, 1);
  return;
}

/* TODO: Make sure the number of elements in the header is equal
 * to the size of the dimension to be selected
 */
static void
parse_header( int **_positions, //out
              char **_args,     //out
              char *_hdr_str,   //in
              int _hdr_len,     //in
              int _numargs )    //in
{
  //copy header string to not modify it
  char deststr[100];
  strcpy (deststr, _hdr_str);
  
  //allocate mem for position
  *_positions = malloc(sizeof(int) * _numargs);
  int num_fetched = 0;
  //char tempstr[30];
  //memset(tempstr, 0, 30);
  int position_in_header = 0;
  //int k,l;
  //l = 0;//current position in temparg
  //char tmp;//currently read char
  char *token;
  char delim[2] = ":";

  token = strtok(deststr, delim);
  
  while (token != NULL){
    if (string_is_member(token, _args, _numargs)){
      (*_positions)[num_fetched] = position_in_header;
      num_fetched++;
    }
    token = strtok(NULL, delim);
    position_in_header++;
  }

  if (num_fetched != _numargs){
    fprintf(stderr, "ERROR: header parsing.\n");
    MPI_Abort(comm, -1);
  }

  return;
}

static int
string_is_member (char *str1, char **str_arr, int arr_len)
{
  int z;
  for (z=0; z<arr_len; z++){
    if (strcmp(str1, str_arr[z]) == 0){
      return 1;
    }
  }
  return 0;
}

/* getMasterIndex
 *
 * Given the 1D index of an element in the output array we are filling,
 * obtains the 1D index of that element in the input array.
 */
static uint64_t
getMasterIndex( int _arrdims,
                uint64_t *_oldDims,
                int _dim_index,
                int *_positions,
                uint64_t *_newDims,
                uint64_t _l )
{
  uint64_t temp_inds[_arrdims];
  memset(temp_inds, 0, sizeof(uint64_t) * _arrdims);
  uint64_t div, thisMod;
  uint64_t numLeft = _l;
  int ii, ij;
  /* Find multi-dimensional index of the element in the output
   * array
   */
  for (ii=0; ii<_arrdims; ii++){
    uint64_t sizeofblock = 1;
    for (ij=(ii+1); ij<_arrdims; ij++){
      sizeofblock *= _newDims[ij];
    }
    div = numLeft / sizeofblock;
    //printf("div = %d\n", (int) div);
    temp_inds[ii] = div;
    thisMod = numLeft % sizeofblock;
    //printf("thisMod = %d\n", (int) thisMod);
    if (thisMod == 0) break;
    numLeft = thisMod;
  }
  /* we now have the multi-D indices of the element in question
   * in the output array. Now, obtain the multi-D indices of the element
   * in the input array. Only the index in the "selected" will change. */
  int tmp = (int) temp_inds[_dim_index];
  temp_inds[_dim_index] = (uint64_t) _positions[tmp];
  /* Now calculate 1D index of the element in the input array */
  uint64_t out = 0;
  for (ii=0; ii<_arrdims; ii++){
    uint64_t prod = 1;
    for (ij = ii+1; ij < _arrdims; ij++){
      prod *= _oldDims[ij];
    } 
    out += prod * temp_inds[ii];
  }
  return out;
}

static void 
build_output_string( char  *_output,    //out
                     int    _numargs,   //in
                     char **_args)      //in
{
  char tmps[20];
  memset (tmps, 0, 20);
  sprintf(_output, "%s", _args[0]);
  int i;
  for (i=1; i<_numargs; i++){
    sprintf ( tmps, ":%s", _args[i]);
    strcat (_output, tmps);
  }
  return;
}
