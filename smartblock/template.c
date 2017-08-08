/* 
 * template.c
 * based on select.c
 *
 * Copyright 2017  National Technology and Engineering Solutions of Sandia,
 * LLC. Under the terms of Contract DE-NA0003525, there is a non-exclusive
 * license for use of this work by or on behalf of the U.S.  Government. Export
 * of this program may require a license from the United States Government.
 *
 * 
 * A documented template for a generic component to be used
 * with the rest of the Superglue components.
 * Illustrates the input, data partitioning,
 * and output stages of the Superglue components.
 *
 * Usage:
 *  <exec> input-arr-name input-stream-name \
 *         ouput-arr-name output-stream-name \
 *         [arg1] [arg2] [arg3] ...
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

//debug
static double area;
static struct sigaction act;
static void sighandler(int signum, siginfo_t *siginfo, void *ptr) {
  printf("[SEL%d] received SEGFAULT in area %f\n", rank, area);
  MPI_Abort(comm, -1);
}

static struct rusage ruse;

static void parse( int _argc, char **_argv, 
    char **_in_stream, char **_arr_name,
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

  /* MPI Initialization */
  MPI_Init (&argc, &argv);
  MPI_Comm_rank (comm, &rank);
  MPI_Comm_size (comm, &size);

  /* Parse command line arguments */
  /* These are the Flexpath streams and data array names within the 
     streams used by Superglue.
     If there are other command-line parameters
     used by this generic component, they should be
     parsed here as well.
  */
  char *in_stream, *arr_name, *out_stream, *out_arr_name;
  char **args; //names of the variables to be extracted from the array
  parse(argc, argv, &in_stream, &arr_name,
        &out_stream, &out_arr_name,
        &args);//fill in the above vars

  /* This executable will use Flexpath for reading */
  adios_read_init_method(ADIOS_READ_METHOD_FLEXPATH, comm, "");
  /* Open the input stream */
  ADIOS_FILE* f = adios_read_open(in_stream,
      ADIOS_READ_METHOD_FLEXPATH,
      comm,
      ADIOS_LOCKMODE_CURRENT, 0.0);

  if (rank == 0) printf("[SEl%d] adios_read_open returned\n", rank);

  /* ADIOS output setup */
  adios_init_noxml (comm);
  adios_set_max_buffer_size(60);
  adios_declare_group (&g_handle, "selectiongroup", "", adios_flag_yes);
  adios_select_method (g_handle, "MPI", "", "");

  /* Read global info from stream */
  ADIOS_VARINFO * hdr_len_info = adios_inq_var(f, "header_len");
  header_len = *((int *)(hdr_len_info->value));

  step = 0;
  while(adios_errno != err_end_of_stream) {
    /* Lets us inquire on the metadata for the main array
       that will be read later. */
    ADIOS_VARINFO * arr_info = adios_inq_var(f, arr_name);//total (all procs) this step
    /* read the number of dimensions of the input array */
    arrdims = arr_info->ndim;

    tstep = 0;
    global_size = 0;
    /* the index of the splitting (largest) dimension (see below) */
    split_index = 0;
    /* the size of the largest dimension */
    maxdim = 0;

    /* Read the timestep number */
    adios_schedule_read (f, NULL, "ntimestep", 0, 1, &tstep);
    adios_perform_reads(f, 1);

    /* Figure out the split dimension: the dimension that will be used to split the
     * data among the processes (for particle simulation data, the dimension that spans
     * over the number of particles). 
     * For this template, use the largest dimension.
     */
    i=0;
    for ( ; i<arrdims; i++){
        if (maxdim < arr_info->dims[i]){
            maxdim = arr_info->dims[i];
            split_index = i;
        }
    }

    /* Get size of array in the splitting dimension */
    global_size = (uint64_t)(arr_info->dims[split_index]);

    /* Determine what portion of the split dimension 
     * this process is responsible
     * for during this timestep */
    get_check_global_range(global_size, &mysize, 
                           &mystart, &common_size, &mod);
    
    /* Allocate space for the incoming data */
    uint64_t mydoublesin = 1;
    for (i=0; i<arrdims; i++){
        if (i == split_index)
            mydoublesin *= mysize;
        else
            mydoublesin *= arr_info->dims[i];
    }
    uint64_t my_bytes = sizeof(double) * mydoublesin;
    data_in = (double *)malloc(my_bytes);
    memset(data_in, 0, my_bytes);

    /* Below are arrays that ADIOS uses
       to specify the starting positions
       and counts of elements in the array to be 
       read, so it can be partitioned among
       all procs involved */
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

    /* Use the above arrays to specify a bounding box 
     * over the main data array for split-reading
     */
    ADIOS_SELECTION *global_select = 
        adios_selection_boundingbox(arrdims, starts, counts);
    /* Read the array */
    adios_schedule_read (f,
                         global_select,
                         arr_name,
                         0, 1, data_in);
    adios_perform_reads (f, 1);

    /* Data has been read: indicate to writer
     * that the data in this step is no longer
     * needed 
     */
    adios_release_step(f);


    /* Allocate array that will be used for outputting data.
       Since this is a template component, the
       output will be a copy of the input (no data processing).
    */
    my_bytes = sizeof(double) * mydoublesin;
    data_out = (double *)malloc(my_bytes);
    memcpy(data_out, data_in, my_bytes);
    
    /* Output the data */
    /* Create the strings that will describe to ADIOS the dimensions
     * of the global array and define the variables to ADIOS */
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
        /* This loop defines the variables that will be used
         * to write the dimensions of the
         * output array to ADIOS.
         */
        for (i=0; i<arrdims; i++) {
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
        /* Here we define the output array, using the dimension strings
         * defined in the loop above, even though the strings
         * are not written yet.
         */
        adios_define_var (g_handle, out_arr_name, "", adios_double, 
                          out_local_dims_str, out_global_dims_str, out_offsets_str);
    }


    /* Check the dimension strings */
      printf("[%d] ADIOS dim strings: %s %s %s\n", rank, out_local_dims_str,
      out_global_dims_str, out_offsets_str);
      sleep(800);
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
    //printf("allocated %" PRIu64 " bytes\n",
    //    ints + ulongs + doubles + str_size);
    adios_write (out_handle, "me", &rank);
    adios_write (out_handle, "mpi_size", &size);
    adios_write (out_handle, "ntimestep", &tstep);//original tstep num from simluation

    /* The strings that defined the dimensions of the main array
       are just strings at this point. We need to write them
       to tell ADIOS the actual dimensions for this run.
    */
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

    /*** WRITE DATA ARRAY ***/      
    adios_write (out_handle, out_arr_name, valid_data);
    adios_close (out_handle);

    if (rank == 0) printf("[TEMPLATE%d]: read and wrote data for timestep %" PRIu64 "\n", rank, tstep);
    free(data_in);
    free(selection_inds);
    free(valid_data);
    step++; 
    int ret;
    //t2 = wfgettimeofday();
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
       char **_out_stream, 
       char **_out_arr_name,
       int *numargs,
       char ***_args )
{
  if (rank == 0 && _argc < 7) {
    fprintf(stderr, "\nUsage: <exec> input-stream input-array-name \\\n"
            "output-stream output-array-name ...\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  *_in_stream = _argv[1];
  *_arr_name = _argv[2];
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



/*


  for (k=0; k<_hdr_len; k++){
    tmp = _hdr_str[k];
    if (tmp == '\0'){//reached end of header
      tempstr[l] = '\0';
      if (string_is_member(tempstr, _args, _numargs)){//last string is a wanted arg
        (*_positions)[num_fetched] = position_in_header;
        num_fetched++;
        if (num_fetched == _numargs){
          return;
        }
        else {
          char errormsg[256];
          sprintf(errormsg, "[SEL%d] in parse_header, with tempstr=%s is last read string in header\n",
            rank, tempstr);
          fprintf(stderr, "%s", errormsg);
          fprintf(stderr, "select: ERROR: arg not found in header\n");
          MPI_Abort(MPI_COMM_WORLD, -1);
        }
      }
      else{
        fprintf(stderr, "error: arg %s not found in header! num_fetched=%d\n", 
            tempstr, num_fetched);
        MPI_Abort(MPI_COMM_WORLD, -1);
      }
    }
    
    if (tmp == ':'){//current header member string end reached
      if (k == 0){
        fprintf(stderr, "error: header cannot start with \':\'\n");
        MPI_Abort (MPI_COMM_WORLD, -1);
      }
      tempstr[l] = '\0';
      if (string_is_member(tempstr, _args, _numargs)){ //this string matches
        (*_positions)[num_fetched] = position_in_header;
        num_fetched++;
        //printf("string %s found, num_fetched = %d _numargs=%d\n", 
        //    tempstr, num_fetched, _numargs);
        if (num_fetched == _numargs){
          return;
        }
        l = 0;
        num_fetched++;
        //printf("string %s found, num_fetched = %d _numargs=%d\n", 
        //    tempstr, num_fetched, _numargs);
        if (num_fetched == _numargs){
          return;
        }
        l = 0;
        memset(tempstr, 0, 30);
      }
      else {//string does not match desired
        l = 0;
        memset(tempstr, 0, 30);
      }
      position_in_header++;
    }

    else {//current header character is not end of a string or argument
      tempstr[l] = tmp;
      l++;
    }
  }
  return;
}

*/

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
  //printf("[%d] temp_inds[0]: %" PRIu64 " temp_inds[1]: %" PRIu64 
  //    " temp_inds[2]: %" PRIu64 "\n",
  //    rank, temp_inds[0], temp_inds[1], temp_inds[2]);
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
