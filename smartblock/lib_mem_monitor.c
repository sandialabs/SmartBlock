/*
 *
 * Copyright 2017  National Technology and Engineering Solutions of Sandia,
 * LLC. Under the terms of Contract DE-NA0003525, there is a non-exclusive
 * license for use of this work by or on behalf of the U.S.  Government. Export
 * of this program may require a license from the United States Government.
 *
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <math.h>
#include "mpi.h"
#include "adios.h"
#include "adios_read.h"
#include "lib_mem_monitor.h"

#define MAX_TIMERS 10

long memcounts;
double mem_vm_ave, mem_res_ave;
long this_mem_vm;
long this_mem_res;
struct sigaction act_mem;

double outer_start_time, outer_end_time;
static double start_times[MAX_TIMERS];
static double end_times[MAX_TIMERS];
static double total_times[MAX_TIMERS];
static double squares[MAX_TIMERS];
static char *routine_names[MAX_TIMERS];
static int timers_used = 0;


double
wfgettimeofday( void )
{
    double timestamp;
    struct timeval now;
    gettimeofday(&now, NULL);
    timestamp = now.tv_sec + now.tv_usec* 1.0e-6 ;
    return timestamp;
}

/* Outer timer functions */

void outer_timer_init(int rank, char *component_name)
{
  if (rank != 0) return;
  outer_start_time = wfgettimeofday();
  FILE *tfp;
  tfp = fopen("time.log", "a");
  fprintf(tfp, "outer start time for %s: %f\n", component_name, outer_start_time);
  fclose(tfp);
}

void outer_timer_end(int rank, char *component_name)
{
  printf("histogram component: outer_timer_end()\n");
  if (rank != 0)
  {
    return;
  }
  outer_end_time = wfgettimeofday();
  FILE *tfp;
  tfp = fopen("time.log", "a");
  fprintf(tfp, "outer end time for %s: %f\n", component_name, outer_end_time);
  fclose(tfp);
}
  

/*
static void sighandler_mem(int signum, siginfo_t *siginfo, void *ptr) {
  memcounts++;
  if (memcounts == 1)
    printf("[sel-%d] sighandler_mem\n", rank);
  main_loop("proc/self/status", &this_mem_vm, &this_mem_res);
  if (memcounts == 1){
    mem_vm_ave = this_mem_vm;
    mem_res_ave = this_mem_res;
  }
  mem_vm_ave = (mem_vm_ave * (memcounts - 1) + this_mem_vm) / ((double) memcounts);
  mem_vm_ave = (mem_res_ave * (memcounts - 1) + this_mem_res) / ((double) memcounts);
  //debug
  if (memcounts == 5) {
    printf("[sel-%d] timer debug memcounts=5 this_mem_vm=%ld this_mem_res=%ld\n",
            rank, this_mem_vm, this_mem_res);
  }
}
*/

void ind_timer_start (int ix, char *routine)
{
  if (ix == MAX_TIMERS - 1){
    fprintf(stderr, "too many timers started in lib_mem_monitor library\n");
    return;
  }
  start_times[ix] = wfgettimeofday();
  routine_names[ix] = routine;
}
  
void ind_timer_end (int ix)
{
  if (ix == MAX_TIMERS - 1){
    fprintf(stderr, "too many timers started in lib_mem_monitor library\n");
    return;
  }
  end_times[ix] = wfgettimeofday();
  timers_used++;
}

void lib_mem_init()
{
  memcounts = 0;
  this_mem_vm = 0;
  this_mem_res = 0;
  mem_vm_ave = 0;
  mem_res_ave = 0;
  timers_used = 0;
}


void nohandler_mem(int rank) {
  memcounts++;
  main_loop("/proc/self/status", &this_mem_vm, &this_mem_res);
  /*
  if (rank == 0)
  {
    printf("[%d] nohandler got this_mem_vm=%ld and this_mem_res=%ld\n", 
            rank, this_mem_vm, this_mem_res);
  }
  */
  if (memcounts == 1){
    mem_vm_ave = (double) this_mem_vm;
    mem_res_ave = (double) this_mem_res;
  }
  else { //new average
    mem_vm_ave = (mem_vm_ave * (memcounts - 1) + this_mem_vm) / memcounts;
    mem_res_ave = (mem_res_ave * (memcounts - 1) + this_mem_res) / memcounts;
  }
}

int main_loop(char *pidstatus, long *vmsizeout, long *vmrssout)
{
  char *line;
  char *vmsize;
  char *vmpeak;
  char *vmrss;
  char *vmhwm;
  
  size_t len;
  
  FILE *f;

  vmsize = NULL;
  vmpeak = NULL;
  vmrss = NULL;
  vmhwm = NULL;
  line = malloc(128);
  len = 128;
  
  f = fopen(pidstatus, "r");
  if (!f) 
  {
    printf("could not fopen?\n");
    return 1;
  }
  
  /* Read memory size data from /proc/pid/status */
  while (!vmsize || !vmpeak || !vmrss || !vmhwm)
  {
    if (getline(&line, &len, f) == -1)
    {
      /* Some of the information isn't there, die */
      return 1;
    }
    /* Find VmPeak */
    if (!strncmp(line, "VmPeak:", 7))
    {
      vmpeak = strdup(&line[7]);
    }
    /* Find VmSize */
    else if (!strncmp(line, "VmSize:", 7))
    {
      vmsize = strdup(&line[7]);
    }
    /* Find VmRSS */
    else if (!strncmp(line, "VmRSS:", 6))
    {
      vmrss = strdup(&line[7]);
    }
    /* Find VmHWM */
    else if (!strncmp(line, "VmHWM:", 6))
    {
      vmhwm = strdup(&line[7]);
    }
  }
  free(line);
  
  fclose(f);

  /* Get rid of " kB\n" */
  len = strlen(vmsize);
  vmsize[len - 4] = 0;
  len = strlen(vmpeak);
  vmpeak[len - 4] = 0;
  len = strlen(vmrss);
  vmrss[len - 4] = 0;
  len = strlen(vmhwm);
  vmhwm[len - 4] = 0;
   
  /* Output results to stderr */
  /*
  printf("[SEL-r%d-a%f] main_loop, mem use: %s\t%s\t%s\t%s\n",
          rank, area, vmsize, vmpeak, vmrss, vmhwm);
  */
  *vmsizeout = atol(vmsize);
  *vmrssout = atol(vmrss);
   
  free(vmpeak);
  free(vmsize);
  free(vmrss);
  free(vmhwm);
  
  /* Success */
  return 0;
}    

double mem_vm_ave_sum, mem_res_ave_sum, mem_vm_ave_ave, mem_res_ave_ave;
void monitor_out(int rank, int size, uint64_t tstep, uint64_t my_bytes,
  MPI_Comm comm, char *component)
{
  int mpi_size;
  MPI_Comm_size(comm, &mpi_size);
  MPI_Reduce (&mem_vm_ave, &mem_vm_ave_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce (&mem_res_ave, &mem_res_ave_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  int j;
  double *ave_times, *ave_squares, *stdev_times, *min_times, *max_times;
  if (rank == 0 ){
    ave_times = malloc(sizeof(double) * timers_used);
    ave_squares = malloc(sizeof(double) * timers_used);
    stdev_times = malloc(sizeof(double) * timers_used);
    min_times = malloc(sizeof(double) * timers_used);
    max_times = malloc(sizeof(double) * timers_used);
  }
  for (j=0; j<timers_used; j++){
    total_times[j] = end_times[j] - start_times[j];
    squares[j] = (total_times[j]) * (total_times[j]);
    MPI_Reduce (&total_times[j], &ave_times[j], 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce (&total_times[j], &min_times[j], 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce (&total_times[j], &max_times[j], 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0){
      ave_times[j] = (ave_times[j])/((double) mpi_size);
    }
    MPI_Reduce (&squares[j], &ave_squares[j], 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    if (rank == 0){
      ave_squares[j] = (ave_squares[j]) / ((double) mpi_size);
      stdev_times[j] = sqrt(ave_squares[j] - (ave_times[j] * ave_times[j]));
    }
  }
  if (rank == 0)
  {
    mem_vm_ave_ave = mem_vm_ave_sum / ((double) size);
    mem_res_ave_ave = mem_res_ave_sum / ((double) size);
    
    
    FILE *memfp;
    char log[30];
    sprintf(log, "%s_monitor.log", component);
    memfp = fopen(log, "a");
    fprintf(memfp, "\n%s step %" PRIu64"\nave vm = %lf, ave resident memory = %lf\n",
                    component, tstep, mem_vm_ave_ave, mem_res_ave_ave);
    fprintf(memfp, "my array bytes (rank 0): %" PRIu64"\n", my_bytes);
    int j;
    fprintf(memfp, "Routine completion times below...\n"
      "%-20s%-15s%-15s%-15s%s\n",
      "routine name", "mean", "min", "max", "stdev");
    for (j=0; j<timers_used; j++){
      fprintf(memfp, "%-20s%-15lf%-15lf%-15lf%-15lf\n",
        routine_names[j], ave_times[j], min_times[j], max_times[j], stdev_times[j]);
    }
    fclose(memfp);
  }
}
