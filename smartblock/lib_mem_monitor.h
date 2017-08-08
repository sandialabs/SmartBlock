#ifndef __LIB_MEM_MONITOR
#define __LIB_MEM_MONITOR 

/*
 *
 * Copyright 2017  National Technology and Engineering Solutions of Sandia,
 * LLC. Under the terms of Contract DE-NA0003525, there is a non-exclusive
 * license for use of this work by or on behalf of the U.S.  Government. Export
 * of this program may require a license from the United States Government.
 *
 *
 */

#include <signal.h>
#include <stdint.h>
#include "mpi.h"

//extern long memcounts;
//extern double mem_vm_ave, mem_res_ave;
//extern long this_mem_vm;
//extern long this_mem_res;
//extern struct sigaction act_mem;
//extern double mem_vm_ave_sum, mem_res_ave_sum, mem_vm_ave_ave, mem_res_ave_ave;

double wfgettimeofday(void);
void monitor_out(int rank, int size, uint64_t tstep, uint64_t my_bytes, 
  MPI_Comm comm, char *component);
void sighandler_mem(int signum, siginfo_t *siginfo, void *ptr);
void lib_mem_init(void);
void nohandler_mem(int rank);
void ind_timer_start(int ix, char *routine);
void ind_timer_end(int ix);

int main_loop(char *pidstatus, long *vmsizeout, long *vmrssout);
//void mem_out(int rank, uint64_t tstep);
void outer_timer_end(int rank, char *component_name);
void outer_timer_init(int rank, char *component_name);

#endif
