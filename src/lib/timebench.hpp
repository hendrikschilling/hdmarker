#ifndef __MICROBENCH_H
#define __MICROBENCH_H
  
 /** 
* @file timebench.hpp 
* @brief To determine the processing time of different code fragments
*
* @author Hendrik Schilling (implementation)
* @author Maximilian Diebold (documentation)
* @date 01/15/2018
* 
*
* Example Usage:
* @code
*    microbench_init();
* 
*   //CODE to get measured ...;
* 
*    microbench_measure_output('app startup');
*   //or
*    microbench_measure_run('app runtime');
* @endcode
* 
*/ 
  
namespace hdmarker {
  
extern struct timespec mb_realtime;
extern struct timespec mb_cputime;

void microbench_init(void);
void microbench_measure_output(char *msg);
void microbench_measure_run(char *msg);


}

#endif
