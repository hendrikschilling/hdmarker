#ifndef __MICROBENCH_H
#define __MICROBENCH_H
  
extern struct timespec mb_realtime;
extern struct timespec mb_cputime;

void microbench_init(void);
void microbench_measure_output(char *msg);
void microbench_measure_run(char *msg);
void microbench_measure_stow(void);
void microbench_output_stowed(char *msg);

#endif