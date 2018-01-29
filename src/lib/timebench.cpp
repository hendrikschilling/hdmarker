#include "timebench.hpp"
#include <map>

#include <time.h>
#include <stdio.h>

namespace hdmarker {
  
#ifdef MICROBENCH
struct timespec mb_realtime;
struct timespec mb_cputime;
struct timespec mb_init_realtime;
struct timespec mb_init_cputime;

typedef std::map<std::string, double> timemap;
timemap timings;

double msdiff(struct timespec &start, struct timespec &stop)
{
  return (stop.tv_sec-start.tv_sec)*1000.0 + (stop.tv_nsec-start.tv_nsec)/1000000.0;
}
#endif

void microbench_init(void)
{
#ifdef MICROBENCH
  clock_gettime(CLOCK_MONOTONIC, &mb_realtime);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &mb_cputime);
  mb_init_cputime = mb_cputime;
  mb_init_realtime = mb_realtime;
  
  timings.clear();
#endif
}

void microbench_measure_output(char *msg)
{
#ifdef MICROBENCH
  struct timespec real;
  struct timespec cpu;
  
  clock_gettime(CLOCK_MONOTONIC, &real);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu);
  
  double rd = msdiff(mb_realtime, real);
  double cd = msdiff(mb_cputime, cpu);
  
  mb_realtime = real;
  mb_cputime = cpu;
  
  printf("%s - real: %.3gms cpu: %.3gms threading: %.3g\n", msg, rd, cd, cd/rd);
  
  if (timings.find(msg) != timings.end()) {
    timings[msg] += rd;
  }
  else
    timings[msg] = rd;
#endif
}

void microbench_measure_run(char *msg)
{
#ifdef MICROBENCH
  struct timespec real;
  struct timespec cpu;
  
  clock_gettime(CLOCK_MONOTONIC, &real);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu);
  
  double rd = msdiff(mb_init_realtime, real);
  double cd = msdiff(mb_init_cputime, cpu);
  
  printf("%s - real: %.3gms cpu: %.3gms threading: %.3g\n", msg, rd, cd, cd/rd);
  
  printf("\nfull list %.1fms: \n---------------------------------\n", rd);
  
  for(timemap::iterator it = timings.begin(); it != timings.end(); it++) {
    printf("%20s - %6.1fms %5.1f%%\n", it->first.c_str(), it->second, it->second/rd*100.0);
  }
#endif
}

} //namespace hdmarker

