#ifndef __GIRDSTORE_H
#define __GIRDSTORE_H

#include <vector>
#include <opencv2/core/core.hpp>
  
class Gridstore {
  int w, h, d;
  std::vector<void*> elements;
  std::vector<std::vector<void*>*> buckets;
public :
  
  Gridstore();
  Gridstore(int w, int h, int dist);
  
  void add(void *data, cv::Point2f p);
  std::vector<void*> getWithin(cv::Point2f p, int dist);
  int size();
  void *operator[](int i);
  ~Gridstore();
};


#endif