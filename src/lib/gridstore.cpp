#include "gridstore.hpp"

using namespace std;
using namespace cv;

typedef unsigned int uint;

Gridstore::Gridstore()
{
  w = 0;
  h = 0;
  d = -1;
}


Gridstore::Gridstore(int wi, int hi, int di)
{
  d = di;
  w = (wi+d-1)/d;
  h = (hi+d-1)/d;
  buckets.resize(w*h);
  elements.resize(0);
}


void Gridstore::add(void *data, Point2f p)
{
  int x = p.x/d;
  int y = p.y/d;
  
  if (x >= w || y >= h) return;
  
  if (!buckets[y*w+x])
    buckets[y*w+x] = new vector<void*>();
  
  buckets[y*w+x]->push_back(data);
  elements.push_back(data);
}

/**
 *  Destructor
 */
Gridstore::~Gridstore()
{
  for(int i=0;i<w*h;i++)
    if (buckets[i])
      delete buckets[i];
}


vector<void*> Gridstore::getWithin(Point2f p, int dist)
{
  vector<void*>result;
  
  int y = p.y;
  int x = p.x;
  
  for(int j=max(0,(y-dist)/d);j<min((y+dist)/d+1,h);j++)
    for(int i=max(0,(x-dist)/d);i<min((x+dist)/d+1,w);i++)
      if (buckets[j*w+i])
	for(uint n=0;n<buckets[j*w+i]->size();n++) {
	  result.push_back((*buckets[j*w+i])[n]);
	}
	
  //printf("result: %d\n", result.size());
	
  return result;
}



int Gridstore::size(){
  return elements.size();
}


void * Gridstore::operator[](int i)
{
  return elements[i];
}
