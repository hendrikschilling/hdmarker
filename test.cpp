#include <stdio.h>

#include "hdmarker.hpp"
#include "timebench.hpp"
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv2/imgproc/imgproc.hpp>

//#include <callgrind.h>

using namespace std;
using namespace cv;

void usage(const char *c)
{
  printf("usage: %s <input image> <output image>", c);
  exit(EXIT_FAILURE);
}

int main(int argc, char* argv[])
{
  microbench_init();
  char buf[64];
  Mat img, paint;
  Point2f p1, p2;
  int x1, y1, x2, y2;
  vector<Corner> corners;
  Corner c;
  
  if (argc != 3 && argc != 4)
    usage(argv[0]);
  
  img = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
  paint = cv::imread(argv[1]);
  
  Marker::init();
  
  microbench_measure_output("app startup");
  //CALLGRIND_START_INSTRUMENTATION;
  if (argc == 4)
    Marker::detect(img, corners,0,0,atof(argv[3]),100);
  else
    Marker::detect(img, corners,0,0,0.5);
  //CALLGRIND_STOP_INSTRUMENTATION;
    
  microbench_init();
  paint = cv::imread(argv[1]);
  
  printf("final score %d corners\n", corners.size());
  
  for(uint32_t i=0;i<corners.size();i++) {
    c = corners[i];
    //if (c.page != 256) continue;
    sprintf(buf, "%d/%d", c.id.x, c.id.y);
    circle(paint, c.p, 1, Scalar(0,0,0,0), 2);
    circle(paint, c.p, 1, Scalar(0,255,0,0));
    putText(paint, buf, c.p, FONT_HERSHEY_PLAIN, 0.5, Scalar(0,0,0,0), 2, CV_AA);
    putText(paint, buf, c.p, FONT_HERSHEY_PLAIN, 0.5, Scalar(255,255,255,0), 1, CV_AA);
  }
  
  imwrite(argv[2], paint);
  

  microbench_measure_output("app finish");
  return EXIT_SUCCESS;
}