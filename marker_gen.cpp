#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <assert.h>
#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <tclap/CmdLine.h>

using namespace cv;
using namespace std;

const int recursive_markers = 3;
const int subsampling = 1;
const int ss_border = 2;

void setnumber(Mat &m, int n)
{
  m.at<uchar>(1, 1) = (n/1 & 0x01) * 255;
  m.at<uchar>(1, 2) = (n/2 & 0x01) * 255;
  m.at<uchar>(1, 3) = (n/4 & 0x01) * 255;
  m.at<uchar>(2, 1) = (n/8 & 0x01) * 255;
  m.at<uchar>(2, 2) = (n/16 & 0x01) * 255;
  m.at<uchar>(2, 3) = (n/32 & 0x01) * 255;
  m.at<uchar>(3, 1) = (n/64 & 0x01) * 255;
  m.at<uchar>(3, 2) = (n/128 & 0x01) * 255;
  m.at<uchar>(3, 3) = (n/256 & 0x01) * 255;
}

int smallidtomask(int id, int x, int y)
{
  int j = (id / 32) * 2 + (id % 2) + y;
  int i = (id % 32) + x;
  
  return (j*13 + i * 7) % 512;
}

int idtomask(int id)
{
  if ((id&2==2)) 
    return id ^ 170;
  else
    return id ^ 340;
}

int masktoid(int mask)
{
  if ((mask&2==2)) 
    return mask ^ 170;
  else
    return mask ^ 340;
}

void writemarker(Mat &img, int page, int offx = 0, int offy = 0)
{
  Mat marker = Mat::zeros(5, 5, CV_8UC1);
  marker.at<uchar>(2, 4) = 255;
  
  for(int j=0;j<16;j++)
    for(int i=0;i<32;i++) {
      setnumber(marker, idtomask(j*32+i));
      marker.copyTo(img.colRange(i*5+offx,i*5+5+offx).rowRange(j*10+(i%2)*5+offy, j*10+(i%2)*5+5+offy));
      
      setnumber(marker, page ^ smallidtomask(j*32+i, 0, 2*((i+1)%2)-1));
      marker = 255 - marker;
      marker.copyTo(img.colRange(i*5+offx,i*5+5+offx).rowRange(j*10+((i+1)%2)*5+offy, j*10+((i+1)%2)*5+5+offy));
      marker = 255 - marker;
    }
}

void checker_recurse(Mat &img, Mat &checker)
{
  Mat hr;
  int w = img.size().width;
  int h = img.size().height;
  int ws = subsampling+2*ss_border;
  int w_hr = w*ws;;
  uint8_t *ptr_hr, *ptr_img;
  
  if (!recursive_markers) {
    checker = img;
    return;
  }
  
  resize(img, hr, Point2i(img.size())*ws, 0, 0, INTER_NEAREST);
  
  ptr_img = img.ptr<uchar>(0);
  ptr_hr = hr.ptr<uchar>(0);
  
  for(int y=0;y<h;y++)
    for(int x=0;x<w;x++) {
      for(int j=ss_border;j<ws-ss_border;j++)
        for(int i=j%2+ss_border;i<ws-ss_border;i+=2)
          ptr_hr[(y*ws+j)*w_hr+x*ws+i] = 255-ptr_hr[(y*ws+j)*w_hr+x*ws+i];
    }
    
  checker = hr;
}

void checker_add_recursive(Mat &img, Mat &checker)
{
  for(int i=0;i<recursive_markers;i++)
    checker_recurse(img, img);
}

int main(int argc, char* argv[])
{
  int page = atoi(argv[1]); 
  
  if (argc == 3) {
    Mat img = Mat::zeros(16*10, 32*5, CV_8UC1);
    img += 255;
    writemarker(img, page);
    //resize(img, img, Size(16*10*8, 16*10*8), 0, 0, INTER_NEAREST);
    checker_add_recursive(img, img);
    imwrite(argv[2], img);
  }
  else if (argc == 5) {
    int w = atoi(argv[2]);
    int h = atoi(argv[3]);
    assert(w && h);
    Mat img = Mat::zeros(h*32*5, w*32*5, CV_8UC1);
    img += 255;
    
    for(int j=0;j<h;j++)
      for(int i=0;i<w;i++)
	writemarker(img, page+j*w+i, 32*5*i, 32*5*j);
    //resize(img, img, Size(w*32*5*1, h*32*5*1), 0, 0, INTER_NEAREST); 
    checker_add_recursive(img, img);
    imwrite(argv[4], img);
  }

  
  return 0;
  
}