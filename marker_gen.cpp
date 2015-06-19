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

using namespace cv;
using namespace std;

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

int main(int argc, char* argv[])
{
  int page = atoi(argv[1]); 
  
  if (argc == 3) {
    Mat img = Mat::zeros(16*10, 32*5, CV_8UC1);
    img += 255;
    writemarker(img, page);
    //resize(img, img, Size(16*10*8, 16*10*8), 0, 0, INTER_NEAREST);
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
    imwrite(argv[4], img);
  }

  
  return 0;
  
}