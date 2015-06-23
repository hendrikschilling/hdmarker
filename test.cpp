#include <stdio.h>

#include "hdmarker.hpp"
#include "timebench.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

const int grid_width = 16;
const int grid_height = 14;

void usage(const char *c)
{
  printf("usage: %s <input image> <output image>", c);
  exit(EXIT_FAILURE);
}

Point3f add_zero_z(Point2f p2)
{
  Point3f p = {p2.x, p2.y, 0.0};
  
  return p;
}

Point2f grid_to_world(Corner &c, int grid_w)
{
  Point2f p;
  int px, py;
  
  px = c.page % grid_w;
  py = c.page / grid_w;
  px*=32;
  py*=32;
  
  p.x = c.id.x+px;
  p.y = c.id.y+py;
  
  return p;
}

bool calib_savepoints(vector<vector<Point2f> > &all_img_points, vector<vector<Point3f> > &all_world_points, vector<Corner> corners, int hpages, int vpages)
{
  if (!corners.size()) return false;
  
  vector<Point2f> plane_points;
  vector<Point2f> img_points;
  vector<Point3f> world_points;
  vector<Point2f> img_points_check;
  vector<Point2f> world_points_check;
  vector <uchar> inliers;
  
  for(uint i=0;i<corners.size();i++) {
    int px, py;
    if (corners[i].page > hpages*vpages)
      continue;
    /*if (corners[i].page != 0)
      continue;*/
    px = corners[i].page % hpages;
    py = corners[i].page / hpages;
    px*=32;
    py*=32;
    img_points_check.push_back(corners[i].p);
    world_points_check.push_back(grid_to_world(corners[i], hpages));
  }
  
  inliers.resize(world_points_check.size());
  findHomography(world_points_check, img_points_check, CV_RANSAC, 100, inliers);
  
  for(uint i=0;i<inliers.size();i++) {
    if (!inliers[i])
      continue;
    img_points.push_back(img_points_check[i]);
    world_points.push_back(add_zero_z(world_points_check[i]));
  }
  
  printf("findHomography: %d inliers of %d calibration points (%.2f%%)\n", img_points.size(),img_points_check.size(),img_points.size()*100.0/img_points_check.size());

  all_img_points[0] = img_points;
  all_world_points[0] = world_points;
  
  return true;
}

void check_calibration(vector<Corner> &corners, int w, int h, Mat &img)
{
  vector<Mat> rvecs, tvecs;
  Mat cameraMatrix(3,3,cv::DataType<double>::type);
  Mat distCoeffs;
  double rms;
  vector<Point2f> projected;
  Mat paint;
  
  vector<vector<Point3f>> world_points(1);
  vector<vector<Point2f>> img_points(1);
  
  if (!calib_savepoints(img_points, world_points, corners, grid_width, grid_height)) {
    return;
  }
  
  distCoeffs = Mat::zeros(1, 8, CV_64F);
  rms = calibrateCamera(world_points, img_points, Size(w, h), cameraMatrix, distCoeffs, rvecs, tvecs, CV_CALIB_RATIONAL_MODEL);
  printf("rms %f with full distortion correction\n", rms);
    
  projectPoints(world_points[0], rvecs[0], tvecs[0], cameraMatrix, distCoeffs, projected);
  cvtColor(img, paint, CV_GRAY2BGR);
  for(int i=0;i<projected.size();i++) {
    Point2f d = projected[i] - img_points[0][i];
    line(paint, img_points[0][i], img_points[0][i]+100*d, Scalar(0,0,255));
  }
  imwrite("off_hdm.png", paint);
  
  cornerSubPix(img, img_points[0], Size(4,4), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 100, 0.001));
  
  distCoeffs = Mat::zeros(1, 8, CV_64F);
  rms = calibrateCamera(world_points, img_points, Size(w, h), cameraMatrix, distCoeffs, rvecs, tvecs, CV_CALIB_RATIONAL_MODEL);
  printf("rms %f with full distortion correction, opencv cornerSubPix\n", rms);
  
  projectPoints(world_points[0], rvecs[0], tvecs[0], cameraMatrix, distCoeffs, projected);
  cvtColor(img, paint, CV_GRAY2BGR);
  for(int i=0;i<projected.size();i++) {
    Point2f d = projected[i] - img_points[0][i];
    line(paint, img_points[0][i], img_points[0][i]+100*d, Scalar(0,0,255));
  }
  imwrite("off_ocv.png", paint);
}


void check_precision(vector<Corner> &corners, int w, int h, Mat &img, const char *ref)
{
  vector<Mat> rvecs, tvecs;
  Mat cameraMatrix(3,3,cv::DataType<double>::type);
  Mat distCoeffs;
  double rms;
  vector<Point2f> projected;
  Mat paint;
  Mat refimg = imread(ref, CV_LOAD_IMAGE_GRAYSCALE);
  
  vector<vector<Point3f>> world_points(1);
  vector<vector<Point2f>> img_points(1);
  
  if (!calib_savepoints(img_points, world_points, corners, grid_width, grid_height)) {
    return;
  }
  
  distCoeffs = Mat::zeros(1, 8, CV_64F);
  rms = calibrateCamera(world_points, img_points, Size(w, h), cameraMatrix, distCoeffs, rvecs, tvecs, CV_CALIB_RATIONAL_MODEL);
  printf("rms %f with full distortion correction\n", rms);
    
  projectPoints(world_points[0], rvecs[0], tvecs[0], cameraMatrix, distCoeffs, projected);
  cvtColor(img, paint, CV_GRAY2BGR);
  for(int i=0;i<projected.size();i++) {
    Point2f d = projected[i] - img_points[0][i];
    line(paint, img_points[0][i], img_points[0][i]+100*d, Scalar(0,0,255));
  }
  imwrite("off_hdm.png", paint);
  
  cornerSubPix(refimg, img_points[0], Size(6,6), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 100, 0.001));
  
  distCoeffs = Mat::zeros(1, 8, CV_64F);
  rms = calibrateCamera(world_points, img_points, Size(w, h), cameraMatrix, distCoeffs, rvecs, tvecs, CV_CALIB_RATIONAL_MODEL);
  printf("rms %f with full distortion correction, opencv cornerSubPix\n", rms);
  
  projectPoints(world_points[0], rvecs[0], tvecs[0], cameraMatrix, distCoeffs, projected);
  cvtColor(img, paint, CV_GRAY2BGR);
  for(int i=0;i<projected.size();i++) {
    Point2f d = projected[i] - img_points[0][i];
    line(paint, img_points[0][i], img_points[0][i]+100*d, Scalar(0,0,255));
  }
  imwrite("off_ocv.png", paint);
}

void corrupt(Mat &img)
{
  GaussianBlur(img, img, Size(9,9), 0);
  Mat noise = Mat(img.size(), CV_32F);
  img. convertTo(img, CV_32F);
  randn(noise, 0, 3.0);
  img += noise;
  img.convertTo(img, CV_8U);
  cvtColor(img, img, COLOR_BayerRG2BGR_VNG);
  cvtColor(img, img, CV_BGR2GRAY);
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
  
  /*if (argc != 3 && argc != 4)
    usage(argv[0]);*/
  
  img = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
  paint = cv::imread(argv[1]);
  corrupt(img);
  imwrite("corrupted.png", img);
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
  
  check_calibration(corners, img.size().width, img.size().height, img);
  //check_precision(corners, img.size().width, img.size().height, img, argv[3]);
  
  imwrite(argv[2], paint);
  

  microbench_measure_output("app finish");
  return EXIT_SUCCESS;
}