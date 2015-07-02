#include <stdio.h>

#include "hdmarker.hpp"
#include "timebench.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "ceres/ceres.h"

#include <iostream>

using namespace std;
using namespace cv;

const int grid_width = 16;
const int grid_height = 14;

const bool use_rgb = false;

const bool demosaic = false;

const float subfit_oversampling = 2.0;
const float min_fit_contrast = 3.0;

const float refine_rms_limit = 10000.0;

const float rms_use_limit = 2.0;

double max_accept_dist = 3.0;

#include <stdarg.h>

void printprogress(int curr, int max, int &last, const char *fmt = NULL, ...)
{
  last = (last + 1) % 4;
  int pos = curr*60/max;
  char unf[] = "                                                             ]";
  char fin[] = "[============================================================]";
  char buf[100];
  
  char cs[] = "-\\|/";
  memcpy(buf, fin, pos+1);
  buf[pos+1] = cs[last];
  memcpy(buf+pos+2, unf+pos+2, 62-pos-2+1);
  if (!fmt) {
    printf("%s\r", buf);
  }
  else {
    printf("%s", buf);
    va_list arglist;
    va_start(arglist, fmt);
    vprintf(fmt, arglist);
    va_end(arglist);
    printf("\r");
  }
  fflush(NULL);
}

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

bool calib_savepoints(vector<vector<Point2f> > all_img_points[4], vector<vector<Point3f> > &all_world_points, vector<Corner> &corners, int hpages, int vpages, vector<Corner> &corners_filtered)
{
  if (!corners.size()) return false;
  
  vector<Point2f> plane_points;
  vector<Point2f> img_points[4];
  vector<Point3f> world_points;
  vector<Point2f> img_points_check[4];
  vector<Point2f> world_points_check;
  vector<int> pos;
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
    img_points_check[0].push_back(corners[i].p);
    pos.push_back(i);
    for(int c=1;c<4;c++)
      if (all_img_points[c].size())
        img_points_check[c].push_back(corners[i].pc[c-1]);
    world_points_check.push_back(grid_to_world(corners[i], hpages));
  }
  
  inliers.resize(world_points_check.size());
  Mat hom = findHomography(img_points_check[0], world_points_check, CV_RANSAC, 30, inliers);
  
  //vector<Point2f> proj;
  //perspectiveTransform(img_points_check[0], proj, hom);
  
  for(uint i=0;i<inliers.size();i++) {
    //printf("input %d distance %f\n", pos[i], norm(world_points_check[i]-proj[i]));
    if (!inliers[i])
      continue;
    corners_filtered.push_back(corners[pos[i]]);
    img_points[0].push_back((img_points_check[0])[i]);
    for(int c=1;c<4;c++)
      if (all_img_points[c].size())
        img_points[c].push_back((img_points_check[c])[i]);
    world_points.push_back(add_zero_z(world_points_check[i]));
  }
  
  printf("findHomography: %d inliers of %d calibration points (%.2f%%)\n", img_points[0].size(),img_points_check[0].size(),img_points[0].size()*100.0/img_points_check[0].size());

  (all_img_points[0])[0] = img_points[0];
  for(int c=1;c<4;c++)
    if (all_img_points[c].size())
      (all_img_points[c])[0] = img_points[c];
  all_world_points[0] = world_points;
  
  return true;
}

void calibrate_channel(vector<vector<Point2f> > &img_points, vector<vector<Point3f> > &world_points, int w, int h, Mat &img)
{
  vector<Mat> rvecs, tvecs;
  Mat cameraMatrix(3,3,cv::DataType<double>::type);
  Mat distCoeffs;
  double rms;
  vector<Point2f> projected;
  Mat paint;
  
  distCoeffs = Mat::zeros(1, 8, CV_64F);
  rms = calibrateCamera(world_points, img_points, Size(w, h), cameraMatrix, distCoeffs, rvecs, tvecs, CV_CALIB_RATIONAL_MODEL);
  printf("rms %f with full distortion correction\n", rms);
    
  projectPoints(world_points[0], rvecs[0], tvecs[0], cameraMatrix, distCoeffs, projected);
  if (img.channels() == 1)
    cvtColor(img, paint, CV_GRAY2BGR);
  else
    paint = img.clone();
  for(int i=0;i<projected.size();i++) {
    Point2f d = projected[i] - img_points[0][i];
    line(paint, img_points[0][i], img_points[0][i]+10*d, Scalar(0,0,255));
  }
  imwrite("off_hdm.png", paint);
}

void check_calibration(vector<Corner> &corners, int w, int h, Mat &img, vector<Corner> &corners_filtered)
{
  vector<Mat> rvecs, tvecs;
  Mat cameraMatrix(3,3,cv::DataType<double>::type);
  Mat distCoeffs;
  double rms;
  vector<Point2f> projected;
  Mat paint;
  
  vector<vector<Point3f>> world_points(1);
  vector<vector<Point2f>> img_points[4];
  
  img_points[0].resize(1);
  if (use_rgb)
    for(int c=1;c<4;c++)
      img_points[c].resize(1);
    
  printf("corners: %d\n", corners.size());
    
  if (!calib_savepoints(img_points, world_points, corners, grid_width, grid_height, corners_filtered)) {
    return;
  }
  
  calibrate_channel(img_points[0], world_points, w, h, img);
  if (use_rgb)
    for(int c=1;c<4;c++)
      calibrate_channel(img_points[c], world_points, w, h, img);

  
  /*cornerSubPix(img, img_points[0], Size(4,4), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 100, 0.001));
  
  distCoeffs = Mat::zeros(1, 8, CV_64F);
  rms = calibrateCamera(world_points, img_points, Size(w, h), cameraMatrix, distCoeffs, rvecs, tvecs, CV_CALIB_RATIONAL_MODEL);
  printf("rms %f with full distortion correction, opencv cornerSubPix\n", rms);
  
  projectPoints(world_points[0], rvecs[0], tvecs[0], cameraMatrix, distCoeffs, projected);
  cvtColor(img, paint, CV_GRAY2BGR);
  for(int i=0;i<projected.size();i++) {
    Point2f d = projected[i] - img_points[0][i];
    line(paint, img_points[0][i], img_points[0][i]+100*d, Scalar(0,0,255));
  }
  imwrite("off_ocv.png", paint);*/
}

/*
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
}*/

bool corner_cmp(Corner a, Corner b)
{
  if (a.id.y != b.id.y)
    return a.id.y < b.id.y;
  else
    return a.id.x < b.id.x;
}

bool corner_find_off_save(vector<Corner> &corners, Corner ref, int x, int y, Point2f &out)
{
  std::vector<Corner>::iterator found;
  ref.id.x += x;
  ref.id.y += y;
  //FIXME doesn't work ?!
  /*bounds=equal_range(corners.begin(), corners.end(), ref, corner_cmp);
  if (bounds.first != bounds.second)
    return true;*/
  /*bool found = false;
  for(int i=0;i<corners.size();i++)
    if (corners[i].id == ref.id) {
      found = true;
      out = corners[i].p;
      break;
    }*/
  found = lower_bound(corners.begin(), corners.end(), ref, corner_cmp);
  if (found->id != ref.id)
    return true;
  out = found->p;
  return false;
}

//FIXME width must be allowed non-integer values!
//FIXME this does not actually do any subpixel processing!
//netter code aber quatsch :-(
/*double fit_1d_rect(Mat &oned, int width)
{
  int size = oned.total();
  uint8_t *ptr = oned.ptr<uchar>(0);
  
  int64_t sum_bg = 0, sum_fg = 0;
  
  for(int i=width;i<2*width;i++)
    sum_fg += ptr[i];
  
  for(int i=0;i<width;i++)
    sum_bg += ptr[i];
  for(int i=2*width;i<size;i++)
    sum_bg += ptr[i];
  
  sum_fg *= subfit_refine_subpix;
  sum_bg *= subfit_refine_subpix;
  
  //FIXME init????
  uint64_t best = abs(sum_fg*(size-width)-sum_bg*width);
  uint64_t cand;
  uint64_t bestpos = 0;
  
  //FIXME first step wrong?
  //i is position of box
  for(int i=width;i<size-2*width;i++) {
    uint8_t left = ptr[i];
    uint8_t right = ptr[i+width];
    for(int s=0;s<subfit_refine_subpix;s++) {
      sum_fg -= left;
      sum_bg += left;
      sum_bg -= right;
      sum_fg += right;
      cand = abs(sum_fg*(size-width)-sum_bg*width);
      if (cand > best) {
        best = cand;
        bestpos = i*subfit_refine_subpix+s;
        printf("new %f %f\n", sum_fg*(1.0/width/subfit_refine_subpix), sum_bg*(1.0/(size-width)/subfit_refine_subpix));
      }
    }
  }
  
  return bestpos*(1.0/subfit_refine_subpix);
}*/



struct Gauss2dError {
  Gauss2dError(int val, int x, int y, double sw, double w)
      : val_(val), x_(x), y_(y), sw_(sw), w_(w) {}

/**
 * used function: 
 */
  template <typename T>
  bool operator()(const T* const p,
                  T* residuals) const {
    T x2 = T(x_) - p[0];
    T y2 = T(y_) - p[1];
    T sx2 = T(2.0)*p[3]*p[3];
    T sy2 = T(2.0)*p[4]*p[4];
    x2 = x2*x2;
    y2 = y2*y2;
    
    //want to us sqrt(x2+y2)+T(1.0) but leads to invalid jakobian?
    //T d = sqrt(x2+y2+T(0.0001)) + T(w_);
    T d = exp(-(x2/T(w_)+y2/T(w_)))+T(sw_);
    //non-weighted leads to better overall estimates?
    residuals[0] = (T(val_) - (p[5] + p[2]*exp(-(x2/sx2+y2/sy2))))*d;
    
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(int val, int x, int y, double sw, double w) {
    return (new ceres::AutoDiffCostFunction<Gauss2dError, 1, 6>(
                new Gauss2dError(val, x, y, sw, w)));
  }

  int x_, y_, val_;
  double w_, sw_;
};


struct GaussBorder2dError {
  GaussBorder2dError(int val, int x, int y, double sw, double w, int size, double border[4])
      : val_(val), x_(x), y_(y), sw_(sw), w_(w), size_(size) {
        for(int i=0;i<4;i++)
          border_[i] = border[i];
      }

/**
 * used function: 
 */
  template <typename T>
  bool operator()(const T* const p,
                  T* residuals) const {
    T x2 = T(x_) - p[0];
    T y2 = T(y_) - p[1];
    T sx2 = T(2.0)*p[3]*p[3];
    T sy2 = T(2.0)*p[4]*p[4];
    x2 = x2*x2;
    y2 = y2*y2;
    
    //possible lower border
    T lby2 = T(y_) - T(size_);
    lby2 = lby2*lby2;
    T lb = exp(-lby2/sy2);
    lb = lb * (T(border_[0])-p[5]);
    
    T rbx2 = T(x_) - T(size_);
    rbx2 = rbx2*rbx2;
    T rb = exp(-rbx2/sx2);
    rb = rb * (T(border_[1])-p[5]);
    
    T uby2 = T(y_);
    uby2 = uby2*uby2;
    T ub = exp(-uby2/sy2);
    ub = ub * (T(border_[2])-p[5]);
    
    T lbx2 = T(x_);
    lbx2 = lbx2*lbx2;
    T leb = exp(-lbx2/sx2);
    leb = leb * (T(border_[3])-p[5]);
    
    //want to us sqrt(x2+y2)+T(1.0) but leads to invalid jakobian?
    //T d = sqrt(x2+y2+T(0.0001)) + T(w_);
    T d = exp(-(x2/T(w_)+y2/T(w_)))+T(sw_)+T(1.0);
    //non-weighted leads to better overall estimates?
    residuals[0] = (T(val_) - (p[5] + lb +rb + leb + ub + p[2]*exp(-(x2/sx2+y2/sy2))))*d;
    
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(int val, int x, int y, double sw, double w, int size, double border[4]) {
    return (new ceres::AutoDiffCostFunction<GaussBorder2dError, 1, 6>(
                new GaussBorder2dError(val, x, y, sw, w, size, border)));
  }

  int x_, y_, val_, size_;
  double w_, sw_, border_[4];
};


struct GaussBorder2dError2 {
  GaussBorder2dError2(int val, int x, int y, double sw, double w, int size, double border[4])
      : val_(val), x_(x), y_(y), sw_(sw), w_(w), size_(size) {
        for(int i=0;i<4;i++)
          border_[i] = border[i];
      }

/**
 * used function: 
 */
  template <typename T>
  bool operator()(const T* const p,
                  T* residuals) const {
    T x2 = T(x_) - p[0];
    T y2 = T(y_) - p[1];
    T sx2 = T(2.0)*p[3]*p[3];
    T sy2 = T(2.0)*p[4]*p[4];
    T bsx2 = T(2.0)*p[6]*p[6];
    T bsy2 = T(2.0)*p[7]*p[7];
    x2 = x2*x2;
    y2 = y2*y2;
    
    //possible lower border
    T lby2 = T(y_) - T(size_);
    lby2 = lby2*lby2;
    T lb = exp(-lby2/bsy2);
    lb = lb * (T(border_[0])-p[5]);
    
    T rbx2 = T(x_) - T(size_);
    rbx2 = rbx2*rbx2;
    T rb = exp(-rbx2/bsx2);
    rb = rb * (T(border_[1])-p[5]);
    
    T uby2 = T(y_);
    uby2 = uby2*uby2;
    T ub = exp(-uby2/bsy2);
    ub = ub * (T(border_[2])-p[5]);
    
    T lbx2 = T(x_);
    lbx2 = lbx2*lbx2;
    T leb = exp(-lbx2/bsx2);
    leb = leb * (T(border_[3])-p[5]);
    
    //want to us sqrt(x2+y2)+T(1.0) but leads to invalid jakobian?
    //T d = sqrt(x2+y2+T(0.0001)) + T(w_);
    T d = /*exp(-(x2/T(w_)+y2/T(w_)))*/ T(sw_)+T(1.0);
    //non-weighted leads to better overall estimates?
    residuals[0] = (T(val_) - (p[5] + lb +rb + leb + ub + p[2]*exp(-(x2/sx2+y2/sy2))))*d;
    
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(int val, int x, int y, double sw, double w, int size, double border[4]) {
    return (new ceres::AutoDiffCostFunction<GaussBorder2dError2, 1, 8>(
                new GaussBorder2dError2(val, x, y, sw, w, size, border)));
  }

  int x_, y_, val_, size_;
  double w_, sw_, border_[4];
};


struct GaussBorder2dError3 {
  GaussBorder2dError3(int val, int x, int y, double sw, double w, int size)
      : val_(val), x_(x), y_(y), sw_(sw), w_(w), size_(size) {}

/**
 * used function: 
 */
  template <typename T>
  bool operator()(const T* const p, const T* const b,
                  T* residuals) const {
    T x2 = T(x_) - p[0];
    T y2 = T(y_) - p[1];
    T sx2 = T(2.0)*p[3]*p[3];
    T sy2 = T(2.0)*p[4]*p[4];
    T bsx2 = T(2.0)*p[6]*p[6];
    T bsy2 = T(2.0)*p[7]*p[7];
    x2 = x2*x2;
    y2 = y2*y2;
    
    //possible lower border
    T lby2 = T(y_) - T(size_);
    lby2 = lby2*lby2;
    T lb = exp(-lby2/bsy2);
    lb = lb * (b[0]-p[5]);
    
    T rbx2 = T(x_) - T(size_);
    rbx2 = rbx2*rbx2;
    T rb = exp(-rbx2/bsx2);
    rb = rb * (b[1]-p[5]);
    
    T uby2 = T(y_);
    uby2 = uby2*uby2;
    T ub = exp(-uby2/bsy2);
    ub = ub * (b[2]-p[5]);
    
    T lbx2 = T(x_);
    lbx2 = lbx2*lbx2;
    T leb = exp(-lbx2/bsx2);
    leb = leb * (b[3]-p[5]);
    
    //want to us sqrt(x2+y2)+T(1.0) but leads to invalid jakobian?
    //T d = sqrt(x2+y2+T(0.0001)) + T(w_);
    T d = /*exp(-(x2/T(w_)+y2/T(w_)))*/ T(sw_)+T(1.0);
    //non-weighted leads to better overall estimates?
    residuals[0] = (T(val_) - (p[5] + lb +rb + leb + ub + (p[2]-p[5])*exp(-(x2/sx2+y2/sy2))));
    
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(int val, int x, int y, double sw, double w, int size) {
    return (new ceres::AutoDiffCostFunction<GaussBorder2dError3, 1, 8, 4>(
                new GaussBorder2dError3(val, x, y, sw, w, size)));
  }

  int x_, y_, val_, size_;
  double w_, sw_;
};


struct GaussBorder2dError4 {
  GaussBorder2dError4(int val, int x, int y, double sw, double w, int size)
      : val_(val), x_(x), y_(y), sw_(sw), w_(w), size_(size) {}

/**
 * used function: 
 */
  template <typename T>
  bool operator()(const T* const p, const T* const b, const T* const pos,
                  T* residuals) const {
    T x2 = T(x_) - p[0];
    T y2 = T(y_) - p[1];
    T sx2 = T(2.0)*p[3]*p[3];
    T sy2 = T(2.0)*p[4]*p[4];
    T bsx2 = T(2.0)*p[6]*p[6];
    T bsy2 = T(2.0)*p[7]*p[7];
    x2 = x2*x2;
    y2 = y2*y2;
    
    //T fg = p[2]-p[5];
    
    //possible lower border
    T lby2 = T(y_) - T(size_) - sin(pos[0])*T(size_)*T(0.2);
    lby2 = lby2*lby2;
    T lb = exp(-lby2/bsy2);
    lb = lb * (b[0]-p[5]);
    
    T rbx2 = T(x_) - T(size_) - sin(pos[1])*T(size_)*T(0.2);
    rbx2 = rbx2*rbx2;
    T rb = exp(-rbx2/bsx2);
    rb = rb * (b[1]-p[5]);
    
    T uby2 = T(y_) - sin(pos[2])*T(size_)*T(0.2);
    uby2 = uby2*uby2;
    T ub = exp(-uby2/bsy2);
    ub = ub * (b[2]-p[5]);
    
    T lbx2 = T(x_) - sin(pos[3])*T(size_)*T(0.2);
    lbx2 = lbx2*lbx2;
    T leb = exp(-lbx2/bsx2);
    leb = leb * (b[3]-p[5]);
    
    //want to us sqrt(x2+y2)+T(1.0) but leads to invalid jakobian?
    //T d = sqrt(x2+y2+T(0.0001)) + T(w_);
    T d = /*exp(-(x2/T(w_)+y2/T(w_)))*/ T(sw_)+T(1.0);
    //non-weighted leads to better overall estimates?
    residuals[0] = (T(val_) - (p[5] + lb +rb + leb + ub + (p[2]-p[5])*exp(-(x2/sx2+y2/sy2))));
    
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(int val, int x, int y, double sw, double w, int size) {
    return (new ceres::AutoDiffCostFunction<GaussBorder2dError4, 1, 8, 4, 4>(
                new GaussBorder2dError4(val, x, y, sw, w, size)));
  }

  int x_, y_, val_, size_;
  double w_, sw_;
};


struct GaussBorder2dError5 {
  GaussBorder2dError5(int val, int x, int y, double sw, double w, int size)
      : val_(val), x_(x), y_(y), sw_(sw), w_(w), size_(size) {}

/**
 * used function: 
 */
  template <typename T>
  bool operator()(const T* const p, const T* const b,
                  T* residuals) const {
    T x2 = T(x_) - p[0];
    T y2 = T(y_) - p[1];
    T sx2 = T(2.0)*p[3]*p[3];
    T sy2 = T(2.0)*p[4]*p[4];
    x2 = x2*x2;
    y2 = y2*y2;
    
    //move extra border gauss distribution some way away from center
    T d2 = T(size_)*1.0+b[0]*b[0];
    T bx = d2*cos(b[1]);
    T by = d2*sin(b[1]);
    
    bx += T(size_)*0.5;
    by += T(size_)*0.5;
    
    bx = T(x_) - bx;
    by = T(y_) - by;
    bx = bx*bx;
    by = by*by;
    
    //spread at least equal to avg of point spread x,y
    T bs2 = (p[3]+p[4])*T(0.5)+b[2];
    bs2 = bs2*bs2;
    
    T border = exp(-(bx/bs2+by/bs2));
    
    T d = T(sw_)+T(1.0);
    residuals[0] = (T(val_) - (p[5] + (b[3]-p[5])*border + (p[2]-p[5])*exp(-(x2/sx2+y2/sy2))))*d;
    
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(int val, int x, int y, double sw, double w, int size) {
    return (new ceres::AutoDiffCostFunction<GaussBorder2dError5, 1, 6, 4>(
                new GaussBorder2dError5(val, x, y, sw, w, size)));
  }

  int x_, y_, val_, size_;
  double w_, sw_;
};


struct Border2dError {
  Border2dError(int val, int x, int y, double sw, int size, double border[4])
      : val_(val), x_(x), y_(y), sw_(sw), size_(size) {
        for(int i=0;i<4;i++)
          border_[i] = border[i];
      }

/**
 * used function: 
 */
  template <typename T>
  bool operator()(const T* const p,
                  T* residuals) const {
    T sx2 = T(2.0)*p[0]*p[0];
    T sy2 = T(2.0)*p[1]*p[1];
    
    //possible lower border
    T lby2 = T(y_) - T(size_);
    lby2 = lby2*lby2;
    T lb = exp(-lby2/sy2);
    lb = lb * (T(border_[0])-p[2]);
    
    T rbx2 = T(x_) - T(size_);
    rbx2 = rbx2*rbx2;
    T rb = exp(-rbx2/sx2);
    rb = rb * (T(border_[1])-p[2]);
    
    T uby2 = T(y_);
    uby2 = uby2*uby2;
    T ub = exp(-uby2/sy2);
    ub = ub * (T(border_[2])-p[2]);
    
    T lbx2 = T(x_);
    lbx2 = lbx2*lbx2;
    T leb = exp(-lbx2/sx2);
    leb = leb * (T(border_[3])-p[2]);
    
    //want to us sqrt(x2+y2)+T(1.0) but leads to invalid jakobian?
    //T d = sqrt(x2+y2+T(0.0001)) + T(w_);
    //T d = exp(-(x2/T(w_)+y2/T(w_)))+T(sw_)+T(1.0);
    //non-weighted leads to better overall estimates?
    residuals[0] = (T(val_) - (p[2] + lb + rb + ub + leb))*T(sw_);
    
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(int val, int x, int y, double sw, int size, double border[4]) {
    return (new ceres::AutoDiffCostFunction<Border2dError, 1, 3>(
                new Border2dError(val, x, y, sw, size, border)));
  }

  int val_, x_, y_, size_;
  double sw_;
  double border_[4];
};

template <typename T> T clamp(const T& n, const T& lower, const T& upper)
{
  return std::max(lower, std::min(n, upper));
}

void draw_gauss(Mat &img, double *p)
{
  int size = img.size().width;
  
  img = Mat(size, size, CV_8U);
  
  for(int y=0;y<size;y++)
    for(int x=0;x<size;x++) {
      double x2 = x-p[0];
      double y2 = y-p[1];
      x2 = x2*x2;
      y2 = y2*y2;
      double sx2 = 2.0*p[3]*p[3];
      double sy2 = 2.0*p[4]*p[4];
      //printf("%dx%d %f\n", );
      img.at<uchar>(y, x) = p[5] + p[2]*exp(-(x2/sx2+y2/sy2));
    }
}

void draw_gauss_border(Mat &img, double *p, double *border)
{
  int size = img.size().width;
  
  img = Mat(size, size, CV_8U);
  
  for(int y=0;y<size;y++)
    for(int x=0;x<size;x++) {
      double x2 = x-p[0];
      double y2 = y-p[1];
      x2 = x2*x2;
      y2 = y2*y2;
      double sx2 = 2.0*p[3]*p[3];
      double sy2 = 2.0*p[4]*p[4];
      //printf("%dx%d %f\n", );
      
      //possible lower border
      double lby2 = y - size+1;
      lby2 = lby2*lby2;
      double lb = exp(-lby2/sy2);
      lb = lb*(border[0]-p[5]);
      
      double uby2 = y;
      uby2 = uby2*uby2;
      double ub = exp(-uby2/sy2);
      ub = ub*(border[2]-p[5]);
      
      double rbx2 = x - size+1;
      rbx2 = rbx2*rbx2;
      double rb = exp(-rbx2/sx2);
      rb = rb*(border[1]-p[5]);
      
      double lbx2 = x;
      lbx2 = lbx2*lbx2;
      double leb = exp(-lbx2/sx2);
      leb = leb*(border[3]-p[5]);
      
      img.at<uchar>(y, x) = p[5] + lb+ub+rb+leb + p[2]*exp(-(x2/sx2+y2/sy2));
    }
}


void draw_gauss_border2(Mat &img, double *p, double *border)
{
  int size = img.size().width;
  
  img = Mat(size, size, CV_8U);
  
  for(int y=0;y<size;y++)
    for(int x=0;x<size;x++) {
      double x2 = x-p[0];
      double y2 = y-p[1];
      x2 = x2*x2;
      y2 = y2*y2;
      double sx2 = 2.0*p[3]*p[3];
      double sy2 = 2.0*p[4]*p[4];
      double bsx2 = 2.0*p[6]*p[6];
      double bsy2 = 2.0*p[7]*p[7];
      //printf("%dx%d %f\n", );
      
      //possible lower border
      double lby2 = y - size+1;
      lby2 = lby2*lby2;
      double lb = exp(-lby2/bsy2);
      lb = lb*(border[0]-p[5]);
      
      double uby2 = y;
      uby2 = uby2*uby2;
      double ub = exp(-uby2/bsy2);
      ub = ub*(border[2]-p[5]);
      
      double rbx2 = x - size+1;
      rbx2 = rbx2*rbx2;
      double rb = exp(-rbx2/bsx2);
      rb = rb*(border[1]-p[5]);
      
      double lbx2 = x;
      lbx2 = lbx2*lbx2;
      double leb = exp(-lbx2/bsx2);
      leb = leb*(border[3]-p[5]);
      
      img.at<uchar>(y, x) = p[5] + lb+ub+rb+leb + p[2]*exp(-(x2/sx2+y2/sy2));
    }
}


void draw_gauss_border3(Mat &img, double *p, double *border)
{
  int size = img.size().width;
  
  img = Mat(size, size, CV_8U);
  
  for(int y=0;y<size;y++)
    for(int x=0;x<size;x++) {
      double x2 = x-p[0];
      double y2 = y-p[1];
      x2 = x2*x2;
      y2 = y2*y2;
      double sx2 = 2.0*p[3]*p[3];
      double sy2 = 2.0*p[4]*p[4];
      double bsx2 = 2.0*p[6]*p[6];
      double bsy2 = 2.0*p[7]*p[7];
      //printf("%dx%d %f\n", );
      
      //possible lower border
      double lby2 = y - size+1;
      lby2 = lby2*lby2;
      double lb = exp(-lby2/bsy2);
      lb = lb*(border[0]-p[5]);
      
      double uby2 = y;
      uby2 = uby2*uby2;
      double ub = exp(-uby2/bsy2);
      ub = ub*(border[2]-p[5]);
      
      double rbx2 = x - size+1;
      rbx2 = rbx2*rbx2;
      double rb = exp(-rbx2/bsx2);
      rb = rb*(border[1]-p[5]);
      
      double lbx2 = x;
      lbx2 = lbx2*lbx2;
      double leb = exp(-lbx2/bsx2);
      leb = leb*(border[3]-p[5]);
      
      img.at<uchar>(y, x) = p[5] + lb+ub+rb+leb + (p[2]-p[5])*exp(-(x2/sx2+y2/sy2));
    }
}


void draw_gauss_border4(Mat &img, double *p, double *border, double *b_pos)
{
  int size = img.size().width;
  
  img = Mat(size, size, CV_8U);
  
  for(int y=0;y<size;y++)
    for(int x=0;x<size;x++) {
      double x2 = x-p[0];
      double y2 = y-p[1];
      x2 = x2*x2;
      y2 = y2*y2;
      double sx2 = 2.0*p[3]*p[3];
      double sy2 = 2.0*p[4]*p[4];
      double bsx2 = 2.0*p[6]*p[6];
      double bsy2 = 2.0*p[7]*p[7];
      //printf("%dx%d %f\n", );
      
      //possible lower border
      double lby2 = y - (size-1) - sin(b_pos[0])*(size-1)*0.2;
      lby2 = lby2*lby2;
      double lb = exp(-lby2/bsy2);
      lb = lb*(border[0]-p[5]);
      
      double uby2 = y - sin(b_pos[2])*(size-1)*0.2;
      uby2 = uby2*uby2;
      double ub = exp(-uby2/bsy2);
      ub = ub*(border[2]-p[5]);
      
      double rbx2 = x - (size-1) - sin(b_pos[1])*(size-1)*0.2;
      rbx2 = rbx2*rbx2;
      double rb = exp(-rbx2/bsx2);
      rb = rb*(border[1]-p[5]);
      
      double lbx2 = x - sin(b_pos[3])*(size-1)*0.2;
      lbx2 = lbx2*lbx2;
      double leb = exp(-lbx2/bsx2);
      leb = leb*(border[3]-p[5]);
      
      img.at<uchar>(y, x) = p[5] + lb+ub+rb+leb + (p[2]-p[5])*exp(-(x2/sx2+y2/sy2));
    }
}


void draw_border(Mat &img, double *p, double border[4])
{
  int size = img.size().width;
  
  img = Mat(size, size, CV_8U);
  
  for(int y=0;y<size;y++)
    for(int x=0;x<size;x++) {
      double x2 = x-p[0];
      double y2 = y-p[1];
      x2 = x2*x2;
      y2 = y2*y2;
      double sx2 = 2.0*p[3]*p[3];
      double sy2 = 2.0*p[4]*p[4];
      //printf("%dx%d %f\n", );
  
      //possible lower border
      double lby2 = y - size+1;
      lby2 = lby2*lby2;
      double lb = exp(-lby2/sy2);
      lb = lb*(border[0]-p[5]);
      
      double uby2 = y;
      uby2 = uby2*uby2;
      double ub = exp(-uby2/sy2);
      ub = ub*(border[2]-p[5]);
      
      double rbx2 = x - size+1;
      rbx2 = rbx2*rbx2;
      double rb = exp(-rbx2/sx2);
      rb = rb*(border[1]-p[5]);
      
      double lbx2 = x;
      lbx2 = lbx2*lbx2;
      double leb = exp(-lbx2/sx2);
      leb = leb*(border[3]-p[5]);
      
      img.at<uchar>(y, x) = clamp<double>(p[5] + lb+ub+rb+leb,0,255.0);
    }
}

int debug_counter = 0;

/**
 * Fit 2d gaussian to image, 5 parameter: \f$x_0\f$, \f$y_0\f$, amplitude, spread, background
 * disregards a border of \f$\lfloor \mathit{size}/5 \rfloor\f$ pixels
 */
double fit_gauss(Mat &img, double *params)
{
  int size = img.size().width;
  assert(img.size().height == size);
  int b = size/5;
  uint8_t *ptr = img.ptr<uchar>(0);
  
  assert(img.depth() == CV_8U);
  assert(img.channels() == 1);
  
  double params_cpy[6];
  
  //x,y
  params[0] = size*0.5;
  params[1] = size*0.5;
  
  int sum = 0;
  int x, y = b;
  for(x=b;x<size-b-1;x++)
    sum += ptr[y*size+x];
  y = size-b-1;
  for(x=b+1;x<size-b;x++)
    sum += ptr[y*size+x];
  x = b;
  for(y=b+1;y<size-b;y++)
    sum += ptr[y*size+x];
  x = size-b-1;
  for(y=b;y<size-b-1;y++)
    sum += ptr[y*size+x];
  
  //background
  params[5] = sum / (4*(size-2*b-1));
  
  //amplitude
  params[2] = ptr[size/2*(size+1)]-params[5];
  
  if (abs(params[2]) < min_fit_contrast) {
    /*printf("%d-%f = %f, %d\n", ptr[size/2*(size+1)],params[5], params[2], debug_counter);
    char buf[64];
    sprintf(buf, "lowc%07d.png", debug_counter);
    imwrite(buf, img);
    debug_counter++;*/
    
    return FLT_MAX;
  }
  
  //spread
  params[3] = size*0.2;
  params[4] = size*0.2;
  
  for(int i=0;i<6;i++)
    params_cpy[i] = params[i];

  
  ceres::Problem problem_gauss;
  for(y=0;y<size-0;y++)
    for(x=0;x<size-0;x++) {
      double x2 = x-size*0.5;
      double y2 = y-size*0.5;
      x2 = x2*x2;
      y2 = y2*y2;
      double s2 = size*0.25;
      s2=s2*s2;
      double s = exp(-x2/s2-y2/s2);
      ceres::CostFunction* cost_function = Gauss2dError::Create(ptr[y*size+x], x, y, s, size*size*0.25);
      problem_gauss.AddResidualBlock(cost_function, NULL, params);
    }
  
  ceres::Solver::Options options;
  options.max_num_iterations = 1000;
  //options.linear_solver_type = ceres::DENSE_SCHUR;
  //options.minimizer_progress_to_stdout = true;

  /*options.num_threads = 2;
  options.parameter_tolerance = 1e-20;
  options.gradient_tolerance = 1e-20;
  options.function_tolerance = 1e-20;*/
  
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem_gauss, &summary);
  
  Point2f v(params[0]-size*0.5, params[1]-size*0.5);
  
  //if (false) {
  if (summary.termination_type != ceres::CONVERGENCE || norm(v)/subfit_oversampling > max_accept_dist) {
    for(int i=0;i<6;i++)
      params[i] = params_cpy[i];
    
    double border[5];
    //angle
    border[0] = 0.0;
    //distance
    border[1] = 0.0;
    //soread
    border[2] = size*0.5;
    border[3] = params[2] + params[5];
    
    params[2] = params[2] + params[5];
    
    ceres::Problem problem_gauss_border5;
    for(y=0;y<size-0;y++)
      for(x=0;x<size-0;x++) {
        double x2 = x-size*0.5;
        double y2 = y-size*0.5;
        x2 = x2*x2;
        y2 = y2*y2;
        double s2 = size*0.5;
        s2=s2*s2;
        double s = exp(-x2/s2-y2/s2);
        ceres::CostFunction* cost_function = GaussBorder2dError5::Create(ptr[y*size+x], x, y, s, size*size, size);
        problem_gauss_border5.AddResidualBlock(cost_function, NULL, params, border);
      }
      
    ceres::Solve(options, &problem_gauss_border5, &summary);
    
    //printf("%f %f -> ", params[0], params[1]);
    /*ceres::Solver::Summary summary2;
    
    for(int i=0;i<6;i++)
      params[i] = params_cpy[i];
    
    int b3 = size/3;
    int size_b = size-b3-b3;
    
    double border[4];
    
    double sum = 0;
    int y=size-1;
    int x;
    for(x=b3;x<size-b3;x++)
      sum += ptr[y*size+x];
    border[0] = sum/size_b;
    
    sum = 0;
    x=size-1;
    for(y=b3;y<size-b3;y++)
      sum += ptr[y*size+x];
    border[1] = sum/size_b;
    
    sum = 0;
    y=0;
    for(x=b3;x<size-b3;x++)
      sum += ptr[y*size+x];
    border[2] = sum/size_b;
    
    sum = 0;
    x=0;
    for(y=b3;y<size-b3;y++)
      sum += ptr[y*size+x];
    border[3] = sum/size_b;*/
    
    /*if (params[2] > 0)
      for(int i=0;i<4;i++)
        border[i] = max(border[i], params[5]);
    else
      for(int i=0;i<4;i++)
        border[i] = min(border[i], params[5]);*/
      
    //double params_common;
    //double params_border;
      
    /*ceres::Problem problem_border;
    for(y=0;y<size-0;y++)
      for(x=0;x<size-0;x++) {
        double x2 = x-size*0.5;
        double y2 = y-size*0.5;
        x2 = x2*x2;
        y2 = y2*y2;
        double s2 = size;
        s2=s2*s2;
        double s = 1.0-exp(-x2/s2-y2/s2);
        ceres::CostFunction* cost_function = Border2dError::Create(ptr[y*size+x], x, y, s, size-1, border);
        problem_border.AddResidualBlock(cost_function, NULL, params+3);
      }
    ceres::Solve(options, &problem_border, &summary2);
    //std::cout << summary2.FullReport() << "\n";*/
    
    /*char buf[64];
    sprintf(buf, "point%07d.png", debug_counter);
    imwrite(buf, img);
    
    Mat paint = img.clone();*/
    /*draw_border(paint, params, border);
    sprintf(buf, "point%07d_fit.png", debug_counter);
    imwrite(buf, paint);*/
    
    //spread
   /* params[3] = size*0.1;
    params[4] = size*0.1;
    //printf("start amplitude %f\n", params[2]);
    
    ceres::Problem problem_gauss_border;
    for(y=0;y<size-0;y++)
      for(x=0;x<size-0;x++) {
        double x2 = x-size*0.5;
        double y2 = y-size*0.5;
        x2 = x2*x2;
        y2 = y2*y2;
        double s2 = size;
        s2=s2*s2;
        double s = exp(-x2/s2-y2/s2);
        ceres::CostFunction* cost_function = GaussBorder2dError::Create(ptr[y*size+x], x, y, s, size*size*0.25, size-1, border);
        problem_gauss_border.AddResidualBlock(cost_function, NULL, params);
      }
    ceres::Solve(options, &problem_gauss_border, &summary2);*/
    //std::cout << summary2.FullReport() << "\n";
    
    /*draw_gauss_border(paint, params, border);
    sprintf(buf, "point%07d_fitg.png", debug_counter);
    imwrite(buf, paint);*/
    
    //spread
    /*params[3] = size*0.1;
    params[4] = size*0.1;
    params[6] = params[3];
    params[7] = params[4];
    
    ceres::Problem problem_gauss_border2;
    for(y=0;y<size-0;y++)
      for(x=0;x<size-0;x++) {
        double x2 = x-size*0.5;
        double y2 = y-size*0.5;
        x2 = x2*x2;
        y2 = y2*y2;
        double s2 = size*0.5;
        s2=s2*s2;
        double s = exp(-x2/s2-y2/s2);
        ceres::CostFunction* cost_function = GaussBorder2dError2::Create(ptr[y*size+x], x, y, s, size*size, size-1, border);
        problem_gauss_border2.AddResidualBlock(cost_function, NULL, params);
      }
    ceres::Solve(options, &problem_gauss_border2, &summary2);
    //std::cout << summary2.FullReport() << "\n";*/
    
    //printf("%f %f %f (%d) b:%f a:%f\n", params[3], params[4], params[6],params[7],debug_counter, params[5], params[2]);
    //summary = summary2;
    
    /*draw_gauss_border2(paint, params, border);
    sprintf(buf, "point%07d_fitg2.png", debug_counter);
    imwrite(buf, paint);*/
    
    //fit border color!
    /*double border_col[8];
    for(int i=0;i<4;i++)
      border_col[i] = border[i];
    
    //make amplitude absolute
    params[2] = params[2]+params[5];*/
    
    //reset spread
    /*params[3] = size*0.1;
    params[4] = size*0.1;
    params[6] = params[3];
    params[7] = params[4];
    
    ceres::Problem problem_gauss_border3;
    for(y=0;y<size-0;y++)
      for(x=0;x<size-0;x++) {
        double x2 = x-size*0.5;
        double y2 = y-size*0.5;
        x2 = x2*x2;
        y2 = y2*y2;
        double s2 = size*0.5;
        s2=s2*s2;
        double s = exp(-x2/s2-y2/s2);
        ceres::CostFunction* cost_function = GaussBorder2dError3::Create(ptr[y*size+x], x, y, s, size*size, size-1);
        problem_gauss_border3.AddResidualBlock(cost_function, NULL, params, border_col);
      }
      
    ceres::Solve(options, &problem_gauss_border3, &summary2);
    //std::cout << summary2.FullReport() << "\n";
    
    draw_gauss_border3(paint, params, border_col);
    sprintf(buf, "point%07d_fitg3.png", debug_counter);
    imwrite(buf, paint);*/
    
    //reset spread
    /*params[3] = size*0.1;
    params[4] = size*0.1;
    params[6] = params[3];
    params[7] = params[4];
    
    double border_pos[4];
    for(int i=0;i<4;i++)
      border_pos[i] = 0;
    
    ceres::Problem problem_gauss_border4;
    for(y=0;y<size-0;y++)
      for(x=0;x<size-0;x++) {
        double x2 = x-size*0.5;
        double y2 = y-size*0.5;
        x2 = x2*x2;
        y2 = y2*y2;
        double s2 = size*0.5;
        s2=s2*s2;
        double s = exp(-x2/s2-y2/s2);
        ceres::CostFunction* cost_function = GaussBorder2dError4::Create(ptr[y*size+x], x, y, s, size*size, size-1);
        problem_gauss_border4.AddResidualBlock(cost_function, NULL, params, border_col, border_pos);
      }
      
    ceres::Solve(options, &problem_gauss_border4, &summary2);*/
    //std::cout << summary2.FullReport() << "\n";
    
    //printf("%f %f %f %f\n", border_pos[0], border_pos[1], border_pos[2], border_pos[3]);
    
    /*draw_gauss_border4(paint, params, border_col, border_pos);
    sprintf(buf, "point%07d_fitg4.png", debug_counter);
    imwrite(buf, paint);*/
    
    //summary = summary2;
    
    //debug_counter++;
  }
  
  //std::cout << summary.FullReport() << "\n";
  
  v = Point2f(params[0]-size*0.5, params[1]-size*0.5);
  
  if (summary.termination_type == ceres::CONVERGENCE && norm(v)/subfit_oversampling <= max_accept_dist /*params[0] > 0 && params[0] < size-0-1 && params[1] > 0 && params[1] < size-0-1*/) {
    //printf("%f %f (%d)- %f %f\n", params[0], params[1], size, params[3], params[4]);
    return sqrt(summary.final_cost/problem_gauss.NumResiduals());
  }
  else {
    //std::cout << summary.FullReport() << "\n";
    //printf("%f x %f\n", params[0], params[1]);
    //printf("%f\n",sqrt(summary.final_cost/problem_gauss.NumResiduals()));
    //printf("%f %f (%d)- %f %f, %d\n", params[0], params[1], size, params[3], params[4], debug_counter);
    
    /*char buf[64];
    sprintf(buf, "point%07d.png", debug_counter);
    imwrite(buf, img);
    draw_gauss(img, params);
    sprintf(buf, "point%07d_fit.png", debug_counter);
    imwrite(buf, img);
    debug_counter++;*/
    
    return FLT_MAX;
  }
}

void detect_sub_corners(Mat &img, vector<Corner> corners, vector<Corner> &corners_out, Mat &paint)
{
  int counter = 0;
  vector<Point2f> ipoints(4);
  sort(corners.begin(), corners.end(), corner_cmp);
  
  
  for(int i=0;i<corners.size();i++) {
    printprogress(i, corners.size(), counter, " %d subs", corners_out.size());
    Corner c = corners[i];
    int size;
    
    ipoints[0] = corners[i].p;
    if (corner_find_off_save(corners, c, 1, 0, ipoints[1])) continue;
    if (corner_find_off_save(corners, c, 1, 1, ipoints[2])) continue;
    if (corner_find_off_save(corners, c, 0, 1, ipoints[3])) continue;
    
    float maxlen = 0;
    for(int i=0;i<4;i++) {
      Point2f v = ipoints[i]-ipoints[(i+1)%4];
      float len = v.x*v.x+v.y*v.y;
      if (len > maxlen)
        maxlen = len;
      v = ipoints[(i+3)%4]-ipoints[i];
      len = v.x*v.x+v.y*v.y;
      if (len > maxlen)
        maxlen = len;
    }
    maxlen = sqrt(maxlen);
    //printf("longest side %f\n", maxlen);
    
    if (maxlen < 5*4)
      continue;
    
    size = maxlen*subfit_oversampling/5;
    
#pragma omp parallel for
    for(int y=0;y<5;y++)
      for(int x=0;x<5;x++) {
        vector<Point2f> cpoints(4);
        cpoints[0] = Point2f(-x*size,-y*size);
        cpoints[1] = Point2f((5-x)*size,-y*size);
        cpoints[2] = Point2f((5-x)*size,(5-y)*size);
        cpoints[3] = Point2f(-x*size,(5-y)*size);
        
        Mat proj = Mat(size, size, CV_8U);
        Mat pers = getPerspectiveTransform(ipoints, cpoints);
        Mat pers_i = getPerspectiveTransform(cpoints, ipoints);
        warpPerspective(img, proj, pers, Size(size, size), INTER_LINEAR);
        Mat oned;
        resize(proj, oned, Size(size, 1), INTER_AREA);
        
        //char buf[64];
        //sprintf(buf, "point%07d_%0d_%d.png", i, x, y);
        //imwrite(buf, proj);
        double params[14];
        double rms = fit_gauss(proj, params);
        if (rms >= 100.0)
          continue;
        //draw_gauss(proj, params);
        //sprintf(buf, "point%07d_%0d_%d_fit.png", i, x, y);
        //imwrite(buf, proj);
        //printf("refined position for %d %d %d: %fx%f, rms %f a %f\n", i, x, y, params[0], params[1], rms, params[2]);
        vector<Point2f> coords(1);
        coords[0] = Point2f(params[0], params[1]);
        vector<Point2f> res(1);
        perspectiveTransform(coords, res, pers_i);
        circle(paint, Point2i(res[0].x*16.0, res[0].y*16.0), 1, Scalar(0,255,0,0), 2, CV_AA, 4);
        
        Corner c_o(res[0], Point2i(c.id.x*10+2*x+1, c.id.y*10+2*y+1), 0);
        //printf("found corner %f!\n", rms);
#pragma omp critical
        corners_out.push_back(c_o);
      }

  }
  printf("\n");
}


void detect_sub_corners2(Mat &img, vector<Corner> corners, vector<Corner> &corners_out, Mat &paint)
{
  int counter = 0;  
  vector<Point2f> ipoints(4);
  sort(corners.begin(), corners.end(), corner_cmp);
  
  for(int i=0;i<corners.size();i++) {
    printprogress(i, corners.size(), counter, " %d subs", corners_out.size());
    //printf("."); fflush(NULL);
    Corner c = corners[i];
    int size;
    
    ipoints[0] = corners[i].p;
    if (corner_find_off_save(corners, c, 2, 0, ipoints[1])) continue;
    if (corner_find_off_save(corners, c, 2, 2, ipoints[2])) continue;
    if (corner_find_off_save(corners, c, 0, 2, ipoints[3])) continue;
    
    float maxlen = 0;
    for(int i=0;i<4;i++) {
      Point2f v = ipoints[i]-ipoints[(i+1)%4];
      float len = v.x*v.x+v.y*v.y;
      if (len > maxlen)
        maxlen = len;
      v = ipoints[(i+3)%4]-ipoints[i];
      len = v.x*v.x+v.y*v.y;
      if (len > maxlen)
        maxlen = len;
    }
    maxlen = sqrt(maxlen);
    
    if (maxlen < 5*4)
      continue;
    
    size = maxlen*subfit_oversampling/5;
    
#pragma omp parallel for
    for(int y=0;y<5;y++)
      for(int x=0;x<5;x++) {
        vector<Point2f> cpoints(4);
        cpoints[0] = Point2f((0.5-x)*size,(0.5-y)*size);
        cpoints[1] = Point2f((5.5-x )*size,(0.5-y)*size);
        cpoints[2] = Point2f((5.5-x )*size,(5.5-y )*size);
        cpoints[3] = Point2f((0.5-x)*size,(5.5-y )*size);
        
        Mat proj = Mat(size, size, CV_8U);
        Mat pers = getPerspectiveTransform(ipoints, cpoints);
        Mat pers_i = getPerspectiveTransform(cpoints, ipoints);
        warpPerspective(img, proj, pers, Size(size, size), INTER_LINEAR);
        Mat oned;
        resize(proj, oned, Size(size, 1), INTER_AREA);
        
        //char buf[64];
        //sprintf(buf, "point%07d_%0d_%d.png", i, x, y);
        //imwrite(buf, proj);
        double params[14];
        //printf("debug id %d = output point %d ", debug_counter, corners_out.size());
        double rms = fit_gauss(proj, params);
        //printf("rms %f\n", rms);
        if (rms >= rms_use_limit)
          continue;
        //draw_gauss(proj, params);
        //sprintf(buf, "point%07d_%0d_%d_fit.png", i, x, y);
        //imwrite(buf, proj);
        //printf("refined position for %d %d %d: %fx%f, rms %f a %f\n", i, x, y, params[0], params[1], rms, params[2]);
        vector<Point2f> coords(1);
        coords[0] = Point2f(params[0], params[1]);
        vector<Point2f> res(1);
        perspectiveTransform(coords, res, pers_i);
        circle(paint, Point2i(res[0].x*16.0, res[0].y*16.0), 1, Scalar(0,255,0,0), 2, CV_AA, 4);
        
        
        Corner c_o(res[0], Point2i(c.id.x*5+2*x, c.id.y*5+2*y), 0);
        
        
        /*if (!x && y < 2) {
          char buf[64];
          sprintf(buf, "%d/%d", c_o.id.x, c_o.id.y);
          circle(paint, c_o.p, 1, Scalar(0,0,0,0), 2);
          circle(paint, c_o.p, 1, Scalar(0,255,0,0));
          putText(paint, buf, c_o.p, FONT_HERSHEY_PLAIN, 0.5, Scalar(0,0,0,0), 2, CV_AA);
          putText(paint, buf, c_o.p, FONT_HERSHEY_PLAIN, 0.5, Scalar(255,255,255,0), 1, CV_AA);
        }*/
        //printf("found corner %f!\n", rms);
#pragma omp critical
        corners_out.push_back(c_o);
      }

  }
  printf("\n");
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
  
  if (argc != 3 && argc != 4)
    usage(argv[0]);
  
  if (demosaic) {
    img = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    paint = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    cvtColor(img, img, COLOR_BayerRG2BGR);
    cvtColor(paint, paint, COLOR_BayerRG2BGR);
  }
  else {
    img = cv::imread(argv[1]);
    paint = cv::imread(argv[1]);
  }  
  
  //corrupt(img);
  //imwrite("corrupted.png", img);
  Marker::init();
  
  microbench_measure_output("app startup");
  //CALLGRIND_START_INSTRUMENTATION;
  if (argc == 4)
    Marker::detect(img, corners,use_rgb,0,0,atof(argv[3]),100);
  else
    Marker::detect(img, corners,use_rgb,0,0,0.5);
  //CALLGRIND_STOP_INSTRUMENTATION;
    
  microbench_init();
  
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
  
  vector<Corner> corners_f;
  check_calibration(corners, img.size().width, img.size().height, img, corners_f);
  //check_precision(corners, img.size().width, img.size().height, img, argv[3]);
  
  Mat gray;
  if (img.channels() != 1)
    cvtColor(img, gray, CV_BGR2GRAY);
  else
    gray = img;
  
  vector<Corner> corners_sub; 
  detect_sub_corners(gray , corners_f, corners_sub, paint);
  
  vector<Corner> corners_f2;
  check_calibration(corners_sub, img.size().width, img.size().height, img, corners_f2);
  
  vector<Corner> corners_sub2; 
  detect_sub_corners2(gray , corners_f2, corners_sub2, paint);
  
  vector<Corner> corners_f3;
  check_calibration(corners_sub2, img.size().width, img.size().height, img, corners_f3);
  
  imwrite(argv[2], paint);
  

  microbench_measure_output("app finish");
  return EXIT_SUCCESS;
}