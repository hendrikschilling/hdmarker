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

const int subfit_oversampling = 2;
const int subfit_refine_subpix = 100;

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

bool calib_savepoints(vector<vector<Point2f> > all_img_points[4], vector<vector<Point3f> > &all_world_points, vector<Corner> corners, int hpages, int vpages)
{
  if (!corners.size()) return false;
  
  vector<Point2f> plane_points;
  vector<Point2f> img_points[4];
  vector<Point3f> world_points;
  vector<Point2f> img_points_check[4];
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
    img_points_check[0].push_back(corners[i].p);
    for(int c=1;c<4;c++)
      if (all_img_points[c].size())
        img_points_check[c].push_back(corners[i].pc[c-1]);
    world_points_check.push_back(grid_to_world(corners[i], hpages));
  }
  
  inliers.resize(world_points_check.size());
  findHomography(world_points_check, img_points_check[0], CV_RANSAC, 200, inliers);
  
  for(uint i=0;i<inliers.size();i++) {
    if (!inliers[i])
      continue;
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
    line(paint, img_points[0][i], img_points[0][i]+100*d, Scalar(0,0,255));
  }
  imwrite("off_hdm.png", paint);
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
  vector<vector<Point2f>> img_points[4];
  
  img_points[0].resize(1);
  if (use_rgb)
    for(int c=1;c<4;c++)
      img_points[c].resize(1);
    
  printf("corners: %d\n", corners.size());
    
  if (!calib_savepoints(img_points, world_points, corners, grid_width, grid_height)) {
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
  Gauss2dError(int val, int x, int y)
      : val_(val), x_(x), y_(y) {}

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
    //T d = sqrt(x2+y2+T(1.0));
    //non-weighted leads to better overall estimates?
    residuals[0] = (T(val_) - p[5] - p[2]*exp(-(x2/sx2+y2/sy2)));
    
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(int val, int x, int y) {
    return (new ceres::AutoDiffCostFunction<Gauss2dError, 1, 6>(
                new Gauss2dError(val, x, y)));
  }

  int x_, y_, val_;
};

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
  
  //spread
  params[3] = size/5;
  params[4] = size/5;

  
  ceres::Problem problem_gauss;
  for(y=b;y<size-b;y++)
    for(x=b;x<size-b;x++) {
      ceres::CostFunction* cost_function = Gauss2dError::Create(ptr[y*size+x], x, y);
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
  
  //std::cout << summary.FullReport() << "\n";
  
  if (summary.termination_type == ceres::CONVERGENCE && params[0] > b && params[0] < size-b-1 && params[1] > b && params[1] < size-b-1) 
    return sqrt(summary.final_cost/problem_gauss.NumResiduals());
  else {
    //std::cout << summary.FullReport() << "\n";
    //printf("%f\n",sqrt(summary.final_cost/problem_gauss.NumResiduals()));
    return FLT_MAX;
  }
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

void detect_sub_corners(Mat &img, vector<Corner> corners, vector<Corner> &corners_out, Mat &paint)
{
  sort(corners.begin(), corners.end(), corner_cmp);
  
  vector<Point2f> ipoints(4);
  vector<Point2f> cpoints(4);
  
  for(int i=0;i<corners.size();i++) {
    printf("."); fflush(NULL);
    Mat proj;
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
    
    if (maxlen < 5*5)
      continue;
    
    size = maxlen*subfit_oversampling/5;
    
#pragma omp parallel for
    for(int y=0;y<5;y++)
      for(int x=0;x<5;x++) {
        cpoints[0] = Point2f(-x*size,-y*size);
        cpoints[1] = Point2f((5-x)*size,-y*size);
        cpoints[2] = Point2f((5-x)*size,(5-y)*size);
        cpoints[3] = Point2f(-x*size,(5-y)*size);
        
        proj = Mat(size, size, CV_8U);
        Mat pers = getPerspectiveTransform(ipoints, cpoints);
        Mat pers_i = getPerspectiveTransform(cpoints, ipoints);
        warpPerspective(img, proj, pers, Size(size, size), INTER_LINEAR);
        Mat oned;
        resize(proj, oned, Size(size, 1), INTER_AREA);
        
        //char buf[64];
        //sprintf(buf, "point%07d_%0d_%d.png", i, x, y);
        //imwrite(buf, proj);
        double params[6];
        double rms = fit_gauss(proj, params);
        if (rms >= 5.0)
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
  
  img = cv::imread(argv[1]);
  paint = cv::imread(argv[1]);
  //corrupt(img);
  imwrite("corrupted.png", img);
  Marker::init();
  
  //FIXME hack - remove detection problems
  Mat pre;
  GaussianBlur(img, pre, Size(5,5), 0);
  
  microbench_measure_output("app startup");
  //CALLGRIND_START_INSTRUMENTATION;
  if (argc == 4)
    Marker::detect(pre, corners,use_rgb,0,0,atof(argv[3]),100);
  else
    Marker::detect(pre, corners,use_rgb,0,0,0.5);
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
  
  Mat gray;
  if (img.channels() != 1)
    cvtColor(img, gray, CV_BGR2GRAY);
  else
    gray = img;
  
  vector<Corner> corners_sub; 
  detect_sub_corners(gray , corners, corners_sub, paint);
  
  check_calibration(corners_sub, img.size().width, img.size().height, img);
  
  imwrite(argv[2], paint);
  

  microbench_measure_output("app finish");
  return EXIT_SUCCESS;
}