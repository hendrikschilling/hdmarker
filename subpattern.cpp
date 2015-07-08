#include "subpattern.hpp"

#include <stdio.h>

#include "hdmarker.hpp"
#include "timebench.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "ceres/ceres.h"

#include <iostream>
#include <unordered_map>

using namespace std;
using namespace cv;

static const float subfit_oversampling = 2.0;
static const int subfit_max_size = 30;
static const int subfit_min_size = 1;
static const float min_fit_contrast = -1.0;
static const float min_fitted_contrast = 5.0; //minimum amplitude of fitted gaussian
static const float rms_use_limit = 100.0;
static const float recurse_min_len = 4.0;
static const int int_search_range = 11;
static const int int_extend_range = 11;
static const double subfit_max_range = 0.2;
static const double fit_gauss_max_tilt = 0.1;
static double max_accept_dist = 3.0;

#include <stdarg.h>

static void printprogress(int curr, int max, int &last, const char *fmt = NULL, ...)
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

static bool corner_cmp(Corner a, Corner b)
{
  if (a.id.y != b.id.y)
    return a.id.y < b.id.y;
  else
    return a.id.x < b.id.x;
}

static bool corner_find_off_save(vector<Corner> &corners, Corner ref, int x, int y, Point2f &out)
{
  std::vector<Corner>::iterator found;
  ref.id.x += x;
  ref.id.y += y;
  found = lower_bound(corners.begin(), corners.end(), ref, corner_cmp);
  if (found == corners.end() || found->id != ref.id)
    return true;
  out = found->p;
  return false;
}

int sign(int x) 
{
  if (x > 0) return 1;
  if (x < 0) return -1;
  return 0;
}

static bool corner_find_off_save_int(vector<Corner> &corners, Corner ref, int x, int y, Point2f &out, int range)
{
  Corner search = ref;
  std::vector<Corner>::iterator found;
  search.id.x += x;
  search.id.y += y;
  found = lower_bound(corners.begin(), corners.end(), search, corner_cmp);
  
  if (found != corners.end() && found->id == search.id) {
    out = found->p;
    return false;
  }
  
  int r_l, r_h, r_diff = 1000000000;
  Point2f l, h;
  
  bool succ1 = false;
  for(int r=1;r<=range;r++)
    if (!corner_find_off_save(corners, ref, x*(1+r), y, h)) {
      r_h = r;
      for(int r=1;r<=range;r++)
        if (!corner_find_off_save(corners, ref, x*(1-r), y, l)) {
          r_l = r;
          r_diff = r_h+r_l;
          succ1 = true;
          goto found1;
        }
    }
  found1 :
  
  if (succ1)
    out = l + (h-l)*(1.0/(r_h+r_l))*r_l;
  
  bool succ2 = false;
  for(int r=1;r<=range;r++)
    if (!corner_find_off_save(corners, ref, x, y*(1+r), h)) {
      r_h = r;
      for(int r=1;r<=range;r++)
        if (!corner_find_off_save(corners, ref, x, y*(1-r), l)) {
          r_l = r;
          if (r_h+r_l > r_diff)
            return false;
          else {
            succ2 = true;
            goto found2;
          }
        }
    }
  found2 :
  
  if (succ2) {
    out = l + (h-l)*(1.0/(r_h+r_l))*r_l;
    return false;
  }
  else if (succ1)
    return false;
  
  return true;
}

struct Gauss2dError {
  Gauss2dError(int val, int x, int y, double m, double sw)
      : val_(val), x_(x), y_(y), m_(m), sw_(sw) {}

/**
 * used function: 
 */
  template <typename T>
  bool operator()(const T* const p,
                  T* residuals) const {
    T x2 = T(x_) - (T(m_)+sin(p[0])*T(m_*2.0*subfit_max_range));
    T y2 = T(y_) - (T(m_)+sin(p[1])*T(m_*2.0*subfit_max_range));
    T sx2 = T(2.0)*p[3]*p[3];
    T sy2 = T(2.0)*p[4]*p[4];
    x2 = x2*x2;
    y2 = y2*y2;

    residuals[0] = (T(val_) - (p[5] + (p[2]-p[5])*exp(-(x2/sx2+y2/sy2))))*T(sw_);
    
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(int val, int x, int y, double m, double sw) {
    return (new ceres::AutoDiffCostFunction<Gauss2dError, 1, 6>(
                new Gauss2dError(val, x, y, m, sw)));
  }

  int x_, y_, val_;
  double sw_, m_;
};


struct Gauss2dDirectError {
  Gauss2dDirectError(int val, int x, int y, double w, double h, double px, double py, double sw)
      : val_(val), x_(x), y_(y), w_(w), h_(h), px_(px), py_(py), sw_(sw){}

/**
 * used function: 
 */
  template <typename T>
  bool operator()(const T* const p,
                  T* residuals) const {
    T x2 = T(x_) - (T(px_)+sin(p[0])*T(w_*subfit_max_range));
    T y2 = T(y_) - (T(py_)+sin(p[1])*T(h_*subfit_max_range));
    T sx2 = T(2.0)*p[3]*p[3];
    T sy2 = T(2.0)*p[4]*p[4];
    x2 = x2*x2;
    y2 = y2*y2;
    
    residuals[0] = (T(val_) - (p[5] + (p[2]-p[5])*exp(-(x2/sx2+y2/sy2))))*T(sw_);
    
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(int val, int x, int y, double w, double h, double px, double py, double sw) {
    return (new ceres::AutoDiffCostFunction<Gauss2dDirectError, 1, 6>(
                new Gauss2dDirectError(val, x, y, w, h, px, py, sw)));
  }

  int x_, y_, val_;
  double w_, sw_, px_, py_, h_;
};


struct Gauss2dPlaneDirectError {
  Gauss2dPlaneDirectError(int val, int x, int y, double w, double h, double px, double py)
      : val_(val), x_(x), y_(y), w_(w), h_(h), px_(px), py_(py){}

/**
 * used function: 
 */
  template <typename T>
  bool operator()(const T* const p,
                  T* residuals) const {
    T x2 = T(x_) - (T(px_)+sin(p[0])*T(w_*subfit_max_range));
    T y2 = T(y_) - (T(py_)+sin(p[1])*T(h_*subfit_max_range));
    T dx = T(x_) - T(px_);
    T dy = T(y_) - T(py_);
    T sx2 = T(2.0)*p[3]*p[3];
    T sy2 = T(2.0)*p[4]*p[4];
    x2 = x2*x2;
    y2 = y2*y2;

    residuals[0] = (T(val_) - (p[5] + p[6]*dx + p[7]*dy + (p[2]-p[5])*exp(-(x2/sx2+y2/sy2))));
    
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(int val, int x, int y, double w, double h, double px, double py) {
    return (new ceres::AutoDiffCostFunction<Gauss2dPlaneDirectError, 1, 8>(
                new Gauss2dPlaneDirectError(val, x, y, w, h, px, py)));
  }

  int x_, y_, val_;
  double w_, px_, py_, h_;
};



struct Gauss2dCenterError {
  Gauss2dCenterError(int val, int x, int y, double m, double sw)
      : val_(val), x_(x), y_(y), m_(m), sw_(sw) {}

/**
 * used function: 
 */
  template <typename T>
  bool operator()(const T* const p,
                  T* residuals) const {
    T x2 = T(x_) - T(m_);
    T y2 = T(y_) - T(m_);
    T sx2 = T(2.0)*p[1]*p[1];
    T sy2 = T(2.0)*p[2]*p[2];
    x2 = x2*x2;
    y2 = y2*y2;

    residuals[0] = (T(val_) - (p[3] + (p[0]-p[3])*exp(-(x2/sx2+y2/sy2))))*T(sw_);
    
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(int val, int x, int y, double m, double sw) {
    return (new ceres::AutoDiffCostFunction<Gauss2dCenterError, 1, 4>(
                new Gauss2dCenterError(val, x, y, m, sw)));
  }

  int x_, y_, val_;
  double sw_, m_;
};


struct Gauss2dDirectCenterError {
  Gauss2dDirectCenterError(int val, int x, int y, double px, double py, double sw)
      : val_(val), x_(x), y_(y), px_(px), py_(py), sw_(sw) {}

/**
 * used function: 
 */
  template <typename T>
  bool operator()(const T* const p,
                  T* residuals) const {
    T x2 = T(x_) - T(px_);
    T y2 = T(y_) - T(py_);
    T sx2 = T(2.0)*p[1]*p[1];
    T sy2 = T(2.0)*p[2]*p[2];
    x2 = x2*x2;
    y2 = y2*y2;

    residuals[0] = (T(val_) - (p[3] + (p[0]-p[3])*exp(-(x2/sx2+y2/sy2))))*T(sw_);
    
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(int val, int x, int y, double px, double py, double sw) {
    return (new ceres::AutoDiffCostFunction<Gauss2dDirectCenterError, 1, 4>(
                new Gauss2dDirectCenterError(val, x, y, px, py, sw)));
  }

  int x_, y_, val_;
  double sw_, px_, py_;
};

template <typename T> inline T clamp(const T& n, const T& lower, const T& upper)
{
  return std::max<T>(lower, std::min<T>(n, upper));
}

static void draw_gauss2d(Mat &img, double *p)
{
  uint8_t *ptr = img.ptr<uchar>(0);
  double m = img.size().width * 0.5;
  
  for(int y=0;y<img.size().height;y++)
    for(int x=0;x<img.size().width;x++) {
      double x2 = x - p[0];
      double y2 = y - p[1];
      double sx2 = 2.0*p[3]*p[3];
      double sy2 = 2.0*p[4]*p[4];
      x2 = x2*x2;
      y2 = y2*y2;

      ptr[y*img.size().width+x] = clamp<int>(p[5] + (p[2]-p[5])*exp(-(x2/sx2+y2/sy2)), 0, 255);
    }
}


static void draw_gauss2d_direct(Mat &img, Point2f c, Point2f res, Point2i size, double *p)
{
  uint8_t *ptr = img.ptr<uchar>(0);
  int w = img.size().width;
  
  for(int y=c.y-size.y/2;y<=c.y+size.y/2;y++)
    for(int x=c.x-size.x/2;x<=c.x+size.x/2;x++) {
      double x2 = x - res.x;
      double y2 = y - res.y;
      double sx2 = 2.0*p[3]*p[3];
      double sy2 = 2.0*p[4]*p[4];
      x2 = x2*x2;
      y2 = y2*y2;
      ptr[y*w+x] = clamp<int>(p[5] + (p[2]-p[5])*exp(-(x2/sx2+y2/sy2)), 0, 255);
    }
}


static void draw_gauss2d_plane_direct(Mat &img, Point2f c, Point2f res, Point2i size, double *p)
{
  uint8_t *ptr = img.ptr<uchar>(0);
  int w = img.size().width;
  
  for(int y=c.y-size.y/2;y<=c.y+size.y/2;y++)
    for(int x=c.x-size.x/2;x<=c.x+size.x/2;x++) {
      double x2 = x - res.x;
      double y2 = y - res.y;
      double dx = x - c.x;
      double dy = y - c.y;
      double sx2 = 2.0*p[3]*p[3];
      double sy2 = 2.0*p[4]*p[4];
      x2 = x2*x2;
      y2 = y2*y2;
      ptr[y*w+x] = clamp<int>(p[5] + p[6]*dx + p[7]*dy + (p[2]-p[5])*exp(-(x2/sx2+y2/sy2)), 0, 255);
    }
}

/**
 * Fit 2d gaussian to image, 5 parameter: \f$x_0\f$, \f$y_0\f$, amplitude, spread, background
 * disregards a border of \f$\lfloor \mathit{size}/5 \rfloor\f$ pixels
 */
static double fit_gauss(Mat &img, double *params)
{
  int size = img.size().width;
  assert(img.size().height == size);
  int b = size/5;
  uint8_t *ptr = img.ptr<uchar>(0);
  
  assert(img.depth() == CV_8U);
  assert(img.channels() == 1);
  
  double params_cpy[6];
  
  //x,y
  params[0] = 0.0;
  params[1] = 0.0;
  
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
  params[2] = ptr[size/2*(size+1)];
  
  if (abs(params[2]-params[5]) < min_fit_contrast)
    return FLT_MAX;
  
  //spread
  params[3] = size*0.2;
  params[4] = size*0.2;
  
  for(int i=0;i<6;i++)
    params_cpy[i] = params[i];

  
  ceres::Problem problem_gauss_center;
  for(y=0;y<size-0;y++)
    for(x=0;x<size-0;x++) {
      double x2 = x-size*0.5;
      double y2 = y-size*0.5;
      x2 = x2*x2;
      y2 = y2*y2;
      double s2 = size*0.5;
      s2=s2*s2;
      double sw = exp(-x2/s2-y2/s2);
      ceres::CostFunction* cost_function = Gauss2dCenterError::Create(ptr[y*size+x], x, y, size*0.5, sw);
      problem_gauss_center.AddResidualBlock(cost_function, NULL, params+2);
    }
  
  ceres::Solver::Options options;
  options.max_num_iterations = 200;
  options.linear_solver_type = ceres::DENSE_QR;
  
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem_gauss_center, &summary);
  
  ceres::Problem problem_gauss;
  for(y=0;y<size-0;y++)
    for(x=0;x<size-0;x++) {
      double x2 = x-size*0.5;
      double y2 = y-size*0.5;
      x2 = x2*x2;
      y2 = y2*y2;
      double s2 = size*0.25;
      s2=s2*s2;
      double sw = exp(-x2/s2-y2/s2);
      ceres::CostFunction* cost_function = Gauss2dError::Create(ptr[y*size+x], x, y, size*0.5, sw);
      problem_gauss.AddResidualBlock(cost_function, NULL, params);
    }
  
  ceres::Solver::Summary summary2;
  ceres::Solve(options, &problem_gauss, &summary2);
  
  params[0] = size*0.5+sin(params[0])*(size*subfit_max_range);
  params[1] = size*0.5+sin(params[1])*(size*subfit_max_range);
  
  //if (abs(params[2]-params[5]) <= min_fitted_contrast)
    //return FLT_MAX;
  if (summary.termination_type != ceres::CONVERGENCE)
    return FLT_MAX;
  if (params[5] <= 0 || params[5] >= 255)
    return FLT_MAX;
  //nonsensical background?
  //spread to large?
  if (abs(params[3]) >= size*0.5 || abs(params[4]) >= size*0.5)
    return FLT_MAX;
  if (abs(params[3]) <= size*0.05 || abs(params[4]) <= size*0.05)
    return FLT_MAX;
  
  //rms scaled with amplitude (small amplitude needs lower rms!
  //printf("scale %f a %f b %f spread %f/%f ", 255.0/abs(params[2]-params[5]), params[2], params[5], params[3], params[4]);
  double contrast = abs(params[2]-params[5]);
  return sqrt(summary2.final_cost/problem_gauss.NumResiduals())*255.0/contrast + rms_use_limit*min_fitted_contrast/contrast;
}


/**
 * Fit 2d gaussian to image, 5 parameter: \f$x_0\f$, \f$y_0\f$, amplitude, spread, background
 * disregards a border of \f$\lfloor \mathit{size}/5 \rfloor\f$ pixels
 */
static double fit_gauss_direct(Mat &img, Point2f size, Point2f &p, Mat *paint = NULL, double *params = NULL)
{
  int w = img.size().width;
  Point2i hw = size*0.5;
  Point2i b = Point2i(size.x, size.y)*0.2;
  uint8_t *ptr = img.ptr<uchar>(0);
  
  assert(img.depth() == CV_8U);
  assert(img.channels() == 1);
  
  double params_static[8];
  
  if (!params)
    params = params_static;
  
  //x,y
  params[0] = 0.0;
  params[1] = 0.0;
  
  Rect area(p.x-hw.x+b.x, p.y-hw.y+b.y, size.x-2*b.x, size.y-2*b.y);
  
  int sum = 0;
  int x, y = area.y;
  for(x=area.x;x<area.br().x-1;x++)
    sum += ptr[y*w+x];
  y = area.br().y-2;
  for(x=area.x+1;x<area.br().x;x++)
    sum += ptr[y*w+x];
  x = area.x;
  for(y=area.y+1;y<area.br().y;y++)
    sum += ptr[y*w+x];
  x = area.br().x-2;
  for(y=area.y;y<area.br().y-1;y++)
    sum += ptr[y*w+x];
  
  //background
  params[5] = sum / (2*(size.x-b.x) + 2*(size.y-b.y));
  
  //amplitude
  //TODO use (small area?)
  params[2] = ptr[int(p.y)*w + int(p.x)];
  
  if (abs(params[2]-params[5]) < min_fit_contrast)
    return FLT_MAX;
  
  //spread
  params[3] = size.x*0.1;
  params[4] = size.y*0.1;
  
  //tilt
  params[6] = 0.0;
  params[7] = 0.0;
  
  ceres::Problem problem_gauss_center;
  for(y=area.y;y<area.br().y;y++)
    for(x=area.x;x<area.br().x;x++) {
      double x2 = x-p.x;
      double y2 = y-p.y;
      x2 = x2*x2;
      y2 = y2*y2;
      double s2 = sqrt(size.x*size.x+size.y*size.y)*0.5;
      s2=s2*s2;
      double sw = exp(-x2/s2-y2/s2);
      ceres::CostFunction* cost_function = Gauss2dDirectCenterError::Create(ptr[y*w+x], x, y, p.y, p.y, sw);
      problem_gauss_center.AddResidualBlock(cost_function, NULL, params+2);
    }
  
  ceres::Solver::Options options;
  options.max_num_iterations = 200;
  options.linear_solver_type = ceres::DENSE_QR;
  
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem_gauss_center, &summary);
  
  //std::cout << summary.FullReport() << "\n";
  
  ceres::Problem problem_gauss;
  for(y=area.y;y<area.br().y;y++)
    for(x=area.x;x<area.br().x;x++) {
      double x2 = x-p.x;
      double y2 = y-p.y;
      x2 = x2*x2;
      y2 = y2*y2;
      double s2 = sqrt(size.x*size.x+size.y*size.y)*0.5;
      s2=s2*s2;
      double sw = exp(-x2/s2-y2/s2);
      ceres::CostFunction* cost_function = Gauss2dDirectError::Create(ptr[y*w+x], x, y, size.x, size.y, p.x, p.y, sw);
      problem_gauss.AddResidualBlock(cost_function, NULL, params);
    }
    
  ceres::Solve(options, &problem_gauss, &summary);
  
  ceres::Problem problem_gauss_plane;
  for(y=area.y;y<area.br().y;y++)
    for(x=area.x;x<area.br().x;x++) {
      ceres::CostFunction* cost_function = Gauss2dPlaneDirectError::Create(ptr[y*w+x], x, y, size.x, size.y, p.x, p.y);
      problem_gauss_plane.AddResidualBlock(cost_function, NULL, params);
    }
    

  ceres::Solve(options, &problem_gauss_plane, &summary);
  
  //std::cout << summary2.FullReport() << "\n";
  
  //printf("%fx%f\n", params[6], params[7]);
  
  Point2f c = p;
  
  p.x += sin(params[0])*(size.x*subfit_max_range);
  p.y += sin(params[1])*(size.y*subfit_max_range);
  
  //printf(" -> %fx%f rms %f\n", p.x, p.y, sqrt(summary2.final_cost/problem_gauss.NumResiduals()));
  
  double contrast = abs(params[2]-params[5]);
  
  //if (abs(params[2]-params[5]) <= min_fitted_contrast)
    //return FLT_MAX;
  if (summary.termination_type != ceres::CONVERGENCE)
    return FLT_MAX;
  if (params[5] <= 0 || params[5] >= 255)
    return FLT_MAX;
  //nonsensical background?
  //spread to large?
  if (abs(params[3]) >= size.x*0.5 || abs(params[4]) >= size.y*0.5)
    return FLT_MAX;
  if (abs(params[3]) <= size.x*0.05 || abs(params[4]) <= size.y*0.05)
    return FLT_MAX;
  if (abs(params[6])*size.x >= contrast*0.5 || abs(params[7])*size.x >= contrast*0.5)
    return FLT_MAX;
  
  if (paint)
    draw_gauss2d_plane_direct(*paint, c, p, size, params);

  //rms scaled with amplitude (small amplitude needs lower rms!
  //printf("scale %f a %f b %f spread %f/%f ", 255.0/abs(params[2]-params[5]), params[2], params[5], params[3], params[4]);
  //return sqrt(summary2.final_cost/problem_gauss.NumResiduals())*255.0/contrast + rms_use_limit*min_fitted_contrast/contrast;
  
  return sqrt(summary.final_cost/problem_gauss.NumResiduals());
}

class Interpolated_Corner {
public:
  Point2i id;
  Point2f p;
  bool used_as_start_corner = false;
  
  Interpolated_Corner() {};
  Interpolated_Corner(Point2i id_, Point2f p_, bool used)
    : id(id_), p(p_), used_as_start_corner(used) {};
};

uint64_t id_to_key(Point2i id)
{
  uint64_t key = id.x;
  key = key << 32;
  key = key | id.y;
  
  return key;
}

typedef unordered_map<uint64_t, Interpolated_Corner> IntCMap;

void addcorners(Rect_<float> area, Point2f c)
{
  area.x = min(c.x, area.x);
  area.y = min(c.y, area.y);
  area.width = max(c.x-area.x, area.width);
  area.height = max(c.y-area.y, area.height);
}

int hdmarker_subpattern_checkneighbours(Mat &img, vector<Corner> &corners, int idx_step, Mat *paint = NULL)
{
  int counter = 0;
  int added = 0;
  
  IntCMap int_corners;
  IntCMap corners_map;
  
  for(int i=0;i<corners.size();i++) {
    Corner c = corners[i];
    Interpolated_Corner c_i(c.id, c.p, false);
    corners_map[id_to_key(corners[i].id)] = c_i;
  }
  
#pragma omp parallel for schedule(dynamic)
  for(int i=0;i<corners.size();i++) {
    Corner c;
#pragma omp critical
    {
      printprogress(i, corners.size(), counter, " %d corners", corners.size());
      c = corners[i];
    }
    
    for(int sy=-int_extend_range;sy<=int_extend_range;sy++)
      for(int sx=-int_extend_range;sx<=int_extend_range;sx++) {
        
        Point2i extr_id = Point2i(c.id)-Point2i(sx,sy)*idx_step;
        Point2f second;
        bool do_continue = false;
#pragma omp critical
        {
          if (corners_map.find(id_to_key(extr_id)) != corners_map.end())
            do_continue = true;
          
          if (!do_continue) {
            Point2i search_id = Point2i(c.id)+Point2i(sx,sy)*idx_step;
            IntCMap::iterator it = corners_map.find(id_to_key(search_id));
            
            if (it == corners_map.end())
              do_continue = true;
            else
              second = it->second.p;
          }
        }
        if (do_continue)
          continue;
        
        Point2f refine_p = c.p + (c.p-second);
        
        float len = norm(c.p-second);
        float maxlen = sqrt(len*len / (sy*sy + sx*sx))*10.0;
        
        if (refine_p.x - maxlen*0.1 <= 0 || refine_p.y - maxlen*0.1 <= 0)
          continue;
        if (refine_p.x + maxlen*0.1 >= img.size().width || refine_p.y + maxlen*0.1 >= img.size().height)
          continue;
        
        double params[8];
        double rms = fit_gauss_direct(img, Point2f(maxlen*0.2, maxlen*0.2), refine_p, paint, params);
        
        if (rms >= rms_use_limit)
          continue;
        
        Corner c_o(refine_p, extr_id, 0);
        Interpolated_Corner c_i(extr_id, refine_p, false);
#pragma omp critical
        {
          added++;
          corners_map[id_to_key(extr_id)] = c_i;
          corners.push_back(c_o); 
        }
        goto finish_add;
      }
      finish_add:;
  }
  
  printf("added %d corners\n", added);
  
  return added;
}

void hdmarker_subpattern_step(Mat &img, vector<Corner> corners, vector<Corner> &corners_out, int in_idx_step, float in_c_offset, int out_idx_scale, int out_idx_offset, bool ignore_corner_neighbours)
{  
  int counter = 0;
  sort(corners.begin(), corners.end(), corner_cmp);
  
  IntCMap corners_interpolated;
  vector<Mat> blurimgs;
  
  Mat paint = Mat::zeros(img.size(), CV_8U);
  paint = 127;
  
#pragma omp parallel for schedule(dynamic)
  for(int i=0;i<corners.size();i++) {
#pragma omp critical (_print_)
    printprogress(i, corners.size(), counter, " %d subs", corners_out.size());
    int sy = 0;
    for(int sy=-int_search_range;sy<=int_search_range;sy++)
      for(int sx=-int_search_range;sx<=int_search_range;sx++) {
        vector<Point2f> ipoints(4);
        Corner c = corners[i];
        int size;
        
        if (!sy && !sx)
          ipoints[0] = corners[i].p;
        else {
          Corner newc;
          //exists in corners
          if (!corner_find_off_save_int(corners, c, sx*in_idx_step, sy*in_idx_step, ipoints[0], int_search_range))
            continue;
          
          bool do_continue = false;
#pragma omp critical (_map_)
          {
            IntCMap::iterator it;
            Point2i id(c.id.x+sx*in_idx_step, c.id.y+sy*in_idx_step);
            it = corners_interpolated.find(id_to_key(id));
            if (it != corners_interpolated.end() && (*it).second.used_as_start_corner)
              do_continue = true;
            
            if (!do_continue)
              //interpolate from corners
              if (corner_find_off_save(corners, c, sx*in_idx_step, sy*in_idx_step, ipoints[0]))
                do_continue = true;
            
            if (!do_continue) {
              //set c to our interpolated corner id
              c.id.x += sx*in_idx_step;
              c.id.y += sy*in_idx_step;
              
              Interpolated_Corner c_i(c.id, c.p, true);
              corners_interpolated[id_to_key(c.id)] = c_i;
            }
          }
          if (do_continue)
            continue;
        }
        if (corner_find_off_save_int(corners, c, in_idx_step, 0, ipoints[1], int_search_range)) continue;
        if (corner_find_off_save_int(corners, c, in_idx_step, in_idx_step, ipoints[2], int_search_range)) continue;
        if (corner_find_off_save_int(corners, c, 0, in_idx_step, ipoints[3], int_search_range)) continue;
        
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
        
        //if (maxlen >= 50 && maxlen <= 100)
          //abort();
        
        //if (maxlen < 5*recurse_min_len)
          //continue;
        
        //FIXME need to use scale-space for perspective transform?
        //size = std::max<int>(std::min<int>(maxlen*subfit_oversampling/5, subfit_max_size), subfit_min_size);
        /*int undersampling = max(maxlen / size, 1);
        int undersampling_idx = log2(undersampling);*/
        
        for(int y=0;y<5;y++)
          for(int x=0;x<5;x++) {
            if (ignore_corner_neighbours) {
              if (x + y <= 1)
                continue;
              if ((x == 4 && y == 0) || (x == 0 && y == 4))
                continue;
            }
            //FIXME use proper (perspective?) center
            Point2f refine_p = ipoints[0] 
                                + (x+in_c_offset)*(ipoints[1]-ipoints[0])*0.2
                                + (y+in_c_offset)*(ipoints[3]-ipoints[0])*0.2;
                                
            if (refine_p.x - maxlen*0.1 <= 0 || refine_p.y - maxlen*0.1 <= 0)
              continue;
            if (refine_p.x + maxlen*0.1 >= img.size().width || refine_p.y + maxlen*0.1 >= img.size().height)
              continue;
            
            double params[8];
            double rms = fit_gauss_direct(img, Point2f(maxlen*0.2, maxlen*0.2), refine_p, &paint, params);
            
            if (rms >= rms_use_limit)
              continue;
            
            Corner c_o(refine_p, Point2i(c.id.x*out_idx_scale+2*x+out_idx_offset, c.id.y*out_idx_scale+2*y+out_idx_offset), 0);
    #pragma omp critical
            {
              corners_out.push_back(c_o); 
            }
          }

      }
  }
  printf("\n");
  
  printf("found %d valid corners                                                  \n", corners_out.size());

  int found = 1;
  while (found) {
    imwrite("fitted.tif", paint);
    found = hdmarker_subpattern_checkneighbours(img, corners_out, in_idx_step, &paint);
  }
  
}

void hdmarker_detect_subpattern(Mat &img, vector<Corner> corners, vector<Corner> &corners_out, int depth, double *size)
{
  vector<Corner> ca, cb;
  if (depth <= 0) {
    corners_out = corners;
    return;
  }
  
  int mul = 1;
  
  ca = corners;
  hdmarker_subpattern_step(img, ca, cb, 1, 0.5, 10, 1);
  mul *= 10;
  
  for(int i=2;i<=depth;i++) {
    ca = cb;
    cb.resize(0);
    hdmarker_subpattern_step(img , ca, cb, 2, 0.0, 5, 0, true);
    if (!cb.size()) {
      cb = ca;
      break;
    }
    mul *= 5;
  }
    
  corners_out = cb;
  *size /= mul;
}
