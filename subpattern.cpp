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
#include <stdarg.h>

using namespace std;
using namespace cv;

namespace hdmarker {
  
static const float min_fit_contrast = 1.0;
static const float min_fitted_contrast = 3.0; //minimum amplitude of fitted gaussian
//static const float min_fit_contrast_gradient = 0.05;
static const float rms_use_limit = 25.0;
static const float recurse_min_len = 3.0;
static const int int_search_range = 11;
static const int int_extend_range = 2;
static const float extent_range_limit_size = 8;
static const double subfit_max_range = 0.1;
static const double fit_gauss_max_tilt = 2.0;
static const float max_size_diff = 1.0;

static int safety_border = 2;

static const double bg_weight = 0.0;
static const double s2_mul = sqrt(1.0);
static const double tilt_max_rms_penalty = 9.0;
static const double border_frac = 0.15;

static const float max_retry_dist = 0.1;

static const float fit_size_min = 5.0;
static const float max_sigma = 0.25;
static const float min_sigma_px = 0.65;
//FIXME add possibility to reject too small sigma (less than ~one pixel (or two for bayer))

static const int min_fit_data_points = 9;


class SimpleCloud2d
{
public:
  
  SimpleCloud2d(int w, int h)
  {
    _w = w;
    _h = h;
    
    points = new std::pair<Point2i,Point2f>[_w*_h];
    for(int i=0;i<_w*_h;i++)
      points[i].first = Point2i(-1,-1);
  }
  
  ~SimpleCloud2d()
  {
    delete[] points;
  }
  
  bool CheckRad(Point2f p, int mindist, Point2i idx)
  {
    int miny = std::max(0, (int)p.y-mindist);
    int maxy = std::min(_h, (int)p.y+mindist+1);
    int minx = std::max(0, (int)p.x-mindist);
    int maxx = std::min(_w, (int)p.x+mindist+1);
    
    for(int j=miny;j<maxy;j++)
      for(int i=minx;i<maxx;i++)
        if (points[j*_w+i].first != Point2i(-1,-1) && points[j*_w+i].first != idx && norm(points[j*_w+i].second-p) <= (double)mindist) {
          printf("ERROR: area already covered from different idx!\n");
          cout << points[j*_w+i].first << " @ " << points[j*_w+i].second << " vs " << idx << " @ " << p << endl;
          return false;
        }
        
    return true;
  }
  
  void add(Point2i idx, Point2f pos)
  {
    if (!CheckRad(pos, 2, idx))
      abort();
    
    points[((int)pos.y)*_w+(int)(pos.x)] = std::pair<Point2i,Point2f>(idx,pos);
  }
  
  std::pair<Point2i,Point2f> *points;
  
private:
  int _w, _h;
};


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

class Interpolated_Corner {
public:
  Point2i id;
  Point2f p;
  float size;
  bool used_as_start_corner = false;
  int dist_searched;
  
  Interpolated_Corner() {};
  Interpolated_Corner(Point2i id_, Point2f p_, bool used)
    : id(id_), p(p_), used_as_start_corner(used) {};
    
  Interpolated_Corner(Corner c)
    : id(c.id), p(c.p), size(c.size) {};
};

static bool corner_cmp(Corner a, Corner b)
{
  if (a.id.y != b.id.y)
    return a.id.y < b.id.y;
  else
    return a.id.x < b.id.x;
}

//FIXME use score!
static bool pointf_cmp(Point2f a, Point2f b)
{
  if (a.y != b.y)
    return a.y < b.y;
  else
    return a.x < b.x;
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
    T s2 = T(2.0)*p[3]*p[3];
    x2 = x2*x2;
    y2 = y2*y2;
    
    residuals[0] = (T(val_) - (p[4] + (p[2]-p[4])*exp(-(x2/s2+y2/s2))))*T(sw_);
    
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(int val, int x, int y, double w, double h, double px, double py, double sw) {
    return (new ceres::AutoDiffCostFunction<Gauss2dDirectError, 1, 5>(
                new Gauss2dDirectError(val, x, y, w, h, px, py, sw)));
  }

  int x_, y_, val_;
  double w_, sw_, px_, py_, h_;
};


struct Gauss2dPlaneDirectError {
  Gauss2dPlaneDirectError(int val, int x, int y, double w, double h, double px, double py, double sw)
      : val_(val), x_(x), y_(y), w_(w), h_(h), px_(px), py_(py), sw_(sw){}

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
    x2 = x2*x2;
    y2 = y2*y2;

    residuals[0] = (T(val_) - (p[4] + p[5]*dx + p[6]*dy + (p[2]-p[4])*exp(-(x2/sx2+y2/sx2))))*T(sw_);
    
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(int val, int x, int y, double w, double h, double px, double py, double sw) {
    return (new ceres::AutoDiffCostFunction<Gauss2dPlaneDirectError, 1, 7>(
                new Gauss2dPlaneDirectError(val, x, y, w, h, px, py, sw)));
  }

  int x_, y_, val_;
  double w_, px_, py_, h_, sw_;
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
    T s2 = T(2.0)*p[1]*p[1];
    x2 = x2*x2;
    y2 = y2*y2;

    residuals[0] = (T(val_) - (p[2] + (p[0]-p[2])*exp(-(x2/s2+y2/s2))))*(T(sw_)+T(0.5));
    
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(int val, int x, int y, double px, double py, double sw) {
    return (new ceres::AutoDiffCostFunction<Gauss2dDirectCenterError, 1, 3>(
                new Gauss2dDirectCenterError(val, x, y, px, py, sw)));
  }

  int x_, y_, val_;
  double sw_, px_, py_;
};

template <typename T> inline T clamp(const T& n, const T& lower, const T& upper)
{
  return std::max<T>(lower, std::min<T>(n, upper));
}

static bool p_area_in_img_border(Mat &img, Point2f p, float extend)
{
  if (p.x - extend <= safety_border 
   || p.y - extend <= safety_border
   || p.x + extend >= img.size().width-safety_border-1
   || p.y + extend >= img.size().height-safety_border-1)
    return false;
  return true;
}

static void draw_gauss2d_plane_direct(Mat &img, Point2f c, Point2f res, Point2i size, double *p)
{
  uint8_t *ptr = img.ptr<uchar>(0);
  int w = img.size().width;
  
  for(int y=c.y-size.y/2;y<c.y+size.y/2;y++)
    for(int x=c.x-size.x/2;x<c.x+size.x/2;x++) {
      double x2 = x - res.x;
      double y2 = y - res.y;
      double dx = x - c.x;
      double dy = y - c.y;
      double s2 = 2.0*p[3]*p[3];
      x2 = x2*x2;
      y2 = y2*y2;
      
      ptr[y*w+x] = clamp<int>(p[4] + p[5]*dx + p[6]*dy + (p[2]-p[4])*exp(-(x2/s2+y2/s2)), 0, 255);
    }
}

/**
 * Fit 2d gaussian to image, 5 parameter: \f$x_0\f$, \f$y_0\f$, amplitude, spread, background
 * disregards a border of \f$\lfloor \mathit{size}/5 \rfloor\f$ pixels
 */
static double fit_gauss_direct(Mat &img, Point2f size, Point2f &p, double *params = NULL, bool *mask_2x2 = NULL, bool retry_allowed = true)
{  
  Point2f r_size = size;
  if (r_size.x < fit_size_min)
    r_size.x = fit_size_min;
  if (r_size.y < fit_size_min)
    r_size.y = fit_size_min;
  
  int w = img.size().width;
  Point2i hw = r_size*0.5;
  Point2i b = Point2f(r_size.x, r_size.y)*border_frac;
  uint8_t *ptr = img.ptr<uchar>(0);
  
  assert(img.depth() == CV_8U);
  assert(img.channels() == 1);
  
  double params_static[7];
  
  if (!params)
    params = params_static;
  
  //x,y
  params[0] = 0.0;
  params[1] = 0.0;
  
  Rect area(p.x+0.5-hw.x+b.x, p.y+0.5-hw.y+b.y, r_size.x-2*b.x, r_size.y-2*b.y);
  
  int y, x;
  
  int min_v = 255;
  int max_v = 0;
  for(y=area.y;y<=area.br().y;y++)
    for(x=area.x;x<=area.br().x;x++)
    if (!mask_2x2 || mask_2x2[(y%2)*2+(x%2)]) {
      min_v = std::min<int>(ptr[y*w+x],min_v);
      max_v = std::max<int>(ptr[y*w+x],max_v);
    }
  
  int center_v;
  for(y=int(p.y);y<=int(p.y)+1;y++)
    for(x=int(p.x);x<=int(p.x)+1;x++)
      if (!mask_2x2 || mask_2x2[(y%2)*2+(x%2)]) {
        center_v = ptr[y*w + x];
        break;
      }

  if (abs(center_v-max_v) < abs(center_v-min_v)) {
    params[2] = max_v;
    params[4] = min_v;
  }
  else {
    params[2] = min_v;
    params[4] = max_v;
  }
  
  if (abs(params[2]-params[4]) < min_fit_contrast)
    return FLT_MAX;
  
  //spread
  params[3] = size.x*0.15;
  
  //tilt
  params[5] = 0.0;
  params[6] = 0.0;
  
  int count = 0;
  for(y=area.y;y<=area.br().y;y++)
    for(x=area.x;x<=area.br().x;x++)
      if (!mask_2x2 || mask_2x2[(y%2)*2+(x%2)])
        count++;

  if (count < min_fit_data_points)
    return FLT_MAX;
  
  ceres::Problem problem_gauss_center;
  for(y=area.y;y<=area.br().y;y++)
    for(x=area.x;x<=area.br().x;x++)
      if (!mask_2x2 || mask_2x2[(y%2)*2+(x%2)]) {
        double x2 = x-p.x;
        double y2 = y-p.y;
        x2 = x2*x2;
        y2 = y2*y2;
        double s2 = s2_mul*(size.x*size.x+size.y*size.y);
        double sw = exp(-x2/s2-y2/s2) + bg_weight;
        ceres::CostFunction* cost_function = Gauss2dDirectCenterError::Create(ptr[y*w+x], x, y, p.x, p.y, sw);
        problem_gauss_center.AddResidualBlock(cost_function, NULL, params+2);
      }
  
  ceres::Solver::Options options;
  options.max_num_iterations = 1000;
  options.logging_type = ceres::LoggingType::SILENT;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.preconditioner_type = ceres::IDENTITY;
  
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem_gauss_center, &summary);
  
  ceres::Problem problem_gauss_plane;
  for(y=area.y-1;y<=area.br().y;y++)
    for(x=area.x-1;x<=area.br().x;x++) 
      if (!mask_2x2 || mask_2x2[(y%2)*2+(x%2)]) {
        double x2 = x-p.x;
        double y2 = y-p.y;
        x2 = x2*x2;
        y2 = y2*y2;
        double s2 = s2_mul*(size.x*size.x+size.y*size.y);
        double sw = exp(-x2/s2-y2/s2) + bg_weight;
        ceres::CostFunction* cost_function = Gauss2dPlaneDirectError::Create(ptr[y*w+x], x, y, size.x, size.y, p.x, p.y, sw);
        problem_gauss_plane.AddResidualBlock(cost_function, NULL, params);
      }
      
  ceres::Solve(options, &problem_gauss_plane, &summary);
  
  Point2f c = p;
  
  p.x += sin(params[0])*(size.x*subfit_max_range);
  p.y += sin(params[1])*(size.y*subfit_max_range);

  
  //minimal possible contrast
  double contrast = abs(params[2]-params[4])*exp(-(0.25/(2.0*params[3]*params[3])+0.25/(2.0*params[3]*params[3])));
  contrast = std::min(255.0, contrast);
  
  
  if (contrast <= min_fitted_contrast)
    return FLT_MAX;
  if (params[4] <= -1 || params[4] >= 256)
    return FLT_MAX;
  if (abs(params[3]) >= size.x*max_sigma)
    return FLT_MAX;
  if (abs(params[3]) <= min_sigma_px)
    return FLT_MAX;
  
  if (retry_allowed) 
    return fit_gauss_direct(img, size, p, params, mask_2x2, false);
  
  return sqrt(summary.final_cost/problem_gauss_plane.NumResiduals())*255.0/contrast*(1.0+tilt_max_rms_penalty*(abs(params[5])+abs(params[6]))/fit_gauss_max_tilt);
}

uint64_t id_to_key(Point2i id)
{
  uint64_t key = id.x;
  key = key << 32;
  key = key | id.y;
  
  return key;
}

//FIXME define our own key and hash functios for unoredered map!
uint64_t id_pair_to_key(Point2i id, Point2i id2)
{
  uint64_t key = id2.x;
  key = key << 32;
  key = key ^ id2.y;
  
  int dx = id.x-id2.x;
  int dy = id.y-id2.y;
  
  key = key ^ (((uint64_t)dx) << 16);
  key = key ^ (((uint64_t)dy) << 48);
  
  return key;
}

typedef unordered_map<uint64_t, Interpolated_Corner> IntCMap;
typedef unordered_map<uint64_t, std::vector<Point2f>> IntCLMap;

void addcorners(Rect_<float> area, Point2f c)
{
  area.x = min(c.x, area.x);
  area.y = min(c.y, area.y);
  area.width = max(c.x-area.x, area.width);
  area.height = max(c.y-area.y, area.height);
}

bool is_diff_larger(float a, float b, float th)
{
  if (max(a/b,b/a) >= 1.0 +th) {
    return true;
  }
  else
    return false;
}

static bool marker_corner_valid(const Corner &c, int page, bool checkrange, const Rect & limit)
{
  if (page != -1 && c.page != page)
    return false;
  
  if (checkrange && !limit.contains(c.id))
    return false;
  
  return true;
}

static bool check_limit(Point2i c, bool checkrange, const Rect & limit)
{ 
  if (checkrange && !limit.contains(c))
    return false;
  
  return true;
}

//FIXME this is still no perfectly repeatable!
int hdmarker_subpattern_checkneighbours(Mat &img, const vector<Corner> corners, vector<Corner> &corners_out, IntCMap &blacklist_rec, IntCLMap &blacklist, int idx_step, int int_extend_range, SimpleCloud2d &points, Mat *paint = NULL, bool *mask_2x2 = NULL, bool checkrange = false, const cv::Rect limit = Rect())
{
  int counter = 0;
  int added = 0;
  int checked = 0;
  int skipped = 0;
  
  IntCMap corners_map;
  IntCMap corners_out_map;
  
  for(int i=0;i<corners.size();i++) {
    Corner c = corners[i];
    Interpolated_Corner c_i(c);
    corners_map[id_to_key(corners[i].id)] = c_i;
  }
  
  int done = 0;
  
//#pragma omp parallel for schedule(dynamic, 8)
  for(int i=0;i<corners.size();i++) {
    Corner c;
    
    c = corners[i];
  
#pragma omp atomic 
    done++;

    int extend_range = int_extend_range;
    
    if (c.size < extent_range_limit_size)
      extend_range = 1;
          
    for(int sy=-extend_range;sy<=extend_range;sy++)
      for(int sx=-extend_range;sx<=extend_range;sx++) {
        
        Point2i extr_id = Point2i(c.id)-Point2i(sx,sy)*idx_step;
        
        if (!check_limit(extr_id, checkrange, limit))
          continue;
        
        Point2f second;
        bool do_continue = false;
        if (corners_map.find(id_to_key(extr_id)) != corners_map.end())
          continue;
        
        Point2i search_id = Point2i(c.id)+Point2i(sx,sy)*idx_step;
        IntCMap::iterator it = corners_map.find(id_to_key(search_id));
        //FIXME size calculation might be off for heavily tilted targets...
        if (it == corners_map.end())
          continue;
        else
          second = it->second.p;
        
        if (is_diff_larger(it->second.size, c.size, max_size_diff)) {
          continue;
        }

#pragma omp critical
        if (corners_out_map.count(id_to_key(extr_id)) && corners_out_map[id_to_key(extr_id)].dist_searched < int_extend_range)
          do_continue = true;
        if (do_continue)
          continue;
        
        Point2f refine_p = c.p + (c.p-second);
        Point2f v = c.p-second;
        //FIXME take min from x and y
        float maxlen = abs(v.y/sy*10.0/idx_step);
        float len = abs(v.x/sx*10.0/idx_step);
        if (len < maxlen)
          maxlen = len;

        const std::vector<Point2f> *tried = NULL;
        
#pragma omp critical (__blacklist__)
        if (blacklist.count(id_to_key(extr_id)))
          tried = &blacklist[id_to_key(extr_id)];
          
        if (tried)
          for(int i=0;i<tried->size();i++) {
            Point2f d = (*tried)[i]-refine_p;
            if (d.x*d.x+d.y*d.y < max_retry_dist*max_retry_dist*maxlen*maxlen) {
              do_continue = true;
#pragma omp atomic
              skipped++;
              break;
            }
          }

        if (do_continue)
          continue;
    
#pragma omp critical
        {
          if (!points.CheckRad(refine_p, 2, extr_id)) {
            imwrite("fitted.tif", *paint);
            abort();
          }
        }
        
        if (!p_area_in_img_border(img, refine_p, maxlen*0.2)
          || is_diff_larger(maxlen*0.2, c.size, max_size_diff)) {
            Interpolated_Corner c_i(extr_id, refine_p, false);
          continue;
        }
        
        double params[7];
        Point2f p_cp = refine_p;
#pragma omp atomic
        checked++;
        double rms = fit_gauss_direct(img, Point2f(maxlen*0.2, maxlen*0.2), refine_p, params, mask_2x2);
        
        if (rms >= rms_use_limit*min(maxlen*0.2,10.0)) {
            Interpolated_Corner c_i(extr_id, refine_p, false);
#pragma omp critical (__blacklist__)
            blacklist[id_to_key(extr_id)].push_back(p_cp);
          continue;
        }
        
        
        Corner c_o(refine_p, extr_id, 0);
        Interpolated_Corner c_i(extr_id, refine_p, false);
        c_i.size = sqrt(norm(c.p-c_o.p)*norm(c.p-c_o.p) / (sy*sy + sx*sx))*2/idx_step;
        c_i.dist_searched = int_extend_range;
        
        if (paint) 
#pragma omp critical (_paint_)
        {
          draw_gauss2d_plane_direct(*paint, p_cp, refine_p, Point2f(c_i.size, c_i.size), params);
          /*char buf[64];
          sprintf(buf, "%d",extr_id.x);
          putText(*paint, buf, refine_p, FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(127,127,127));
          sprintf(buf, "%d",extr_id.y);
          putText(*paint, buf, refine_p+Point2f(0,7), FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(127,127,127));*/
        }
          
#pragma omp critical
        {
          if (corners_out_map.count(id_to_key(extr_id)) != 0) {
            Interpolated_Corner c_i_old = corners_out_map[id_to_key(extr_id)];
            assert(c_i_old.id == c_i.id);
            //FIXME use rms score!
            if (pointf_cmp(c_i_old.p, c_i.p)) {
              corners_out_map[id_to_key(extr_id)] = c_i;
            }
          }
          else {
            added++;
            if (!points.CheckRad(refine_p, 2, extr_id)) {
              imwrite("fitted.tif", *paint);
              abort();
            }
            points.add(extr_id, Point2i(c_i.p.x+0.5, c_i.p.y+0.5));
            corners_out_map[id_to_key(extr_id)] = c_i;
            //FIXME add same corner multiple times if found from different direction?
            //corners_out.push_back(c_o);
          }
        }
        //else
        //count multiply found processed points?
      }
      
      if (done % ((corners.size()/100)+1) == 0)
#pragma omp critical(__print__)
        printprogress(done, corners.size(), counter, " %d corners, range %d, added %d checked %d skipped %d", corners.size(), int_extend_range, added, checked, skipped);
  }
  
  //FIXME push all corners from corners_out_map
  //corners_out.push_back(c_o);
  for(auto it=corners_out_map.begin();it!=corners_out_map.end();++it) {
    Corner c_o(it->second.p, it->second.id, 0);
    c_o.size = it->second.size;
    corners_out.push_back(c_o);
  }
  
  printf("added %d corners\n", added);
  
  return added;
}

void hdmarker_subpattern_step(Mat &img, vector<Corner> corners, vector<Corner> &corners_out, int in_idx_step, float in_c_offset, int out_idx_scale, int out_idx_offset, bool ignore_corner_neighbours, Mat *paint, bool *mask_2x2, int page, bool checkrange, const cv::Rect limit)
{  
  int counter = 0;
  sort(corners.begin(), corners.end(), corner_cmp);
  
  IntCMap corners_interpolated;
  vector<Mat> blurimgs;
  
  IntCMap blacklist;
  IntCLMap blacklist_neighbours;
  
  int done = 0;
  
  float minsize_fac = 1.0;
  
  if (mask_2x2) {
    int count = 0;
    minsize_fac = 2;
    for(int i=0;i<4;i++)
      if (mask_2x2)
        count++;
    if (count >= 2)
      minsize_fac = sqrt(2);
  }
  
#pragma omp parallel for schedule(dynamic, 8)
  for(int i=0;i<corners.size();i++) {

#pragma omp critical (_print_)
  {
    done++;
    if (done % ((corners.size()/100)+1) == 0)
      printprogress(done, corners.size(), counter, " %d subs", corners_out.size());
  }
    int sy = 0;
    
    /*if (in_idx_step == 2)
      if (norm(corners[i].p-Point2f(4000,1000)) >= 500)
        continue;*/
    
  Corner c = corners[i];
    
//   if (std::max(c.p.x, c.p.y) > 800 || std::max(c.p.x, c.p.y) < 700)
//     continue;
    
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
        
        float minlen = 1000000000000;
        for(int i=0;i<4;i++) {
          Point2f v = ipoints[i]-ipoints[(i+1)%4];
          float len = v.x*v.x+v.y*v.y;
          if (len < minlen)
            minlen = len;
          v = ipoints[(i+3)%4]-ipoints[i];
          len = v.x*v.x+v.y*v.y;
          if (len < minlen)
            minlen = len;
        }
        minlen = sqrt(minlen);
        
        
        if (minlen < 5*minsize_fac*recurse_min_len)
          continue;
        
        //FIXME need to use scale-space for perspective transform?
        //size = std::max<int>(std::min<int>(maxlen*subfit_oversampling/5, subfit_max_size), subfit_min_size);
        /*int undersampling = max(maxlen / size, 1);
        int undersampling_idx = log2(undersampling);*/
        
        for(int y=0;y<5;y++)
          for(int x=0;x<5;x++) {
            Point2i target_id(c.id.x*out_idx_scale+2*x+out_idx_offset, c.id.y*out_idx_scale+2*y+out_idx_offset);
            
            if (!check_limit(target_id, checkrange, limit))
              continue;
            
            if (ignore_corner_neighbours) {
              
              if (x + y <= 1 || (x == 4 && y == 0) || (x == 0 && y == 4)) {
                Interpolated_Corner c_i(target_id, Point2f(0,0), false);
    #pragma omp critical
                blacklist[id_to_key(c_i.id)] = c_i;
                continue;
              }
            }
            //FIXME use proper (perspective?) center
            Point2f refine_p = ipoints[0] 
                                + (x+in_c_offset)*(ipoints[1]-ipoints[0])*0.2
                                + (y+in_c_offset)*(ipoints[3]-ipoints[0])*0.2;
                                
            if (!p_area_in_img_border(img, refine_p, minlen*0.1)) {
                Interpolated_Corner c_i(target_id, Point2f(0,0), false);
    #pragma omp critical
                blacklist[id_to_key(c_i.id)] = c_i;
                continue;
              }
            
            double params[8];
            Point2f p_cp = refine_p;
            double rms = fit_gauss_direct(img, Point2f(minlen*0.2, minlen*0.2), refine_p, params, mask_2x2);
            
            if (rms >= rms_use_limit*min(minlen*0.2,10.0)){
                Interpolated_Corner c_i(target_id, Point2f(0,0), false);
    #pragma omp critical
                blacklist[id_to_key(c_i.id)] = c_i;
                continue;
              }
              
#pragma omp critical (_paint_)
          if (paint) {
            draw_gauss2d_plane_direct(*paint, p_cp, refine_p, Point2f(minlen*0.2, minlen*0.2), params);
            Point2i extr_id(c.id.x*out_idx_scale+2*x+out_idx_offset, c.id.y*out_idx_scale+2*y+out_idx_offset);
            /*char buf[64];
            sprintf(buf, "%d",extr_id.x);
            putText(*paint, buf, refine_p, FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(127,127,127));
            sprintf(buf, "%d",extr_id.y);
            putText(*paint, buf, refine_p+Point2f(0,7), FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(127,127,127));*/
          }
            
            Corner c_o(refine_p, Point2i(c.id.x*out_idx_scale+2*x+out_idx_offset, c.id.y*out_idx_scale+2*y+out_idx_offset), 0);
            c_o.size = minlen*0.2;
    #pragma omp critical
            {
              corners_out.push_back(c_o); 
            }
          }

      }
  }
  printf("\n");
  
  printf("found %d intitial corners                                                  \n", corners_out.size());
  
  SimpleCloud2d points(img.size().width, img.size().height);
  
  if (corners_out.size()) {
    //imwrite("fitted.tif", *paint);
    
    std::sort(corners_out.begin(), corners_out.end(), corner_cmp);
    for(int r=1;r<=int_extend_range;r++) {
      int found = corners_out.size();
      while (found > (corners_out.size()-found)*0.01 || (found && r == int_extend_range)) {
        //hdmarker_subpattern_checkneighbours results depend on corner ordering - make repeatable for threading!
        found = hdmarker_subpattern_checkneighbours(img, corners_out, corners_out, blacklist, blacklist_neighbours, 2, r, points, paint, mask_2x2, checkrange, limit);
        std::sort(corners_out.begin(), corners_out.end(), corner_cmp);
        if (found > (corners_out.size()-found)*0.01) {
          r = 1;
        }
        //imwrite("fitted.tif", *paint);
      }
    }
    
    printf("found %d corners with extra search                                                 \n", corners_out.size());
  }
}

void hdmarker_detect_subpattern(Mat &img, vector<Corner> corners, vector<Corner> &corners_out, int depth, double *size, Mat *paint, bool *mask_2x2, int page, const cv::Rect limit)
{
  bool checkrange = true;
  
  vector<Corner> ca, cb, all;
  if (depth <= 0) {
    corners_out = corners;
    return;
  }
  
  if (!limit.tl().x && !limit.tl().y && !limit.br().x &&!limit.br().y)
    checkrange = false;
  
  //imwrite("orig.tif", img);
  
  if (paint) {
    paint->create(img.size(), CV_8U);
    paint->setTo(Scalar(0));
  }
  
  
  for (auto & c : corners)
    if (marker_corner_valid(c, page, checkrange, limit))
      ca.push_back(c);
        
  corners_out = corners;
  int in_idx_step = 1;
  int mul = 10;
  hdmarker_subpattern_step(img, ca, cb, in_idx_step, 0.5, 10, 1, false, paint, mask_2x2, page, checkrange, Rect(limit.tl().x*mul,limit.tl().y*mul,limit.size().width*mul,limit.size().height*mul));
  in_idx_step = 2;
  
  //imwrite("debug.tif", *paint);
  
  if (cb.size() <= ca.size()) {
    corners_out = ca;
    return;
  }
  
  
  //FIXME what if we don't detect enough in this step?
  
  for(int i=0;i<corners_out.size();i++)
    corners_out[i].id *= 10;
  corners_out.resize(corners_out.size()+cb.size());
  for(int i=0;i<cb.size();i++)
    corners_out[corners_out.size()-cb.size()+i] = cb[i];
  
  for(int i=2;i<=depth;i++) {
    ca = cb;
    cb.resize(0);
    hdmarker_subpattern_step(img , ca, cb, in_idx_step, 0.0, 5, 0, true, paint, mask_2x2, page, checkrange, Rect(limit.tl().x*mul,limit.tl().y*mul,limit.size().width*mul,limit.size().height*mul));
    in_idx_step = 1;
    printf("stepped!\n");
    if (cb.size() <= ca.size()) {
      //reset
      cb = ca;
      break;
    }
    mul *= 5;
    for(int i=0;i<corners_out.size();i++)
      corners_out[i].id *= 5;
    corners_out.resize(corners_out.size()+cb.size());
    for(int i=0;i<cb.size();i++)
      corners_out[corners_out.size()-cb.size()+i] = cb[i];
    
    printf("write debug!\n");
    //imwrite("debug.tif", *paint);
  }
    
  *size /= mul;
  
  //imwrite("debug_last.tif", *paint);
}

} //namespace hdmarker

