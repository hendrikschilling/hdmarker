#ifndef _LUTCALIB_LOESS_H
#define _LUTCALIB_LOESS_H

#include "settings.hpp"

#include "ceres/ceres.h"
#include "opencv2/core/core.hpp"

template <int x_degree, int y_degree> struct Poly2dError {
  Poly2dError(double x, double y, double val, double w)
      : x_(x), y_(y), val_(val), w_(w) {}
      
  template <typename T>
  bool operator()(const T* const p,
                  T* residuals) const {
    T xvars[x_degree];
    T yvar = T(1.0);
    xvars[0] = T(1.0);
    for(int i=1;i<x_degree;i++)
        xvars[i] = xvars[i-1]*T(x_);
    
    T res = T(0.0);
    for(int j=0;j<y_degree;j++) {
      for(int i=0;i<x_degree;i++) {
        res += p[j*x_degree+i]*(yvar*xvars[i]);
      }
      yvar = yvar*T(y_);
    }
        
    residuals[0] = (T(val_) - res)*T(w_);
    
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(double x, double y, double val, double w) {
    return (new ceres::AutoDiffCostFunction<Poly2dError, 1, x_degree*y_degree>(
                new Poly2dError(x, y, val, w)));
  }

  double x_,y_,val_, w_;
};


struct PersError {
  PersError(double x, double y, double vx, double vy)
      : x_(x), y_(y), vx_(vx), vy_(vy) {}
      
  template <typename T>
  bool operator()(const T* const p,
                  T* residuals) const {
    
    T res[3];
    
    res[0] = p[0]*T(x_) + p[1]*T(y_) + p[2];
    res[1] = p[3]*T(x_) + p[4]*T(y_) + p[5];
    res[2] = p[6]*T(x_) + p[7]*T(y_) + p[8];
                    
    residuals[0] = (T(vx_) - res[0]/res[2]);
    residuals[1] = (T(vy_) - res[1]/res[2]);
    
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(double x, double y, double vx, double vy) {
    return (new ceres::AutoDiffCostFunction<PersError, 2, 9>(
                new PersError(x, y, vx, vy)));
  }

  double x_,y_,vx_,vy_;
};

//NOTE residuals below 0.05 are ignored
//NOTE: z coordinate of wps is ignored (assumed to be constant - e.g. flat target)
template<int x_degree, int y_degree> double fit_2d_poly_2d(std::vector<cv::Point2f> &ips, std::vector<cv::Point3f> &wps, cv::Point2f center, double *coeffs, double sigma, int *count = NULL)
{
  ceres::Solver::Options options;
  options.max_num_iterations = 1000;
  options.num_threads = 1;
  //options.minimizer_progress_to_stdout = true;
  //options.trust_region_strategy_type = ceres::DOGLEG;
  options.linear_solver_type = ceres::DENSE_QR;
  
  double w_sum = 0.0;
  
  Point2f wc(0, 0);
  

  coeffs[0] = wps[0].x;
  for(int i=1;i<x_degree*y_degree;i++)
    coeffs[i] = 0.0;
  coeffs[9] = wps[0].y;
  for(int i=1;i<x_degree*y_degree;i++)
    coeffs[i+9] = 0.0;
  
  ceres::Problem problem_x;
  ceres::Problem problem_y;
  if (count)
    (*count) = 0;
  for(int i=0;i<ips.size();i++) {
      cv::Point2f ip = ips[i]-center;
      double w = exp(-(ip.x*ip.x+ip.y*ip.y)/(2.0*sigma*sigma));
      if (w <= 0.05)
        continue;
      if (count)
        (*count)++;
      w_sum += w;
      wc += w*ip;
      w = sqrt(w); //root to get w after squaring by ceres!
        ceres::CostFunction* cost_function =
            Poly2dError<x_degree,y_degree>::Create(ip.x, ip.y,wps[i].x, w);
            
        problem_x.AddResidualBlock(cost_function,
                                 NULL,
                                coeffs);
        
        cost_function =
            Poly2dError<x_degree,y_degree>::Create(ip.x, ip.y,wps[i].y, w);
            
        problem_y.AddResidualBlock(cost_function,
                                 NULL,
                                coeffs+x_degree*y_degree);
      }
      
  if (norm(wc*(1.0/w_sum)) >= 50.0)
    return std::numeric_limits<double>::quiet_NaN();  
  
  ceres::Solver::Summary summary_x;
  ceres::Solver::Summary summary_y;
  ceres::Solve(options, &problem_x, &summary_x);
  ceres::Solve(options, &problem_y, &summary_y);
 
  //std::cout << summary_x.FullReport() << "\n";
  //std::cout << summary_y.FullReport() << "\n";
  
  return sqrt((summary_x.final_cost+summary_y.final_cost)/w_sum);
}

template<int x_degree, int y_degree> cv::Point2f eval_2d_poly_2d(cv::Point2f p, double *coeffs)
{
  double xvars[x_degree];
  double yvar = 1.0;
  xvars[0] = 1.0;
  for(int i=1;i<x_degree;i++)
    xvars[i] = xvars[i-1]*p.x;
  
  double res_x = 0.0;
  double res_y = 0.0;
  for(int j=0;j<y_degree;j++) {
    for(int i=0;i<x_degree;i++) {
      res_x += coeffs[j*x_degree+i]*(yvar*xvars[i]);
      res_y += coeffs[j*x_degree+i+x_degree*y_degree]*(yvar*xvars[i]);
    }
    yvar = yvar*p.y;
  }
      
  return cv::Point2f(res_x, res_y);
}


#endif
