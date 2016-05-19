#include <iostream>


#include <metamat/mat.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <assert.h>
#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "hdmarker/hdmarker.hpp"
#include "hdmarker/subpattern.hpp"

#include <ucalib/proxy.hpp>

using namespace cv;
using namespace std;
using namespace hdmarker;

Mat read_double_m(string path, int rows, int cols)
{
  double m;
  Mat out = Mat::zeros(rows, cols, CV_64FC1);
  
  ifstream ifs(path);
  int i = 0;
  while (ifs >> m)
  {
    out.at<double>(i / cols, i % cols) = m;
    i++;
  }
  
  return out;
}

#define PATTERN_CHECKER 0
#define PATTERN_HDMARKER 1

void detect_pattern(cv::Mat &img, int pattern_type, std::vector<cv::Point2f> &ipoints, std::vector<cv::Point3f> &wpoints)
{
  //FIXME load hardcoded values from config file?
  if (pattern_type == PATTERN_CHECKER) {
    ipoints.resize(0);
    wpoints.resize(0);
    findChessboardCorners(img, cv::Size(15, 9), ipoints, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
    if (!ipoints.size())
      return;
    //cornerSubPix(img, ipoints,  cv::Size(15, 15), cv::Size(-1, -1), TermCriteria( CV_TERMCRIT_EPS, 40, 0.001 ));
    printf("opencv found %d checker corners\n", ipoints.size());
    for(int j=0;j<9;j++)
      for(int i=0;i<15;i++) {
        wpoints.push_back(Point3f((14-i)+2, (8-j)+6, 0));
        printf("%dx%d = %fx%f\n", i, j, ipoints[j*15+i].x, ipoints[j*15+i].y);
      }
  }
  else {
    std::vector<Corner> corners_rough;
    std::vector<Corner> corners;
    double unit_size_res = 1.0;
    
    Marker::detect(img, corners_rough);
    
    if (!corners_rough.size())
      return;
    
    hdmarker_detect_subpattern(img, corners_rough, corners, 3, &unit_size_res);
    
    for(uint ci=0;ci<corners.size();ci++) {    
      wpoints.push_back(Point3f(corners[ci].id.x*unit_size_res,corners[ci].id.y*unit_size_res, 0));
      ipoints.push_back(Point2f(corners[ci].p.x, corners[ci].p.y));
    }
  }
}

void normalize_rot(cv::Mat &rot_v)
{
  double rot = norm(rot_v);
  if (rot > M_PI*0.5) {
    double newrot = M_PI-rot;
    rot_v *= -newrot/rot;
  }
}

void eval_pnp(std::vector<cv::Point2f> &ipoints, std::vector<cv::Point3f> &wpoints, const Matx33d &cam_to_img, const cv::Mat &rot_ref, const cv::Mat &trans_ref, const cv::Size img_size, const std::string prefix = std::string())
{
  Mat r_m;
  Rodrigues(rot_ref, r_m);
  
  if (!ipoints.size())
    return;
  
  double rms = 0.0;
  Point2d diff(0, 0);
  
  for(uint i=0;i<wpoints.size();i++) {
    Matx31d p_w(wpoints[i].x, wpoints[i].y, wpoints[i].z);
    Matx31d p_c = Matx33d(r_m)*p_w;
    p_c += Matx31d(trans_ref);
    Matx31d p_ih = cam_to_img*p_c;
    Point2d p(p_ih(0)/p_ih(2), p_ih(1)/p_ih(2));

    p -= Point2d(0.5,0.5);
    
    Point2d d = Point2d(ipoints[i]) - p;

    //line(dirs, p*16, (p+d*50)*16, CV_RGB(255,255,255), 1, CV_AA, 4);
    
    diff += d;
    
    rms += d.x*d.x+d.y*d.y;
    
    //circle(debug, p*16, 3, CV_RGB(255,255,255), -1, CV_AA, 4);
    //circle(debug2, corners[ci].p*16, 3, CV_RGB(255,255,255), -1, CV_AA, 4);
  }
  
  cout << prefix << " avg: " << diff*(1.0/wpoints.size()) << "\n";
  cout << prefix << " rms: " << sqrt(rms*(1.0/wpoints.size())) << "\n";
  
  Mat rot;
  Mat trans;
  solvePnP(wpoints, ipoints, cam_to_img, noArray(), rot, trans, false);
  
  //normalize_rot(rot);
  
  cout << "\n" << "rot ref:\n" << rot_ref << "\n";
  cout << "trans ref:\n" << trans_ref << "\n";
  
  cout << "rot:\n" << rot << "\n";
  cout << "trans:\n" << trans << "\n";
  
  cout << "trans diff:\n" << trans-trans_ref << "\n";
  cout << "rot diff:\n" << rot-rot_ref << "\n";
}

int main(int argc, char* argv[])
{  
  std::vector<Point3f> wpoints;
  std::vector<Point2f> ipoints;
  /*Matx34d world_to_cam(-0.0000, -1.0000,  0.0000,  10.1065,
                       -0.8937,  0.0000,  0.4488,   8.2506,
                        0.4488, -0.0000,  0.8937, -13.6151);*/
  
  /*Matx34d world_to_cam(6.322289778450813e-05, -0.999999994230047, -8.684912835035807e-05, 9.890795956340014,
  0.8937274826921973, 1.75426293043332e-05, 0.448610283400776, -8.253638241149725,
  -0.4486102792887537, -0.0001059818949470104, 0.8937274786445278, 13.61469197324652);*/
  
  /*Matx34d world_to_cam(1.0, 0.0, 0.0, -12.0,
  0.0, 1.0, 0.0, -12.0,
  0.0, 0.0, 1.0, -10.0);*/
  
  /*Matx34d world_to_cam(1.0000, -0.0000,  0.0000, -12.0000,
            0.0000,  0.7071,  0.7071,  -5.661583810168157,
             0.0000, -0.7071,  0.7071,  22.62450874187406);*/
  
/*Matx34d world_to_cam(0.94763702, 0.31785029,  -0.03090816, -10.84295086,
                     -0.16547957,  0.5715158,   0.80373275, -6.13332222,
                    0.27313116, -0.75653219,  0.59418726, 20.48268994);*/

  Mat rot_ref = read_double_m("rot.txt", 3, 3);
  Rodrigues(rot_ref, rot_ref);
  rot_ref.at<double>(1) *= -1;
  rot_ref.at<double>(2) *= -1;
  
  Mat trans_ref = read_double_m("trans.txt", 3, 1);
  trans_ref.at<double>(0) *= -1;
  
  
  Matx33d cam_to_img(2100.0000,    0.0000, 960.0000,
               0.0000, 2100.0000, 540.0000,
              0.0000,    0.0000,   1.0000);
  
  std::vector<Corner> corners_rough;
  std::vector<Corner> corners;
  double unit_size_res = 1.0;
  
  assert(argc == 2);
  
  Mat gray = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
  
  /*std::vector<cv::Point2f> ipoints;
  std::vector<cv::Point3f> wpoints;
  findChessboardCorners(gray, cv::Size(15, 9), ipoints, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
  //cornerSubPix(gray, ipoints,  cv::Size(15, 15), cv::Size(-1, -1), TermCriteria( CV_TERMCRIT_EPS, 40, 0.001 ));
  printf("opencv found %d checker corners\n", ipoints.size());
  for(int j=0;j<9;j++)
    for(int i=0;i<15;i++)
      wpoints.push_back(Point3f(i+2, j+6, 0));
    
    
  Mat rot;
  Rodrigues(rot_ref, rot);
  Mat trans = trans_ref.clone();
  trans.setTo(-10);
  trans.at<double>(2) = -10;
  
  solvePnP(wpoints, ipoints, cam_to_img, noArray(), rot, trans, true);
  
  Mat dist;
  
  std::vector<Mat> rmats;
  std::vector<Mat> tmats;
  
  Rodrigues(rot, rot);
  
  cout << "rotmat:\n" << rot << "\n";
  cout << "trans:\n" << trans << "\n";
  
  cout << "rot ref:\n" << rot_ref << "\n";
  cout << "rot diff:\n" << rot-rot_ref << "\n";
  cout << "trans ref:\n" << trans_ref << "\n";
  cout << "trans diff:\n" << trans-trans_ref << "\n";*/
  
  Marker::init();
  
  detect_pattern(gray, PATTERN_CHECKER, ipoints, wpoints);
  eval_pnp(ipoints, wpoints, cam_to_img, rot_ref, trans_ref, gray.size());
  
  detect_pattern(gray, PATTERN_HDMARKER, ipoints, wpoints);
  eval_pnp(ipoints, wpoints, cam_to_img, rot_ref, trans_ref, gray.size());
  
  /*Marker::detect(gray, corners_rough);
  Mat paint;
  Mat debug;
  Mat debug2;
  Mat dirs;
  
  hdmarker_detect_subpattern(gray, corners_rough, corners, 2, &unit_size_res, &paint);
  
  imwrite("points.tif", paint);
  debug.create(paint.size().height, paint.size().width, CV_8UC1);
  debug.setTo(0);
  debug2.create(paint.size().height, paint.size().width, CV_8UC1);
  debug2.setTo(0);
  dirs.create(paint.size().height, paint.size().width, CV_8UC1);
  dirs.setTo(0);
  
  
  Point2d diff(0,0);
  
  std::vector<Point3f> wpoints, wpoints2;
  std::vector<Point2f> ipoints, ipoints2;
  
  std::vector<std::vector<Point3f>> wpoints_v;
  std::vector<std::vector<Point2f>> ipoints_v;
  
  double rms = 0.0;
  
  for(uint ci=0;ci<corners.size();ci++) {
    Matx31d p_w((double)corners[ci].id.x*unit_size_res, (double)corners[ci].id.y*unit_size_res, 0);
    Matx31d p_c = Matx33d(rot_ref)*p_w;
    p_c += Matx31d(trans_ref);
    Matx31d p_ih = cam_to_img*p_c;
    Point2d p(p_ih(0)/p_ih(2), p_ih(1)/p_ih(2));
  
  
    //cout << corners[ci].id << "\n";
    //cout << corners[ci].id.x*unit_size_res << " " << corners[ci].id.y*unit_size_res << "\n";
    
    p -= Point2d(0.5,0.5);
    
    Point2d d = Point2d(corners[ci].p.x, corners[ci].p.y) - p;
    
    
    //cout << p << "\n";
    //cout << corners[ci].p << " " << d << "\n\n";
    
    line(dirs, p*16, (p+d*50)*16, CV_RGB(255,255,255), 1, CV_AA, 4);
    
    diff += d;
    
    rms += d.x*d.x+d.y*d.y;
    
    circle(debug, p*16, 3, CV_RGB(255,255,255), -1, CV_AA, 4);
    circle(debug2, corners[ci].p*16, 3, CV_RGB(255,255,255), -1, CV_AA, 4);
  
    wpoints.push_back(Point3f(corners[ci].id.x*unit_size_res,corners[ci].id.y*unit_size_res, 0));
    ipoints.push_back(Point2f(corners[ci].p.x, corners[ci].p.y));
  }
  
  cout << "avg: " << diff*(1.0/corners.size()) << "\n";
  cout << "rms: " << sqrt(rms*(1.0/corners.size())) << "\n";
  
  wpoints_v.push_back(wpoints);
  ipoints_v.push_back(ipoints);
  
  imwrite("dirs.tif", dirs);
  imwrite("debug.tif", debug);
  imwrite("debug2.tif", debug2);
  
  Mat rot;
  Mat trans;
  
  solvePnP(wpoints, ipoints, cam_to_img, noArray(), rot, trans);
  
  Mat dist;
  
  std::vector<Mat> rmats;
  std::vector<Mat> tmats;
  
  //double rms = calibrateCamera(wpoints_v, ipoints_v, paint.size(), cam_to_img, dist, rmats, tmats, CV_CALIB_FIX_K1 | CV_CALIB_FIX_K2 | CV_CALIB_FIX_K3 | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5 | CV_CALIB_FIX_K6 | CV_CALIB_FIX_ASPECT_RATIO | CV_CALIB_FIX_PRINCIPAL_POINT | CV_CALIB_ZERO_TANGENT_DIST);
  
  //printf("rms: %f\n", rms);
  
  //cout << "rot:\n" << rot << "\n";
  
  Rodrigues(rot, rot);
  
  cout << "rotmat:\n" << rot << "\n";
  cout << "trans:\n" << trans << "\n";
  
  cout << "rot ref:\n" << rot_ref << "\n";
  cout << "rot diff:\n" << rot-rot_ref << "\n";
  cout << "trans ref:\n" << trans_ref << "\n";
  cout << "trans diff:\n" << trans-trans_ref << "\n";
  
  
  //cout << "reverse:\n" << -rot.t()*trans << "\n";

  
  clif::Mat_<float> proxy({2, 33, 19});
  proxy_backwards_pers_poly_generate<0,0>(proxy, ipoints, wpoints, Point2i(paint.size().width, paint.size().height),paint.size().width/proxy[1]*0.5, 10);
  
  diff = Point2d(0,0);
  rms = 0;
  dirs.setTo(0);
  
  for(auto idx : clif::Idx_It_Dims(proxy,1,2)) {
    if (isnan(proxy(0, idx[1], idx[2])))
      continue;
    
    Matx31d p_w((double)proxy(0, idx[1], idx[2]), (double)proxy(1, idx[1], idx[2]), 0);
    Matx31d p_c = Matx33d(rot_ref)*p_w;
    p_c += Matx31d(trans_ref);
    
    Matx31d p_ih = cam_to_img*p_c;
    Point2d p(p_ih(0)/p_ih(2), p_ih(1)/p_ih(2));
    
    Point2d fit_p = Point2f((idx[1]+0.5)*paint.size().width/proxy[1],(idx[2]+0.5)*paint.size().height/proxy[2]);
  
  
    //cout << corners[ci].id << "\n";
    //cout << corners[ci].id.x*unit_size_res << " " << corners[ci].id.y*unit_size_res << "\n";
    
    p -= Point2d(0.5,0.5);
    
    Point2d d = fit_p - p;
    
    
    //cout << p << "\n";
    //cout << corners[ci].p << " " << d << "\n\n";
    
    line(dirs, p*16, (p+d*200)*16, CV_RGB(255,255,255), 1, CV_AA, 4);
    
    diff += d;
    
    rms += d.x*d.x+d.y*d.y;
    
    //circle(debug, p*16, 3, CV_RGB(255,255,255), -1, CV_AA, 4);
    //circle(debug2, corners[ci].p*16, 3, CV_RGB(255,255,255), -1, CV_AA, 4);
    
    wpoints2.push_back(Point3f(proxy(0, idx[1], idx[2]),proxy(1, idx[1], idx[2]), 0));
    ipoints2.push_back(fit_p);
  }
  
  imwrite("dirs_proxy.tif", dirs);
  
  cout << "avg: " << diff*(1.0/corners.size()) << "\n";
  cout << "rms: " << sqrt(rms*(1.0/corners.size())) << "\n";
  
  
  solvePnP(wpoints2, ipoints2, cam_to_img, noArray(), rot, trans);
  
  Rodrigues(rot, rot);
  cout << "rotmat:\n" << rot << "\n";
  cout << "trans:\n" << trans << "\n";
  
  
  cout << "r diff:\n" << rot-rot_ref << "\n";
  cout << "t diff:\n" << trans - trans_ref << "\n";*/
  
  return EXIT_SUCCESS;
}
