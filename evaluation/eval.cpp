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

int main(int argc, char* argv[])
{  
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
  rot_ref.at<double>(0, 1) *= -1;
  rot_ref.at<double>(0, 2) *= -1;
  rot_ref.at<double>(1, 0) *= -1;
  rot_ref.at<double>(2, 0) *= -1;
  
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
  
  Marker::init();
  
  Marker::detect(gray, corners_rough);
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
  cout << "t diff:\n" << trans - trans_ref << "\n";
  
  return EXIT_SUCCESS;
}
