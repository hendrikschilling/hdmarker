#ifndef __LIBHDMARKER_H
#define __LIBHDMARKER_H

/** 
* @file hdmarker.hpp 
* @brief Header of hdmarker class
*
* @author Hendrik Schilling (implementation)
* @author Maximilian Diebold (documentation)
* @date 01/15/2018
*/

#include <vector>
#include <opencv2/core/core.hpp>

#include "gridstore.hpp"

namespace hdmarker {

    
    
    
/**
* @class Marker_Corner
*
* @brief Used internally in markers class to detect marker corners
*
* TODO
*/    
class Marker_Corner {
public :
  cv::Point2f p, dir[2], pc[3];
  int mask;
  float size;
  float size_x, size_y;
  int scale = 0;
  int x, y, page = -1;
  float dir_rad[2];
  cv::Point2i coord;
  double score = 10000.0;
  bool estimated = false;
  bool refined = false;
  
  Marker_Corner();
  Marker_Corner(cv::Point2f point, float s);
  Marker_Corner(cv::Point2f point, int m, float s);
  
  Marker_Corner operator=(Marker_Corner m);
  Marker_Corner operator*(float s);
  Marker_Corner operator*=(float s);
  
  //double scoreCorner(cv::Mat &img, cv::Point2f p, cv::Point2f dir[2]);
  //double scoreCorner(cv::Mat &img, cv::Point2f p, cv::Point2f dir[2], int size);
  void paint(cv::Mat &paint);
  void refineDir(cv::Mat img, float range);
  void refineDirIterative(cv::Mat img, int min_step, int max_step);
  void refineDirIterative_size(cv::Mat img, int min_step, int max_step, int size, int dead);
  void refineDirIterative(cv::Mat img);
  void refine(cv::Mat img, bool force = false, int dir_step_refine = 0);
  void refine(cv::Mat img, float max_step, bool force = false, int dir_step_refine = 0);
  void refine_size(cv::Mat img, float refine_max, bool force, int dir_step_refine, int size, int dead);
  void refine_gradient(cv::Mat &img, float scale);
  void estimateDir(cv::Mat img);
  void estimateDir(cv::Mat img, cv::Mat &paint);
  void cornerSubPixCPMask(cv::InputArray _image, cv::Point2f &p,
                      cv::Size win, cv::Size zeroZone, cv::TermCriteria criteria);
};



/**
* @class Corner
*
* @brief contain the intersection points of the checkerboard and the fractal marker points, this information is used for the calibration in ucalib
*
* TODO
*/ 
class Corner {
public :
  cv::Point2f p, pc[3];
  cv::Point2i id;
  int page;
  float size;
  
  Corner()
  {
    page = -1;
  }
  
  Corner(cv::Point2f cp, cv::Point2i cid, int cpage)
  {
    p = cp;
    pc[0] = cp;
    pc[1] = cp;
    pc[2] = cp;
    id = cid;
    page = cpage;
  }
  
  Corner(Marker_Corner &c)
  {
    p = c.p;
    pc[0] = c.pc[0];
    pc[1] = c.pc[1];
    pc[2] = c.pc[2];
    id = c.coord;
    page = c.page;
    size = c.size;
  }
  
  Corner *operator=(Corner c)
  {
    p = c.p;
    pc[0] = c.pc[0];
    pc[1] = c.pc[1];
    pc[2] = c.pc[2];
    id = c.id;
    page = c.page;
    size = c.size;
    
    return this;
  }
  
  void paint(cv::Mat &img);
  void paint_text(cv::Mat &paint);


  void write(cv::FileStorage& fs) const                        //Write serialization for this class
  {
      fs << "{"
         << "p" << p
         << "pc0" << pc[0]
         << "pc1" << pc[1]
         << "pc2" << pc[2]
         << "id" << id
         << "page" << page
         << "size" << size
         << "}";
  }
  void read(const cv::FileNode& node)                          //Read serialization for this class
  {
      node["p"] >> p;
      node["pc0"] >> pc[0];
      node["pc1"] >> pc[1];
      node["pc2"] >> pc[2];
      page = (int)node["page"];
      size = (float)node["page"];
  }
};

void write(cv::FileStorage& fs, const std::string&, const Corner& x);

void read(const cv::FileNode& node, Corner& x, const Corner& default_value = Corner());

/**
* @class Marker
*
* @brief used to detect the coded markers and the checkerboard intersection corners
*
* TODO
*/ 
class Marker {
  private:
  public:
    std::vector<Marker_Corner> corners;
    double score;
    int id;
    int page;
    int neighbours = 0;
    float scale = 0;
    bool filtered = false;
    
    Marker() {};
    int pointMarkerTest(cv::Point2f p);
    int calcId(cv::Mat &input);
    void bigId_affine(cv::Mat img, cv::Point2f start, cv::Point2f h, cv::Point2f v, int &big_id, double &big_score);
    void bigId(cv::Mat img, std::vector<Marker_Corner> &corners, int &big_id, double &big_score);
    void getPoints(cv::Point2f &p1, int &x1, int &y1, cv::Point2f &p2, int &x2, int &y2);
    void getPoints(cv::Point2f p[4], cv::Point2i c[4]);
    void getCorners(Marker_Corner c[4]);
    Marker(cv::Mat input, cv::Mat img, double marker_score, Marker_Corner *p1, Marker_Corner *p2, Marker_Corner *p3, float scale, int inpage = -1, int inid = -1);
    Marker(cv::Mat input, cv::Mat img, double marker_score, Marker_Corner *p1, Marker_Corner *p2, Marker_Corner *p3, Marker_Corner *p4, float scale, int inpage = -1, int inid = -1);
    void filterPoints(Gridstore *candidates, float scale);
    void filter(Gridstore *candidates, std::vector<Marker> &markers, float scale);
    void neighbours_inc(Gridstore *candidates, float scale);
    void neighbour_check(Marker *n,  Gridstore *candidates, float scale);
    void paint(cv::Mat &paint);
    Marker operator=(Marker m);
    
    static void init(void);
    static void detect_scale(std::vector<cv::Mat> imgs, std::vector<cv::Mat> norms, std::vector<cv::Mat> checkers, std::vector<Marker> &markers, float scale, float effort = 0.5, int inpage = -1);
    //static void detect(cv::Mat &img, std::vector<Marker> &markers);
    //static void detect(cv::Mat &img, std::vector<Corner> &corners);
    //static void detect_minscale(cv::Mat &img, cv::Mat &paint, std::vector<Corner> &corners, int scale_min = 8);
    static void detect(cv::Mat &img, std::vector<Marker> &markers, int marker_size_max = 0, int marker_size_min = 5, float effort = 0.5, int mincount = 10, std::vector<cv::Mat> *scales = NULL, std::vector<cv::Mat> *scales_border = NULL, int inpage = -1);
    static void detect_minscale(cv::Mat &img, cv::Mat &paint, std::vector<Corner> &corners, int scale_min = 8);
};


void detect(cv::Mat img, 
            std::vector<Corner> &corners, 
            bool use_rgb = false, 
            int marker_size_max = 0, 
            int marker_size_min = 5, 
            float effort = 0.5, 
            int mincount = 5, 
            int inpage = -1);

}

#endif
