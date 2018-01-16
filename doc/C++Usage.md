C++Usage
================

## Marker Detection

In almost all cases detection should be performed hdmarker::detect() method:

~~~~~~~~~~~~~{.cpp}
#include <hdmarker/hdmarker.hpp>

detect(cv::Mat img,
       std::vector<Corner> &corners,
       bool use_rgb = false,
       int marker_size_max = 0,
       int marker_size_min = 5,
       float effort = 0.5,
       int mincount = 1);
~~~~~~~~~~~~~

This function simply generates a vector of \a hdmarker::Corners, which desribes for each marker corner, its 2D marker coordinate to the subpixel accurate corner location and additionally a page ID which is not considered in the further calibration but present in the vector.

## Fractal Refinement

The fractal refinement recursively adds calibration points at the next higher scale, until no calibration points can be detected:
~~~~~~~~~~~~~{.cpp}
#include <hdmarker/subpattern.hpp>

void refine_recursive(cv::Mat &img,
                      std::vector<Corner> corners, 
                      std::vector<Corner> &corners_out, 
                      int depth, 
                      double *size, 
                      cv::Mat *paint = NULL, 
                      bool *mask_2x2 = NULL, 
                      int page = -1, 
                      const std::vector<cv::Rect> &limits = std::vector<cv::Rect>(), 
                      int flags = KEEP_ALL_LEVELS);
~~~~~~~~~~~~~
@copydoc hdmarker::refine_recursive(cv::Mat &img, std::vector<Corner> corners, std::vector<Corner> &corners_out, int depth, double *size, cv::Mat *paint = NULL, bool *mask_2x2 = NULL, int page = -1, const std::vector<cv::Rect> &limits = std::vector<cv::Rect>(), int flags = KEEP_ALL_LEVELS)
