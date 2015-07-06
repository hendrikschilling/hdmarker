#ifndef HDMARKER_SUBPATTERN_H
#define HDMARKER_SUBPATTERN_H

#include <opencv2/core/core.hpp>

#include "hdmarker.hpp"

void detect_sub_corners(cv::Mat &img, std::vector<Corner> corners, std::vector<Corner> &corners_out, int in_idx_step, float in_c_offset, int out_idx_scale, int out_idx_offset);

#endif