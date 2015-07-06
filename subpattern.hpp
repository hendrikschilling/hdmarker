#ifndef HDMARKER_SUBPATTERN_H
#define HDMARKER_SUBPATTERN_H

#include <opencv2/core/core.hpp>

#include "hdmarker.hpp"

void hdmarker_subpattern_step(cv::Mat &img, std::vector<Corner> corners, std::vector<Corner> &corners_out, int in_idx_step, float in_c_offset, int out_idx_scale, int out_idx_offset);

void hdmarker_detect_subpattern(cv::Mat &img, std::vector<Corner> corners, std::vector<Corner> &corners_out, int depth);

#endif