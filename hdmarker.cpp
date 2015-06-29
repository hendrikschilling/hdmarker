#include "hdmarker.hpp"

#include <iostream>
#include <assert.h>
#include <stdio.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "gridstore.hpp"
#include "timebench.hpp"
#include "intrinsics.h"

#ifndef CV_RGB
#define CV_RGB( r, g, b )  Scalar( (b), (g), (r), 0 )
#endif

//#define WRITE_PROJECTED_IDS
//#define PAINT_CANDIDATE_CORNERS
//#define DEBUG_SAVESTEPS
//#define USE_SLOW_CORNERs

/*
TODO: - false positive corners in post-detection
      - unify scale space and filtering
      - implement + check/verify final high accuracy refine!
      - make corner threshold locally adaptive?
      - use valid marker/corner count and not raw count for early finish
      - adaptive minconstrast (first high then low then next scale)
      - use small and large blur (from scaling?) for last part of chess corner metric
      - 123_3_4.png should work at scale 8!
      - use corner (candidates and/or estimation) from one scale higher?
 * */

using namespace cv;
using namespace std;

static int inits = 0;
      
      /*float ix, fx;
      float iy, fy;*/
const double pattern_score_ok = 0.0;
const double pattern_score_good = 0.35;
const double pattern_score_early = 0.5;
const double pattern_score_sure = 10000000.0;

const float norm_avg_value = 140.0;
const float norm_avg_sub = 0.1;
const int hist_simplify = 1;
const int marker_minsize = 5;
const int marker_basesize = 25;
const int marker_maxsize = 35;
const float prescore_corner_limit = 0.25;
const float prescore_marker_limit = 0.25;
const int linescore_min = 50;
const int minconstrast = 3;
const float corner_ok = 0.1;
const float corner_good = 0.1;
const int corner_score_size = 4;
const int corner_score_dead = 2;
//const int corner_threshold_low = 1;
//const int corner_threshold_high = 20;
//FIXME adapt dirstep for effort 
const int dir_step = 16;
const int dir_step_sub = 64;
//FIXME global var in header...
const int dir_step_refine = 64;
//FIXME post detection refinement + adapt with effort
const float refine_min_step = 1.0/64.0;
const float refine_max_step = 0.5;
const int marker_neighbour_valid_count = 2;
const int marker_neighbour_sure_count = 5;
const int chess_dist = 1;
const float min_angle = 0.1; //0.1 ==  45deg?
const float id_sharpen = 0.25;
const bool replace_overlapping = false; //buggy
const float marker_corner_dir_dist_max = 0.5*corner_score_size;
const float marker_dir_corner_dist_max = 0.25;
const float marker_corner_dir_rad_dist = M_PI/16*1.1;
const float detection_scale_min = 0.5;
const int post_detection_range = 5;
const float corner_score_oversampling = 1;
const int border = marker_maxsize*2;

const Point2f b_o(border, border);

uint16_t corner_patt_x_b[24];
uint16_t corner_patt_y_b[24];
uint16_t corner_patt_x_w[24];
uint16_t corner_patt_y_w[24];

//#ifdef WRITE_PROJECTED_IDS
int global_counter = 0;
//#endif

typedef unsigned int uint;

typedef char v16si __attribute__ ((vector_size (16)));
typedef unsigned char v16qi __attribute__ ((vector_size (16)));
typedef int16_t v8si __attribute__ ((vector_size(16)));
typedef uint16_t v8qi __attribute__ ((vector_size (16)));
typedef long long v2di __attribute__ ((vector_size (16)));
typedef int32_t v4si __attribute__ ((vector_size (16)));
typedef float v4f __attribute__ ((vector_size (16), aligned(16))); 

vector<Point2f> box_corners;
vector<Point2f> box_corners_pers;
vector<Point2f> refine_corners;

    Marker_Corner::Marker_Corner(){
      refined = false;
      estimated = false;
      scale = 0;
      mask = 7;
      score = -1;
    }
    Marker_Corner::Marker_Corner(Point2f point, float s){
      refined = false;
      estimated = false;
      scale = s;
      p = point;
      mask = 7;
      score = -1;
    }
    Marker_Corner::Marker_Corner(Point2f point, int m, float s){
      refined = false;
      estimated = false;
      scale = s;
      p = point;
      mask = m;
      score = -1;
    }
    
    Marker_Corner Marker_Corner::operator=(Marker_Corner m)
    {
      p = m.p;
      pc[0] = m.pc[0];
      pc[1] = m.pc[1];
      pc[2] = m.pc[2];
      dir[0] = m.dir[0];
      dir[1] = m.dir[1];
      coord = m.coord;
      mask = m.mask;
      page = m.page;
      scale = m.scale;
      dir_rad[0] = m.dir_rad[0];
      dir_rad[1] = m.dir_rad[1];
      refined = m.refined;
      estimated = m.estimated;
      score = m.score;
      size = m.size;
      return *this;
    }
    
    Marker_Corner Marker_Corner::operator*(float s)
    {
      Marker_Corner nc = *this;
      nc.p *= s;
      return nc;
    }
    
    Marker_Corner Marker_Corner::operator*=(float s)
    {
      p *= s;
      return *this;
    }
    
    //warp to square
    static inline void simplewarp(Mat &src, Mat &dst, vector<Point2f> &c_src, int size)
    {
      int i,j;
      int w = src.size().width;
      Point2f d1, d2;
      uchar *srcptr = src.ptr<uchar>(0);
      uchar *dstptr = dst.ptr<uchar>(0);
      
      d1 = (c_src[1]-c_src[2])*(1.0/size);
      d2 = (c_src[0]-c_src[1])*(1.0/size);
      c_src[2] += d1*0.5+d2*0.5;
      
      for(j=0;j<size;j++)
	for(i=0;i<size;i++)
	  dstptr[j*size+i] = srcptr[(int)((c_src[2].y+i*d1.y+j*d2.y))*w+(int)(c_src[2].x+i*d1.x+j*d2.x)];
    }
    
    
    //warp to square
    static inline void simplewarp_bilin(Mat &src, Mat &dst, vector<Point2f> &c_src, int size)
    {
      int i,j;
      int w = src.size().width;
      Point2f d1, d2, c;
      uchar *srcptr = src.ptr<uchar>(0);
      uchar *dstptr = dst.ptr<uchar>(0);
      
      d1 = (c_src[1]-c_src[2])*(1.0/size);
      d2 = (c_src[0]-c_src[1])*(1.0/size);
      c = c_src[2] + d1*0.5+d2*0.5;
      
      int ix, iy;
      float fx, fy;
      
      for(j=0;j<size;j++)
	for(i=0;i<size;i++) {
	  fx = c.x+i*d1.x+j*d2.x;
	  fy = c.y+i*d1.y+j*d2.y;
	  ix = fx;
	  iy = fy;
	  fx -= ix;
	  fy -= iy;
	  dstptr[j*size+i] = srcptr[iy*w+ix]*(1.0-fx)*(1.0-fy)
			    +srcptr[iy*w+ix+1]*(fx)*(1.0-fy)
			    +srcptr[(iy+1)*w+ix]*(1.0-fx)*(fy)
			    +srcptr[(iy+1)*w+ix+1]*(fx)*(fy);
	}
    }
    
    
    inline int warp_getpoint(uchar *srcptr, int w, Point2f c, Point2f d1, Point2f d2, int x, int y)
    {
      
      int ix, iy;
      float fx, fy;
      
      c += 0.5*d1+0.5*d2;

	  fx = c.x+x*d1.x+y*d2.x;
	  fy = c.y+x*d1.y+y*d2.y;
	  ix = fx;
	  iy = fy;
	  fx -= ix;
	  fy -= iy;
	  return srcptr[iy*w+ix]*(1.0-fx)*(1.0-fy)
			    +srcptr[iy*w+ix+1]*(fx)*(1.0-fy)
			    +srcptr[(iy+1)*w+ix]*(1.0-fx)*(fy)
			    +srcptr[(iy+1)*w+ix+1]*(fx)*(fy);
    }
    
static inline int warp_getpoint2(uchar *srcptr, int w, Point2f c, Point2f d1, Point2f d2, int x, int y)
{
    
    int ix, iy;
    float fx, fy;
    
    fx = c.x+x*d1.x+y*d2.x;
    fy = c.y+x*d1.y+y*d2.y;
    ix = fx;
    iy = fy;
    fx -= ix;
    fy -= iy;
    int pos = iy*w+ix;
    float fxm = 1.0-fx;
    float fym = 1.0-fy;
    return srcptr[pos]*fxm*fym
    +srcptr[pos+1]*fx*fym
    +srcptr[pos+w]*fxm*fy
    +srcptr[pos+w+1]*fx*fy;
}

const uint16_t getpoints_intscale = 32;
const uint16_t getpoints_intscale2 = 256;

//ATTENTION max image dimension 2**16/getpoints_intscale for this function!
static inline int warp_getpoints_x8(uchar *srcptr, int w, Point2f c, Point2f d1, Point2f d2, uint16_t *x, uint16_t *y)
{
    int val = 0;
    uint16_t gi = getpoints_intscale;
    uint16_t gi2 = getpoints_intscale2;
    int ix, iy;
    float fx, fy;
    int pos;
    int fxm;
    int fym;
    v8si gid = (v8si)_mm_set1_epi16(gi2/gi);
    v8si v_one_16 = (v8si)_mm_set1_epi16(1);
    v8si v_gi = (v8si)_mm_set1_epi16(gi);
    
    uint16_t d1sx = d1.x*gi2;
    uint16_t d1sy = d1.y*gi2;
    uint16_t d2sx = d2.x*gi2;
    uint16_t d2sy = d2.y*gi2;
    uint16_t fxs;
    uint16_t fys;
    uint16_t cx = c.x*gi;
    uint16_t cy = c.y*gi;
    
    v8qi fxs_v, fys_v;
    v8si v8s;
    v8qi fxsr_v, fysr_v;
    v8si mul_v;
    v8qi cx_v, cy_v;
    v8si d1x_v, d1y_v, d2x_v, d2y_v;
    v8qi fxm_v,fym_v;
    v8qi ix_v,iy_v;
    v8qi gi_mask;
    v8qi val_v;
    int pos_v[8];
    
    for(int i=0;i<8;i++) cx_v[i] = cx;
    for(int i=0;i<8;i++) cy_v[i] = cy;
    for(int i=0;i<8;i++) d1x_v[i] = d1sx;
    for(int i=0;i<8;i++) d1y_v[i] = d1sy;
    for(int i=0;i<8;i++) d2x_v[i] = d2sx;
    for(int i=0;i<8;i++) d2y_v[i] = d2sy;
    for(int i=0;i<8;i++) gi_mask[i] = getpoints_intscale-1;
       
    v8s = *(v8si*)x * d1x_v;
    mul_v = *(v8si*)y * d2x_v;
    v8s = v8s + mul_v;
    v8s /= gid;
    fxs_v = cx_v + v8s;
    
    v8s = *(v8si*)x * d1y_v;
    mul_v = *(v8si*)y * d2y_v;
    v8s = v8s + mul_v;
    v8s /= gid;
    fys_v = cy_v + v8s;
    
    fxsr_v = fxs_v & gi_mask;
    fysr_v = fys_v & gi_mask;
    
    fxm_v = gi_mask + v_one_16 - fxsr_v;
    fym_v = gi_mask + v_one_16 - fysr_v;
    
    ix_v = fxs_v / v_gi;
    iy_v = fys_v / v_gi;
    
    for(int i=0;i<8;i++)
        pos_v[i] = iy_v[i]*w+ix_v[i];
    
    for(int i=0;i<8;i++)
        val_v[i] = srcptr[pos_v[i]];
    val_v = val_v * fxm_v * fym_v;
    for(int i=0;i<8;i++) {
       /* fx = c.x+x[i]*d1.x+y[i]*d2.x;
        fy = c.y+x[i]*d1.y+y[i]*d2.y;
        ix = fx;
        iy = fy;
        fx -= ix;
        fy -= iy;
        int pos = iy*w+ix;
        float fxm = 1.0-fx;
        float fym = 1.0-fy;
        assert(srcptr[pos]*fxm*fym*gi*gi == val_v[i]);*/
        val += val_v[i];
    }
    
    for(int i=0;i<8;i++)
        pos_v[i] += 1;
    for(int i=0;i<8;i++)
        val_v[i] = srcptr[pos_v[i]];
    val_v = val_v * fxsr_v * fym_v;
    for(int i=0;i<8;i++)
        val += val_v[i];

    for(int i=0;i<8;i++)
        pos_v[i] += w-1;
    for(int i=0;i<8;i++)
        val_v[i] = srcptr[pos_v[i]];
    val_v = val_v * fxm_v * fysr_v;
    for(int i=0;i<8;i++)
        val += val_v[i];
    
    for(int i=0;i<8;i++)
        pos_v[i] += 1;
    for(int i=0;i<8;i++)
        val_v[i] = srcptr[pos_v[i]];
    val_v = val_v * fxsr_v * fysr_v;
    for(int i=0;i<8;i++)
        val += val_v[i];
        
    return val*(1.0/(gi*gi));
}
    
    /*double Marker_Corner::scoreCorner(Mat &img, Point2f p, Point2f dir[2])
    {
      int x, y;
      int white = 0, black = 0;
      Mat affine;
      Mat transformed(Size(2*corner_score_size, 2*corner_score_size),CV_8UC1);
      vector<Point2f> test_corners(3);
   
      test_corners[0] = p+dir[0]-dir[1];
      test_corners[1] = p+dir[0]+dir[1];
      test_corners[2] = p-dir[0]+dir[1];
      
      simplewarp_bilin(img, transformed, test_corners, 2*corner_score_size);
      
      for(y=0;y<corner_score_size;y++)
	for(x=0;x<corner_score_size;x++)
	  black += transformed.at<uchar>(y, x);
      for(y=corner_score_size;y<2*corner_score_size;y++)
	for(x=0;x<corner_score_size;x++)
	  white += transformed.at<uchar>(y, x);
      for(y=0;y<corner_score_size;y++)
	for(x=corner_score_size;x<2*corner_score_size;x++)
	  white += transformed.at<uchar>(y, x);
      for(y=corner_score_size;y<2*corner_score_size;y++)
	for(x=corner_score_size;x<2*corner_score_size;x++)
	  black += transformed.at<uchar>(y, x);
	
      for(y=0;y<corner_score_dead;y++)
	for(x=0;x<corner_score_dead;x++)
	  black -= transformed.at<uchar>(y, x);
      for(y=2*corner_score_size-corner_score_dead;y<2*corner_score_size;y++)
	for(x=0;x<corner_score_dead;x++)
	  white -= transformed.at<uchar>(y, x);
      for(y=0;y<corner_score_dead;y++)
	for(x=2*corner_score_size-corner_score_dead;x<2*corner_score_size;x++)
	  white -= transformed.at<uchar>(y, x);
      for(y=2*corner_score_size-corner_score_dead;y<2*corner_score_size;y++)
	for(x=2*corner_score_size-corner_score_dead;x<2*corner_score_size;x++)
	  black -= transformed.at<uchar>(y, x);
	
	//norm(Point2f(870,709)-p) < 3
	
	return (float)(white-black);
    }*/
    
    
    void writecorner(Mat &img, Point2f p, Point2f dir[2])
    {
      int x, y;
      int white = 0, black = 0;
      Mat affine;
      Mat transformed(Size(2*corner_score_size, 2*corner_score_size),CV_8UC1);
      vector<Point2f> test_corners(3);
   
      p += Point2f(0.5,0.5);
      
      test_corners[0] = p+dir[0]-dir[1];
      test_corners[1] = p+dir[0]+dir[1];
      test_corners[2] = p-dir[0]+dir[1];
      
      simplewarp_bilin(img, transformed, test_corners, 2*corner_score_size);
      
      char buf[64];
      sprintf(buf, "c%d_%d.png", (int)p.x, (int)p.y);
      imwrite(buf, transformed);
    }
    
    void writecorner(Mat &img, Point2f p, Point2f dir[2], char s[64])
    {
      int x, y;
      int white = 0, black = 0;
      Mat affine;
      Mat transformed(Size(2*corner_score_size, 2*corner_score_size),CV_8UC1);
      vector<Point2f> test_corners(3);
   
      p += Point2f(0.5,0.5);
      
      test_corners[0] = p+dir[0]-dir[1];
      test_corners[1] = p+dir[0]+dir[1];
      test_corners[2] = p-dir[0]+dir[1];
      
      simplewarp_bilin(img, transformed, test_corners, 2*corner_score_size);
      
      imwrite(s, transformed);
    }
    
    
    
    double scoreCorner(Mat &img, Point2f p, Point2f dir[2], int corner_score_size, int corner_score_dead)
    {
      int s = corner_score_size;
      int d = corner_score_dead;
      int x, y;
      int white = 0, black = 0, wh = 0, bl = 0;
      vector<Point2f> test_corners(3);
      
      test_corners[0] = p+dir[0]-dir[1]+b_o;
      test_corners[1] = p+dir[0]+dir[1]+b_o;
      test_corners[2] = p-dir[0]+dir[1]+b_o;
      
      int w = img.size().width;
      Point2f d1, d2;
      uchar *srcptr = img.ptr<uchar>(0);
      
      d1 = (test_corners[1]-test_corners[2])*(1.0/(2*corner_score_size));
      d2 = (test_corners[0]-test_corners[1])*(1.0/(2*corner_score_size));
      Point2f c = test_corners[2];
      
      c += 0.5*d1+0.5*d2;
	
      for(y=0;y<s;y++)
	for(x=d;x<s;x++)
	  bl += warp_getpoint2(srcptr, w, c, d1, d2, x, y);
      for(y=0;y<s;y++)
	for(x=s;x<s+d;x++)
	  wh += warp_getpoint2(srcptr, w, c, d1, d2, x, y);
	
      for(y=d;y<s;y++)
	for(x=0;x<d;x++)
	  bl += warp_getpoint2(srcptr, w, c, d1, d2, x, y);
      for(y=d;y<s;y++)
	for(x=2*s-d;x<2*s;x++)
	  wh += warp_getpoint2(srcptr, w, c, d1, d2, x, y);
	
      for(y=s;y<2*s-d;y++)
	for(x=0;x<d;x++)
	  wh += warp_getpoint2(srcptr, w, c, d1, d2, x, y);
      for(y=s;y<2*s-d;y++)
	for(x=2*s-d;x<2*s;x++)
	  bl += warp_getpoint2(srcptr, w, c, d1, d2, x, y);

      for(y=s;y<2*s;y++)
	for(x=d;x<s;x++)
	  wh += warp_getpoint2(srcptr, w, c, d1, d2, x, y);
      for(y=s;y<2*s;y++)
	for(x=s;x<s+d;x++)
	  bl += warp_getpoint2(srcptr, w, c, d1, d2, x, y);
	
	return (double)(wh-bl) / (2*(corner_score_size*corner_score_size-corner_score_dead*corner_score_dead)*256);
    }
    
    double scoreCorner(Mat &img, Point2f p, Point2f dir[2])
    {
      int s = corner_score_size;
      int d = corner_score_dead;
      int x, y;
      int white = 0, black = 0, wh = 0, bl = 0;
      vector<Point2f> test_corners(3);
      
      test_corners[0] = p+dir[0]-dir[1]+b_o;
      test_corners[1] = p+dir[0]+dir[1]+b_o;
      test_corners[2] = p-dir[0]+dir[1]+b_o;
      
      int w = img.size().width;
      Point2f d1, d2;
      uchar *srcptr = img.ptr<uchar>(0);
      
      d1 = (test_corners[1]-test_corners[2])*(1.0/(2*corner_score_size));
      d2 = (test_corners[0]-test_corners[1])*(1.0/(2*corner_score_size));
      Point2f c = test_corners[2];
      
      c += 0.5*d1+0.5*d2;
	
      for(y=0;y<s;y++)
	for(x=d;x<s;x++)
	  bl += warp_getpoint2(srcptr, w, c, d1, d2, x, y);
      for(y=0;y<s;y++)
	for(x=s;x<s+d;x++)
	  wh += warp_getpoint2(srcptr, w, c, d1, d2, x, y);
	
      for(y=d;y<s;y++)
	for(x=0;x<d;x++)
	  bl += warp_getpoint2(srcptr, w, c, d1, d2, x, y);
      for(y=d;y<s;y++)
	for(x=2*s-d;x<2*s;x++)
	  wh += warp_getpoint2(srcptr, w, c, d1, d2, x, y);
	
      for(y=s;y<2*s-d;y++)
	for(x=0;x<d;x++)
	  wh += warp_getpoint2(srcptr, w, c, d1, d2, x, y);
      for(y=s;y<2*s-d;y++)
	for(x=2*s-d;x<2*s;x++)
	  bl += warp_getpoint2(srcptr, w, c, d1, d2, x, y);

      for(y=s;y<2*s;y++)
	for(x=d;x<s;x++)
	  wh += warp_getpoint2(srcptr, w, c, d1, d2, x, y);
      for(y=s;y<2*s;y++)
	for(x=s;x<s+d;x++)
	  bl += warp_getpoint2(srcptr, w, c, d1, d2, x, y);
	
	return (double)(wh-bl) / (2*(corner_score_size*corner_score_size-corner_score_dead*corner_score_dead)*256);
    }

static double scoreCorner_SIMD(Mat &img, Point2f p, Point2f dir[2])
{
    int s = corner_score_size;
    int d = corner_score_dead;
    int x, y;
    int white = 0, black = 0, wh = 0, bl = 0;
    vector<Point2f> test_corners(3);
    
    test_corners[0] = p+dir[0]-dir[1];
    test_corners[1] = p+dir[0]+dir[1];
    test_corners[2] = p-dir[0]+dir[1];
    
    int w = img.size().width;
    Point2f d1, d2;
    uchar *srcptr = img.ptr<uchar>(0);
    
    d1 = (test_corners[1]-test_corners[2])*(1.0/(2*corner_score_size));
    d2 = (test_corners[0]-test_corners[1])*(1.0/(2*corner_score_size));
    Point2f c = test_corners[2];
    
    c += 0.5*d1+0.5*d2;
    
    assert(s == 4 && d == 2);
    
    /*for(y=0;y<s;y++)
        for(x=d;x<s;x++) {
            bl += warp_getpoint2(srcptr, w, c, d1, d2, x, y);
        }
    int test = warp_getpoints_x8(srcptr,w,c,d1,d2,corner_patt_x_b,corner_patt_y_b);
    if (test != bl)
        printf("%d - %d\n",test,bl);*/
    bl += warp_getpoints_x8(srcptr,w,c,d1,d2,corner_patt_x_b,corner_patt_y_b);
    /*for(y=0;y<s;y++)
        for(x=s;x<s+d;x++)
            wh += warp_getpoint2(srcptr, w, c, d1, d2, x, y);*/
    wh += warp_getpoints_x8(srcptr,w,c,d1,d2,corner_patt_x_w,corner_patt_y_w);
    
    //16
    
    for(y=d;y<s;y++)
        for(x=0;x<d;x++)
            bl += warp_getpoint2(srcptr, w, c, d1, d2, x, y);
    for(y=d;y<s;y++)
        for(x=2*s-d;x<2*s;x++)
            wh += warp_getpoint2(srcptr, w, c, d1, d2, x, y);
    
    for(y=s;y<2*s-d;y++)
        for(x=0;x<d;x++)
            wh += warp_getpoint2(srcptr, w, c, d1, d2, x, y);
    for(y=s;y<2*s-d;y++)
        for(x=2*s-d;x<2*s;x++)
            bl += warp_getpoint2(srcptr, w, c, d1, d2, x, y);

    //16
    
    for(y=s;y<2*s;y++)
        for(x=d;x<s;x++)
            wh += warp_getpoint2(srcptr, w, c, d1, d2, x, y);
    for(y=s;y<2*s;y++)
        for(x=s;x<s+d;x++)
            bl += warp_getpoint2(srcptr, w, c, d1, d2, x, y);
    
    //16
    
    return (double)(wh-bl) / (2*(corner_score_size*corner_score_size-corner_score_dead*corner_score_dead)*256);
}
    
    
    void Marker_Corner::refineDir(Mat img, float range)
    {
      double test_score;
      int a, b, besta, bestb;
      //angle
      Point2f test_dir[2];
      
      besta = dir_rad[0];
      bestb = dir_rad[1];
      
      for(a=dir_rad[0]-range;a<=dir_rad[0]+range;a+=M_PI/dir_step_sub) 
	for(b=dir_rad[1]-range;b<=dir_rad[1]+range;b+=M_PI/dir_step_sub)  {
	  test_dir[0].x = corner_score_size*cos(a);
	  test_dir[0].y = corner_score_size*sin(a);
	  test_dir[1].x = corner_score_size*cos(b);
	  test_dir[1].y = corner_score_size*sin(b);
	  //test_dir[0].x = -test_dir[1].y;
	  //test_dir[0].y = test_dir[1].x;
	  
	  test_score = scoreCorner(img, p, test_dir);
	  if (test_score > score) {
	    score = test_score;
	    dir[0] = test_dir[0];
	    dir[1] = test_dir[1];
	    besta = a;
	    bestb = b;
	  }
      }
      
      dir_rad[0] = besta;
      dir_rad[1] = bestb;
    
      //printf("%f direction: %fx%f\n", score, dir[0].x, dir[0].y);
    }
    
    
    void Marker_Corner::refineDirIterative_size(Mat img, int min_step, int max_step, int size, int dead)
    {
      bool change = true;
      double test_score;
      float a, b;
      Point2f test_dir[2];
      
      for(float range=M_PI/min_step;range>=M_PI/max_step;range*=0.5) {
	change = true;
	for(int i=0;i<100 && change;i++) {
	  change = false;
	  for(int sign=-1;sign<=1;sign+=2)
	    for(int coord=0;coord<2;coord++) {
	      if (!coord) {
		a = dir_rad[0]+sign*range;
		b = dir_rad[1];
	      }
	      else {
		a = dir_rad[0];
		b = dir_rad[1]+sign*range;
	      }
	      
	      if (fabs(a-b) < M_PI*min_angle || fabs(a-b) > M_PI*(1.0-min_angle))
		continue;
	      
	      test_dir[0].x = size*cos(a);
	      test_dir[0].y = size*sin(a);
	      test_dir[1].x = size*cos(b);
	      test_dir[1].y = size*sin(b);
	      
              
	      test_score = scoreCorner(img, p, test_dir, corner_score_oversampling*size, corner_score_oversampling*dead);
              //printf("dir score %f == %f -> %f\n", scoreCorner(img, p, dir), score, test_score);
	      if (test_score > score) {
                //printf("yay better %f!\n", test_score);
		change = true;
		score = test_score;
		dir[0] = test_dir[0];
		dir[1] = test_dir[1];
		dir_rad[0] = a;
		dir_rad[1] = b;
	      }
	    }
	}
      }
    }
    
    void Marker_Corner::refineDirIterative(Mat img, int min_step, int max_step)
    {
      bool change = true;
      double test_score;
      float a, b;
      Point2f test_dir[2];
      
      for(float range=M_PI/min_step;range>=M_PI/max_step;range*=0.5) {
	change = true;
	for(int i=0;i<100 && change;i++) {
	  change = false;
	  for(int sign=-1;sign<=1;sign+=2)
	    for(int coord=0;coord<2;coord++) {
	      if (!coord) {
		a = dir_rad[0]+sign*range;
		b = dir_rad[1];
	      }
	      else {
		a = dir_rad[0];
		b = dir_rad[1]+sign*range;
	      }
	      
	      if (fabs(a-b) < M_PI*min_angle || fabs(a-b) > M_PI*(1.0-min_angle))
		continue;
	      
	      test_dir[0].x = corner_score_size*cos(a);
	      test_dir[0].y = corner_score_size*sin(a);
	      test_dir[1].x = corner_score_size*cos(b);
	      test_dir[1].y = corner_score_size*sin(b);
	      
              
	      test_score = scoreCorner(img, p, test_dir);
              //printf("dir score %f == %f -> %f\n", scoreCorner(img, p, dir), score, test_score);
	      if (test_score > score) {
                //printf("yay better %f!\n", test_score);
		change = true;
		score = test_score;
		dir[0] = test_dir[0];
		dir[1] = test_dir[1];
		dir_rad[0] = a;
		dir_rad[1] = b;
	      }
	    }
	}
      }
    }
    
    
    void Marker_Corner::refineDirIterative(Mat img)
    {
      refineDirIterative(img, dir_step, dir_step_sub);
    }
    
   void Marker_Corner::estimateDir(Mat img)
    {      
      double test_score;
      float a, b;
      //angle
      Point2f test_dir[2];
      Point2f test_p;
      
      if (estimated)
	return;
      
      score = -1000;
      
      for(a=0;a<M_PI;a+=M_PI/dir_step) {
	b = a+M_PI/2;
	test_dir[0].x = corner_score_size*cos(a);
	test_dir[0].y = corner_score_size*sin(a);
	test_dir[1].x = corner_score_size*cos(b);
	test_dir[1].y = corner_score_size*sin(b);
	
	test_score = scoreCorner(img, p, test_dir);
	if (test_score > score) {
	  score = test_score;
	  dir[0] = test_dir[0];
	  dir[1] = test_dir[1];
	  dir_rad[0] = a;
	  dir_rad[1] = b;
	}
      }
      
      refineDirIterative(img, dir_step/2, dir_step_sub);
      
      estimated = true;
    }
    
    /*void Marker_Corner::estimateDir(Mat img)
    {      
      double test_score;
      float a, b;
      //angle
      Point2f test_dir[2];
      
      if (estimated)
	return;
      
      score = -1000;
      
      for(a=0;a<M_PI;a+=M_PI/dir_step)
	for(b=a+M_PI/2*min_angle;b<a+M_PI/2+M_PI/2*(1.0-min_angle);b+=M_PI/dir_step) {
	//b = a+M_PI/2;
	test_dir[0].x = corner_score_size*cos(a);
	test_dir[0].y = corner_score_size*sin(a);
	test_dir[1].x = corner_score_size*cos(b);
	test_dir[1].y = corner_score_size*sin(b);
	
	test_score = scoreCorner_SIMD(img, p, test_dir);
	if (test_score > score) {
	  score = test_score;
	  dir[0] = test_dir[0];
	  dir[1] = test_dir[1];
	  dir_rad[0] = a;
	  dir_rad[1] = b;
	}
      }
      
      refineDirIterative(img, 2, dir_step*dir_step_sub);
      
      if (score > corner_good && abs(p.x-177)<=2 && abs(p.y-160)<=2) {
	cout << score << endl;
	writecorner(img, p, dir);
      }
      
      estimated = true;
    }*/
    
    void Marker_Corner::refine(Mat img, float refine_max, bool force, int dir_step_refine)
    {
      bool change;
      int i, sign, coord;
      float step;
      //angle
      Marker_Corner c;
      /*bool debug = false;
      
      Point2f testp(680/pow(2, scale)*2, 480/pow(2, scale)*2);
      
      testp = testp-p;
      if (testp.x*testp.x + testp.y*testp.y < 25) {
        printf("enable debug at scale %d\n", scale);
        debug = true;
      }*/
      
      if (refined && force == false)
	return;
      
      refined = true;

      score = scoreCorner(img, p, dir);
      
      for(step=refine_max;step>=refine_min_step;step*=0.5) {
	change = true;
	for(i=0;i<1000 && change;i++) {
	  change = false;
	  for(coord=0;coord<2;coord++)
	    for(sign=-1;sign<=1;sign+=2) {
	      c = *this;
	      if (!coord)
		c.p.x +=sign*step;
	      else
		c.p.y +=sign*step;
	      
	      if (c.p.x <= 2*corner_score_size || c.p.y <= 2*corner_score_size || c.p.x >= img.size().width-2*corner_score_size || c.p.y >= img.size().height-2*corner_score_size) 
		continue;
	      
	      c.score = scoreCorner(img, c.p, c.dir);
	      if (dir_step_refine)
 		c.refineDirIterative(img, dir_step_refine, dir_step_refine);
	      
              //if (debug)
                //printf("old score %f new score %f\n", score, c.score);
              
	      if (c.score > score) {
		change = true;
		*this = c;
	      }
	    }
	}
	//if (debug)
          //printf("step %f %d iters\n", step, i);
      }
    }
    
    
void cornerSubPixCP( InputArray _image, Point2f &p,
                      Size win, Size zeroZone, TermCriteria criteria )
{
  const int MAX_ITERS = 100;
  int win_w = win.width * 2 + 1, win_h = win.height * 2 + 1;
  int i, j, k;
  int max_iters = (criteria.type & CV_TERMCRIT_ITER) ? MIN(MAX(criteria.maxCount, 1), MAX_ITERS) : MAX_ITERS;
  double eps = (criteria.type & CV_TERMCRIT_EPS) ? MAX(criteria.epsilon, 0.) : 0;
  eps *= eps; // use square of error in comparsion operations

  cv::Mat src = _image.getMat();

  CV_Assert( win.width > 0 && win.height > 0 );
  CV_Assert( src.cols >= win.width*2 + 5 && src.rows >= win.height*2 + 5 );
  CV_Assert( src.channels() == 1 );

  Mat maskm(win_h, win_w, CV_32F), subpix_buf(win_h+2, win_w+2, CV_32F);
  float* mask = maskm.ptr<float>();

  for( i = 0; i < win_h; i++ )
  {
      float y = (float)(i - win.height)/win.height;
      float vy = std::exp(-y*y);
      for( j = 0; j < win_w; j++ )
      {
          float x = (float)(j - win.width)/win.width;
          mask[i * win_w + j] = (float)(vy*std::exp(-x*x));
      }
  }

  // make zero_zone
  if( zeroZone.width >= 0 && zeroZone.height >= 0 &&
      zeroZone.width * 2 + 1 < win_w && zeroZone.height * 2 + 1 < win_h )
  {
      for( i = win.height - zeroZone.height; i <= win.height + zeroZone.height; i++ )
      {
          for( j = win.width - zeroZone.width; j <= win.width + zeroZone.width; j++ )
          {
              mask[i * win_w + j] = 0;
          }
      }
  }

    Point2f cT = p;
    Point2f cI = p;
    int iter = 0;
    double err = 0;

    do
    {
        Point2f cI2;
        double a = 0, b = 0, c = 0, bb1 = 0, bb2 = 0;

        getRectSubPix(src, Size(win_w+2, win_h+2), cI, subpix_buf, subpix_buf.type());
        const float* subpix = &subpix_buf.at<float>(1,1);

        // process gradient
        for( i = 0, k = 0; i < win_h; i++, subpix += win_w + 2 )
        {
            double py = i - win.height;

            for( j = 0; j < win_w; j++, k++ )
            {
                double m = mask[k];
                double tgx = subpix[j+1] - subpix[j-1];
                double tgy = subpix[j+win_w+2] - subpix[j-win_w-2];
                double gxx = tgx * tgx * m;
                double gxy = tgx * tgy * m;
                double gyy = tgy * tgy * m;
                double px = j - win.width;

                a += gxx;
                b += gxy;
                c += gyy;

                bb1 += gxx * px + gxy * py;
                bb2 += gxy * px + gyy * py;
            }
        }

        double det=a*c-b*b;
        if( fabs( det ) <= DBL_EPSILON*DBL_EPSILON )
            break;

        // 2x2 matrix inversion
        double scale=1.0/det;
        cI2.x = (float)(cI.x + c*scale*bb1 - b*scale*bb2);
        cI2.y = (float)(cI.y - b*scale*bb1 + a*scale*bb2);
        err = (cI2.x - cI.x) * (cI2.x - cI.x) + (cI2.y - cI.y) * (cI2.y - cI.y);
        cI = cI2;
        if( cI.x < 0 || cI.x >= src.cols || cI.y < 0 || cI.y >= src.rows )
            break;
    }
    while( ++iter < max_iters && err > eps );

    // if new point is too far from initial, it means poor convergence.
    // leave initial point as the result
    if( fabs( cI.x - cT.x ) > win.width || fabs( cI.y - cT.y ) > win.height )
        cI = cT;

    p = cI;
}

float ldist(Point2f base, Point2f dir_u, Point2f p)
{
  Point2f v = p-base;
  return abs(dir_u.x*v.y - dir_u.y*v.x);
}

void Marker_Corner::cornerSubPixCPMask( InputArray _image, Point2f &p,
                      Size win, Size zeroZone, TermCriteria criteria )
{
  const int MAX_ITERS = 100;
  int win_w = win.width * 2 + 1, win_h = win.height * 2 + 1;
  int i, j, k;
  int max_iters = (criteria.type & CV_TERMCRIT_ITER) ? MIN(MAX(criteria.maxCount, 1), MAX_ITERS) : MAX_ITERS;
  double eps = (criteria.type & CV_TERMCRIT_EPS) ? MAX(criteria.epsilon, 0.) : 0;
  eps *= eps; // use square of error in comparsion operations

  cv::Mat src = _image.getMat();

  CV_Assert( win.width > 0 && win.height > 0 );
  CV_Assert( src.cols >= win.width*2 + 5 && src.rows >= win.height*2 + 5 );
  CV_Assert( src.channels() == 1 );

  Mat maskm(win_h, win_w, CV_32F), subpix_buf(win_h+2, win_w+2, CV_32F);
  float* mask = maskm.ptr<float>();

  Point2f d1 = Point2f(cos(dir_rad[0]), sin(dir_rad[0]));
  Point2f d2 = Point2f(cos(dir_rad[1]), sin(dir_rad[1]));
  Point2f m = Point2f(win.width, win.height);
  
  float scaler = sqrt(m.x*m.x+m.y*m.y);
  
  for( i = 0; i < win_h; i++ )
  {
      float y = (float)(i - win.height)/win.height;
      float vy = std::exp(-y*y);
      for( j = 0; j < win_w; j++ )
      {
          float x = (float)(j - win.width)/win.width;
          mask[i * win_w + j] = (float)(vy*std::exp(-x*x))*1.0/max(min(ldist(m, d1, Point2f(i, j)), ldist(m, d2, Point2f(i, j))), (float)1.0);
          if (ldist(m, d1, Point2f(i, j)) > size/10 && ldist(m, d2, Point2f(i, j)) > size/10)
            mask[i*win_w+j] = 0;
      }
  }
  
  //imwrite("mask.png", maskm*256);

  // make zero_zone
  if( zeroZone.width >= 0 && zeroZone.height >= 0 &&
      zeroZone.width * 2 + 1 < win_w && zeroZone.height * 2 + 1 < win_h )
  {
      for( i = win.height - zeroZone.height; i <= win.height + zeroZone.height; i++ )
      {
          for( j = win.width - zeroZone.width; j <= win.width + zeroZone.width; j++ )
          {
              mask[i * win_w + j] = 0;
          }
      }
  }

    Point2f cT = p;
    Point2f cI = p;
    int iter = 0;
    double err = 0;

    do
    {
        Point2f cI2;
        double a = 0, b = 0, c = 0, bb1 = 0, bb2 = 0;

        getRectSubPix(src, Size(win_w+2, win_h+2), cI, subpix_buf, subpix_buf.type());
        const float* subpix = &subpix_buf.at<float>(1,1);

        // process gradient
        for( i = 0, k = 0; i < win_h; i++, subpix += win_w + 2 )
        {
            double py = i - win.height;

            for( j = 0; j < win_w; j++, k++ )
            {
                double m = mask[k];
                double tgx = subpix[j+1] - subpix[j-1];
                double tgy = subpix[j+win_w+2] - subpix[j-win_w-2];
                double gxx = tgx * tgx * m;
                double gxy = tgx * tgy * m;
                double gyy = tgy * tgy * m;
                double px = j - win.width;

                a += gxx;
                b += gxy;
                c += gyy;

                bb1 += gxx * px + gxy * py;
                bb2 += gxy * px + gyy * py;
            }
        }

        double det=a*c-b*b;
        if( fabs( det ) <= DBL_EPSILON*DBL_EPSILON )
            break;

        // 2x2 matrix inversion
        double scale=1.0/det;
        cI2.x = (float)(cI.x + c*scale*bb1 - b*scale*bb2);
        cI2.y = (float)(cI.y - b*scale*bb1 + a*scale*bb2);
        err = (cI2.x - cI.x) * (cI2.x - cI.x) + (cI2.y - cI.y) * (cI2.y - cI.y);
        cI = cI2;
        if( cI.x < 0 || cI.x >= src.cols || cI.y < 0 || cI.y >= src.rows )
            break;
    }
    while( ++iter < max_iters && err > eps );

    // if new point is too far from initial, it means poor convergence.
    // leave initial point as the result
    if( fabs( cI.x - cT.x ) > win.width || fabs( cI.y - cT.y ) > win.height )
        cI = cT;

    p = cI;
}
    
    
    void Marker_Corner::refine_gradient(Mat &img, float scale)
    {
      //cornerSubPixCP(img, p, Size(size/6,size/6), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 100, 0.001));
      cornerSubPixCPMask(img, p, Size(size/4,size/4), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 100, 0.001));
    }
    
    
    void Marker_Corner::refine_size(Mat img, float refine_max, bool force, int dir_step_refine, int size, int dead)
    {
      bool change;
      int i, sign, coord;
      float step;
      //angle
      Marker_Corner c;
      float cso = corner_score_oversampling;
      
      if (refined && force == false)
	return;
      
      refined = true;

      dir[0].x = size*cos(dir_rad[0]);
      dir[0].y = size*sin(dir_rad[0]);
      dir[1].x = size*cos(dir_rad[1]);
      dir[1].y = size*sin(dir_rad[1]);
      
      if (p.x <= size || p.y <= size || p.x >= img.size().width - size || p.y >= img.size().height-size) {
        printf("error: too close to border, could not refine\n");
        return;
      }
        
      score = scoreCorner(img, p, dir, cso*size, cso*dead);
      
      for(step=refine_max;step>=refine_min_step;step*=0.5) {
	change = true;
	for(i=0;i<1000 && change;i++) {
	  change = false;
	  for(coord=0;coord<2;coord++)
	    for(sign=-1;sign<=1;sign+=2) {
	      c = *this;
	      if (!coord)
		c.p.x +=sign*step;
	      else
		c.p.y +=sign*step;
	      
	      if (c.p.x <= 2*cso*size || c.p.y <= 2*cso*size || c.p.x >= img.size().width-2*cso*size || c.p.y >= img.size().height-2*cso*size) 
		continue;
	      
	      c.score = scoreCorner(img, c.p, c.dir, cso*size, cso*dead);
	      if (dir_step_refine)
 		c.refineDirIterative_size(img, dir_step_refine, dir_step_refine, size, dead);
              
	      if (c.score > score) {
		change = true;
		*this = c;
	      }
	    }
	}
      }
    }
    
    void Marker_Corner::refine(Mat img, bool force, int dir_step_refine)
    {
      refine(img, refine_max_step, force, dir_step_refine);
    }
    
    void Marker_Corner::estimateDir(Mat img, Mat &paint)
    {
      estimateDir(img);
      
      if (score > corner_ok) {
	line(paint, p, p-15.0/corner_score_size*dir[1], CV_RGB(0,255,0), 1);
	line(paint, p, p-15.0/corner_score_size*dir[0], CV_RGB(255,0,0), 1);
      }
    }


void Corner::paint(Mat &img)
{
  Size s;
  char buf[64];
  circle(img, p, 1, CV_RGB(0,0,0), 2);
  circle(img, p, 1, CV_RGB(255,255,255));
  sprintf(buf, "%d/%d", id.x, id.y);
  putText(img, buf, p, FONT_HERSHEY_PLAIN, 0.7, CV_RGB(0,0,0), 2);
  putText(img, buf, p, FONT_HERSHEY_PLAIN, 0.7, CV_RGB(127,255,127), 1);
  s = getTextSize(buf, FONT_HERSHEY_PLAIN, 0.7, 2, NULL);
  sprintf(buf, "%d", page);
  putText(img, buf, p+Point2f(0,s.height+1), FONT_HERSHEY_PLAIN, 0.7, CV_RGB(0,0,0), 2);
  putText(img, buf, p+Point2f(0,s.height+1), FONT_HERSHEY_PLAIN, 0.7, CV_RGB(127,255,127), 1);
}
    
double pattern_score(Mat patt)
{
  int white = 0, black = 0;
  
  //top
  white += patt.at<uchar>(0, 2);
  white += patt.at<uchar>(1, 2);
  white += patt.at<uchar>(1, 3);
  white += patt.at<uchar>(1, 4);
  white += patt.at<uchar>(1, 5);
  white += patt.at<uchar>(1, 6);
  white += patt.at<uchar>(0, 6);
  //right
  white += patt.at<uchar>(2, 8);
  white += patt.at<uchar>(2, 7);
  white += patt.at<uchar>(3, 7);
  white += patt.at<uchar>(4, 7);
  white += patt.at<uchar>(5, 7);
  white += patt.at<uchar>(6, 7);
  white += patt.at<uchar>(6, 8);
  //bottom
  white += patt.at<uchar>(8, 2);
  white += patt.at<uchar>(7, 2);
  white += patt.at<uchar>(7, 3);
  white += patt.at<uchar>(7, 4);
  white += patt.at<uchar>(7, 5);
  white += patt.at<uchar>(7, 6);
  white += patt.at<uchar>(8, 6);
  //left
  white += patt.at<uchar>(2, 0);
  white += patt.at<uchar>(2, 1);
  white += patt.at<uchar>(3, 1);
  //4/1 is negative dir hole
  white += patt.at<uchar>(5, 1);
  white += patt.at<uchar>(6, 1);
  white += patt.at<uchar>(6, 0);
  //direction hole
  white += patt.at<uchar>(6, 4);
  
  //top-left
  black += patt.at<uchar>(0, 1);
  black += patt.at<uchar>(1, 1);
  black += patt.at<uchar>(1, 0);
  //top-right
  black += patt.at<uchar>(0, 7);
  black += patt.at<uchar>(1, 7);
  black += patt.at<uchar>(1, 8);
  //bottom-right
  black += patt.at<uchar>(8, 7);
  black += patt.at<uchar>(7, 7);
  black += patt.at<uchar>(7, 8);
  //bottom-left
  black += patt.at<uchar>(7, 0);
  black += patt.at<uchar>(7, 1);
  black += patt.at<uchar>(8, 1);
  
  //inner square
  black += patt.at<uchar>(2, 2);
  black += patt.at<uchar>(2, 3);
  black += patt.at<uchar>(2, 4);
  black += patt.at<uchar>(2, 5);
  black += patt.at<uchar>(2, 6);
  black += patt.at<uchar>(3, 6);
  //4x6 is dir hole
  black += patt.at<uchar>(5, 6);
  black += patt.at<uchar>(6, 6);
  black += patt.at<uchar>(6, 5);
  black += patt.at<uchar>(6, 4);
  black += patt.at<uchar>(6, 3);
  black += patt.at<uchar>(6, 2);
  black += patt.at<uchar>(5, 2);
  black += patt.at<uchar>(4, 2);
  black += patt.at<uchar>(3, 2);
  
  //negative direction hole
  black += patt.at<uchar>(4, 1);
  
  //outer /= 28;
  //inner /= 28;
  
  if (white-black <= 28*5)
    return -1.0;
  
  for(int y=0;y<9;y++)
    for(int x=0;x<9;x++)
	patt.at<uchar>(y, x) = min(max(0, (patt.at<uchar>(y, x) - black/28)*28*256/(white-black)), 255);

  black = 0;
  white = 0;

  //top
  white += patt.at<uchar>(0, 2);
  white += patt.at<uchar>(1, 2);
  white += patt.at<uchar>(1, 3);
  white += patt.at<uchar>(1, 4);
  white += patt.at<uchar>(1, 5);
  white += patt.at<uchar>(1, 6);
  white += patt.at<uchar>(0, 6);
  //right
  white += patt.at<uchar>(2, 8);
  white += patt.at<uchar>(2, 7);
  white += patt.at<uchar>(3, 7);
  white += patt.at<uchar>(4, 7);
  white += patt.at<uchar>(5, 7);
  white += patt.at<uchar>(6, 7);
  white += patt.at<uchar>(6, 8);
  //bottom
  white += patt.at<uchar>(8, 2);
  white += patt.at<uchar>(7, 2);
  white += patt.at<uchar>(7, 3);
  white += patt.at<uchar>(7, 4);
  white += patt.at<uchar>(7, 5);
  white += patt.at<uchar>(7, 6);
  white += patt.at<uchar>(8, 6);
  //left
  white += patt.at<uchar>(2, 0);
  white += patt.at<uchar>(2, 1);
  white += patt.at<uchar>(3, 1);
  //4/1 is negative dir hole
  black += patt.at<uchar>(4, 1);
  white += patt.at<uchar>(5, 1);
  white += patt.at<uchar>(6, 1);
  white += patt.at<uchar>(6, 0);
  //direction hole
  white += patt.at<uchar>(6, 4);
  
  //top-left
  black += patt.at<uchar>(0, 1);
  black += patt.at<uchar>(1, 1);
  black += patt.at<uchar>(1, 0);
  //top-right
  black += patt.at<uchar>(0, 7);
  black += patt.at<uchar>(1, 7);
  black += patt.at<uchar>(1, 8);
  //bottom-right
  black += patt.at<uchar>(8, 7);
  black += patt.at<uchar>(7, 7);
  black += patt.at<uchar>(7, 8);
  //bottom-left
  black += patt.at<uchar>(7, 0);
  black += patt.at<uchar>(7, 1);
  black += patt.at<uchar>(8, 1);
  
  //inner square
  black += patt.at<uchar>(2, 2);
  black += patt.at<uchar>(2, 3);
  black += patt.at<uchar>(2, 4);
  black += patt.at<uchar>(2, 5);
  black += patt.at<uchar>(2, 6);
  black += patt.at<uchar>(3, 6);
  //dir hole
  white += patt.at<uchar>(4, 6);
  black += patt.at<uchar>(5, 6);
  black += patt.at<uchar>(6, 6);
  black += patt.at<uchar>(6, 5);
  black += patt.at<uchar>(6, 4);
  black += patt.at<uchar>(6, 3);
  black += patt.at<uchar>(6, 2);
  black += patt.at<uchar>(5, 2);
  black += patt.at<uchar>(4, 2);
  black += patt.at<uchar>(3, 2);
  
  //negative direction hole
  black += patt.at<uchar>(4, 1);
  
  //outer /= 28;
  //inner /= 28;
  //printf("= %d %d\n", white, black);

  //return  (double)white / (black+1) * (patt.at<uchar>(4, 6)-patt.at<uchar>(4, 1)) / (white-black+1);
  return  (double)white / (black+1) * (patt.at<uchar>(4, 6)-patt.at<uchar>(4, 1)) / (28.0*256.0);
  //return  (double)white / (black+1) * patt.at<uchar>(4, 6) / (patt.at<uchar>(4, 1)+1);
}

    /*Marker(double ix, double iy, double iscore)
    {
      score = iscore;
    }*/
    
    int Marker::pointMarkerTest(Point2f p)
    {      
      int mask = 7;
      
      if (!marker_neighbour_sure_count && score < pattern_score_sure)
	return mask;
      
      if (neighbours < marker_neighbour_sure_count && score < pattern_score_sure)
	return mask;
      
      if (id == -1 || page == -1)
	return mask;
      
      if (norm(corners[0].p - p) > marker_maxsize*2)
	return mask ;
      
      Point2f h = corners[1].p - corners[2].p;
      Point2f v = corners[1].p - corners[0].p;
      
      vector<Point2f> box = vector<Point2f>(4);
      vector<Point2f> inner = vector<Point2f>(4);
      
      inner[0] = corners[0].p+(-h+v)*0.2;
      inner[1] = corners[1].p+(-h-v)*0.2;
      inner[2] = corners[2].p+(h-v)*0.2;
      inner[3] = corners[3].p+(h+v)*0.2;
      
      box[0] = inner[0]+h;
      box[1] = inner[1]+h;
      box[2] = inner[2]-h;
      box[3] = inner[3]-h;
      
      if (pointPolygonTest(box, p, false) >= 0)
	return 0;
      
      box[0] = inner[0]-v;
      box[1] = inner[1]+v;
      box[2] = inner[2]+v;
      box[3] = inner[3]-v;
      
      if (pointPolygonTest(box, p, false) >= 0)
	return 0;
      
      if (norm(corners[0].p - p) < marker_minsize)
	mask &= 4;
      
      if (norm(corners[1].p - p) < marker_minsize)
	return 0;
      
      if (norm(corners[2].p - p) < marker_minsize)
	mask &= 1;
      
      return mask;
    }
    
    int Marker::calcId(Mat &patt)
    {
      int white = 0, black = 0;
      int newid;
      Mat small;
      
#ifdef WRITE_PROJECTED_IDS
      char buf[64];
      sprintf(buf, "id_%d_r.png", global_counter);
      imwrite(buf, patt);
#endif
      
      if (id_sharpen) {
	float sharpen = 1.0;
	Mat kern = (Mat_<float>(3, 3) << 0,-sharpen,0,-sharpen,4.0*sharpen+1.0,-sharpen,0,-sharpen,0);
	Mat sharp;
	filter2D(patt, sharp, patt.depth(), kern);
#ifdef WRITE_PROJECTED_IDS
	sprintf(buf, "id_%d_s.png", global_counter);
	imwrite(buf, sharp);
#endif
	sharp(Rect(2, 2, 5, 5)).copyTo(small);
      }
      else
	patt(Rect(2, 2, 5, 5)).copyTo(small);
      
      //top
      white += patt.at<uchar>(0, 2);
      white += patt.at<uchar>(1, 2);
      white += patt.at<uchar>(1, 3);
      white += patt.at<uchar>(1, 4);
      white += patt.at<uchar>(1, 5);
      white += patt.at<uchar>(1, 6);
      white += patt.at<uchar>(0, 6);
      //right
      white += patt.at<uchar>(2, 8);
      white += patt.at<uchar>(2, 7);
      white += patt.at<uchar>(3, 7);
      white += patt.at<uchar>(4, 7);
      white += patt.at<uchar>(5, 7);
      white += patt.at<uchar>(6, 7);
      white += patt.at<uchar>(6, 8);
      //bottom
      white += patt.at<uchar>(8, 2);
      white += patt.at<uchar>(7, 2);
      white += patt.at<uchar>(7, 3);
      white += patt.at<uchar>(7, 4);
      white += patt.at<uchar>(7, 5);
      white += patt.at<uchar>(7, 6);
      white += patt.at<uchar>(8, 6);
      //left
      white += patt.at<uchar>(2, 0);
      white += patt.at<uchar>(2, 1);
      white += patt.at<uchar>(3, 1);
      //4/1 is negative dir hole
      white += patt.at<uchar>(5, 1);
      white += patt.at<uchar>(6, 1);
      white += patt.at<uchar>(6, 0);
      //direction hole
      white += patt.at<uchar>(6, 4);
      
      //top-left
      black += patt.at<uchar>(0, 1);
      black += patt.at<uchar>(1, 1);
      black += patt.at<uchar>(1, 0);
      //top-right
      black += patt.at<uchar>(0, 7);
      black += patt.at<uchar>(1, 7);
      black += patt.at<uchar>(1, 8);
      //bottom-right
      black += patt.at<uchar>(8, 7);
      black += patt.at<uchar>(7, 7);
      black += patt.at<uchar>(7, 8);
      //bottom-left
      black += patt.at<uchar>(7, 0);
      black += patt.at<uchar>(7, 1);
      black += patt.at<uchar>(8, 1);
      
      //inner square
      black += patt.at<uchar>(2, 2);
      black += patt.at<uchar>(2, 3);
      black += patt.at<uchar>(2, 4);
      black += patt.at<uchar>(2, 5);
      black += patt.at<uchar>(2, 6);
      black += patt.at<uchar>(3, 6);
      //4x6 is dir hole
      black += patt.at<uchar>(5, 6);
      black += patt.at<uchar>(6, 6);
      black += patt.at<uchar>(6, 5);
      black += patt.at<uchar>(6, 4);
      black += patt.at<uchar>(6, 3);
      black += patt.at<uchar>(6, 2);
      black += patt.at<uchar>(5, 2);
      black += patt.at<uchar>(4, 2);
      black += patt.at<uchar>(3, 2);
      
      //negative direction hole
      black += patt.at<uchar>(4, 1);
      
      //outer /= 28;
      //inner /= 28;
      
      if (white-black <= 28*5)
	return -1;
      
      for(int y=0;y<5;y++)
	for(int x=0;x<5;x++)
	small.at<uchar>(y, x) = min(max(0, (small.at<uchar>(y, x) - black/28)*28*256/(white-black)), 255);
      
	Mat small_cp;
      small_cp = small.clone();
	
      threshold(small, small, 127, 1, THRESH_BINARY);
      
      newid = 0;
      newid += small.at<uchar>(1, 1);
      newid += small.at<uchar>(1, 2)*2;
      newid += small.at<uchar>(1, 3)*4;
      newid += small.at<uchar>(2, 1)*8;
      newid += small.at<uchar>(2, 2)*16;
      newid += small.at<uchar>(2, 3)*32;
      newid += small.at<uchar>(3, 1)*64;
      newid += small.at<uchar>(3, 2)*128;
      newid += small.at<uchar>(3, 3)*256;
      
#ifdef WRITE_PROJECTED_IDS
      sprintf(buf, "id_%d_t=%d.png", global_counter++, newid);
      imwrite(buf, small);
#endif
	
      return newid;
    }
    
    void Marker::bigId_affine(Mat img, Point2f start, Point2f h, Point2f v, int &big_id, double &big_score)
    {      
      Mat box, affine;
      vector<Point2f> test_box = vector<Point2f>(3);
      big_id = -1;
      
      test_box[0] = start;
      test_box[1] = start+v;
      test_box[2] = start+v-h;
      //test_box[3] = corners[0];
      
      affine = getAffineTransform(test_box, box_corners);
      warpAffine(img, box, affine, Size(9, 9), INTER_LINEAR);
      
      box = 255 - box;
      
      big_score = pattern_score(box);
      
      if (big_score > pattern_score_ok)
	big_id = calcId(box);
    }
    
    void Marker::bigId(Mat img, vector<Marker_Corner> &corners, int &big_id, double &big_score)
    {      
      Mat box, pers;
      big_id = -1;
      
      vector<Point2f> points(4);

      for(int i=0;i<4;i++) {
	//corners[i].estimateDir(img);
	//corners[i].refine(img);
	points[i] = corners[i].p+b_o;
      }
      
      pers = getPerspectiveTransform(points, box_corners_pers);
      warpPerspective(img, box, pers, Size(9, 9), INTER_LINEAR); 
      
      box = 255 - box;
      
      big_score = pattern_score(box);
      
      big_id = calcId(box);
    }
    
    void Marker::getPoints(Point2f &p1, int &x1, int &y1, Point2f &p2, int &x2, int &y2)
    {     
      /*if (dir_step_refine) {
	corners[0].refine(img);
	corners[1].refine(img);
      }*/
      
      if (id % 2 == 0) {
	x1 = id % 32;
	x2 = x1;
	y1 = id / 32*2;
	y2 = id / 32*2+1;
	p1 = corners[1].p;
	p2 = corners[0].p;
      }
      else {
	x1 = id % 32;
	x2 = x1;
	y1 = id / 32*2+1;
	y2 = id / 32*2+2;
	p1 = corners[1].p;
	p2 = corners[0].p;
      }
    }
    
    void Marker::getPoints(cv::Point2f p[4], cv::Point2i c[4])
    {          
      if (id % 2 == 0) {
	c[0].x = id % 32+1;
	c[1].x = id % 32+1;
	c[2].x = id % 32;
	c[3].x = id % 32;
	c[0].y = id / 32*2;
	c[1].y = id / 32*2+1;
	c[2].y = id / 32*2;
	c[3].y = id / 32*2+1;
	p[0] = corners[1].p;
	p[1] = corners[0].p;
	p[2] = corners[2].p;
	p[3] = corners[3].p;
      }
      else {
	c[0].x = id % 32+1;
	c[1].x = id % 32+1;
	c[3].x = id % 32;
	c[2].x = id % 32;
	c[0].y = id / 32*2+1;
	c[1].y = id / 32*2+2;
	c[2].y = id / 32*2+1;
	c[3].y = id / 32*2+2;
	p[0] = corners[1].p;
	p[1] = corners[0].p;
	p[2] = corners[2].p;
	p[3] = corners[3].p;
      }
    }
    
    void Marker::getCorners(Marker_Corner c[4])
    {          
      float d;
      Point2f v;
      float mindist;
      
      for(int i=0;i<4;i++) {
	c[i] = corners[i];
        v = corners[(i+3)%4].p-corners[i].p;
        mindist = sqrt(v.x*v.x+v.y*v.y);
        c[i].dir_rad[1] = atan2(v.y, v.x);
        v = corners[i].p-corners[(i+1)%4].p;
        mindist += sqrt(v.x*v.x+v.y*v.y);
        c[i].dir_rad[0] = atan2(v.y, v.x);
        c[i].size = mindist*0.5;
      }
      
      
      if (id % 2 == 0) {
	c[1].coord.x = id % 32+1;
	c[0].coord.x = id % 32+1;
	c[2].coord.x = id % 32;
	c[3].coord.x = id % 32;
	c[1].coord.y = id / 32*2;
	c[0].coord.y = id / 32*2+1;
	c[2].coord.y = id / 32*2;
	c[3].coord.y = id / 32*2+1;
      }
      else {
	c[1].coord.x = id % 32+1;
	c[0].coord.x = id % 32+1;
	c[3].coord.x = id % 32;
	c[2].coord.x = id % 32;
	c[1].coord.y = id / 32*2+1;
	c[0].coord.y = id / 32*2+2;
	c[2].coord.y = id / 32*2+1;
	c[3].coord.y = id / 32*2+2;
      }
    }
    
    /*void refine(Mat img)
    {      
      double next_score;
      bool change;
      
      Mat box, pers;
      vector<Point2f> test_box = vector<Point2f>(4);
      
      test_box[0] = corners[0].p;
      test_box[1] = corners[1].p;
      test_box[2] = corners[2].p;
      test_box[3] = corners[3].p;
      
      //printf("input score %f\n", score);
      
      for(double step=4.0;step>=0.16;step*=0.5) {
	change = true;
	for(int i=0;i<100 && change;i++) {
	  change = false;
	  int index = 3;
	    for(int coord=0;coord<2;coord++)
	      for(int sign1=-1;sign1<=1;sign1++) 
	      for(int sign2=-1;sign2<=1;sign2++){
		test_box[index].x = corners[index].p.x+sign1*step;
		test_box[index].y = corners[index].p.y+sign2*step;
		
		pers = getPerspectiveTransform(test_box, box_corners_pers);
		warpPerspective(img, box, pers, Size(9, 9), INTER_LINEAR);
		
		next_score = pattern_score(box);
		
		if (next_score > score) {
		  change = true;
		  corners[index].p = test_box[index];
		  score = next_score;
		}
	      }
	}
      }
      
      for(double step=0.512;step>=0.16;step*=0.5) {
	change = true;
	for(int i=0;i<100 && change;i++) {
	  change = false;
	  for(int index=0;index<4;index++)
	    for(int coord=0;coord<2;coord++)
	      for(int sign1=-1;sign1<=1;sign1++) 
	      for(int sign2=-1;sign2<=1;sign2++){
		test_box[index].x = corners[index].p.x+sign1*step;
		test_box[index].y = corners[index].p.y+sign2*step;
		
		pers = getPerspectiveTransform(test_box, box_corners_pers);
		warpPerspective(img, box, pers, Size(9, 9), INTER_LINEAR);
		
		next_score = pattern_score(box);
		
		if (next_score > score) {
		  change = true;
		  corners[index].p = test_box[index];
		  score = next_score;
		}
	      }
	}
      }
	    
      //printf("refined score %f\n", score);
    }*/
    
    
void nomax(Mat &img, Mat &out)
{
  Mat dil;
  out.create(img.size(), CV_8UC1);
  
  Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    
  GaussianBlur(img, out, Size(3, 3), 0.8);

  dilate(out, dil, element);

  for(int y=0;y<dil.size().height;y++)
    for(int x=0;x<dil.size().width;x++)
      if (out.at<uchar>(y, x) != dil.at<uchar>(y, x))
	out.at<uchar>(y, x) = 0;
}

static inline float dir_rad_diff(float rada, float radb)
{ 
  float diff = fmod(abs(rada - radb),M_PI);
  
  diff= min(diff, (float)abs(M_PI-diff));
  
  return diff;
}
    

    int smallidtomask(int id, int x, int y)
    {
      int j = (id / 32) * 2 + (id % 2) + y;
      int i = (id % 32) + x;
      
      return (j*13 + i * 7) % 512;
    }

    int idtomask(int id)
    {
      if ((id&2==2)) 
	return id ^ 170;
      else
	return id ^ 340;
    }

    int masktoid(int mask)
    {
      if ((mask&2==2)) 
	return mask ^ 170;
      else
	return mask ^ 340;
    }
    
    Marker::Marker(Mat input, Mat img, double marker_score, Marker_Corner *p1, Marker_Corner *p2, Marker_Corner *p3, float s, int inpage, int inid)
    {            
      int big_id, big_newid;
      Mat pers;
      scale = s;
      double big_score, big_newscore;
      corners = vector<Marker_Corner>(4);
      vector<Point2f> points = vector<Point2f>(4);
      vector<Marker_Corner> corners_page(4);
      vector<int> pages;
      vector<float> scores;
      Marker_Corner c;
      corners[0] = *p1;
      corners[1] = *p2;
      corners[2] = *p3;
      neighbours = 0;
      
      corners[3] = Marker_Corner(p1->p + (p3->p-p2->p), scale);
      if (corners[3].p.x < 2*corner_score_size || corners[3].p.y < 2*corner_score_size
          || corners[3].p.x > img.size().width-2*corner_score_size || corners[3].p.y > img.size().height-2*corner_score_size){
	id = -1;
	page = -1;
	score -1;
	return;
      }
      corners[3].estimateDir(img);
      if (dir_rad_diff(corners[0].dir_rad[0], corners[3].dir_rad[1]) > marker_corner_dir_rad_dist
       || dir_rad_diff(corners[0].dir_rad[1], corners[3].dir_rad[0]) > marker_corner_dir_rad_dist) {
	id = -1;
	page = -1;
	score -1;
	return;
      }
      //TODO can we replace estimatedir with dir from p1?
      /*corners[3] = *p1;
      corners[3].p += (p3->p-p2->p);
      corners[3].dir_rad[0] += M_PI/2;
      corners[3].dir_rad[1] -= M_PI/2;*/
      corners[3].refine(img, true, 0);
      
      corners[0].scale = scale;
      corners[1].scale = scale;
      corners[2].scale = scale;
      corners[3].scale = scale;
      
      points[0] = corners[0].p+b_o;
      points[1] = corners[1].p+b_o;
      points[2] = corners[2].p+b_o;
      points[3] = corners[3].p+b_o;
      
      pers = getPerspectiveTransform(points, box_corners_pers);
      warpPerspective(img, input, pers, Size(9, 9), INTER_LINEAR);
      
      id = masktoid(calcId(input));
      //printf("found id %d\n", id);
      if (inid != -1 && id != inid) {
	id = -1;
	page = -1;
	score -1;
	return;
      }
      score = pattern_score(input);
      
      Point2f h = corners[1].p - corners[2].p;
      Point2f v = corners[1].p - corners[0].p;
      
      //to the right
      corners_page[0] = Marker_Corner(corners[0].p+(corners[0].p-corners[3].p), scale);
      corners_page[1] = Marker_Corner(corners[1].p+(corners[1].p-corners[2].p), scale);
      corners_page[2] = Marker_Corner(corners[1].p, scale);
      corners_page[3] = Marker_Corner(corners[0].p, scale);
      bigId(img, corners_page, big_id, big_score);
      big_id = big_id ^ smallidtomask(id, 1, 0);
      if (big_id >= 0 && big_id <= 512) {
	if (inpage != -1 && inpage == big_id && big_score > pattern_score_ok)
	  goto bigid_correct;
	pages.push_back(big_id);
	scores.push_back(big_score);
      }
      
      corners_page[0] = Marker_Corner(corners[1].p, scale);
      corners_page[1] = Marker_Corner(corners[1].p+(corners[1].p-corners[0].p), scale);
      corners_page[2] = Marker_Corner(corners[2].p+(corners[2].p-corners[3].p), scale);
      corners_page[3] = Marker_Corner(corners[2].p, scale);
      bigId(img, corners_page, big_id, big_score);
      big_id = big_id ^ smallidtomask(id, 0, -1);
      for(int i=0;i<pages.size();i++)
	if (pages[i] == big_id) {
	  scores[i] += big_score;
	  break;
	}
      if (big_id >= 0 && big_id <= 512) {
	if (inpage != -1 && inpage == big_id && big_score > pattern_score_ok)
	  goto bigid_correct;
	pages.push_back(big_id);
	scores.push_back(big_score);
      }
      
      corners_page[0] = Marker_Corner(corners[3].p, scale);
      corners_page[1] = Marker_Corner(corners[2].p, scale);
      corners_page[2] = Marker_Corner(corners[2].p+(corners[2].p-corners[1].p), scale);
      corners_page[3] = Marker_Corner(corners[3].p+(corners[3].p-corners[0].p), scale);
      bigId(img, corners_page, big_id, big_score);
      big_id = big_id ^ smallidtomask(id, -1, 0);
      for(int i=0;i<pages.size();i++)
	if (pages[i] == big_id) {
	  scores[i] += big_score;
	  break;
	}
      if (big_id >= 0 && big_id <= 512) {
	if (inpage != -1 && inpage == big_id && big_score > pattern_score_ok)
	  goto bigid_correct;
	pages.push_back(big_id);
	scores.push_back(big_score);
      }
      
      corners_page[0] = Marker_Corner(corners[0].p+(corners[0].p-corners[1].p), scale);
      corners_page[1] = Marker_Corner(corners[0].p, scale);
      corners_page[2] = Marker_Corner(corners[3].p, scale);
      corners_page[3] = Marker_Corner(corners[3].p+(corners[3].p-corners[2].p), scale);
      bigId(img, corners_page, big_id, big_score);
      big_id = big_id ^ smallidtomask(id, 0, 1);
      for(int i=0;i<pages.size();i++)
	if (pages[i] == big_id) {
	  scores[i] += big_score;
	  break;
	}
      if (big_id >= 0 && big_id <= 512) {
	if (inpage != -1 && inpage == big_id && big_score > pattern_score_ok)
	  goto bigid_correct;
	pages.push_back(big_id);
	scores.push_back(big_score);
      }
      
	
      big_score = 0.0;
      for(int i=0;i<pages.size();i++)
	if (scores[i] > big_score) {
	  big_score = scores[i];
	  big_id = pages[i];
	}
	
      bigid_correct :
      
      if (big_score > pattern_score_ok) {
	score += big_score;
	page = big_id;
	corners[0].page = page;
	corners[1].page = page;
	corners[2].page = page;
	corners[3].page = page;
	corners[0].score += score;
	corners[1].score += score;
	corners[2].score += score;
	corners[3].score += score;
      }
      else {
	id = -1;
	page = -1;
      }
      
      corners[0].p = corners[0].p * scale + Point2f(scale/2, scale/2);
      corners[1].p = corners[1].p * scale + Point2f(scale/2, scale/2);
      corners[2].p = corners[2].p * scale + Point2f(scale/2, scale/2);
      corners[3].p = corners[3].p * scale + Point2f(scale/2, scale/2);
    }
    
    void Marker::filterPoints(Gridstore *candidates, float scale) 
    {      
      if (filtered)
	return;
      
      if (!marker_neighbour_sure_count && score < pattern_score_sure)
	return;
      
      assert(id != -1 && page != -1 && (neighbours >= marker_neighbour_sure_count || score >= pattern_score_sure));
      
      filtered = true;
      
      vector<void*> allcorners = candidates->getWithin(corners[0].p, marker_maxsize);
      
      for(uint32_t i=0;i<allcorners.size();i++)
	if (((Marker_Corner*)allcorners[i])->mask)
	  ((Marker_Corner*)allcorners[i])->mask &= pointMarkerTest(((Marker_Corner*)allcorners[i])->p*scale);

    }
    
    void Marker::neighbours_inc(Gridstore *candidates, float scale) 
    {
      neighbours++;
      
      if (candidates && neighbours >= marker_neighbour_sure_count && neighbours-1 < marker_neighbour_sure_count)
	filterPoints(candidates, scale);
    }
    
    void Marker::neighbour_check(Marker *n, Gridstore *candidates, float scale) 
    {
      Point2f vx = corners[1].p-corners[2].p;
      Point2f vy = corners[1].p-corners[0].p;
      Point2f p;
      float d, maxd = norm(vx+vy)*0.2;
      
      if (n->id == id-2) {
	p = corners[0].p-vx-vx;
	d = norm(p-n->corners[0].p);
	if (d < maxd) {
	  n->neighbours_inc(candidates, scale);
	  neighbours_inc(candidates, scale);
	}
      }
      else if (n->id == id+2) {
	p = corners[0].p+vx+vx;
	d = norm(p-n->corners[0].p);
	if (d < maxd) {
	  n->neighbours_inc(candidates, scale);
	  neighbours_inc(candidates, scale);
	}
      }
      
      if (n->id == id-32) {
	p = corners[0].p+vy+vy;
	d = norm(p-n->corners[0].p);
	if (d < maxd) {
	  n->neighbours_inc(candidates, scale);
	  neighbours_inc(candidates, scale);
	}
      }
      else if (n->id == id+32) {
	p = corners[0].p-vy-vy;
	d = norm(p-n->corners[0].p);
	if (d < maxd) {
	  n->neighbours_inc(candidates, scale);
	  neighbours_inc(candidates, scale);
	}
      }
      
      if (id % 2 == 0) {
	if (n->id == id-1) {
	  p = corners[0].p-vy-vx;
	  d = norm(p-n->corners[0].p);
	  if (d < maxd) {
	    n->neighbours_inc(candidates, scale);
	    neighbours_inc(candidates, scale);
	  }
	}
	else if (n->id == id+1) {
	  p = corners[0].p-vy+vx;
	  d = norm(p-n->corners[0].p);
	  if (d < maxd) {
	    n->neighbours_inc(candidates, scale);
	    neighbours_inc(candidates, scale);
	  }
	}
	if (n->id == id-33) {
	  p = corners[0].p+vy-vx;
	  d = norm(p-n->corners[0].p);
	  if (d < maxd) {
	    n->neighbours_inc(candidates, scale);
	    neighbours_inc(candidates, scale);
	  }
	}
	else if (n->id == id-31) {
	  p = corners[0].p+vy+vx;
	  d = norm(p-n->corners[0].p);
	  if (d < maxd) {
	    n->neighbours_inc(candidates, scale);
	    neighbours_inc(candidates, scale);
	  }
	}
      }
      else {
	if (n->id == id-1) {
	  p = corners[0].p+vy-vx;
	  d = norm(p-n->corners[0].p);
	  if (d < maxd) {
	    n->neighbours_inc(candidates, scale);
	    neighbours_inc(candidates, scale);
	  }
	}
	else if (n->id == id+1) {
	  p = corners[0].p+vy+vx;
	  d = norm(p-n->corners[0].p);
	  if (d < maxd) {
	    n->neighbours_inc(candidates, scale);
	    neighbours_inc(candidates, scale);
	  }
	}
	if (n->id == id+31) {
	  p = corners[0].p-vy-vx;
	  d = norm(p-n->corners[0].p);
	  if (d < maxd) {
	    n->neighbours_inc(candidates, scale);
	    neighbours_inc(candidates, scale);
	  }
	}
	else if (n->id == id+33) {
	  p = corners[0].p-vy+vx;
	  d = norm(p-n->corners[0].p);
	  if (d < maxd) {
	    n->neighbours_inc(candidates, scale);
	    neighbours_inc(candidates, scale);
	  }
	}
      }
      
      /*if (abs(m->id - id) <= 2) {
	m->neighbours_inc(canditate_corners, scale);
	neighbours_inc(canditate_corners, scale);
      }
      if (abs(m->id-32-id) <= 1) {
	m->neighbours_inc(canditate_corners, scale);
	neighbours_inc(canditate_corners, scale);
      }
      if (abs(m->id+32-id) <= 1) {
	m->neighbours_inc(canditate_corners, scale);
	neighbours_inc(canditate_corners, scale);
      }*/
    }
    
    void Marker::filter(Gridstore *candidates, vector<Marker> &markers, float scale) 
    {
      if (!marker_neighbour_sure_count && score < pattern_score_sure)
	return;
      
      if (candidates && score >= pattern_score_sure)
	filterPoints(candidates, scale);
      
      Marker *m;
      /*if (id != -1)
	for(uint32_t i=0;i<canditate_corners.size();i++) {
	  if (canditate_corners[i].mask)
	    canditate_corners[i].mask &= pointMarkerTest(canditate_corners[i].p*scale);
	}*/
      for(uint i=0;i<markers.size();i++) {
	m = &markers[i];
	if (m->page != page) continue;
	if (m->id == id) continue; //FIXME mark both as invalid?
	if (abs(m->id - id) <= 2)
	  neighbour_check(m, candidates, scale);
	if (abs(m->id-32-id) <= 1)
	  neighbour_check(m, candidates, scale);
	if (abs(m->id+32-id) <= 1)
	  neighbour_check(m, candidates, scale);
      }
    }
    
    void Marker::paint(Mat &paint) {
      if (id != -1) {
	line(paint, corners[0].p, corners[1].p, CV_RGB (255, 0, 255), 1);
	line(paint, corners[1].p, corners[2].p, CV_RGB (200, 0, 200), 1);
	line(paint, corners[2].p, corners[3].p, CV_RGB (150, 0, 150), 1);
	line(paint, corners[3].p, corners[0].p, CV_RGB (50, 0, 50), 1);
      }
      else {
	line(paint, corners[0].p, corners[1].p, CV_RGB (255, 0, 0), 1);
	line(paint, corners[1].p, corners[2].p, CV_RGB (200, 0, 0), 1);
	line(paint, corners[2].p, corners[3].p, CV_RGB (150, 0, 0), 1);
	line(paint, corners[3].p, corners[0].p, CV_RGB (50, 0, 0), 1);
      }
      
      Point p = (corners[0].p + corners[1].p + corners[2].p + corners[3].p)*0.25-Point2f(10,0);
      Size s;
      char buf[64];
      sprintf(buf, "%d/%d", id, page);
      putText(paint, buf, p, FONT_HERSHEY_PLAIN, 0.7, CV_RGB(0,0,0), 2);
      putText(paint, buf, p, FONT_HERSHEY_PLAIN, 0.7, CV_RGB(127,255,127), 1);
      s = getTextSize(buf, FONT_HERSHEY_PLAIN, 0.7, 2, NULL);
    }
    
    Marker Marker::operator=(Marker m)
    {
      score = m.score;
      corners = m.corners;
      id = m.id;
      page = m.page;
      neighbours = m.neighbours;
      scale = m.scale;
      filtered = m.filtered;
	    
      return *this;
    }


float pattern_prescore(Mat img, Point2f p1, Point2f p2)
{
  Point2f h = (p2 - p1); // /5?
  Point2f v = Point2f(-h.y, h.x);
  h *= 0.15;
  v *= 0.15;
  
  int white = img.at<uchar>(p1 + h + v) + img.at<uchar>(p1 - h - v);
  int black = img.at<uchar>(p1 - h + v) + img.at<uchar>(p1 + h - v);
  
  return (float)(white-black) / (256);
}

static inline float snorm(Point2f p)
{
  return p.x*p.x+p.y*p.y;
}

void detect_marker(Marker_Corner *start, Mat harry, Gridstore &candidates, vector<Marker> &markers, Mat img, Mat smallblur, int scale)
{
  Marker marker;
  vector<Point2f> last_corner;
  Mat box(Size(9, 9), CV_8UC1), affine, best_affine, best_box;
  vector<Marker_Corner*> corners1;
  vector<Marker_Corner*> corners2;
  vector<Marker_Corner*> test_corners;
  double score, best_score;
  double direction;
  //int paintx = (markers.size() % (paint.size().width/24))*24, painty = (markers.size() / (paint.size().width/24))*24;
  vector<Marker_Corner*> best_corners;
  vector<Point2f> test_points;
  
  Point2f d1, d2;
  float l1, l2;
  
  test_points.resize(3);
  
  test_corners.resize(3);
  best_corners.resize(3);
  last_corner.resize(1);
  
  test_corners[0] = start;
  if (!(start->mask & 1))
    return;
  
  //FIXME bug in gridstore?
  vector<void*> allcorners = candidates.getWithin(start->p, marker_maxsize);
  
  for (uint32_t i=0;i<allcorners.size();i++) {
    Point2f v = start->p-((Marker_Corner*)allcorners[i])->p;
    float l = v.x*v.x+v.y*v.y;
    if (l >= marker_maxsize*1.5*marker_maxsize*1.5)
      continue;
    if (l < marker_minsize*marker_minsize)
      continue;
    if (dir_rad_diff(start->dir_rad[0], ((Marker_Corner*)allcorners[i])->dir_rad[1]) <= marker_corner_dir_rad_dist
     && dir_rad_diff(start->dir_rad[1], ((Marker_Corner*)allcorners[i])->dir_rad[0]) <= marker_corner_dir_rad_dist
    )
      corners1.push_back((Marker_Corner*)allcorners[i]);
    else if (dir_rad_diff(start->dir_rad[0], ((Marker_Corner*)allcorners[i])->dir_rad[0]) <= marker_corner_dir_rad_dist
     && dir_rad_diff(start->dir_rad[1], ((Marker_Corner*)allcorners[i])->dir_rad[1]) <= marker_corner_dir_rad_dist
    )
      corners2.push_back((Marker_Corner*)allcorners[i]);
  }
    
  best_score = 0.0;
  for(uint32_t s1=0;s1<corners1.size();s1++) {
    for(uint32_t s2=0;s2<corners2.size();s2++) {
      /*if (dir*pattern_prescore(smallblur, corners[s1], corners[s2]) < bigblur.at<uchar>(start)/5)
	continue;*/
      
      test_corners[1] = corners1[s1];
      test_corners[2] = corners2[s2];
      
      d2 = test_corners[2]->p-test_corners[1]->p;
      l2 = d2.x*d2.x+d2.y*d2.y;
      
      if (l2 >= marker_maxsize*marker_maxsize || l2 < marker_minsize*marker_minsize)
	continue;
      
      //test three corners
      Point2f A = test_corners[1]->p - test_corners[0]->p;
      Point2f B = test_corners[2]->p - test_corners[0]->p;
      direction = A.cross(B);
      
      if (abs(direction) < marker_minsize*marker_minsize || abs(direction) > marker_maxsize*marker_maxsize)
	continue;
      
      if (direction > 0)
	continue;
      
      if (!(test_corners[1]->mask & 2))
	continue; 
      if (!(test_corners[2]->mask & 4))
	continue;
      
      d1 = test_corners[1]->p-test_corners[0]->p;
      l1 = d1.x*d1.x+d1.y*d1.y;
      
    if (l1 >= marker_maxsize*marker_maxsize || l1 < marker_minsize*marker_minsize)
	continue;
      
    if (norm(start->dir[0]-test_corners[2]->dir[0]) > marker_corner_dir_dist_max && norm(start->dir[0]+test_corners[2]->dir[0]) > marker_corner_dir_dist_max)
      continue;
    if (norm(start->dir[1]-test_corners[2]->dir[1]) > marker_corner_dir_dist_max && norm(start->dir[1]+test_corners[2]->dir[1]) > marker_corner_dir_dist_max)
      continue;
    
    /*if (norm(start->dir[0]-test_corners[1]->dir[1]) > marker_corner_dir_dist_max2 && norm(start->dir[0]+test_corners[1]->dir[1]) > marker_corner_dir_dist_max2)
      continue;
    if (norm(start->dir[1]-test_corners[1]->dir[0]) > marker_corner_dir_dist_max2 && norm(start->dir[1]+test_corners[1]->dir[0]) > marker_corner_dir_dist_max2)
      continue;*/
    
    Point2f dira, dirb;
    
    dira = start->dir[0];
    dirb = start->dir[1];
    
    if (snorm(dira-test_corners[2]->dir[0]) <= snorm(dira+test_corners[2]->dir[0]))
      dira += test_corners[2]->dir[0];
    else
      dira -= test_corners[2]->dir[0];
    
    if (snorm(dira*0.5-test_corners[1]->dir[1]) <= snorm(dira*0.5+test_corners[1]->dir[1]))
      dira += test_corners[1]->dir[1];
    else
      dira -= test_corners[1]->dir[1];
    
    if (snorm(dirb-test_corners[2]->dir[1]) <= snorm(dirb+test_corners[2]->dir[1]))
      dirb += test_corners[2]->dir[1];
    else
      dirb -= test_corners[2]->dir[1];
    
    if (snorm(dirb*0.5-test_corners[1]->dir[0]) <= snorm(dirb*0.5+test_corners[1]->dir[0]))
      dirb += test_corners[1]->dir[0];
    else
      dirb -= test_corners[1]->dir[0];
    
    dira *= 1.0/3.0;
    dirb *= 1.0/3.0;
    
    if (min(snorm(dira-start->dir[0]),snorm(dira+start->dir[0])) > marker_corner_dir_dist_max*marker_corner_dir_dist_max)
      continue;
    if (min(snorm(dira-test_corners[2]->dir[0]),snorm(dira+test_corners[2]->dir[0])) > marker_corner_dir_dist_max*marker_corner_dir_dist_max)
      continue;
    if (min(snorm(dira-test_corners[1]->dir[1]),snorm(dira+test_corners[1]->dir[1])) > marker_corner_dir_dist_max*marker_corner_dir_dist_max)
      continue;
    
    if (min(snorm(dirb-start->dir[1]),snorm(dirb+start->dir[1])) > marker_corner_dir_dist_max*marker_corner_dir_dist_max)
      continue;
    if (min(snorm(dirb-test_corners[2]->dir[1]),snorm(dirb+test_corners[2]->dir[1])) > marker_corner_dir_dist_max*marker_corner_dir_dist_max)
      continue;
    if (min(snorm(dirb-test_corners[1]->dir[0]),snorm(dirb+test_corners[1]->dir[0])) > marker_corner_dir_dist_max*marker_corner_dir_dist_max)
      continue;
    
    dira = dira * (1.0/norm(dira));
    dirb = dirb * (1.0/norm(dirb));
        
    d1 = d1 * (1.0/norm(d1));
    d2 = d2 * (1.0/norm(d2));
    
    if (min(snorm(dira-d1),snorm(dira+d1)) > marker_dir_corner_dist_max*marker_dir_corner_dist_max)
      continue;
    if (min(snorm(dirb-d2),snorm(dirb+d2)) > marker_dir_corner_dist_max*marker_dir_corner_dist_max)
      continue;

    //FIXME broken?
    /*if (pattern_prescore(smallblur, test_corners[0]->p, test_corners[1]->p) < prescore_marker_limit)
      continue;
    if (pattern_prescore(smallblur, test_corners[1]->p, test_corners[2]->p) < prescore_marker_limit)
      continue;
    if (pattern_prescore(smallblur, test_corners[2]->p, test_corners[2]->p + test_corners[1]->p - test_corners[0]->p)  < prescore_marker_limit)
      continue;*/
    
      double x21, x10, y21, y10;
      double d;
      x21 = test_corners[1]->p.x - test_corners[0]->p.x;
      x10 = test_corners[0]->p.x - test_corners[2]->p.x;
      y21 = test_corners[1]->p.y - test_corners[0]->p.y;
      y10 = test_corners[0]->p.y - test_corners[2]->p.y;
      
      d = (x21*y10 - x10*y21)*(x21*y10 - x10*y21) / (x21*x21 + y21*y21);
      if (d < marker_minsize*marker_minsize || d > marker_maxsize*marker_maxsize)
	continue;
      
      /*last_corner[0] = test_corners[2].p + (test_corners[1].p - start.p);
      cornerSubPix(img, last_corner, cvSize (scale+1, scale+1), cvSize (-1, -1), cvTermCriteria (CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 3, 0.1));
      if (pattern_prescore(smallblur, last_corner[0], start.p)  < 0.0)
	continue;*/

      /*test_points[0] = test_corners[0]->p;
      test_points[1] = test_corners[1]->p;
      test_points[2] = test_corners[2]->p;
      affine = getAffineTransform(test_points, box_corners);
      warpAffine(img, box, affine, Size(9, 9), INTER_LINEAR);*/
      
      Point2f d1, d2;
      d1 = (test_corners[1]->p - test_corners[2]->p)*0.4;
      d2 = (test_corners[1]->p - test_corners[0]->p)*0.4;
      
      test_points[0] = test_corners[0]->p+d1-d2+b_o;
      test_points[1] = test_corners[1]->p+d1+d2+b_o;
      test_points[2] = test_corners[2]->p-d1+d2+b_o;
      
      simplewarp_bilin(img, box, test_points, 9);
      
      score = pattern_score(box);

      if (score > best_score) {
	best_score = score;
	affine.copyTo(best_affine);
	best_corners = test_corners;
	box.copyTo(best_box);
	if (best_score > pattern_score_early)
	  goto finish;
      }
    }
  }
  
  finish :
  
  if (best_score > pattern_score_ok) {
    //FIXME synchronise?!
    best_corners[0]->refine(img);
    best_corners[1]->refine(img);
    best_corners[2]->refine(img);
    if (scale)
      marker = Marker(best_box, img, best_score, best_corners[0], best_corners[1], best_corners[2], scale);
    else
      marker = Marker(best_box, img, best_score, best_corners[0], best_corners[1], best_corners[2], 0.5);
    assert(marker.neighbours == 0);
    //marker.refine(img);
    if (marker.score > pattern_score_good && marker.id != -1) { 
	//marker.corners[0].refine(img);
	//marker.corners[1].refine(img);
#pragma omp critical
      {
	bool valid = true;
      for(int i=0;i<markers.size();i++)
	if (replace_overlapping) {
	    if (markers[i].id == marker.id && markers[i].page == marker.page) {
	      if (norm(markers[i].corners[0].p - marker.corners[0].p) > 2*(scale+1)) {
		valid = false;
		//keep only highest rated one!
		//FIXME what if this one has already valid number of neighbours?!
		//markers[i].neighbours = -1000000000;
		//cout << "duplicate at different position! " << marker.id << endl;
		if (markers[i].score < marker.score)
		  markers[i] = marker;
		break;
	      }
	      else {
		//FIXME something breaks here! Maybe filtering? (check test.jpg)
		//cout << "duplicate at similar position! " << marker.id << endl; 
		valid = false;
		//use higher scale if found
		if (markers[i].score < marker.score)
		  markers[i] = marker; 
	      }
	    }
	  }
	if (valid) {
	  if (scale)
	    marker.filter(&candidates, markers, scale);
	  else
	    marker.filter(&candidates, markers, 0.5);
	  markers.push_back(marker);
	}
      }
    }
  }
    
}



static inline uchar contrFromHist(int *hist, int size, int in)
{
  int i;
  int count, low, high;
  count = 0;
  /*if (minconstrast >= 0) {
    low = 0;
    i = 0;
    count = hist[i];
  }
  else {*/
    for(i=0;i<256/hist_simplify;i++) {
      count += hist[i];
      if (count >= (size*2+1)*(size*2+1)/8) {
	low = (i-1)*hist_simplify;
	break;
      }
    }
  //}
  if (i<0) i = 0;
  count -= hist[i];
  for(;i<256/hist_simplify;i++) {
    count += hist[i];
    if (count >= (size*2+1)*(size*2+1)*7/8) {
      high = i*hist_simplify;
      break;
    }
  }
  if (high-low > abs(minconstrast)) {
    if (minconstrast > 0) low = min(high/4, low);
    return min(max(in - low, 0)*255/(high-low), 255);
  }
  else
    return 0;
}


static inline uchar contrFromHist8(int *hist, int *hist8, int size, int in)
{
  int i;
  int count, low, high;
  count = 0;
  /*if (minconstrast >= 0) {
    low = 0;
    i = 0;
    count = hist[i];
  }
  else {*/
  
  for(i=0;i<256/8;i++) {
    count += hist8[i];
    if (count >= (size*2+1)*(size*2+1)/8) {
      count -= hist8[i];
      break;
    }
  }
  for(i=i*8;i<256/hist_simplify;i++) {
    count += hist[i];
    if (count >= (size*2+1)*(size*2+1)/8) {
      low = (i-1)*hist_simplify;
      break;
    }
  }
  //}
  if (i<0) i = 0;
  count -= hist[i];
  for(;i<256/hist_simplify;i++) {
    count += hist[i];
    if (count >= (size*2+1)*(size*2+1)*7/8) {
      high = i*hist_simplify;
      break;
    }
  }
  if (high-low > abs(minconstrast)) {
    if (minconstrast > 0) low = min(high/4, low);
    return min(max(in - low, 0)*255/(high-low), 255);
  }
  else
    return 0;
}

void facsFromHist(int *hist, int size, uchar &low, uchar &high, int step)
{
  int i;
  int count = 0;
  
  for(i=0;i<256/hist_simplify;i++) {
    count += hist[i];
    if (count >= (size*2+1)/step*(size*2+1)/step/4) {
      low = i*hist_simplify;
      break;
    }
  }
  count -= hist[i];
  for(;i<256/hist_simplify;i++) {
    count += hist[i];
    if (count > (size*2+1)/step*(size*2+1)/step*3/4) {
      high = i*hist_simplify;
      break;
    }
  }
  
  if (high-low <= minconstrast) {
    low = 0;
    high = 255;
  }
}

void localHistCont(Mat img, Mat &out, int size)
{
  int y, x;
  out = Mat::zeros(img.size(), CV_8UC1);
  int hist[256/hist_simplify];
  
  
  for(int i=0;i<256/hist_simplify;i++)
    hist[i] = 0;
  for(int j=0;j<=2*size;j++)
    for(int i=0;i<=2*size;i++)
      hist[img.at<uchar>(j, i)/hist_simplify]++;
  
  for(y=size;y<img.size().height-size-1;y++) {      
    for(x=size;x<img.size().width-size-1;x++) {
      out.at<uchar>(y, x) = contrFromHist(hist, size, img.at<uchar>(y, x));
      
      for(int j=y-size;j<=y+size;j++)
	hist[img.at<uchar>(j, x-size)/hist_simplify]--;
      for(int j=y-size;j<=y+size;j++)
	hist[img.at<uchar>(j, x+size+1)/hist_simplify]++;
    }
    out.at<uchar>(y, x) = contrFromHist(hist, size, img.at<uchar>(y, x));
    
    for(int i=x-size;i<=x+size;i++)
	hist[img.at<uchar>(y-size, i)/hist_simplify]--;
    for(int i=x-size;i<=x+size;i++)
	hist[img.at<uchar>(y+size+1, i)/hist_simplify]++;
    y++;
    
    for(x=img.size().width-size-1;x>size;x--) {
      out.at<uchar>(y, x) = contrFromHist(hist, size, img.at<uchar>(y, x));
      
      for(int j=y-size;j<=y+size;j++)
	hist[img.at<uchar>(j, x+size)/hist_simplify]--;
      for(int j=y-size;j<=y+size;j++)
	hist[img.at<uchar>(j, x-size-1)/hist_simplify]++;
    }
    out.at<uchar>(y, x) = contrFromHist(hist, size, img.at<uchar>(y, x));
    
    for(int i=x-size;i<=x+size;i++)
	hist[img.at<uchar>(y-size, i)/hist_simplify]--;
    for(int i=x-size;i<=x+size;i++)
	hist[img.at<uchar>(y+size+1, i)/hist_simplify]++;
  }
}


void localHistContOmp_slow(Mat img, Mat &out, int size)
{
  int y, x;
  out = Mat::zeros(img.size(), CV_8UC1);
  int hist[256/hist_simplify];
  
  memset(hist, 0, sizeof(int)*256/hist_simplify);
  
  
#pragma omp parallel for private(y, x, hist)
  for (int ychunk=size;ychunk<img.size().height-size-1;ychunk+=128) {
    for(int i=0;i<256/hist_simplify;i++)
      hist[i] = 0;
    for(int j=ychunk-size;j<=ychunk+size;j++)
      for(int i=0;i<=2*size;i++)
	hist[img.at<uchar>(j, i)/hist_simplify]++;
      
      x = size;
    for(y=ychunk;y<min(img.size().height-size-1, ychunk+128);y++) {      
      for(x=size;x<img.size().width-size-1;x++) {
	out.at<uchar>(y, x) = contrFromHist(hist, size, img.at<uchar>(y, x));
	
	for(int j=y-size;j<=y+size;j++)
	  hist[img.at<uchar>(j, x-size)/hist_simplify]--;
	for(int j=y-size;j<=y+size;j++)
	  hist[img.at<uchar>(j, x+size+1)/hist_simplify]++;
      }
      out.at<uchar>(y, x) = contrFromHist(hist, size, img.at<uchar>(y, x));
      
      for(int i=x-size;i<=x+size;i++)
	hist[img.at<uchar>(y-size, i)/hist_simplify]--;
      for(int i=x-size;i<=x+size;i++)
	hist[img.at<uchar>(y+size+1, i)/hist_simplify]++;
      y++;
      
      for(x=img.size().width-size-1;x>size;x--) {
	out.at<uchar>(y, x) = contrFromHist(hist, size, img.at<uchar>(y, x));
	
	for(int j=y-size;j<=y+size;j++)
	  hist[img.at<uchar>(j, x+size)/hist_simplify]--;
	for(int j=y-size;j<=y+size;j++)
	  hist[img.at<uchar>(j, x-size-1)/hist_simplify]++;
      }
      out.at<uchar>(y, x) = contrFromHist(hist, size, img.at<uchar>(y, x));
      
      for(int i=x-size;i<=x+size;i++)
	hist[img.at<uchar>(y-size, i)/hist_simplify]--;
      for(int i=x-size;i<=x+size;i++)
	hist[img.at<uchar>(y+size+1, i)/hist_simplify]++;
    }
  }
}


void localHistContOmp(Mat img, Mat &out, int size)
{
  int y, x;
  out = Mat::zeros(img.size(), CV_8UC1);
  int hist[256/hist_simplify];
  int hist8[256/8];  
  
#pragma omp parallel for private(y, x, hist, hist8)
  for (int ychunk=size;ychunk<img.size().height-size-1;ychunk+=128) {
    for(int i=0;i<256/hist_simplify;i++)
      hist[i] = 0;
    for(int i=0;i<256/8;i++)
      hist8[i] = 0;
    for(int j=ychunk-size;j<=ychunk+size;j++)
      for(int i=0;i<=2*size;i++) {
	hist[img.at<uchar>(j, i)/hist_simplify]++;
	hist8[img.at<uchar>(j, i)/8]++;
      }
      
      x = size;
    for(y=ychunk;y<min(img.size().height-size, ychunk+128);y++) {      
      for(x=size;x<img.size().width-size-1;x++) {
	out.at<uchar>(y, x) = contrFromHist8(hist, hist8,  size, img.at<uchar>(y, x));
	
	for(int j=y-size;j<=y+size;j++) {
	  hist[img.at<uchar>(j, x-size)/hist_simplify]--;
	  hist8[img.at<uchar>(j, x-size)/8]--;
	}
	for(int j=y-size;j<=y+size;j++) {
	  hist[img.at<uchar>(j, x+size+1)/hist_simplify]++;
	  hist8[img.at<uchar>(j, x+size+1)/8]++;
	}
      }
      out.at<uchar>(y, x) = contrFromHist8(hist, hist8, size, img.at<uchar>(y, x));
      
      for(int i=x-size;i<=x+size;i++) {
	hist[img.at<uchar>(y-size, i)/hist_simplify]--;
	hist8[img.at<uchar>(y-size, i)/8]--;
      }
      for(int i=x-size;i<=x+size;i++) {
	hist[img.at<uchar>(y+size+1, i)/hist_simplify]++;
	hist8[img.at<uchar>(y+size+1, i)/8]++;
      }
      y++;
      
      for(x=img.size().width-size-1;x>size;x--) {
	out.at<uchar>(y, x) = contrFromHist8(hist, hist8, size, img.at<uchar>(y, x));
	
	for(int j=y-size;j<=y+size;j++) {
	  hist[img.at<uchar>(j, x+size)/hist_simplify]--;
	  hist8[img.at<uchar>(j, x+size)/8]--;
	}
	for(int j=y-size;j<=y+size;j++) {
	  hist[img.at<uchar>(j, x-size-1)/hist_simplify]++;
	  hist8[img.at<uchar>(j, x-size-1)/8]++;
	}
      }
      out.at<uchar>(y, x) = contrFromHist8(hist, hist8, size, img.at<uchar>(y, x));
      
      for(int i=x-size;i<=x+size;i++) {
	hist[img.at<uchar>(y-size, i)/hist_simplify]--;
	hist8[img.at<uchar>(y-size, i)/8]--;
      }
      for(int i=x-size;i<=x+size;i++) {
	hist[img.at<uchar>(y+size+1, i)/hist_simplify]++;
	hist8[img.at<uchar>(y+size+1, i)/8]++;
      }
    }
  }
}

void norm_avg(Mat img, Mat &out, int size)
{
  Mat blurred;
  int div;
  
  blur(img, blurred, Size(2*size+1,2*size+1));
  
  out.create(img.size(), CV_8UC1);
  //out = img / blurred * 255;
#pragma omp parallel for private(div)
  for(int j=0;j<img.size().height;j++)
    for(int i=0;i<img.size().width;i++) {
      div = blurred.at<uchar>(j,i)-blurred.at<uchar>(j,i)*norm_avg_sub;
      if (div < minconstrast)
	div = 255;
      out.at<uchar>(j,i) = std::max(std::min((int)norm_avg_value*img.at<uchar>(j,i) / div, 255), 0);
    }
}


void norm_avg_SIMD(Mat img, Mat &out, int size)
{
  Mat blurred;
  int div;
  uint8_t *bp, *ip, *o;
  v16si n;
  v16qi in_i, in_b, mask, v_0x80;
  v4f img_f[4], blur_f[4], res_f, v_160, div_fac;
  char mc = minconstrast-128;
  v16si minc = {mc,mc,mc,mc,mc,mc,mc,mc,mc,mc,mc,mc,mc,mc,mc,mc};
  div_fac = (v4f)_mm_set1_ps(1.0-norm_avg_sub);
  v_160 = (v4f)_mm_set1_ps(norm_avg_value);
  v_0x80 = (v16qi)_mm_set1_epi8(0x80);
  
  blur(img, blurred, Size(2*size+1,2*size+1));
  
  out.create(img.size(), CV_8UC1);
  //out = img / blurred * 255;
#pragma omp parallel for private(div)
  for(int j=0;j<img.size().height;j++) {
    bp = blurred.ptr<uchar>(j);
    ip = img.ptr<uchar>(j);
    o = out.ptr<uchar>(j);
    int i;
    for(i=0;i<img.size().width-16;i+=16,ip+=16,bp+=16,o+=16) {
      
      n = n ^ n;
      memcpy(&in_i, ip, 16);
      memcpy(&in_b, bp, 16);
      img_f[1] = (v4f)punpcklbw((v16si)in_i, n);
      img_f[3] = (v4f)punpckhbw((v16si)in_i, n);
      
      img_f[0] = (v4f)punpcklwd((v8si)img_f[1], (v8si)n);
      img_f[1] = (v4f)punpckhwd((v8si)img_f[1], (v8si)n);
       
      img_f[2] = (v4f)punpcklwd((v8si)img_f[3], (v8si)n);
      img_f[3] = (v4f)punpckhwd((v8si)img_f[3], (v8si)n);
      
      blur_f[1] = (v4f)punpcklbw((v16si)in_b, n);
      blur_f[3] = (v4f)punpckhbw((v16si)in_b, n);
      
      blur_f[0] = (v4f)punpcklwd((v8si)blur_f[1], (v8si)n);
      blur_f[1] = (v4f)punpckhwd((v8si)blur_f[1], (v8si)n);
       
      blur_f[2] = (v4f)punpcklwd((v8si)blur_f[3], (v8si)n);
      blur_f[3] = (v4f)punpckhwd((v8si)blur_f[3], (v8si)n);
      
      for(int r=0;r<4;r++) {
	img_f[r] = __builtin_ia32_cvtdq2ps((v4si)img_f[r]);
	blur_f[r] = __builtin_ia32_cvtdq2ps((v4si)blur_f[r]);
        //blur_f[r] = div_fac*blur_f[r];
	blur_f[r] = __builtin_ia32_rcpps(blur_f[r]);
	img_f[r] = v_160 * img_f[r] * blur_f[r];
	img_f[r] = (v4f)__builtin_ia32_cvtps2dq(img_f[r]);
      }
      
      img_f[0] = (v4f)__builtin_ia32_packuswb128((v8si)img_f[0], (v8si)img_f[1]);
      img_f[2] = (v4f)__builtin_ia32_packuswb128((v8si)img_f[2], (v8si)img_f[3]);
      img_f[0] = (v4f)__builtin_ia32_packuswb128((v8si)img_f[0], (v8si)img_f[2]);
      
      //FIXME xor may be faster?
      mask = in_b - v_0x80;
      mask = (v16qi)pcmpgtb((v16si)minc, (v16si)mask);
      img_f[0] = (v4f)por((v2di)img_f[0], (v2di)mask);
      
      /*div = blurred.at<uchar>(j,i);
      if (div < minconstrast)
	div = 255;
      out.at<uchar>(j,i) = max(min(160*img.at<uchar>(j,i) / div, 255), 0);
      
      if (in_i[0])
	abort();*/
      memcpy(o, &img_f[0], 16);
    }
    for(;i<img.size().width;i++,o++)
      *o = 0;
  }
}


void localHistContUnderSampled(Mat img, Mat &out, int size, int step)
{
  int y, x;
  out = Mat::zeros(img.size(), CV_8UC1);
  int hist[256/hist_simplify];
  uchar low, high;  
  
  for(int i=0;i<256/hist_simplify;i++)
    hist[i] = 0;
  for(int j=0;j<=2*size;j+=step)
    for(int i=0;i<=2*size;i+=step)
      hist[img.at<uchar>(j, i)/hist_simplify]++;
    
    x = size;
  
  for(y=size;y<img.size().height-size-step;y+=step) {       
    for(;x<img.size().width-size-step;x+=step) {
      facsFromHist(hist, size, low, high, step);
      for(int j=y-step/2;j<y+step/2;j++)
	for(int i=x-step/2;i<x+step/2;i++)
	  out.at<uchar>(j, i) = min(max(img.at<uchar>(j, i) - low, 0)*255/(high-low), 255);
      
      for(int j=y-size;j<=y+size;j+=step)
	hist[img.at<uchar>(j, x-size)/hist_simplify]--;
      for(int j=y-size;j<=y+size;j+=step)
	hist[img.at<uchar>(j, x+size+step)/hist_simplify]++;
    }
    facsFromHist(hist, size, low, high, step);
      for(int j=y-step/2;j<y+step/2;j++)
	for(int i=x-step/2;i<x+step/2;i++)
	  out.at<uchar>(j, i) = min(max(img.at<uchar>(j, i) - low, 0)*255/(high-low), 255);
  
    for(int i=x-size;i<=x+size;i+=step)
	hist[img.at<uchar>(y-size, i)/hist_simplify]--;
    for(int i=x-size;i<=x+size;i+=step)
	hist[img.at<uchar>(y+size+step, i)/hist_simplify]++;
    y+=step;
    
    for(;x>size+step;x-=step) {
    facsFromHist(hist, size, low, high, step);
      for(int j=y-step/2;j<y+step/2;j++)
	for(int i=x-step/2;i<x+step/2;i++)
	  out.at<uchar>(j, i) = min(max(img.at<uchar>(j, i) - low, 0)*255/(high-low), 255);
      
      for(int j=y-size;j<=y+size;j+=step)
	hist[img.at<uchar>(j, x+size)/hist_simplify]--;
      for(int j=y-size;j<=y+size;j+=step)
	hist[img.at<uchar>(j, x-size-step)/hist_simplify]++;
    }
    facsFromHist(hist, size, low, high, step);
      for(int j=y-step/2;j<y+step/2;j++)
	for(int i=x-step/2;i<x+step/2;i++)
	  out.at<uchar>(j, i) = min(max(img.at<uchar>(j, i) - low, 0)*255/(high-low), 255);
    
    for(int i=x-size;i<=x+size;i+=step)
	hist[img.at<uchar>(y-size, i)/hist_simplify]--;
    for(int i=x-size;i<=x+size;i+=step)
	hist[img.at<uchar>(y+size+step, i)/hist_simplify]++;
  }
}


void localHistContUnderSampledOmp(Mat img, Mat &out, int size, int step)
{
  int ychunk;
  int y, x;
  out = Mat::zeros(img.size(), CV_8UC1);
  int hist[256/hist_simplify];
  uchar low, high;  
  
#pragma omp parallel for private(y, x, hist, low, high)
  for (ychunk=size;ychunk<img.size().height-size-step;ychunk+=step*128) {
    for(int i=0;i<256/hist_simplify;i++)
      hist[i] = 0;
    for(int j=ychunk-size;j<=ychunk+size;j+=step)
      for(int i=0;i<=2*size;i+=step)
	hist[img.at<uchar>(j, i)/hist_simplify]++;
      
    x = size;
    for(y=ychunk;y<min(ychunk+step*128,img.size().height-size-step);y+=step) {       
      for(;x<img.size().width-size-step;x+=step) {
	facsFromHist(hist, size, low, high, step);
	for(int j=y-step/2;j<y+step/2;j++)
	  for(int i=x-step/2;i<x+step/2;i++)
	    out.at<uchar>(j, i) = min(max(img.at<uchar>(j, i) - low, 0)*255/(high-low), 255);
	  
	  for(int j=y-size;j<=y+size;j+=step)
	    hist[img.at<uchar>(j, x-size)/hist_simplify]--;
	  for(int j=y-size;j<=y+size;j+=step)
	    hist[img.at<uchar>(j, x+size+step)/hist_simplify]++;
      }
      facsFromHist(hist, size, low, high, step);
      for(int j=y-step/2;j<y+step/2;j++)
	for(int i=x-step/2;i<x+step/2;i++)
	  out.at<uchar>(j, i) = min(max(img.at<uchar>(j, i) - low, 0)*255/(high-low), 255);
	
	for(int i=x-size;i<=x+size;i+=step)
	  hist[img.at<uchar>(y-size, i)/hist_simplify]--;
	for(int i=x-size;i<=x+size;i+=step)
	  hist[img.at<uchar>(y+size+step, i)/hist_simplify]++;
	y+=step;
      
      for(;x>size+step;x-=step) {
	facsFromHist(hist, size, low, high, step);
	for(int j=y-step/2;j<y+step/2;j++)
	  for(int i=x-step/2;i<x+step/2;i++)
	    out.at<uchar>(j, i) = min(max(img.at<uchar>(j, i) - low, 0)*255/(high-low), 255);
	  
	  for(int j=y-size;j<=y+size;j+=step)
	    hist[img.at<uchar>(j, x+size)/hist_simplify]--;
	  for(int j=y-size;j<=y+size;j+=step)
	    hist[img.at<uchar>(j, x-size-step)/hist_simplify]++;
      }
      facsFromHist(hist, size, low, high, step);
      for(int j=y-step/2;j<y+step/2;j++)
	for(int i=x-step/2;i<x+step/2;i++)
	  out.at<uchar>(j, i) = min(max(img.at<uchar>(j, i) - low, 0)*255/(high-low), 255);
	
	for(int i=x-size;i<=x+size;i+=step)
	  hist[img.at<uchar>(y-size, i)/hist_simplify]--;
	for(int i=x-size;i<=x+size;i+=step)
	  hist[img.at<uchar>(y+size+step, i)/hist_simplify]++;
    }
  }
}


void simpleCorner(Mat img, Mat &out, int d)
{ 
  out = Mat::zeros(img.size(), CV_8UC1);
  
  for(int y=d;y<img.size().height-d;y++)
    for(int x=d;x<img.size().width-d;x++)
      out.at<uchar>(y, x) = min(max(abs((img.at<uchar>(y-d, x-d)-img.at<uchar>(y-d, x+d))*(img.at<uchar>(y-d, x+d)-img.at<uchar>(y+d, x+d))*(img.at<uchar>(y+d, x+d)-img.at<uchar>(y+d, x-d))*(img.at<uchar>(y+d, x-d)-img.at<uchar>(y-d, x-d))),
				abs((img.at<uchar>(y, x-d)-img.at<uchar>(y-d, x))*(img.at<uchar>(y-d, x)-img.at<uchar>(y, x+d))*(img.at<uchar>(y, x+d)-img.at<uchar>(y+d, x))*(img.at<uchar>(y+d, x)-img.at<uchar>(y, x-d))))/500000, 255);				
  
}

static inline int dirscore(int c, int a1, int a2, int b1, int b2)
{
  int avg = a1+a2+b1+b2;
  return max(min(abs((a1+a2)-(b1+b2))-abs(a1-a2)*2-abs(b1-b2)*2 - abs(4*c - avg)*7/8, 255), 0);
}

static inline int dirscore2(int c, int a1, int a2, int b1, int b2)
{
  int avg = a1+a2+b1+b2;
  return max(min(abs((a1+a2)-(b1+b2))-abs(a1-a2)*2-abs(b1-b2)*2, 255), 0);
}

static inline void dirscore_x16(uint8_t *res, uint8_t *c, uint8_t *a1, uint8_t *a2, uint8_t *b1, uint8_t *b2)
{
  v16qi va1, va2, vb1, vb2, vc, vavg, vavg1, vda, vdb, vsa, vsb;
  v16qi vamin, vamax, vbmin, vbmax, vcmax, vcmin, vsmax,vsmin;
  v8qi vcm1, vcm2;
  v8qi v_7, v_4;
  v_7 = (v8qi)_mm_set1_epi16(7);
  v_4 = (v8qi)_mm_set1_epi16(4);
  
  memcpy(&vc, c, 16);
  memcpy(&va1, a1, 16);
  memcpy(&va2, a2, 16);
  memcpy(&vb1, b1, 16);
  memcpy(&vb2, b2, 16);
  
  vavg = (v16qi)__builtin_ia32_pavgb128((v16si)va1, (v16si)va2);
  vavg1 = (v16qi)__builtin_ia32_pavgb128((v16si)vb1, (v16si)vb2);
  vavg = (v16qi)__builtin_ia32_pavgb128((v16si)vavg, (v16si)vavg1);
  
  vsa = (v16qi)__builtin_ia32_pavgb128((v16si)va1, (v16si)va2);
  vsb = (v16qi)__builtin_ia32_pavgb128((v16si)vb1, (v16si)vb2);
  
  vsmax = (v16qi)__builtin_ia32_pmaxub128((v16si)vsa, (v16si)vsb);
  vsmin = (v16qi)__builtin_ia32_pminub128((v16si)vsa, (v16si)vsb);
  
  //avg = a1+a2+b1+b2;
  
  vamax = (v16qi)__builtin_ia32_pmaxub128((v16si)va1, (v16si)va2);
  vamin = (v16qi)__builtin_ia32_pminub128((v16si)va1, (v16si)va2);
  
  vbmax = (v16qi)__builtin_ia32_pmaxub128((v16si)vb1, (v16si)vb2);
  vbmin = (v16qi)__builtin_ia32_pminub128((v16si)vb1, (v16si)vb2);
  
  vcmax = (v16qi)__builtin_ia32_pmaxub128((v16si)vc, (v16si)vavg);
  vcmin = (v16qi)__builtin_ia32_pminub128((v16si)vc, (v16si)vavg);
  
  /*for(int i=0;i<16;i++) {
    res[i] = (uint8_t)max(min(2*(vsmax[i]-vsmin[i]-(vamax[i]-vamin[i])-(vbmax[i]-vbmin[i])) - (vcmax[i] - vcmin[i])*7/2, 255), 0);
    //if (res[i] != dirscore(c[i], a1[i], a2[i], b1[i], b2[i]))
      //printf("%d != %d %d\n",res[i],dirscore(c[i], a1[i], a2[i], b1[i], b2[i]));
  }
  
  return;*/
  
  vsmax = vsmax-vsmin;
  vamax = vamax-vamin;
  vbmax = vbmax-vbmin;
  vcmax = vcmax-vcmin;
  
  v16qi vamax2 = (v16qi)__builtin_ia32_paddusb128((v16si)vamax, (v16si)vbmax);
  
  v16qi vsmax2 = (v16qi)__builtin_ia32_psubusb128((v16si)vsmax, (v16si)vamax2);
  
  vbmax = vbmax ^ vbmax;
  vcm1 = (v8qi)punpcklbw((v16si)vcmax, (v16si)vbmax);
  vcm2 = (v8qi)punpckhbw((v16si)vcmax, (v16si)vbmax);
  
  //printf("%d -> %d\n", vcmax[8], vcm2[0]);
  
  vcm1 *= v_7;
  vcm1 /= v_4;
  vcm2 *= v_7;
  vcm2 /= v_4;
  
  vcmax = (v16qi)__builtin_ia32_packuswb128((v8si)vcm1, (v8si)vcm2);
  v16qi vsmax3 = (v16qi)__builtin_ia32_psubusb128((v16si)vsmax2, (v16si)vcmax);
  memcpy(res, &vsmax3, 16);
  /*for(int i=0;i<16;i++) {
    res[i] = (uint8_t)max(min(vsmax[i] - (vcmax[i]), 255), 0);
  }*/
  
  /*for(int i=0;i<16;i++) {
    if (abs(res[i] - dirscore(c[i], a1[i], a2[i], b1[i], b2[i])/2) >= 5)
      printf("%d %d = %d - %d - %d - %d != %d = %d - %d - %d - %d\n",i, res[i],vsmax[i],vamax[i],vbmax[i],vcmax[i],dirscore(c[i], a1[i], a2[i], b1[i], b2[i]),abs((a1[i]+a2[i])-(b1[i]+b2[i])),abs(a1[i]-a2[i])*2,abs(b1[i]-b2[i])*2,abs(4*c[i] - (a1[i]+a2[i]+b1[i]+b2[i]))*7/8);
  }*/
}


static inline void dirscore2_x16(uint8_t *res, uint8_t *c, uint8_t *a1, uint8_t *a2, uint8_t *b1, uint8_t *b2)
{
  v16qi va1, va2, vb1, vb2, vc, vavg, vavg1, vda, vdb, vsa, vsb;
  v16qi vamin, vamax, vbmin, vbmax, vcmax, vcmin, vsmax,vsmin;
  v8qi vcm1, vcm2;
  
  memcpy(&vc, c, 16);
  memcpy(&va1, a1, 16);
  memcpy(&va2, a2, 16);
  memcpy(&vb1, b1, 16);
  memcpy(&vb2, b2, 16);
  
  vavg = (v16qi)__builtin_ia32_pavgb128((v16si)va1, (v16si)va2);
  vavg1 = (v16qi)__builtin_ia32_pavgb128((v16si)vb1, (v16si)vb2);
  vavg = (v16qi)__builtin_ia32_pavgb128((v16si)vavg, (v16si)vavg1);
  
  vsa = (v16qi)__builtin_ia32_pavgb128((v16si)va1, (v16si)va2);
  vsb = (v16qi)__builtin_ia32_pavgb128((v16si)vb1, (v16si)vb2);
  
  vsmax = (v16qi)__builtin_ia32_pmaxub128((v16si)vsa, (v16si)vsb);
  vsmin = (v16qi)__builtin_ia32_pminub128((v16si)vsa, (v16si)vsb);
  
  //avg = a1+a2+b1+b2;
  
  vamax = (v16qi)__builtin_ia32_pmaxub128((v16si)va1, (v16si)va2);
  vamin = (v16qi)__builtin_ia32_pminub128((v16si)va1, (v16si)va2);
  
  vbmax = (v16qi)__builtin_ia32_pmaxub128((v16si)vb1, (v16si)vb2);
  vbmin = (v16qi)__builtin_ia32_pminub128((v16si)vb1, (v16si)vb2);
  
  /*for(int i=0;i<16;i++) {
    res[i] = (uint8_t)max(min(vsmax[i]-vsmin[i]-(vamax[i]-vamin[i])-(vbmax[i]-vbmin[i]) - (vcmax[i] - vcmin[i])*7/4, 255), 0);
  }*/
  
  vsmax = vsmax-vsmin;
  vamax = vamax-vamin;
  vbmax = vbmax-vbmin;
  vamax = (v16qi)__builtin_ia32_paddusb128((v16si)vamax, (v16si)vbmax);
  vsmax = (v16qi)__builtin_ia32_psubusb128((v16si)vsmax, (v16si)vamax);
  
  memcpy(res, &vsmax, 16);
}

void simpleChessCorner(Mat &img, Mat &out, float d)
{ 
  int d1 = d;
  int d2 = (d*3+1)/2;
  out = Mat::zeros(img.size(), CV_8UC1);
  int val;
  int w = img.size().width;
  
  int h1 = w*d;
  int h2 = w*d2;
  
  uint8_t *c;
  uint8_t *o;
  
#pragma omp parallel for private(c, o, val)
  for(int y=d2;y<img.size().height-d2;y++) {
    c = &img.ptr<uchar>(0)[d2+w*y];
    o = &out.ptr<uchar>(0)[d2+w*y];
    for(int x=d2;x<img.size().width-d2;x++,c++,o++) {
      val = dirscore(c[0], c[-d1], c[d1], c[h2], c[-h2]);
      val = max(val, dirscore(c[0], c[-d2], c[d2], c[h1], c[-h1]));
      val = max(val, dirscore(c[0], c[-d2-h2], c[d2+h2], c[h1-d1], c[-h1+d1]));
      val = max(val, dirscore(c[0], c[-d1-h1], c[d1+h1], c[h2-d2], c[-h2+d2]));
      out.at<uchar>(y, x) = min(max(val, 0), 255);
    }
  }
}


void simpleChessCorner2(Mat &img, Mat &sblur, Mat &bblur, Mat &out, float d)
{ 
  int d1 = d;
  int d2 = (d*3+1)/2;
  out = Mat::zeros(img.size(), CV_8UC1);
  int val;
  int w = img.size().width;
  
  int h1 = w*d;
  int h2 = w*d2;
  
  uint8_t *c;
  uint8_t *o;
  uint8_t *sb;
  uint8_t *bb;
  
#pragma omp parallel for private(c, o, val)
  for(int y=d2;y<img.size().height-d2;y++) {
    c = &img.ptr<uchar>(0)[d2+w*y];
    o = &out.ptr<uchar>(0)[d2+w*y];
    sb = &sblur.ptr<uchar>(0)[d2+w*y];
    bb = &bblur.ptr<uchar>(0)[d2+w*y];
    for(int x=d2;x<img.size().width-d2;x++,c++,o++,sb++,bb++) {
      val = dirscore2(c[0], c[-d1], c[d1], c[h2], c[-h2]);
      val = max(val, dirscore2(c[0], c[-d2], c[d2], c[h1], c[-h1]));
      val = max(val, dirscore2(c[0], c[-d2-h2], c[d2+h2], c[h1-d1], c[-h1+d1]));
      val = max(val, dirscore2(c[0], c[-d1-h1], c[d1+h1], c[h2-d2], c[-h2+d2]));
      out.at<uchar>(y, x) = min(max(val - 4*abs(sb[0]-bb[0]), 0), 255);
    }
  }
}


void simpleChessCorner2_SIMD(Mat &img, Mat &sblur, Mat &bblur, Mat &out, float d)
{ 
  int d1 = d;
  int d2 = (d*3+1)/2;
  out = Mat::zeros(img.size(), CV_8UC1);
  uint8_t val;
  int w = img.size().width;
  v16qi res, res2;
  v16qi v_sb, v_bb, b_min,b_max;
  
  int h1 = w*d;
  int h2 = w*d2;
  
  uint8_t *c;
  uint8_t *o;
  uint8_t *sb;
  uint8_t *bb;
  
#pragma omp parallel for private(c, o, val)
  for(int y=d2;y<img.size().height-d2;y++) {
    c = &img.ptr<uchar>(0)[d2+w*y];
    o = &out.ptr<uchar>(0)[d2+w*y];
    sb = &sblur.ptr<uchar>(0)[d2+w*y];
    bb = &bblur.ptr<uchar>(0)[d2+w*y];
    for(int x=d2;x<img.size().width-d2;x+=16,c+=16,o+=16,sb+=16,bb+=16) {
      dirscore2_x16((uint8_t*)&res, c, c-d1, c+d1, c+h2, c-h2);
      dirscore2_x16((uint8_t*)&res2, c, c-d2, c+d2, c+h1, c-h1);
      res = (v16qi)__builtin_ia32_pmaxub128((v16si)res, (v16si)res2);
      dirscore2_x16((uint8_t*)&res2, c, c-d2-h2, c+d2+h2, c+h1-d1, c+d1-h1);
      res = (v16qi)__builtin_ia32_pmaxub128((v16si)res, (v16si)res2);
      dirscore2_x16((uint8_t*)&res2, c, c-d1-h1, c+d1+h1, c+h2-d2, c+d2-h2);
      res = (v16qi)__builtin_ia32_pmaxub128((v16si)res, (v16si)res2);
      memcpy(&v_sb, sb, 16);
      memcpy(&v_bb, bb, 16);
      b_min = (v16qi)__builtin_ia32_pminub128((v16si)v_sb, (v16si)v_bb);
      b_max = (v16qi)__builtin_ia32_pmaxub128((v16si)v_sb, (v16si)v_bb);
      b_max = b_max - b_min;
      b_max = (v16qi)__builtin_ia32_paddusb128((v16si)b_max, (v16si)b_max);
      
      res = (v16qi)__builtin_ia32_psubusb128((v16si)res, (v16si)b_max);
      memcpy(o, &res, 16);
    }
  }
}

void simpleChessCorner_SIMD(Mat &img, Mat &out, float d)
{ 
  int d1 = d;
  int d2 = (d*3+1)/2;
  out = Mat::zeros(img.size(), CV_8UC1);
  uint8_t val;
  int w = img.size().width;
  v16qi res, res2;
  
  int h1 = w*d;
  int h2 = w*d2;
  
  uint8_t *c;
  uint8_t *o;
  
#pragma omp parallel for private(c, o, val, res, res2)
  for(int y=d2;y<img.size().height-d2;y++) {
    c = &img.ptr<uchar>(0)[d2+w*y];
    o = &out.ptr<uchar>(0)[d2+w*y];
    for(int x=d2;x<img.size().width-d2;x+=16,c+=16,o+=16) {
      dirscore_x16((uint8_t*)&res, c, c-d1, c+d1, c+h2, c-h2);
      dirscore_x16((uint8_t*)&res2, c, c-d2, c+d2, c+h1, c-h1);
      res = (v16qi)__builtin_ia32_pmaxub128((v16si)res, (v16si)res2);
      dirscore_x16((uint8_t*)&res2, c, c-d2-h2, c+d2+h2, c+h1-d1, c+d1-h1);
      res = (v16qi)__builtin_ia32_pmaxub128((v16si)res, (v16si)res2);
      dirscore_x16((uint8_t*)&res2, c, c-d1-h1, c+d1+h1, c+h2-d2, c+d2-h2);
      res = (v16qi)__builtin_ia32_pmaxub128((v16si)res, (v16si)res2);
      
      /*dirscore_x16((uint8_t*)&res2, c, c-d1-h1, c+d1+h1, c+h1-d1, c+d1-h1);
      res = (v16qi)__builtin_ia32_pmaxub128((v16si)res, (v16si)res2);
      dirscore_x16((uint8_t*)&res2, c, c-d1, c+d1, c+h1, c+d1);
      res = (v16qi)__builtin_ia32_pmaxub128((v16si)res, (v16si)res2);*/
      
      memcpy(o, &res, 16);
    }
  }
}

static inline int simplelienscore(Mat &img, int x, int y)
{ 
  return max(abs(img.at<uchar>(y, x)-img.at<uchar>(y, x-1))
    +abs(img.at<uchar>(y, x)-img.at<uchar>(y, x+1))
    +abs(img.at<uchar>(y, x)-img.at<uchar>(y+1, x))
    +abs(img.at<uchar>(y, x)-img.at<uchar>(y-1, x)),
    abs(img.at<uchar>(y, x)-img.at<uchar>(y-1, x-1))
    +abs(img.at<uchar>(y, x)-img.at<uchar>(y-1, x+1))
    +abs(img.at<uchar>(y, x)-img.at<uchar>(y+1, x-1))
    +abs(img.at<uchar>(y, x)-img.at<uchar>(y+1, x+1)));
}



void localHistThres(Mat img, Mat &out, int size, float th)
{
  out = Mat::zeros(img.size(), CV_8UC1);
  int hist[256/hist_simplify], val, count;
  
  for(int y=size;y<img.size().height-size;y++) {
    for(int x=size;x<img.size().width-size;x++) {
      for(int i=0;i<256/hist_simplify;i++)
	hist[i] = 0;
      for(int j=y-size;j<=y+size;j++)
	for(int i=x-size;i<=x+size;i++)
	  hist[img.at<uchar>(j, i)/hist_simplify]++;
      count = 0;
      for(int i=0;i<256/hist_simplify;i++) {
	count += hist[i];
	if (count >= th*(size*2+1)*(size*2+1)) {
	  val = i*hist_simplify;
	  break;
	}
      }
      if (img.at<uchar>(y, x) >= val)
	img.at<uchar>(y, x) = 255;
    }
  }
}


void Marker_Corner::paint(cv::Mat &paint)
{
  line(paint, p, p-3.0/corner_score_size*dir[1], CV_RGB(0,255,0), 1);
  line(paint, p, p-3.0/corner_score_size*dir[0], CV_RGB(255,0,0), 1);
  circle(paint, p, 0, CV_RGB(200, 200, 0), 1, 1);
}

void Marker::detect_scale(vector<Mat> imgs, vector<Mat> norms, vector<Mat> checkers, vector<Marker> &markers, float scale, float effort)
{
  int w, h;
  Mat frame, blur, bw, checker_ls, blurf, box, checker, best_pers, harrblur1, harry_erode, bigblur, small_min, small_max, small_blur, small_hc_blur, big_hc;
  Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
  Mat bigelement = getStructuringElement(MORPH_ELLIPSE, Size(11, 11));
  vector<Point2f> corners;
  vector<Marker_Corner> corners2;
  Gridstore candidates;
  Size s;
  
  int scale_idx = log2(scale)+1;
  
  //printf("detect at %f - %d\n", scale, scale_idx);
  
//  
// --- init ---
//
  w = imgs[scale_idx].size().width;
  h = imgs[scale_idx].size().height;
  s = Size(w, h);
  
  candidates = Gridstore(w, h, marker_maxsize*1.5);
  
  microbench_measure_output("init");
      
//      
// --- corner detection ---
//      
  Mat small_hc_sb;
  GaussianBlur(norms[scale_idx], small_hc_sb, Size(3, 3), 0);
  microbench_measure_output("checkerboard corners");
  
  Mat checker_nomax;
#ifdef USE_SLOW_CORNERs
  checker = checkers[scale_idx].clone();
  resize(checkers[scale_idx+1], checker_ls, checker.size(), INTER_LINEAR);
  for(int y=0;y<checker.size().height;y++)
    for(int x=0;x<checker.size().width;x++)
      checker.at<uchar>(y, x) = std::min(checker.at<uchar>(y, x)*checker_ls.at<uchar>(y, x), 255);
#else
  resize(checkers[scale_idx+1], checker_ls, s, INTER_LINEAR);
  checker = (4*checkers[scale_idx]+2*checker_ls);
  GaussianBlur(checker, checker, Size(3, 3), 0.8);
#endif
  nomax(checker, checker_nomax);
  
  
#ifdef DEBUG_SAVESTEPS
  char buf[64];
  sprintf(buf, "chess_src_%02d.png", scale_idx);
  imwrite(buf, checker);
  sprintf(buf, "chess_nomax_src_%02d.png", scale_idx);
  imwrite(buf, checker_nomax);
#endif
  
  checker = checker_nomax;
  
    corners.resize(0);
    
  //float th = 20;
  //float th = 40;// 100 - 50*min(max((double)effort, 0.0),1.0);
#ifdef USE_SLOW_CORNERs
  float th = 80;
#else
  float th = 10;// 100 - 50*min(max((double)effort, 0.0),1.0);
#endif
  //float th = corner_threshold_high - (corner_threshold_high-corner_threshold_low)*max(min(effort,(float)1.0),(float)0.0);
#pragma omp parallel for
#ifdef USE_SLOW_CORNERs
    for(int y=2*marker_maxsize;y<checker.size().height-2*marker_maxsize;y++)
      for(int x=2*marker_maxsize;x<checker.size().width-2*marker_maxsize;x++)
	if (checker.at<uchar>(y, x) > th /*&& simplelienscore(norms[scale_idx], x, y) >= linescore_min*/) {
	  Point2f p = Point2f(x, y)*0.5;
	  /*if (pattern_prescore(small_hc_sb, p, p+Point2f(0, 10)) >= prescore_corner_limit ||
	    pattern_prescore(small_hc_sb, p, p+Point2f(0, -10)) >= prescore_corner_limit ||
	    pattern_prescore(small_hc_sb, p, p+Point2f(10, 0)) >= prescore_corner_limit ||
	    pattern_prescore(small_hc_sb, p, p+Point2f(-10, 0)) >= prescore_corner_limit ||
	    pattern_prescore(small_hc_sb, p, p+Point2f(5, 5)) >= prescore_corner_limit ||
	    pattern_prescore(small_hc_sb, p, p+Point2f(-5, 5)) >= prescore_corner_limit ||
	    pattern_prescore(small_hc_sb, p, p+Point2f(5, -5)) >= prescore_corner_limit ||
	    pattern_prescore(small_hc_sb, p, p+Point2f(-5, -5)) >= prescore_corner_limit)*/
#pragma omp critical
	    corners.push_back(p-b_o);
	}
#else
    for(int y=2*marker_maxsize;y<checker.size().height-2*marker_maxsize;y++)
      for(int x=2*marker_maxsize;x<checker.size().width-2*marker_maxsize;x++)
	if (checker.at<uchar>(y, x) > th && simplelienscore(norms[scale_idx], x, y) >= linescore_min) {
	  Point2f p = Point2f(x, y);
	  if (pattern_prescore(small_hc_sb, p, p+Point2f(0, 10)) >= prescore_corner_limit ||
	    pattern_prescore(small_hc_sb, p, p+Point2f(0, -10)) >= prescore_corner_limit ||
	    pattern_prescore(small_hc_sb, p, p+Point2f(10, 0)) >= prescore_corner_limit ||
	    pattern_prescore(small_hc_sb, p, p+Point2f(-10, 0)) >= prescore_corner_limit ||
	    pattern_prescore(small_hc_sb, p, p+Point2f(5, 5)) >= prescore_corner_limit ||
	    pattern_prescore(small_hc_sb, p, p+Point2f(-5, 5)) >= prescore_corner_limit ||
	    pattern_prescore(small_hc_sb, p, p+Point2f(5, -5)) >= prescore_corner_limit ||
	    pattern_prescore(small_hc_sb, p, p+Point2f(-5, -5)) >= prescore_corner_limit)
#pragma omp critical
	    corners.push_back(p-b_o);
	}
#endif
  
  if (!corners.size())
    return;
  
#ifdef PAINT_CANDIDATE_CORNERS
  sprintf(buf, "corner_cand_%d.png", scale_idx);
  imwrite(buf, checker);
  Mat paint;
  cvtColor(norms[scale_idx], paint, COLOR_GRAY2BGR);
#endif
  
  microbench_measure_output("corner detection");
  
//  
// --- corner estimation ---
//
  corners2.resize(corners.size());
#pragma omp parallel for
  for (uint32_t ci=0;ci<corners.size();ci++) {
    if (ci%500 == 0)
      cout << ci << "estimated of" << corners.size() << " found " << markers.size() << "markers " <<endl;//<< "                  \r" << flush;
    int mask = 7;
    //FIXME
    for(uint32_t i=0;i<markers.size() && mask;i++) {
      if (!scale)
	mask &= markers[i].pointMarkerTest(corners[ci]*0.5);
      else
	mask &= markers[i].pointMarkerTest(corners[ci]*scale);
    }
    if (!mask)
      continue;
    Marker_Corner c = Marker_Corner(corners[ci], mask);
    c.estimateDir(norms[scale_idx]);
    if (c.score <= corner_ok) 
      continue;
    //FIXME this is worse than disabled!
    //if (effort >= 0.75)
      //c.refine(norms[scale_idx], true, dir_step_refine);
    if (c.score < corner_good)
	continue;
    corners2[ci] = c;
  }
  for (uint i=0;i<corners2.size();i++)
    if (corners2[i].score > max(corner_ok, corner_good)) {
	candidates.add(&corners2[i], corners2[i].p);
#ifdef PAINT_CANDIDATE_CORNERS
      //corners2[i].paint(paint);
      Marker_Corner pc = corners2[i];
      pc.p += b_o;
      pc.paint(paint);
#endif
    }
    
  //printf("\npossible marker corners: %d\n", candidates.size());
    
#ifdef PAINT_CANDIDATE_CORNERS
  sprintf(buf, "corners_%d.png", scale_idx);
  imwrite(buf,paint);
#endif
      
  microbench_measure_output("corner estimation");
  
  //printf("found %lu good corners\n", corners2.size());
  //for (uint32_t i=0;i<corners2.size();i++) {
    //circle(paint, corners2[i].p, 1, CV_RGB(200, 200, 0), 1, 1);
    //line(paint, corners2[i].p, corners2[i].p-15.0/corner_score_size*corners2[i].dir[1], CV_RGB(0,255,0), 1);
    //line(paint, corners2[i].p, corners2[i].p-15.0/corner_score_size*corners2[i].dir[0], CV_RGB(255,0,0), 1);
  //}  
  
  //cout << corners2.size() << " found " << markers.size() << "markers " << endl;
  
  //markers.resize(0);
#pragma omp parallel for schedule(dynamic)
  for (uint32_t i=0;i<candidates.size();i++) {
    if (i%500 == 0)
      cout << i << "/" << candidates.size() << " found " << markers.size() << "markers, scale " << scale << endl;//"                  \r" << flush;
    detect_marker((Marker_Corner*)candidates[i], checker, candidates, markers, imgs[scale_idx], small_hc_sb, scale);
  }
  //cout << i << "/" << candidates.size() << " found " << markers.size() << "markers, scale " << scale << endl;
  microbench_measure_output("marker detection");
}

/*void Marker::detect(Mat &img, vector<Marker> &markers)
{
  Mat paint;
  
  cvtColor(img, paint, COLOR_GRAY2BGR);
  
  Marker::detect_scale(img, paint, markers);
}*/

//FIXME call again for newly recognized markers...
int checkneighbours(vector<Mat> &img, vector<vector<int>*> &allmarkers, vector<Marker> &markers, int checked_idx)
{
  Marker *m;
  int found = 0;
  int pdc = post_detection_range;
  int css = 2*corner_score_oversampling*corner_score_size;
  
#ifdef DEBUG_SAVESTEPS
  //Mat paint_extrasearch;
  //cvtColor(img, paint_extrasearch, COLOR_GRAY2BGR);
#endif
  
  for(uint i=0;i<512;i++) {
    if (!allmarkers[i])
      continue;
    for(uint j=0;j<512;j++)
      if ((*allmarkers[i])[j] != -1 && (*allmarkers[i])[j] >= checked_idx) {
	m = &markers[(*allmarkers[i])[j]];
        int scale = log2(m->scale)+1;
        int w = img[scale].size().width;
        int h = img[scale].size().height;
        
	//check neighbours
	for(int n = -2*pdc;n<=2*pdc;n+=2)
	if (j+n < 512 && j+n >= 0 && (*allmarkers[i])[j+n] == -1) {
	  Marker_Corner c[3];
	  Marker newm;
	  //FIXME correct projection & project all four corners
	  c[0] = m->corners[0];
	  c[0].p += n*(m->corners[0].p - m->corners[3].p);
	  c[1] = m->corners[1];
	  c[1].p += n*(m->corners[1].p - m->corners[2].p);
	  c[2] = m->corners[2];
	  c[2].p += n*(m->corners[1].p - m->corners[2].p);
	  //FIXME use original scale and images?
	  for(int i=0;i<3;i++) {
	    c[i].scale = m->scale;
            c[i] *= 1.0/m->scale;
	    if (c[i].p.x <= css || c[i].p.x >= w-css)
	      goto pos_invalid_h;
	    if (c[i].p.y <= css || c[i].p.y >= h-css)
	      goto pos_invalid_h;
	    c[i].refine(img[scale], true, 0);
	    if (c[i].score < corner_good)
	      goto pos_invalid_h;
	  }
	  newm = Marker(Mat(Size(9, 9), CV_8UC1), img[scale], 0.0, &c[0], &c[1], &c[2], m->scale, m->page, m->id+n);
#ifdef DEBUG_SAVESTEPS
	  newm.paint(paint_extrasearch);
#endif
	  if (newm.id == m->id+n && newm.page == m->page) {
	    found++;
	    newm.filter(NULL, markers, 1);
	    markers.push_back(newm);
	    m = &markers[(*allmarkers[i])[j]];
	    (*allmarkers[newm.page])[newm.id] = markers.size()-1;
	  }
	  pos_invalid_h : ;
	}
	for(int n = -32*pdc;n<=32*pdc;n+=32)
	if (j+n < 512 && j+n >= 0 && (*allmarkers[i])[j+n] == -1) {
	  Marker_Corner c[3];
	  Marker newm;
	  //FIXME correct projection & project all four corners
	  c[0] = m->corners[0];
	  c[0].p += n/16*(m->corners[0].p - m->corners[1].p);
	  c[1] = m->corners[1];
	  c[1].p += n/16*(m->corners[0].p - m->corners[1].p);
	  c[2] = m->corners[2];
	  c[2].p += n/16*(m->corners[3].p - m->corners[2].p);
	  //FIXME use original scale and images?
	  for(int i=0;i<3;i++) {
	    c[i].scale = m->scale;
            c[i] *= 1.0/m->scale;
	    if (c[i].p.x <= css || c[i].p.x >= w-css)
	      goto pos_invalid_v;
	    if (c[i].p.y <= css || c[i].p.y >= h-css)
	      goto pos_invalid_v;
	    c[i].refine(img[scale], true, 0);
	    if (c[i].score < corner_good)
	      goto pos_invalid_v;
	  }
	  newm = Marker(Mat(Size(9, 9), CV_8UC1), img[scale], 0.0, &c[0], &c[1], &c[2], m->scale, m->page, m->id+n);
#ifdef DEBUG_SAVESTEPS
	  newm.paint(paint_extrasearch);
#endif
	  if (newm.id == m->id+n && newm.page == m->page) {
	    found++;
	    newm.filter(NULL, markers, 1);
	    markers.push_back(newm);
	    m = &markers[(*allmarkers[i])[j]];
	    (*allmarkers[newm.page])[newm.id] = markers.size()-1;
	  }
	  pos_invalid_v : ;
	}
      }
  }
  
  //printf("found %d markers in post detection check\n", found);
  
/*#ifdef DEBUG_SAVESTEPS
  imwrite("addsearch.png", paint_extrasearch);
  Mat paintm;
  cvtColor(img, paintm, COLOR_GRAY2BGR);
#endif*/
  
  return found;
}

void corners_offset(Marker_Corner c[3], int offx, int offy, Point2f dx, Point2f dy)
{
  for(int i=0;i<3;i++)
    c[i].p += offx*dx + offy*dy;
}

int try_marker_from_corners(Mat &img, Marker_Corner c[3], int page, int id, vector<vector<int>*> &allmarkers, vector<Marker> &markers, float scale)
{
  int css = 2*corner_score_oversampling*corner_score_size;
  int w = img.size().width;
  int h = img.size().height;
  Marker m;
  
  for(int i=0;i<3;i++) {
    if (c[i].p.x <= css || c[i].p.x >= w-css)
      return 0;
    if (c[i].p.y <= css || c[i].p.y >= h-css)
      return 0;
    c[i].scale = scale;
    c[i].refine(img, true, 0);
    if (c[i].score < corner_good)
      return 0;
  }
  m = Marker(Mat(Size(9, 9), CV_8UC1), img, 0.0, &c[0], &c[1], &c[2], scale, page, id);
  if (m.id == id && m.page == page) {
    m.filter(NULL, markers, 1);
    markers.push_back(m);
    (*allmarkers[m.page])[m.id] = markers.size()-1;
    return 1;
  }
  return 0;
}

void corners_from_marker_offset(Marker_Corner c[3], Marker *m, int x, int y)
{
  float fac = (1.0/m->scale);
  Point2f dx = (m->corners[0].p - m->corners[3].p)*fac;
  Point2f dy =  (m->corners[1].p - m->corners[0].p)*fac;
  
  c[0] = m->corners[0]*fac;
  c[1] = m->corners[1]*fac;
  c[2] = m->corners[2]*fac;
  
  corners_offset(c, x, y, dx, dy);
}

//TODO correct projection & project all four corners
//TODO use original scale and images?
int checkneighbours3(vector<Mat> &img, vector<vector<int>*> &allmarkers, vector<Marker> &markers, int checked_idx)
{
  Marker *m;
  int found = 0;
  //make pdc odd
  int pdc = post_detection_range/2*2+1;
  
#ifdef DEBUG_SAVESTEPS
  //Mat paint_extrasearch;
  //cvtColor(img, paint_extrasearch, COLOR_GRAY2BGR);
#endif
  
  for(uint i=0;i<512;i++) {
    if (!allmarkers[i])
      continue;
    //even ids
    for(uint j=0;j<512;j+=2) {
      if ((*allmarkers[i])[j] == -1 || (*allmarkers[i])[j] < checked_idx)
	continue;
      
      m = &markers[(*allmarkers[i])[j]];
      int scale = log2(m->scale)+1;
      //check neighbours
      for(int n = -pdc;n<=pdc;n+=2) {
	if (j+n >= 512 || j+n < 0 || (*allmarkers[i])[j+n] != -1)
	  continue;
	Marker_Corner c[3];
	corners_from_marker_offset(c, m, n, -1);
	found += try_marker_from_corners(img[scale], c, m->page, m->id+n, allmarkers, markers, m->scale);
	//m may be moved by push to markers array!
	m = &markers[(*allmarkers[i])[j]];
      }
      for(int n = -pdc-32;n<=pdc-32;n+=2) {
	if (j+n >= 512 || j+n < 0 || (*allmarkers[i])[j+n] != -1)
	  continue;
	Marker_Corner c[3];
	corners_from_marker_offset(c, m, n+32, 1);
	found += try_marker_from_corners(img[scale], c, m->page, m->id+n, allmarkers, markers, m->scale);
	//m may be moved by push to markers array!
	m = &markers[(*allmarkers[i])[j]];
      }
    }
    //odd ids
    for(uint j=1;j<512;j+=2) {
      if ((*allmarkers[i])[j] == -1 || (*allmarkers[i])[j] < checked_idx)
	continue;
      
      m = &markers[(*allmarkers[i])[j]];
      int scale = log2(m->scale)+1;
      //check neighbours
      for(int n = -pdc;n<=pdc;n+=2) {
	if (j+n >= 512 || j+n < 0 || (*allmarkers[i])[j+n] != -1)
	  continue;
	Marker_Corner c[3];
	corners_from_marker_offset(c, m, n, 1);
	found += try_marker_from_corners(img[scale], c, m->page, m->id+n, allmarkers, markers, m->scale);
	//m may be moved by push to markers array!
	m = &markers[(*allmarkers[i])[j]];
      }
      for(int n = -pdc+32;n<=pdc+32;n+=2) {
	if (j+n >= 512 || j+n < 0 || (*allmarkers[i])[j+n] != -1)
	  continue;
	Marker_Corner c[3];
	corners_from_marker_offset(c, m, n-32, -1);
	found += try_marker_from_corners(img[scale], c, m->page, m->id+n, allmarkers, markers, m->scale);
	//m may be moved by push to markers array!
	m = &markers[(*allmarkers[i])[j]];
      }
    }
    //FIXME add odd ids
  }
  
  //printf("found %d markers in post detection check\n", found);
  
/*#ifdef DEBUG_SAVESTEPS
  imwrite("addsearch.png", paint_extrasearch);
  Mat paintm;
  cvtColor(img, paintm, COLOR_GRAY2BGR);
#endif*/
  
  return found;
}

//FIXME call again for newly recognized markers...
int checkneighbours2(vector<Mat> &img, vector<vector<int>*> &allmarkers, vector<Marker> &markers, int checked_idx)
{
  Marker *m,*m2;
  int found = 0;
  int pdc = post_detection_range;
  int css = 2*corner_score_oversampling*corner_score_size;
  
#ifdef DEBUG_SAVESTEPS
  //Mat paint_extrasearch;
  //cvtColor(img, paint_extrasearch, COLOR_GRAY2BGR);
#endif
  
  for(uint i=0;i<512;i++) {
    if (!allmarkers[i])
      continue;
    for(uint j=0;j<512;j++)
      if ((*allmarkers[i])[j] != -1) {
	m = &markers[(*allmarkers[i])[j]];
        int scale = log2(m->scale)+1;
        int w = img[scale].size().width;
        int h = img[scale].size().height;
	//check neighbours
	for(int n = -2*pdc;n<=2*pdc;n+=2)
	if (j+n < 512 && j+n >= 0 && j-n < 512 && j-n >= 0 && (*allmarkers[i])[j+n] == -1) {
	  if ((*allmarkers[i])[j-n] == -1)
	    continue;
	  if ((*allmarkers[i])[j] < checked_idx && (*allmarkers[i])[j-n] < checked_idx)
	    continue;
	  m2 = &markers[(*allmarkers[i])[j-n]];
	  Marker_Corner c[3];
	  Marker newm;
	  //FIXME correct projection & project all four corners
	  c[0] = m->corners[0];
	  c[0].p += m->corners[0].p - m2->corners[0].p;
	  c[1] = m->corners[1];
	  c[1].p += m->corners[1].p - m2->corners[1].p;
	  c[2] = m->corners[2];
	  c[2].p += m->corners[2].p - m2->corners[2].p;
	  for(int i=0;i<3;i++) {
	    c[i].scale = m->scale;
            c[i] *= 1.0/m->scale;
	    if (c[i].p.x <= css || c[i].p.x >= w-2*css)
	      goto pos_invalid_h;
	    if (c[i].p.y <= css || c[i].p.y >= h-2*css)
	      goto pos_invalid_h;
	    c[i].refine(img[scale], true, 0);
	    if (c[i].score < corner_good)
	      goto pos_invalid_h;
	  }
	  newm = Marker(Mat(Size(9, 9), CV_8UC1), img[scale], 0.0, &c[0], &c[1], &c[2], m->scale, m->page, m->id+n);
#ifdef DEBUG_SAVESTEPS
	  newm.paint(paint_extrasearch);
#endif
	  if (newm.id == m->id+n && newm.page == m->page && newm.score >= pattern_score_good) {
	    found++;
	    newm.filter(NULL, markers, 1);
	    markers.push_back(newm);
	    m = &markers[(*allmarkers[i])[j]];
	    (*allmarkers[newm.page])[newm.id] = markers.size()-1;
	  }
	  pos_invalid_h : ;
	}
	for(int n = -32*pdc;n<=32*pdc;n+=32)
	if (j+n < 512 && j+n >= 0 && j-n < 512 && j-n >= 0 && (*allmarkers[i])[j+n] == -1) {
	  if ((*allmarkers[i])[j-n] == -1)
	    continue;
	  if ((*allmarkers[i])[j] < checked_idx && (*allmarkers[i])[j-n] < checked_idx)
	    continue;
	  m2 = &markers[(*allmarkers[i])[j-n]];
	  Marker_Corner c[3];
	  Marker newm;
	  //FIXME correct projection & project all four corners
	  c[0] = m->corners[0];
	  c[0].p += m->corners[0].p - m2->corners[0].p;
	  c[1] = m->corners[1];
	  c[1].p += m->corners[1].p - m2->corners[1].p;
	  c[2] = m->corners[2];
	  c[2].p += m->corners[2].p - m2->corners[2].p;
	  //FIXME use original scale and images?
	  for(int i=0;i<3;i++) {
	    c[i].scale = m->scale;
            c[i] *= 1.0/m->scale;
	    if (c[i].p.x <= css || c[i].p.x >= w-2*css)
	      goto pos_invalid_v;
	    if (c[i].p.y <= css || c[i].p.y >= h-2*css)
	      goto pos_invalid_v;
	    c[i].refine(img[scale], true, 0);
	    if (c[i].score < corner_good)
	      goto pos_invalid_v;
	  }
	  newm = Marker(Mat(Size(9, 9), CV_8UC1), img[scale], 0.0, &c[0], &c[1], &c[2], m->scale, m->page, m->id+n);
#ifdef DEBUG_SAVESTEPS
	  newm.paint(paint_extrasearch);
#endif
	  if (newm.id == m->id+n && newm.page == m->page && newm.score >= pattern_score_good) {
	    found++;
	    newm.filter(NULL, markers, 1);
	    markers.push_back(newm);
	    m = &markers[(*allmarkers[i])[j]];
	    (*allmarkers[newm.page])[newm.id] = markers.size()-1;
	  }
	  pos_invalid_v : ;
	}
      }
  }
  
  //printf("found %d markers in post detection check\n", found);
  
/*#ifdef DEBUG_SAVESTEPS
  imwrite("addsearch.png", paint_extrasearch);
  Mat paintm;
  cvtColor(img, paintm, COLOR_GRAY2BGR);
#endif*/
  
  return found;
}

void calc_line_dirs(Mat &img1, Mat &paint)
{
  Mat img;
  float m, a;
  float dx, dy;

  resize(img1, img, Size(img1.size().width*2,img1.size().height*2));
  paint.create(img.size(), CV_8UC3);
  
  for(int j=1;j<img.size().height-1;j++)
    for(int i=1;i<img.size().width-1;i++) {
      dx = (int)img.at<uchar>(j,i+1) - img.at<uchar>(j,i-1);
      dy = (int)img.at<uchar>(j+1,i) - img.at<uchar>(j-1,i);
      m = abs(dx)+abs(dy);
      a = atan2(dy, dx);
      paint.at<Vec3b>(j,i)[0] = 40 * (a+M_PI);
      paint.at<Vec3b>(j,i)[1] = m/2;
      paint.at<Vec3b>(j,i)[2] = 255;
    }
  
}

void line_dir_checker_score(Mat &img, Mat &paint)
{
  int dir_simp = 8;
  int r = 3;
  int b_range = 25/dir_simp;
  float scores[256/dir_simp];
  float blur[256/dir_simp];
  float qscore[64/dir_simp];
  float qscore2[64/dir_simp];
  //Mat img;
  
  //GaussianBlur(img1, img, Size(3, 3), 0.3);
  
  paint.create(img.size(), CV_8UC1);
  
  for(int j=r;j<img.size().height-r-1;j++)
    for(int i=r;i<img.size().width-r-1;i++) {
      for(int i=0;i<256/dir_simp;i++)
	scores[i] = 0;
      for(int l=j-r;l<=j+r;l++)
	for(int k=i-r;k<=i+r;k++) 
	  if ((l-j)*(l-j)+(k-i)*(k-i) > 3*3)
	    continue;
	  else
	    scores[img.at<Vec3b>(l,k)[0]/dir_simp] += img.at<Vec3b>(l,k)[1];

      for(int i=0;i<256/dir_simp;i++) {
	blur[i] = 0;
	for(int n=0;n<b_range;n++)
	  blur[i] += scores[(i+n)%(256/dir_simp)];
	blur[i] *= 1.0/(b_range*dir_simp*(2*r+1)*(2*r+1));
      }
      for(int i=0;i<64/dir_simp;i++) {
	qscore[i] = blur[i];
	qscore[i] = min(qscore[i], blur[(i+64/dir_simp)%(256/dir_simp)]);
	qscore[i] = min(qscore[i], blur[(i+128/dir_simp)%(256/dir_simp)]);
	qscore[i] = min(qscore[i], blur[(i+192/dir_simp)%(256/dir_simp)]);
      }
      for(int i=0;i<64/dir_simp;i++)
	qscore2[i] = qscore[i] - qscore[(i+(32/dir_simp))%(64/dir_simp)];
      
      int max_a = 0;
      float max_v = -1;
      for(int i=0;i<64/dir_simp;i++) {
	if (qscore2[i] > max_v) {
	  max_v = qscore2[i];
	  max_a = i;
	}
      }
	
      paint.at<uchar>(j,i) = std::min((float)255*max_v, (float)255.0);
    }
}

void Marker::detect(cv::Mat &img, std::vector<Marker> &markers, int marker_size_max, int marker_size_min, float effort, int mincount, vector<Mat> *scales)
{
  microbench_init();
  Mat paint;
  vector<Mat> checkers;
  vector<Mat> norms;
  Marker_Corner c[4];
  Marker_Corner mc;
  vector<Marker> markers_raw;
  vector<vector<Marker_Corner>*> allcorners;
  vector<vector<int>*> allmarkers;
  vector<Corner> blub;
  Marker *m;
  int scale_min = 1;
  bool last_try = false;
  int checked_3 = 0;
  int checked_2 = 0;
  int checked_1 = 0;
  int inserted = 0;
  int found;
  int lowest_scale;
  bool free_scales = false;
  vector<Mat> scales_border;
  
  allcorners.resize(512);
  allmarkers.resize(512);
  markers_raw.resize(0);
  
  if (!marker_size_max)
    marker_size_max = max(img.size().height,  img.size().width);
  
  while (scale_min*marker_maxsize < marker_size_max)
    scale_min *= 2;
  
  int lowest_scale_idx = log2(scale_min)+2;
  
  if (!marker_size_min)
    marker_size_min = 5;

  if (!scales)
    scales = new vector<Mat>(log2(scale_min)+3);
  else
    scales->resize(log2(scale_min)+3);
  
  scales_border.resize(log2(scale_min)+3);

  norms.resize(log2(scale_min)+3);
  checkers.resize(log2(scale_min)+3);
  (*scales)[1] = img;
  
  copyMakeBorder((*scales)[1], scales_border[1], border, border, border, border, BORDER_REPLICATE);
  //scales_border[1] = (*scales)[1];
  
  //TODO for existing scales input check if images are already contained!
  for(int idx=1;idx<log2(scale_min)+2;idx++)
  {
    Mat blurred;
    GaussianBlur((*scales)[idx],blurred,Size(3, 3), 0);
    //blur((*scales)[idx],blurred,Size(3, 3));
    resize(blurred, (*scales)[idx+1], Size((*scales)[idx].size().width/2,(*scales)[idx].size().height/2), INTER_NEAREST);
    copyMakeBorder((*scales)[idx+1], scales_border[idx+1], border, border, border, border, BORDER_REPLICATE);
    //scales_border[idx+1] = (*scales)[idx+1];
  }
  microbench_measure_output("scale");
  
  norms[lowest_scale_idx].create(scales_border[lowest_scale_idx].size(), CV_8UC1);
  checkers[lowest_scale_idx].create(scales_border[lowest_scale_idx].size(), CV_8UC1);
  //norm_avg_SIMD((*scales)[lowest_scale_idx], norms[lowest_scale_idx], marker_basesize);
  localHistCont(scales_border[lowest_scale_idx], norms[lowest_scale_idx], marker_basesize);
  simpleChessCorner_SIMD(norms[lowest_scale_idx], checkers[lowest_scale_idx], chess_dist);
  
  
  for(float s=scale_min;marker_size_min<=2*marker_minsize*s && s>=detection_scale_min;s/=2)
    {
      int scale_idx = log2(s)+1;
      
      if (s == 0.5) {
	resize((*scales)[1], (*scales)[0], Size((*scales)[1].size().width*2,(*scales)[1].size().height*2), INTER_LINEAR);
	float rs = 1.0;  
	Mat rkern = (Mat_<float>(3, 3) << 0,-rs,0,-rs,4.0*rs+1.0,-rs,0,-rs,0);
	filter2D((*scales)[0], (*scales)[0], (*scales)[0].depth(), rkern);
        copyMakeBorder((*scales)[0], scales_border[0], border, border, border, border, BORDER_REPLICATE);
        //scales_border[0] = (*scales)[0];
      }
      
      norms[scale_idx].create(scales_border[scale_idx].size(), CV_8UC1);
      checkers[scale_idx].create(scales_border[scale_idx].size(), CV_8UC1);
      //norm_avg_SIMD((*scales)[scale_idx], norms[scale_idx], marker_basesize);
      localHistCont(scales_border[scale_idx], norms[scale_idx], marker_basesize);
      simpleChessCorner_SIMD(norms[scale_idx], checkers[scale_idx], chess_dist);
      
#ifdef USE_SLOW_CORNERs
      Mat linedirs, checker_dir, checker_filtered;
      calc_line_dirs(norms[scale_idx], linedirs);
      line_dir_checker_score(linedirs, checker_dir);
      
      checkers[scale_idx] = checker_dir;
      
      char buf[64];
      cvtColor(linedirs, linedirs, COLOR_HLS2BGR);
      sprintf(buf, "linedir_norm_%d.png", scale_idx);
      imwrite(buf, linedirs);
      
      sprintf(buf, "linedir_checker_%d.png", scale_idx);
      imwrite(buf, checker_dir);
#endif
      
      microbench_measure_output("norm and checker");
      
      Marker::detect_scale(scales_border, norms, checkers, markers_raw, s, effort);
      cout << " count " << markers_raw.size() << " scale " << s << endl; 
      
#ifdef PAINT_CANDIDATE_CORNERS
      cvtColor(img, paint, COLOR_GRAY2BGR);
#endif
      
      //collect markers by page
      for(uint i=inserted;i<markers_raw.size();i++) {
	m = &markers_raw[i];
	if (m->page == -1)  {abort();continue;}
#ifdef PAINT_CANDIDATE_CORNERS
	m->paint(paint);
#endif
	if (!allmarkers[m->page]) {
	  allmarkers[m->page] = new vector<int>(512);
	  for(uint n=0;n<512;n++)
	    (*allmarkers[m->page])[n] = -1;
	}
	(*allmarkers[m->page])[m->id] = i;
      }
      inserted = markers_raw.size();
      
      //find missing markers
      //FIXME either use normalized for corners or change in detect_scale
      found = 1;
      while (found) {
	int checked_new;
	found = 0;
	
	//FIXME refresh allmarkers with newly found markers_raw
	checked_new = markers_raw.size();
	
	if (effort < 0.25 && markers_raw.size() >= mincount)
	  break;
	found += checkneighbours3(scales_border, allmarkers, markers_raw, checked_3);
	checked_3 = checked_new;
	
	if (effort < 0.25 && markers_raw.size() >= mincount)
	  break;
	found = checkneighbours2(scales_border, allmarkers, markers_raw, checked_2);
	checked_2 = checked_new;
	
	if (effort < 0.25 && markers_raw.size() >= mincount)
	  break;
	found += checkneighbours(scales_border, allmarkers, markers_raw, checked_1);
	checked_1 = checked_new;
      }
      
      
    microbench_measure_output("neighbour check");
      
      if (markers_raw.size() >= mincount) {
	if (effort < 0.5)
	  break;
	else if (effort >= 0.5 && effort < 1.0) {
	  if (last_try && (effort < 0.75 || s <= 1.0))
	    break;
	  else
	    last_try = true;
	}
      }
    }
  
#ifdef DEBUG_SAVESTEPS
  Mat paintm;
  cvtColor(img, paintm, COLOR_GRAY2BGR);
#endif
  for(uint i=0;i<512;i++) {
    if (!allmarkers[i])
      continue;
    for(uint j=0;j<512;j++)
      if ((*allmarkers[i])[j] != -1) {
	m = &markers_raw[(*allmarkers[i])[j]];
#ifdef DEBUG_SAVESTEPS
	  m->paint(paintm);
#endif
	if ((m->neighbours >= marker_neighbour_valid_count && m->score >= pattern_score_good) || (m->score >= pattern_score_sure)) {
	  markers.push_back(*m);
	}
      }
    delete allmarkers[i];
  }
  
#ifdef DEBUG_SAVESTEPS
  imwrite("markers.png", paintm);
#endif
  
    microbench_measure_output("refine");
    microbench_measure_run("full run");
}

void Marker::detect(Mat img, vector<Corner> &corners, bool use_rgb, int marker_size_max, int marker_size_min, float effort, int mincount)
{
  Marker_Corner c[4];
  Marker_Corner mc;
  Marker *m;
  vector<Marker> markers_raw;
  vector<vector<Marker_Corner>*> allcorners;
  vector<vector<int>*> allmarkers;
  vector<Corner> blub;
  Mat paint, col;
  vector<Mat> scales;
  vector<Marker> markers(0);
  int scale_min = 1;
  if (!marker_size_max)
    marker_size_max = max(img.size().height,  img.size().width);
  
  while (scale_min*marker_maxsize < marker_size_max)
    scale_min *= 2;
  
  allcorners.resize(512);
  
  if (img.depth() != CV_8U)
    img.convertTo(img, CV_8U);
  
  if (img.channels() != 1) {
    col = img.clone();
    cvtColor(img, img, CV_BGR2GRAY);
  }
  else
    use_rgb = false;
  
  detect(img, markers, marker_size_max, marker_size_min, effort, mincount, &scales);
  
  for(uint i=0;i<markers.size();i++) {
    m = &markers[i];
    m->getCorners(c);
    if (!allcorners[m->page])
      allcorners[m->page] = new vector<Marker_Corner>(35*35);
    if ((*allcorners[m->page])[c[0].coord.y*32+c[0].coord.x].page == -1 || (*allcorners[m->page])[c[0].coord.y*32+c[0].coord.x].score < c[0].score)
      (*allcorners[m->page])[c[0].coord.y*32+c[0].coord.x] = c[0];
    if ((*allcorners[m->page])[c[1].coord.y*32+c[1].coord.x].page == -1 || (*allcorners[m->page])[c[1].coord.y*32+c[1].coord.x].score < c[1].score)
      (*allcorners[m->page])[c[1].coord.y*32+c[1].coord.x] = c[1];
    if ((*allcorners[m->page])[c[2].coord.y*32+c[2].coord.x].page == -1 || (*allcorners[m->page])[c[2].coord.y*32+c[2].coord.x].score < c[2].score)
      (*allcorners[m->page])[c[2].coord.y*32+c[2].coord.x] = c[2];
    if ((*allcorners[m->page])[c[3].coord.y*32+c[3].coord.x].page == -1 || (*allcorners[m->page])[c[3].coord.y*32+c[3].coord.x].score < c[3].score)
      (*allcorners[m->page])[c[3].coord.y*32+c[3].coord.x] = c[3];
  }
  
#ifdef PAINT_CANDIDATE_CORNERS
  cvtColor(img, paint, COLOR_GRAY2BGR);
#endif
  
  resize(scales[1], scales[0], Size(scales[1].size().width*2,scales[1].size().height*2), INTER_LINEAR);
  
  Mat channels[3];
  if (use_rgb) {
    split(col, channels);
  }
  
//FIXME either use normalized for corners or change in detect_scale (and refine for all scales?)
#pragma omp parallel for private(mc)
  for(int j=0;j<allcorners.size();j++)
    if (allcorners[j]) {
      for(int i=0;i<(*allcorners[j]).size();i++)
	if ((*allcorners[j])[i].page != -1 && (*allcorners[j])[i].score > corner_good) {
	  mc = (*allcorners[j])[i];
          //printf("refine %.2fx%.2f->", mc.p.x, mc.p.y);
	  //FIXME more accurate refinement+larger region than global options?
	 //mc.estimated = false;
	  //FIXME check/verify accuracy!
	 //if (mc.scale >= 2)
          //FIXME what if mc.scale == 0!
	  /*for(int s=log2(mc.scale)+1,sd=mc.scale;s>=1;s--,sd/=2) {
              mc.refined = false;
              mc.scale = s;
              mc.estimated = false;
	      mc.p = mc.p*(1.0/sd);
	      //mc.estimateDir(scales[s]);
	      mc.refine(scales[s], true);
              mc.refine_size(scales[s], 1.0, true , 0, (mc.size)/sd, (mc.size)/sd/2);
	      mc.p = mc.p*sd;
	    }
            mc.refined = false;
            mc.scale = 0;
            mc.estimated = false;*/
            //mc.p = mc.p*2.0;
            //mc.estimateDir(scales[s]);
            //mc.refine_size(scales[1], 1.0, true , 0, mc.size, mc.size/2);
            mc.refine_gradient(scales[1], 1.0);

            //printf("%02d %02d  ", (int)(mc.size+1), (int)((mc.size+2)*0.5));
            //mc.refine(scales[0], true);
            //mc.p = (mc.p-Point2f(1.0,1.0))*0.5;
          //printf("%.2fx%.2f\n", mc.p.x, mc.p.y);
          (*allcorners[j])[i].p = mc.p;
          if (use_rgb) {
            Point2f merge = mc.p;
            //green
            mc.refine_gradient(channels[1], 1.0);
            (*allcorners[j])[i].pc[1] = mc.p;
            mc.p = merge;
            //blue
            mc.refine_gradient(channels[0], 1.0);
            (*allcorners[j])[i].pc[2] = mc.p;
            mc.p = merge;
            //red
            mc.refine_gradient(channels[2], 1.0);
            (*allcorners[j])[i].pc[0] = mc.p;
            mc.p = merge;
          }
	  
	  
#pragma omp critical 
	  {
#ifdef PAINT_CANDIDATE_CORNERS
	    line(paint, mc.p, mc.p+3*mc.dir[0], CV_RGB(255,0,0), 1);
	    line(paint, mc.p, mc.p+3*mc.dir[1], CV_RGB(0,255,0), 1);
	    line(paint, (*allcorners[j])[i].p, mc.p, CV_RGB(0,0,255), 1);
	    corners.push_back(Corner((*allcorners[j])[i]));
	    corners[corners.size()-1].paint(paint);
#else
	    corners.push_back(Corner((*allcorners[j])[i]));
#endif
	  }
	}
	delete allcorners[j];
    }
    
    
    //cout << " corners: " << corners.size() << endl; 
    //cout << "                                                                             \r";
    microbench_measure_output("cleanup");
    microbench_measure_run("full run");
}

void Marker::init(void)
{
  if (inits)
    return;
  
  inits = 1;
  
  setNumThreads(0);
  
  box_corners.resize(3);
  refine_corners.resize(3);
  box_corners_pers.resize(4);
  
  box_corners[0] = Point2f(6.5, 6.5);
  box_corners[1] = Point2f(6.5, 1.5);
  box_corners[2] = Point2f(1.5, 1.5);
  
  refine_corners[0] = Point2f(-0.5, corner_score_size-0.5);
  refine_corners[1] = Point2f(corner_score_size-0.5, -0.5);
  refine_corners[2] = Point2f(corner_score_size-0.5, corner_score_size-0.5);
  
  box_corners_pers[0] = Point2f(6.5, 6.5);
  box_corners_pers[1] = Point2f(6.5, 1.5);
  box_corners_pers[2] = Point2f(1.5, 1.5);
  box_corners_pers[3] = Point2f(1.5, 6.5);
  

    int s = corner_score_size;
    int d = corner_score_dead;
    int x, y;
    int i = 0;
    for(y=0;y<s;y++)
        for(x=d;x<s;x++) {
            corner_patt_x_b[i] = x;
            corner_patt_y_b[i] = y;
            i++;
        }
        
    i = 0;    
    for(y=0;y<s;y++)
        for(x=s;x<s+d;x++) {
            corner_patt_x_w[i] = x;
            corner_patt_y_w[i] = y;
            i++;
        }
}