/*
 *  DistanceUI.h
 *  EyeRingOpenCV
 *
 *  Created by Roy Shilkrot on 12/22/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#include <fltk3/fltk3.h>

#include <opencv2/opencv.hpp>

#include <fltk3/Widget.h>
#include <fltk3/draw.h>
#include <fltk3/RGBImage.h>
#include <fltk3/Box.h>
#include <fltk3/names.h> 

class OpenCVImageViewer : public fltk3::Widget {
protected:
	cv::Mat im;
	cv::Ptr<fltk3::RGBImage> img;
public:
	OpenCVImageViewer(const cv::Mat& _im,const char* label = 0):
	im(_im),
	fltk3::Widget(0, 0, _im.cols, _im.rows, label)
	{
		setImage(im);
	}
	
	void setImage(cv::Mat& im) {
		img = new fltk3::RGBImage(im.data,im.cols,im.rows);
	}
	
	virtual void draw() {
		img->draw(x_,y_,w_,h_); //*im.cols/im.rows);
	}
};