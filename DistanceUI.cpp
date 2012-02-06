/*
 *  DistanceUI.cpp
 *  EyeRingOpenCV
 *
 *  Created by Roy Shilkrot on 12/22/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#include "Distance.h"
#include "DistanceUI.h"
#include <fltk3/FileChooser.h> 
#include "Visualization.h"
#include "gui.h"

cv::Ptr<fltk3::Widget> left_w,right_w;
cv::Ptr<fltk3::Window> window;
bool left_loaded,right_loaded,distance_init;
cv::Mat left_im,right_im;
cv::Ptr<Distance> distance;
cv::Ptr<UserInterface> ui;

bool OpenFileAndLoadWidget(cv::Ptr<fltk3::Widget>& _w, cv::Mat& m) {
	char* filename = fltk3::file_chooser("Choose a file","","/Users/royshilkrot/Documents/eyering/coop",0);
	if (filename == NULL) {
		return false;
	}
	cv::Mat_<cv::Vec3b> im = cv::imread(filename);
	if (im.cols > 0 && im.rows > 0) {
		im.copyTo(m);
		cv::resize(im,im,cv::Size(273,273*im.cols/im.rows));
		window->remove(_w);
		_w.release();
		_w = cv::Ptr<fltk3::Widget>(new OpenCVImageViewer(im,filename));
		window->add(_w);
		return true;
	} else {
		fltk3::alert("Cannot load image at %s",filename);
		return false;
	}
}

void leftcb(fltk3::Button *, void * matptr) {
	if(left_loaded = OpenFileAndLoadWidget(left_w, left_im)) {	
		left_w->resize(225,5,273,205);
		window->redraw();
		distance_init = false;
	}
}

void rightcb(fltk3::Button *, void * matptr) {
	if(right_loaded = OpenFileAndLoadWidget(right_w, right_im)) {
		right_w->resize(500,5,273,205);
		window->redraw();
		distance_init = false;
	}
}

void exitcb(fltk3::Button *, void *) {
	fltk3::first_window()->hide();
}

void initD() {
	if (!distance_init && left_loaded && right_loaded) {
		distance = new Distance(left_im,right_im);
		distance_init = true;
	}
}

void matchcb(fltk3::Button *, void *) {
	initD();
	int strategy =	(ui->feature_match_rb->value() ? STRATEGY_USE_FEATURE_MATCH : 0) +
					(ui->optical_flow_rb->value() ? STRATEGY_USE_OPTICAL_FLOW : 0) +
					(ui->dense_of_rb->value() ? STRATEGY_USE_DENSE_OF : 0) +
					(ui->horiz_disparity_rb->value() ? STRATEGY_USE_HORIZ_DISPARITY : 0);
	distance->OnlyMatchFeatures(strategy);
}

void depthcb(fltk3::Button *, void *) {
	initD();
	distance->RecoverDepthFromImages();
}

void mndcb(fltk3::Button *, void *) {
	initD();
	distance->OnlyMatchFeatures();
	distance->RecoverDepthFromImages();
}

void vizcb(fltk3::Button *, void *)
{
	RunVisualization(distance->getpointcloud(), distance->getleft_im_orig(), distance->getright_im_orig(), distance->getcorrespImg1Pt());
}

int runUI(int argc, char** argv) {
	left_loaded = right_loaded = distance_init = false;
	
	window = new fltk3::Window(780,215);
	
	ui = new UserInterface(0,0,780,215);

	window->end();
	window->show(argc, argv);
	return fltk3::run();
}