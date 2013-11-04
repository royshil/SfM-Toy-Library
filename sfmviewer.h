/*
 *  sfmviewer.h
 *  SfMToyLibrary
 *
 *  Created by Roy Shilkrot on 11/3/13.
 *  Copyright 2013 MIT. All rights reserved.
 *
 */
#pragma once


#include <QGLViewer/qglviewer.h>
#include <QFileDialog>
#include <QLineEdit>
#include <QThreadPool>
#include <Eigen/Eigen>

#include "MultiCameraPnP.h"

void open_imgs_dir(const char* dir_name, std::vector<cv::Mat>& images, std::vector<std::string>& images_names, double downscale_factor);

class SFMViewer : public QGLViewer, public SfMUpdateListener, public QRunnable
{
	Q_OBJECT
	cv::Ptr<MultiCameraPnP> 		distance;

	std::vector<cv::Mat> 			images;
	std::vector<std::string> 		images_names;
	std::vector<cv::Point3d> 		m_pcld;
	std::vector<cv::Vec3b> 			m_pcldrgb;
	std::vector<cv::Matx34d> 		m_cameras;
	std::vector<Eigen::Affine3d> 	m_cameras_transforms;
	Eigen::Affine3d 				m_global_transform;

//	QThreadPool 					qtp;

	float 							vizScale;
	double 							scale_cameras_down;

protected :
	virtual void draw();
	virtual void init();

public:
	SFMViewer(QWidget *parent = 0):QGLViewer(QGLFormat::defaultFormat(),parent),vizScale(1.0) {
		distance = new MultiCameraPnP();
		distance->attach(this);
		m_global_transform = Eigen::Affine3d::Identity();
	}
    ~SFMViewer() { saveStateToFile(); }

    virtual void update(std::vector<cv::Point3d> pcld,
			std::vector<cv::Vec3b> pcldrgb,
			std::vector<cv::Point3d> pcld_alternate,
			std::vector<cv::Vec3b> pcldrgb_alternate,
			std::vector<cv::Matx34d> cameras);


	void run() { distance->RecoverDepthFromImages(); }

public slots:
	void openDirectory() {
		images.clear();images_names.clear();
		std::string imgs_path = QFileDialog::getExistingDirectory(this, tr("Open Images Directory"), ".").toStdString();
		double scale_factor = 1.0;
        QLineEdit* l = parentWidget()->findChild<QLineEdit*>("lineEdit_scaleFactor");
        if(l) {
        	scale_factor = l->text().toFloat();
        	std::cout << "downscale to " << scale_factor << std::endl;
        }
		open_imgs_dir(imgs_path.c_str(),images,images_names,scale_factor);
		if(images.size() == 0) {
			std::cerr << "can't get image files" << std::endl;
		} else {
			distance->setImages(images,images_names,imgs_path);
		}
	}
	void setUseRichFeatures(bool b) {distance->use_rich_features = b;}
	void setUseGPU(bool b) {distance->use_gpu = b;}
	void runSFM() {
		this->setAutoDelete(false);
		m_pcld.clear();
		m_pcldrgb.clear();
		m_cameras.clear();
		m_cameras_transforms.clear();
		QThreadPool::globalInstance()->start(this);
	}
	void setVizScale(int i) { vizScale = (float)(i); updateGL(); }
};
