/*
 *  sfmviewer.cpp
 *  SfMToyLibrary
 *
 *  Created by Roy Shilkrot on 11/3/13.
 *  Copyright 2013 MIT. All rights reserved.
 *
 */

#include "sfmviewer.h"

void SFMViewer::draw() {
	glPushMatrix();
	glScalef(vizScale,vizScale,vizScale);

	glPushAttrib(GL_ENABLE_BIT);
	glDisable(GL_LIGHTING);
	glBegin(GL_POINTS);
	for (int i = 0; i < m_pcld.size(); ++i) {
		glColor3b(m_pcldrgb[i][0],m_pcldrgb[i][1],m_pcldrgb[i][2]);
		glVertex3dv(&(m_pcld[i].x));
	}
	glEnd();

	glScaled(scale_cameras_down,scale_cameras_down,scale_cameras_down);
	glEnable(GL_RESCALE_NORMAL);
	glEnable(GL_LIGHTING);
	for (int i = 0; i < m_cameras_transforms.size(); ++i) {

		glPushMatrix();
//		glTranslated(c.x(),c.y(),c.z());
		glMultMatrixd(m_cameras_transforms[i].data());

	    glColor4f(1, 0, 0, 1);
	    QGLViewer::drawArrow(qglviewer::Vec(0,0,0), qglviewer::Vec(3,0,0));
	    glColor4f(0, 1, 0, 1);
	    QGLViewer::drawArrow(qglviewer::Vec(0,0,0), qglviewer::Vec(0,3,0));
	    glColor4f(0, 0, 1, 1);
	    QGLViewer::drawArrow(qglviewer::Vec(0,0,0), qglviewer::Vec(0,0,3));

	    glPopMatrix();

//		Eigen::Matrix3f R = m_cameras[i].block(0,0,3,3);
//		Eigen::Vector3f _t = m_cameras[i].col(3);
//
//		Vector3f t = -R.transpose() * _t;
//
//		Vector3f vright = R.row(0).normalized() * s;
//		Vector3f vup = -R.row(1).normalized() * s;
//		Vector3f vforward = R.row(2).normalized() * s;
//
//		Vector3f rgb(r,g,b);
//
//		pcl::PointCloud<pcl::PointXYZRGB> mesh_cld;
//		mesh_cld.push_back(Eigen2PointXYZRGB(t,rgb));
//		mesh_cld.push_back(Eigen2PointXYZRGB(t + vforward + vright/2.0 + vup/2.0,rgb));
//		mesh_cld.push_back(Eigen2PointXYZRGB(t + vforward + vright/2.0 - vup/2.0,rgb));
//		mesh_cld.push_back(Eigen2PointXYZRGB(t + vforward - vright/2.0 + vup/2.0,rgb));
//		mesh_cld.push_back(Eigen2PointXYZRGB(t + vforward - vright/2.0 - vup/2.0,rgb));
//
//		glBegin(GL_QUADS);
//		glVertex()
	}

	glPopAttrib();
	glPopMatrix();
}

void SFMViewer::init() {
	// Restore previous viewer state.
	restoreStateFromFile();

    setFPSIsDisplayed();

	setSceneBoundingBox(qglviewer::Vec(-50,-50,-50), qglviewer::Vec(50,50,50));

	showEntireScene();
}
