/*
 *  GPUSURFFeatureMatcher.h
 *  ExploringSfMWithOpenCV
 *
 *  Created by Roy Shilkrot on 6/13/12.
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2013 Roy Shilkrot
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 */

#include "IFeatureMatcher.h"
#include <opencv2/opencv_modules.hpp>

#ifdef HAVE_OPENCV_GPU
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/nonfree/gpu.hpp>


class GPUSURFFeatureMatcher : public IFeatureMatcher {
private:
	cv::Ptr<cv::gpu::SURF_GPU> extractor;
	
	std::vector<cv::gpu::GpuMat> descriptors;
	
	std::vector<cv::gpu::GpuMat> imgs; 
	std::vector<cv::gpu::GpuMat> imggpupts;
	std::vector<std::vector<cv::KeyPoint> >& imgpts;

	bool use_ratio_test;
public:
	//c'tor
	GPUSURFFeatureMatcher(std::vector<cv::Mat>& imgs, 
					   std::vector<std::vector<cv::KeyPoint> >& imgpts);
	
	void MatchFeatures(int idx_i, int idx_j, std::vector<cv::DMatch>* matches = NULL);
	
	std::vector<cv::KeyPoint> GetImagePoints(int idx) { return imgpts[idx]; }
};
#else
//empty impl.
class GPUSURFFeatureMatcher : public IFeatureMatcher {
public:
	GPUSURFFeatureMatcher(std::vector<cv::Mat>& imgs, 
						  std::vector<std::vector<cv::KeyPoint> >& imgpts) {}
	
	void MatchFeatures(int idx_i, int idx_j, std::vector<cv::DMatch>* matches = NULL) {}
	
	std::vector<cv::KeyPoint> GetImagePoints(int idx) { return std::vector<cv::KeyPoint>(); }
	
};

#endif
