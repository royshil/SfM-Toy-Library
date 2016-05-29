/*
 * SfM.h
 *
 *  Created on: May 26, 2016
 *      Author: roy_shilkrot
 * 
 *  Copyright @  2016
 */

#ifndef SFMTOYLIB_SFM_H_
#define SFMTOYLIB_SFM_H_

#include "SfMCommon.h"
#include "SfM2DFeatureUtilities.h"

#include <string>
#include <vector>
#include <map>

namespace sfmtoylib {

enum ErrorCode {
    OKAY = 0,
    ERROR
};

class SfM {
    typedef std::vector<std::vector<Matching> > MatchMatrix;

public:
    SfM();
    virtual ~SfM();

    /**
     * Set the directory with images to perform the SfM operation on.
     * Image file with extensions "jpg" and "png" will be used.
     * @return true on success.
     */
    bool setImagesDirectory(const std::string directoryPath);

    /**
     * Run the SfM operation.
     * @return error code.
     */
    ErrorCode runSfM();

private:
    /**
     * Extract features for all images in working set.
     */
    void extractFeatures();

    /**
     * Create a feature-matching matrix between all frames in working set.
     */
    void createFeatureMatchMatrix();

    /**
     * Find the best two views and perform an initial triangulation from their feature matching.
     */
    void findBaselineTriangulation();

    /**
     * Run a bundle adjuster on the current reconstruction.
     */
    void adjustCurrentBundle();

    /**
     * Sort the image pairs for the initial baseline triangulation based on the number of homography-inliers
     * @return scoring of views-pairs
     */
    std::map<float, ImagePair> sortViewsForBaseline();

    /**
     * Add more views from the set to the 3D point cloud
     */
    void addMoreViewsToReconstruction();


    std::vector<std::string>      mImageFilenames;
    std::vector<cv::Mat>          mImages;
    std::vector<Features>         mImageFeatures;
    MatchMatrix                   mFeatureMatchMatrix;
    SfM2DFeatureUtilities         mFeatureUtil;
    Intrinsics                    mIntrinsics;
    PointCloud                    mReconstructionCloud;
};

} /* namespace sfmtoylib */

#endif /* SFMTOYLIB_SFM_H_ */
