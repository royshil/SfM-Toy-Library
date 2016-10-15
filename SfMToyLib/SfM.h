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
#include <set>

namespace sfmtoylib {

enum ErrorCode {
    OKAY = 0,
    ERROR
};

class SfM {
    /**
     * This is a matrix of matches from view i to view j
     */
    typedef std::vector<std::vector<Matching> > MatchMatrix;

    typedef std::map<int, Image2D3DMatch> Images2D3DMatches;

public:
    SfM(const float downscale = 1.0);
    virtual ~SfM();

    /**
     * Set the directory with images to perform the SfM operation on.
     * Image file with extensions "jpg" and "png" will be used.
     * @return true on success.
     */
    bool setImagesDirectory(const std::string& directoryPath);

    /**
     * This is the main function of this class. Start here.
     * Run the SfM operation:
     *  - Extract and match image features.
     *  - Find a baseline triangulation.
     *  - Sequentially add more views to the cloud.
     * @return error code.
     */
    ErrorCode runSfM();

    void saveCloudAndCamerasToPLY(const std::string& filename);

    void setConsoleDebugLevel(unsigned int consoleDebugLevel) {
        mConsoleDebugLevel = MIN(LOG_ERROR, consoleDebugLevel);
    }

    void setVisualDebugLevel(unsigned int visualDebugLevel) {
        mVisualDebugLevel = MIN(LOG_ERROR, visualDebugLevel);
    }

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

    /**
     * For all remaining images to process, find the set of 2D points that correlate to 3D points in the current cloud.
     * This is done by scanning the 3D cloud and checking the originating 2D views of each 3D point to see if they
     * match 2D features in the new views.
     * @return 2D-3D matching from the image features to the cloud
     */
    Images2D3DMatches find2D3DMatches();

    /**
     * Merge the given point cloud into the existing reconstruction, by merging 3D points from multiple views.
     * @param cloud to merge
     * @return number of new points added
     */
    void mergeNewPointCloud(const PointCloud& cloud);

    std::vector<std::string>  mImageFilenames;
    std::vector<cv::Mat>      mImages;
    std::vector<Features>     mImageFeatures;
    std::vector<cv::Matx34f>  mCameraPoses;
    std::set<int>             mDoneViews;
    std::set<int>             mGoodViews;
    MatchMatrix               mFeatureMatchMatrix;
    SfM2DFeatureUtilities     mFeatureUtil;
    Intrinsics                mIntrinsics;
    PointCloud                mReconstructionCloud;
    unsigned int              mConsoleDebugLevel;
    unsigned int              mVisualDebugLevel;
    float                     mDownscaleFactor;
};

} /* namespace sfmtoylib */

#endif /* SFMTOYLIB_SFM_H_ */
