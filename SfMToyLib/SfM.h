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

#include "SfM2DFeatureUtilities.h"

#include <string>
#include <vector>

namespace sfmtoylib {

enum ErrorCode {
    OKAY = 0,
    ERROR
};

class SfM {
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


    std::vector<std::string>              mImageFilenames;
    std::vector<cv::Mat>                  mImages;
    std::vector<FeaturePointsDescriptors> mImageFeatures;
    SfM2DFeatureUtilities                 mFeatureUtilities;
};

} /* namespace sfmtoylib */

#endif /* SFMTOYLIB_SFM_H_ */
