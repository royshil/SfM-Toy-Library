/*
 * SfMStereoUtilities.h
 *
 *  Created on: May 27, 2016
 *      Author: roy_shilkrot
 * 
 *  Copyright @ Roy Shilkrot 2016
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

#ifndef SFMTOYLIB_SFMSTEREOUTILITIES_H_
#define SFMTOYLIB_SFMSTEREOUTILITIES_H_

#include "SfMCommon.h"

namespace sfmtoylib {

class SfMStereoUtilities {
public:
    SfMStereoUtilities();
    virtual ~SfMStereoUtilities();

    /**
     * Find the amount of inlier points in a homography between 2 views.
     * @param left      Left image features
     * @param right     Right image features
     * @param matches   Matching between the features
     * @return number of inliers.
     */
    static int findHomographyInliers(
            const Features& left,
            const Features& right,
            const Matching& matches);

    /**
     * Find camera matrices (3x4 poses) from stereo point matching.
     * @param intrinsics      Camera intrinsics (assuming both cameras have the same parameters)
     * @param featureMatching Matching between left and right features
     * @param featuresLeft    Features in left image
     * @param featuresRight   Features in right image
     * @param prunedMatches   Output: matching after pruning using essential matrix
     * @param Pleft           Output: left image matrix (3x4)
     * @param Pright          Output: right image matrix (3x4)
     * @return true on success.
     */
    static bool findCameraMatricesFromMatch(
            const Intrinsics& intrinsics,
            const Matching&   featureMatching,
            const Features&   featuresLeft,
            const Features&   featuresRight,
			Matching&         prunedMatches,
            cv::Matx34f&      Pleft,
            cv::Matx34f&      Pright);

    /**
     * Triangulate (recover 3D locations) from point matching.
     * @param imagePair     Indices of left and right views
     * @param leftFeatures  Left image features
     * @param rightFeatures Right image features
     * @param Pleft         Left camera matrix
     * @param Pright        Right camera matrix
     * @param pointCloud    Output: point cloud with image associations
     * @return true on success.
     */
    static bool triangulateViews(
            const Intrinsics&  intrinsics,
            const ImagePair    imagePair,
            const Matching&    matches,
            const Features&    leftFeatures,
            const Features&    rightFeatures,
            const cv::Matx34f& Pleft,
            const cv::Matx34f& Pright,
            PointCloud&        pointCloud
            );

    /**
     * Find the camera location based on 2D 3D point correspondence using PnP.
     * @param intrinsics camera intrinsics
     * @param match      a 2d-3d point matching
     * @param cameraPose Output: camera pose
     * @return true on success.
     */
    static bool findCameraPoseFrom2D3DMatch(
            const Intrinsics&     intrinsics,
            const Image2D3DMatch& match,
            cv::Matx34f&          cameraPose
            );
};

} /* namespace sfmtoylib */

#endif /* SFMTOYLIB_SFMSTEREOUTILITIES_H_ */
