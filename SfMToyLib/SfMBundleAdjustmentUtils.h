/*
 * SfMBundleAdjustmentUtils.h
 *
 *  Created on: Jun 6, 2016
 *      Author: roy_shilkrot
 * 
 *  Copyright @  2016
 */

#ifndef SFMTOYLIB_SFMBUNDLEADJUSTMENTUTILS_H_
#define SFMTOYLIB_SFMBUNDLEADJUSTMENTUTILS_H_

#include "SfMCommon.h"

namespace sfmtoylib {

class SfMBundleAdjustmentUtils {
public:
    /**
     *
     * @param pointCloud
     * @param cameraPoses
     * @param intrinsics
     * @param image2dFeatures
     */
    static void adjustBundle(
            PointCloud&                     pointCloud,
            std::vector<cv::Matx34f>&       cameraPoses,
            Intrinsics&                     intrinsics,
            const std::vector<Features>&    image2dFeatures
            );
};

} /* namespace sfmtoylib */

#endif /* SFMTOYLIB_SFMBUNDLEADJUSTMENTUTILS_H_ */
