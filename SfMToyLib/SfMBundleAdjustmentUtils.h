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
    static void adjustBundle(
            const PointCloud&               pointCloud,
            const std::vector<cv::Matx34f>& cameraPoses,
            const Intrinsics&               intrinsics,
            const std::vector<Features>&    image2dFeatures
            );
};

} /* namespace sfmtoylib */

#endif /* SFMTOYLIB_SFMBUNDLEADJUSTMENTUTILS_H_ */
