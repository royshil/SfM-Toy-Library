/*
 * SfMBundleAdjustmentUtils.cpp
 *
 *  Created on: Jun 6, 2016
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

#include "SfMBundleAdjustmentUtils.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <mutex>

using namespace cv;
using namespace std;

namespace sfmtoylib {

namespace BundleAdjustUtils {

/**
 * Use call_once to initialize Google logging only once.
 */
void initLogging() {
    google::InitGoogleLogging("SFM");
}

std::once_flag initLoggingFlag;

}
using namespace BundleAdjustUtils;

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 7 parameters: 3 for rotation, 3 for translation, 1 for
// focal length. The principal point is not modeled (assumed be located at the
// image center, and already subtracted from 'observed'), and focal_x = focal_y.
struct SimpleReprojectionError {
    SimpleReprojectionError(double observed_x, double observed_y) :
            observed_x(observed_x), observed_y(observed_y) {
    }
    template<typename T>
    bool operator()(const T* const camera,
    				const T* const point,
					const T* const focal,
						  T* residuals) const {
        T p[3];
        // Rotate: camera[0,1,2] are the angle-axis rotation.
        ceres::AngleAxisRotatePoint(camera, point, p);

        // Translate: camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // Perspective divide
        const T xp = p[0] / p[2];
        const T yp = p[1] / p[2];

        // Compute final projected point position.
        const T predicted_x = *focal * xp;
        const T predicted_y = *focal * yp;

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);
        return true;
    }
    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double observed_x, const double observed_y) {
        return (new ceres::AutoDiffCostFunction<SimpleReprojectionError, 2, 6, 3, 1>(
                new SimpleReprojectionError(observed_x, observed_y)));
    }
    double observed_x;
    double observed_y;
};

void SfMBundleAdjustmentUtils::adjustBundle(
        PointCloud&                  pointCloud,
        std::vector<Pose>&           cameraPoses,
        Intrinsics&                  intrinsics,
        const std::vector<Features>& image2dFeatures) {

    std::call_once(initLoggingFlag, initLogging);

    // Create residuals for each observation in the bundle adjustment problem. The
    // parameters for cameras and points are added automatically.
    ceres::Problem problem;

    //Convert camera pose parameters from [R|t] (3x4) to [Angle-Axis (3), Translation (3), focal (1)] (1x7)
    typedef cv::Matx<double, 1, 6> CameraVector;
    vector<CameraVector> cameraPoses6d;
    cameraPoses6d.reserve(cameraPoses.size());
    for (size_t i = 0; i < cameraPoses.size(); i++) {
        const Pose& pose = cameraPoses[i];

        if (pose(0, 0) == 0 and pose(1, 1) == 0 and pose(2, 2) == 0) {
            //This camera pose is empty, it should not be used in the optimization
            cameraPoses6d.push_back(CameraVector());
            continue;
        }
        Vec3f t(pose(0, 3), pose(1, 3), pose(2, 3));
        Matx33f R = pose.get_minor<3, 3>(0, 0);
        float angleAxis[3];
        ceres::RotationMatrixToAngleAxis<float>(R.t().val, angleAxis); //Ceres assumes col-major...

        cameraPoses6d.push_back(CameraVector(
                angleAxis[0],
                angleAxis[1],
                angleAxis[2],
                t(0),
                t(1),
                t(2)));
    }

    //focal-length factor for optimization
    double focal = intrinsics.K.at<float>(0, 0);

    vector<cv::Vec3d> points3d(pointCloud.size());

    for (int i = 0; i < pointCloud.size(); i++) {
        const Point3DInMap& p = pointCloud[i];
        points3d[i] = cv::Vec3d(p.p.x, p.p.y, p.p.z);

        for (const auto& kv : p.originatingViews) {
            //kv.first  = camera index
            //kv.second = 2d feature index
            Point2f p2d = image2dFeatures[kv.first].points[kv.second];

            //subtract center of projection, since the optimizer doesn't know what it is
            p2d.x -= intrinsics.K.at<float>(0, 2);
            p2d.y -= intrinsics.K.at<float>(1, 2);

            // Each Residual block takes a point and a camera as input and outputs a 2
            // dimensional residual. Internally, the cost function stores the observed
            // image location and compares the reprojection against the observation.
            ceres::CostFunction* cost_function = SimpleReprojectionError::Create(p2d.x, p2d.y);

            problem.AddResidualBlock(cost_function,
                    NULL /* squared loss */,
                    cameraPoses6d[kv.first].val,
                    points3d[i].val,
					&focal);
        }
    }

    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 500;
    options.eta = 1e-2;
    options.max_solver_time_in_seconds = 10;
    options.logging_type = ceres::LoggingType::SILENT;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";

    if (not (summary.termination_type == ceres::CONVERGENCE)) {
        cerr << "Bundle adjustment failed." << endl;
        return;
    }

    //update optimized focal
    intrinsics.K.at<float>(0, 0) = focal;
    intrinsics.K.at<float>(1, 1) = focal;

    //Implement the optimized camera poses and 3D points back into the reconstruction
    for (size_t i = 0; i < cameraPoses.size(); i++) {
    	Pose& pose = cameraPoses[i];
    	Pose poseBefore = pose;

        if (pose(0, 0) == 0 and pose(1, 1) == 0 and pose(2, 2) == 0) {
            //This camera pose is empty, it was not used in the optimization
            continue;
        }

        //Convert optimized Angle-Axis back to rotation matrix
        double rotationMat[9] = { 0 };
        ceres::AngleAxisToRotationMatrix(cameraPoses6d[i].val, rotationMat);

        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                pose(c, r) = rotationMat[r * 3 + c]; //`rotationMat` is col-major...
            }
        }

        //Translation
        pose(0, 3) = cameraPoses6d[i](3);
        pose(1, 3) = cameraPoses6d[i](4);
        pose(2, 3) = cameraPoses6d[i](5);
    }

    for (int i = 0; i < pointCloud.size(); i++) {
        pointCloud[i].p.x = points3d[i](0);
        pointCloud[i].p.y = points3d[i](1);
        pointCloud[i].p.z = points3d[i](2);
    }
}

} /* namespace sfmtoylib */
