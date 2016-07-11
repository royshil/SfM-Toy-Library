/*
 * SfMBundleAdjustmentUtils.cpp
 *
 *  Created on: Jun 6, 2016
 *      Author: roy_shilkrot
 * 
 *  Copyright @  2016
 */

#include "SfMBundleAdjustmentUtils.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

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
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
struct SnavelyReprojectionError {
    SnavelyReprojectionError(double observed_x, double observed_y) :
            observed_x(observed_x), observed_y(observed_y) {
    }
    template<typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const {
        T p[3];
        // camera[0,1,2] are the angle-axis rotation.
        ceres::AngleAxisRotatePoint(camera, point, p);
        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];
        // Apply second and fourth order radial distortion.
//        const T& l1 = camera[7];
//        const T& l2 = camera[8];
//        T r2 = xp * xp + yp * yp;
        T distortion = T(1.0); // + r2 * (l1 + l2 * r2);
        // Compute final projected point position.
        const T& focal = camera[6];
        T predicted_x = focal * distortion * xp;
        T predicted_y = focal * distortion * yp;
        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);
        return true;
    }
    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double observed_x, const double observed_y) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                new SnavelyReprojectionError(observed_x, observed_y)));
    }
    double observed_x;
    double observed_y;
};

void SfMBundleAdjustmentUtils::adjustBundle(
        PointCloud&                  pointCloud,
        std::vector<Pose>&           cameraPoses,
        const Intrinsics&            intrinsics,
        const std::vector<Features>& image2dFeatures) {

    std::call_once(initLoggingFlag, initLogging);

    // Create residuals for each observation in the bundle adjustment problem. The
    // parameters for cameras and points are added automatically.
    ceres::Problem problem;

    //Convert camera pose parameters from [R|t] (3x4) to [Angle-Axis, Translation, focal, distortion] (1x9)
    typedef cv::Matx<double, 1, 9> CameraVector;
    vector<CameraVector> cameraPoses9d;
    cameraPoses9d.reserve(cameraPoses.size());
    for (size_t i = 0; i < cameraPoses.size(); i++) {
        const Pose& pose = cameraPoses[i];

        if (pose(0, 0) == 0 and pose(1, 1) == 0 and pose(2, 2) == 0) {
            //This camera pose is empty, it should not be used in the optimization
            cameraPoses9d.push_back(CameraVector());
            continue;
        }
        Vec3f t(pose(0, 3), pose(1, 3), pose(2, 3));
        Matx33f R = pose.get_minor<3, 3>(0, 0);
        float angleAxis[3];
        ceres::RotationMatrixToAngleAxis<float>(R.val, angleAxis);

        cameraPoses9d.push_back(CameraVector(
                angleAxis[0],
                angleAxis[1],
                angleAxis[2],
                t(0),
                t(1),
                t(2),
                intrinsics.K.at<float>(0, 0), 0, 0));
    }

    vector<cv::Vec3d> points3d(pointCloud.size());

    for (int i = 0; i < pointCloud.size(); i++) {
        const Point3DInMap& p = pointCloud[i];
        points3d[i] = cv::Vec3d(p.p.x, p.p.y, p.p.z);

        for (const auto& kv : p.originatingViews) {
            //kv.first  = camera index
            //kv.second = 2d feature index
            const Point2f p2d = image2dFeatures[kv.first].points[kv.second];

            // Each Residual block takes a point and a camera as input and outputs a 2
            // dimensional residual. Internally, the cost function stores the observed
            // image location and compares the reprojection against the observation.
            ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(p2d.x, p2d.y);

            problem.AddResidualBlock(cost_function,
                    NULL /* squared loss */,
                    cameraPoses9d[kv.first].val,
                    points3d[i].val);
        }
    }

    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 50;
    options.eta = 1e-2;
    options.max_solver_time_in_seconds = 10;
    options.logging_type = ceres::LoggingType::SILENT;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    //Implement the optimized camera poses and 3D points back into the reconstruction
    for (size_t i = 0; i < cameraPoses.size(); i++) {
        if (cameraPoses[i](0, 0) == 0 and cameraPoses[i](1, 1) == 0 and cameraPoses[i](2, 2) == 0) {
            //This camera pose is empty, it was not used in the optimization
            continue;
        }

        //Convert optimized Angle-Axis back to rotation matrix
        double rotationMat[9] = { 0 };
        ceres::AngleAxisToRotationMatrix(cameraPoses9d[i].val, rotationMat);

        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                cameraPoses[i].val[r * 4 + c] = rotationMat[r * 3 + c];
            }
        }
        //Translation
        cameraPoses[i].val[0 * 3 + 3] = cameraPoses9d[i](3);
        cameraPoses[i].val[1 * 3 + 3] = cameraPoses9d[i](4);
        cameraPoses[i].val[2 * 3 + 3] = cameraPoses9d[i](5);
    }

    for (int i = 0; i < pointCloud.size(); i++) {
//        cout << "pt " << i << " before: " << pointCloud[i].p << endl;
        pointCloud[i].p.x = points3d[i](0);
        pointCloud[i].p.y = points3d[i](1);
        pointCloud[i].p.z = points3d[i](2);
//        cout << "pt " << i << " after: " << pointCloud[i].p << endl;
    }
}

} /* namespace sfmtoylib */
