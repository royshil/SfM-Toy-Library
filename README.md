# Toy Structure From Motion Library using OpenCV

This is a reference implementation of a Structure-from-Motion pipeline in OpenCV, following the work of Snavely et al. [2] and Hartley and Zisserman [1].

*Note:* This is not a complete and robust SfM pipeline implementation. The purpose of this code is to serve as a tutorial and reference for OpenCV users and a soft intro to SfM in OpenCV. If you are looking for a more complete solution with many more options and parameters to tweak, check out the following:

 * OpenMVG http://openmvg.readthedocs.io/en/latest/#
 * libMV https://github.com/libmv/libmv

SfM-Toy-Library is now using OpenCV 3, which introduced many new convenience functions to Structure from Motion, the implementation is much cleaner and simpler. 

Ceres solver was chosen to do bundle adjustment, for its simple API, straightforward modeling of the problem and long-term support.

Also added are doxygen-style documentation comments throughout.

## Compile

To compile use CMake: http://www.cmake.org

### Prerequisite
- OpenCV 3.x: http://www.opencv.org
- Ceres Solver (for bundle adjustment): http://ceres-solver.org/
- Boost C++ libraries (system, chrono, filesystem) v1.54+: http://www.boost.org/
- OPTIONAL: Qt 5.x (for the GUI) and libQGLViewer: http://www.libqglviewer.com/ for the 3D visualization of the point cloud

### How to make

#### On OSX Using XCode

Get Boost and Ceres using homebrew: `brew install boost ceres-solver`

	mkdir build
	cd build
	cmake -G "Xcode" ..
	open SfMToyExample.xcodeproj
	
#### On Linux (or OSX) via a Makefile

	mkdir build
	cd build
	cmake -G "Unix Makefiles" ..
	make 

#### On Windows

Use Cmake's GUI to create a MSVC solution, and build it.

## Use

See http://www.morethantechnical.com/2012/02/07/structure-from-motion-and-3d-reconstruction-on-the-easy-in-opencv-2-3-w-code/

### Execute

	USAGE: SfMToyUI.exe <path_to_images> [use rich features (RICH/OF) = RICH] [use GPU (GPU/CPU) = GPU] [down/upscale factor = 1.0]

### Datasets

Here's a place with some standard datasets for SfM: http://cvlab.epfl.ch/~strecha/multiview/denseMVS.html
Also, you can use the "Crazy Horse" (A national memorial site in South Dakota) dataset, that I pictured myself, included in the repo.

## References

1. Multiple View Geometry in Computer Vision, Hartley, R. I. and Zisserman, A., 2004, Cambridge University Press [http://www.robots.ox.ac.uk/~vgg/hzbook/]
2. Modeling the World from Internet Photo Collections, N. Snavely, S. M. Seitz, R. Szeliski, IJCV 2007 [http://phototour.cs.washington.edu/ModelingTheWorld_ijcv07.pdf]
3. Triangulation, R.I. Hartley, P. Strum, 1997, Computer vision and image understanding
4. Recovering baseline and orientation from essential matrix, B.K.P. Horn, 1990, J. Optical Society of America [http://people.csail.mit.edu/bkph/articles/Essential_Old.pdf]

## Troubleshooting

- If you get linker errors "mismatch detected for '_ITERATOR_DEBUG_LEVEL': value '0' doesn't match value '2' in Visualization.obj", you must make sure you are compiling vs. the right VTK static libs (Debug have "-gd" postfix, Release don't).
