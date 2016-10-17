# Toy Structure From Motion Library using OpenCV

This is a reference implementation of a Structure-from-Motion pipeline in OpenCV, following the work of Snavely et al. [2] and Hartley and Zisserman [1].

*Note:* This is not a complete and robust SfM pipeline implementation. The purpose of this code is to serve as a tutorial and reference for OpenCV users and a soft intro to SfM in OpenCV. If you are looking for a more complete solution with many more options and parameters to tweak, check out the following:

 * OpenMVG http://openmvg.readthedocs.io/en/latest/#
 * libMV https://github.com/libmv/libmv

SfM-Toy-Library is now using OpenCV 3, which introduced many new convenience functions to Structure from Motion, the implementation is much cleaner and simpler. 

Ceres solver was chosen to do bundle adjustment, for its simple API, straightforward modeling of the problem and long-term support.

Doxygen-style documentation comments appear throughout.

## Compile

To compile use CMake: http://www.cmake.org

### Prerequisite
- OpenCV 3.x: http://www.opencv.org
- Ceres Solver (for bundle adjustment): http://ceres-solver.org/
- Boost C++ libraries v1.54+: http://www.boost.org/

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

    USAGE ./build/SfMToyUI [options] <input-directory>
      -h [ --help ]                   produce help message
      -d [ --console-debug ] arg (=2) Debug output to console log level (0 = Trace,
                                      4 = Error).
      -v [ --visual-debug ] arg (=3)  Visual debug output to screen log level (0 = 
                                      All, 4 = None).
      -s [ --downscale ] arg (=1)     Downscale factor for input images
      -p [ --input-directory ] arg    Directory to find input images

### Datasets

Here's a place with some standard datasets for SfM: http://cvlab.epfl.ch/~strecha/multiview/denseMVS.html

Also, you can use the "Crazy Horse" (A national memorial site in South Dakota) dataset, that I pictured myself, included in the repo.

## References

1. Multiple View Geometry in Computer Vision, Hartley, R. I. and Zisserman, A., 2004, Cambridge University Press [http://www.robots.ox.ac.uk/~vgg/hzbook/]
2. Modeling the World from Internet Photo Collections, N. Snavely, S. M. Seitz, R. Szeliski, IJCV 2007 [http://phototour.cs.washington.edu/ModelingTheWorld_ijcv07.pdf]
3. Triangulation, R.I. Hartley, P. Strum, 1997, Computer vision and image understanding
4. Recovering baseline and orientation from essential matrix, B.K.P. Horn, 1990, J. Optical Society of America [http://people.csail.mit.edu/bkph/articles/Essential_Old.pdf]

