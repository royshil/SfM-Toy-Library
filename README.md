# Toy Structure From Motion Library using OpenCV

## Compile

To compile use CMake: http://www.cmake.org

Prerequisite
- OpenCV: http://opencv.willowgarage.com/wiki/
- SSBA & Sparsesuite: http://www.inf.ethz.ch/personal/chzach/opensource.html (bundeled with the library in the '3rdparty' directory) (Now optional when using the USE_SSBA=OFF option)
- FLTK 3.x (Optional for GUI): http://www.fltk.org/

### How to make

- Either build SSBA-3.0 first, by compiling it from '3rdparty' directory, or use "-DUSE_SSBA=OFF" with cmake.

On MacOS

	mkdir build
	cd build
	cmake -DSSBA_LIBRARY_DIR=../../SSBA-3.0/build -G "Xcode" ..
	open SfMToyExample.xcodeproj

On Linux

	mkdir build
	cd build
	cmake -SSBA_LIBRARY_DIR=../../SSBA-3.0/build -G "Unix Makefiles" ..
	make 

On Windows

Use Cmake's GUI to create a MSVC solution, and build it.


## Use

See http://www.morethantechnical.com/2012/02/07/structure-from-motion-and-3d-reconstruction-on-the-easy-in-opencv-2-3-w-code/

Execute

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
