# Toy Structure From Motion Library using OpenCV

## Compile

To compile use CMake: http://www.cmake.org
Prerequisite
- OpenCV: http://opencv.willowgarage.com/wiki/
- SSBA & Sparsesuite: http://www.inf.ethz.ch/personal/chzach/opensource.html
- FLTK 3.x (Optional for GUI): http://www.fltk.org/

### How to make

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


## Use

See http://www.morethantechnical.com/2012/02/07/structure-from-motion-and-3d-reconstruction-on-the-easy-in-opencv-2-3-w-code/
