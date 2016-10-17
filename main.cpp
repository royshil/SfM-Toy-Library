/*
 *  main.cpp
 *  SfMToyUI
 *
 *  Created by Roy Shilkrot on 4/27/12.
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2016 Roy Shilkrot
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
 *
 */

#include "SfMToyLib/SfM.h"

#include <iostream>

#include <boost/program_options.hpp>

using namespace sfmtoylib;
using namespace std;
namespace po = boost::program_options;

int main(int argc, char** argv) {
    //add command line options
    po::options_description od;
    od.add_options()
            ("help,h",                                                          "Produce help message.")
            ("console-debug,d",   po::value<int>()->default_value(LOG_INFO),    "Debug output to console log level (0 = Trace, 4 = Error).")
            ("visual-debug,v",    po::value<int>()->default_value(LOG_WARN),    "Visual debug output to screen log level (0 = All, 4 = None).")
            ("downscale,s",       po::value<double>()->default_value(1.0),      "Downscale factor for input images.")
            ("input-directory,p", po::value<string>()->required(),              "Directory to find input images.")
            ("output-prefix,o",   po::value<string>()->default_value("output"), "Prefix for output files.")
            ;

    po::positional_options_description op;
    op.add("input-directory", 1);

    //parse options
    po::variables_map varMap;
    try {
        po::store(po::command_line_parser(argc, argv).positional(op).options(od).run(), varMap);
        po::notify(varMap);
    } catch (const std::exception& e) {
        cerr << "Error while parsing command line options: " << e.what() << endl
             << "USAGE " << argv[0] << " [options] <" << op.name_for_position(0) << ">" << endl << od;
        exit(0);
    }
    if (varMap.count("help")) {
        cerr << argv[0] << " [options] <" << op.name_for_position(0) << ">" << endl << od << endl;
        exit(0);
    }


	SfM sfm(varMap["downscale"].as<double>());
	sfm.setImagesDirectory(varMap["input-directory"].as<string>());
	sfm.setConsoleDebugLevel(varMap["console-debug"].as<int>());
	sfm.setVisualDebugLevel(varMap["visual-debug"].as<int>());
	sfm.runSfM();

	//save point cloud and cameras to file
	sfm.saveCloudAndCamerasToPLY(varMap["output-prefix"].as<string>());
}
