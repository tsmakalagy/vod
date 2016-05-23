// ViolenceDetection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <string>
#include <iostream>
#include <fstream>
#include <time.h>
#include <vector>
#include <exception>

#include <boost\filesystem.hpp>
#include <boost\thread.hpp>

#include <opencv2\opencv.hpp>

#include "MoFREAKUtilities.h"

#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

using namespace std;
using namespace boost::filesystem;

bool DISTRIBUTED = false;

string MOSIFT_DIR, MOFREAK_PATH, VIDEO_PATH, SVM_PATH, METADATA_PATH; // for file structure

unsigned int NUM_MOTION_BYTES = 8;
unsigned int NUM_APPEARANCE_BYTES = 8;
unsigned int FEATURE_DIMENSIONALITY = NUM_MOTION_BYTES + NUM_APPEARANCE_BYTES;
unsigned int NUM_CLUSTERS, NUMBER_OF_GROUPS, NUM_CLASSES, ALPHA;

vector<int> possible_classes;
std::deque<MoFREAKFeature> mofreak_ftrs;

enum states {
	DETECT_MOFREAK, DETECTION_TO_CLASSIFICATION, // standard recognition states
	PICK_CLUSTERS, COMPUTE_BOW_HISTOGRAMS, DETECT, TRAIN, GET_SVM_RESPONSES,
}; // these states are exclusive to TRECVID

enum datasets { KTH, TRECVID, HOLLYWOOD, UTI1, UTI2, HMDB51, UCF101 };

int dataset = KTH;//UCF101; //KTH;//HMDB51;
int state = DETECT_MOFREAK;//DETECTION_TO_CLASSIFICATION;

MoFREAKUtilities *mofreak;

void setParameters()
{
	// KTH
	if (dataset == KTH)
	{
		NUM_CLUSTERS = 600;
		NUM_CLASSES = 6;
		NUMBER_OF_GROUPS = 25;

		for (unsigned i = 0; i < NUM_CLASSES; ++i)
		{
			possible_classes.push_back(i);
		}

		// structural folder info.
		MOSIFT_DIR = "C:/data/kth/mosift/";
		MOFREAK_PATH = "C:/data/kth/mofreak/";
		VIDEO_PATH = "C:/data/kth/videos/";
		SVM_PATH = "C:/data/kth/svm/";
		METADATA_PATH = "";
	}
}

// given a collection of videos, generate a single mofreak file per video,
// containing the descriptor data for that video.
void computeMoFREAKFiles()
{
	directory_iterator end_iter;

	cout << "Here are the videos: " << VIDEO_PATH << endl;
	cout << "MoFREAK files will go here: " << MOFREAK_PATH << endl;
	cout << "Motion bytes: " << NUM_MOTION_BYTES << endl;
	cout << "Appearance bytes: " << NUM_APPEARANCE_BYTES << endl;
	for (directory_iterator dir_iter(VIDEO_PATH); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
			// parse mosift files so first x characters gets us the video name.
			path current_file = dir_iter->path();
			string video_path = current_file.generic_string();
			string video_filename = current_file.filename().generic_string();

			if ((video_filename.substr(video_filename.length() - 3, 3) == "avi"))
			{
				cout << "AVI: " << VIDEO_PATH << "/" << video_filename << endl;

				string video = VIDEO_PATH + "/" + video_filename;
				string mofreak_path = MOFREAK_PATH + "/" + video_filename + ".mofreak";

				mofreak->computeMoFREAKFromFile(video, mofreak_path, true);
			}
		}
		else if (is_directory(dir_iter->status()))
		{
			// get folder name.
			string video_action = dir_iter->path().filename().generic_string();
			cout << "action: " << video_action << endl;

			// set the mofreak object's action to that folder name.
			mofreak->setCurrentAction(video_action);

			// compute mofreak on all files on that folder.
			string action_video_path = VIDEO_PATH + "/" + video_action;
			cout << "action video path: " << action_video_path << endl;

			for (directory_iterator video_iter(action_video_path);
				video_iter != end_iter; ++video_iter)
			{
				if (is_regular_file(video_iter->status()))
				{
					string video_filename = video_iter->path().filename().generic_string();
					if (video_filename.substr(video_filename.length() - 3, 3) == "avi")
					{
						cout << "filename: " << video_filename << endl;
						cout << "AVI: " << action_video_path << video_filename << endl;

						string mofreak_path = MOFREAK_PATH + "/" + video_action + "/" + video_filename + ".mofreak";

						// create the corresponding directories, then go ahead and compute the mofreak files.
						boost::filesystem::path dir_to_create(MOFREAK_PATH + "/" + video_action + "/");
						boost::system::error_code returned_error;
						boost::filesystem::create_directories(dir_to_create, returned_error);
						if (returned_error)
						{
							std::cout << "Could not make directory " << dir_to_create.string() << std::endl;
							exit(1);
						}

						cout << "mofreak path: " << mofreak_path << endl;
						mofreak->computeMoFREAKFromFile(action_video_path + "/" + video_filename, mofreak_path, true);
					}
				}
			}
		}
	}
}

int _tmain(int argc, _TCHAR* argv[])
{
	setParameters();
	clock_t start, end;
	mofreak = new MoFREAKUtilities(dataset);

	if (state == DETECT_MOFREAK)
	{
		start = clock();
		computeMoFREAKFiles();
		end = clock();
	}

	cout << "Took this long: " << (end - start) / (double)CLOCKS_PER_SEC << " seconds! " << endl;
	cout << "All done.  Press any key to continue..." << endl;
	cout << "Dumping memory leak info" << endl;
	system("PAUSE");
	_CrtDumpMemoryLeaks();
	system("PAUSE");

	return 0;
}

