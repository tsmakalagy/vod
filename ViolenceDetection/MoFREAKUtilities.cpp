#include "MoFREAKUtilities.h"

std::vector<std::string> MoFREAKUtilities::split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    return split(s, delim, elems);
}

unsigned int MoFREAKUtilities::countOnes(unsigned int byte)
{
	unsigned int num_ones = 0;

	for (unsigned int i = 0; i < 8; ++i)
	{
		if ((byte & (1 << i)) != 0)
		{
			++num_ones;
		}
	}
	return num_ones;
}

// Computes the motion interchange pattern between the current and previous frame.
// Assumes both matrices are 19 x 19, and we will check the 8 motion patch locations in the previous frame
// returns a binary descriptor representing the MIP responses for the patch at the SURF descriptor.
// x, y correspond to the location in the 19x19 roi that we are centering a patch around.

unsigned int MoFREAKUtilities::motionInterchangePattern(cv::Mat &current_frame, cv::Mat &prev_frame, int x, int y)
{
	const int THETA = 288;//10368;//5184;//2592;//1296; // 288 means an average intensity difference of at least 32 per pixel.
	// extract patch on current frame.
	cv::Rect roi(x - 1, y - 1, 3, 3);
	cv::Mat patch_t(current_frame, roi);

	// extract patches from previous frame.
	vector<cv::Mat> previous_patches;
	// (-4, 0)
	previous_patches.push_back(prev_frame(cv::Rect((x - 4) - 1, y - 1, 3, 3)));
	// (-3, 3)
	previous_patches.push_back(prev_frame(cv::Rect((x - 3) - 1, (y + 3) - 1, 3, 3)));
	// (0, 4)
	previous_patches.push_back(prev_frame(cv::Rect(x - 1, (y + 4) - 1, 3, 3)));
	// (3, 3)
	previous_patches.push_back(prev_frame(cv::Rect((x + 3) - 1, (y + 3) - 1, 3, 3)));
	// (4, 0)
	previous_patches.push_back(prev_frame(cv::Rect((x + 4) - 1, y - 1, 3, 3)));
	// (3, -3)
	previous_patches.push_back(prev_frame(cv::Rect((x + 3) - 1, (y - 3) - 1, 3, 3)));
	// (0, -4)
	previous_patches.push_back(prev_frame(cv::Rect(x - 1, (y - 4) - 1, 3, 3)));
	// (-3, -3)
	previous_patches.push_back(prev_frame(cv::Rect((x - 3) - 1, (y - 3) - 1, 3, 3)));

	// now do SSD between current patch and all of those patches.
	// opencv might have an optimized ssd, i didn't find it though.
	unsigned int bit = 1;
	unsigned int descriptor = 0;
	for (auto it = previous_patches.begin(); it != previous_patches.end(); ++it)
	{
		int ssd = 0;
        uchar *p = patch_t.data;
        uchar *p2 = it->data;
		for (int row = 0; row < 3; ++row)
		{
			for (int col = 0; col < 3; ++col)
			{
				ssd += (int)pow((float)((*p) - (*p2)), 2);
				p++;
				p2++;
			}
		}

		if (ssd > THETA) 
		{
			descriptor |= bit;
		}
		bit <<= 1;
	}

	return descriptor;
}

void MoFREAKUtilities::extractMotionByMotionInterchangePatterns(cv::Mat &current_frame, cv::Mat &prev_frame,
	vector<unsigned int> &motion_descriptor, 
	float scale, int x, int y)
{
	// get region of interest from frames at keypt + scale
	int tl_x = x - (int)scale/2;
	int tl_y = y - (int)scale/2;
	cv::Rect roi(tl_x, tl_y, ceil(scale), ceil(scale));
	cv::Mat current_roi = current_frame(roi);
	cv::Mat prev_roi = prev_frame(roi);

	// resize to 19x19
	cv::Mat frame_t(19, 19, CV_32F);
	cv::Mat frame_t_minus_1(19, 19, CV_32F);

	cv::resize(current_roi, frame_t, frame_t.size());
	cv::resize(prev_roi, frame_t_minus_1, frame_t_minus_1.size());

	// we will compute the descriptor around these 9 points...
	vector<cv::Point2d> patch_centers;
	patch_centers.push_back(cv::Point2d(5, 5));
	patch_centers.push_back(cv::Point2d(5, 9));
	patch_centers.push_back(cv::Point2d(5, 13));
	patch_centers.push_back(cv::Point2d(9, 5));
	//patch_centers.push_back(cv::Point2d(9, 9));
	patch_centers.push_back(cv::Point2d(9, 13));
	patch_centers.push_back(cv::Point2d(13, 5));
	patch_centers.push_back(cv::Point2d(13, 9));
	patch_centers.push_back(cv::Point2d(13, 13));

	// over each of these patch centers, we compute a 1-byte motion descriptor.
	for (auto it = patch_centers.begin(); it != patch_centers.end(); ++it)
	{
		//cout << "patch centered at " << it->x << ", " << it->y << endl;
		unsigned int descriptor = motionInterchangePattern(frame_t, frame_t_minus_1, it->x, it->y);
		motion_descriptor.push_back(descriptor);
	}
}

// to decide if there is sufficient motion, we compute the MIP on the center location (9, 9)
// If there are a sufficient number of ones, then we have sufficient motion.
bool MoFREAKUtilities::sufficientMotion(cv::Mat &current_frame, cv::Mat prev_frame, float x, float y, float scale)
{
	// get region of interest from frames at keypt + scale
	int tl_x = x - (int)scale/2;
	int tl_y = y - (int)scale/2;
	cv::Rect roi(tl_x, tl_y, ceil(scale), ceil(scale));
	cv::Mat current_roi = current_frame(roi);
	cv::Mat prev_roi = prev_frame(roi);

	// resize to 19x19
	cv::Mat frame_t(19, 19, CV_32F);
	cv::Mat frame_t_minus_1(19, 19, CV_32F);

	cv::resize(current_roi, frame_t, frame_t.size());
	cv::resize(prev_roi, frame_t_minus_1, frame_t_minus_1.size());

	unsigned int descriptor = motionInterchangePattern(frame_t, frame_t_minus_1, 9, 9);

	unsigned int num_ones = countOnes(descriptor);
	
	//return (num_ones > 0);//3);
	return true;
}

bool MoFREAKUtilities::sufficientMotion(cv::Mat &diff_integral_img, float &x, float &y, float &scale, int &motion)
{
	// compute the sum of the values within this patch in the difference image.  It's that simple.
	int radius = ceil((scale));///2);
	const int MOTION_THRESHOLD = 4 * radius * 5;

	// always + 1, since the integral image adds a row and col of 0s to the top-left.
	int tl_x = MAX(0, x - radius + 1);
	int tl_y = MAX(0, y - radius + 1);
	int br_x = MIN(diff_integral_img.cols, x + radius + 1);
	int br_y = MIN(diff_integral_img.rows, y + radius + 1);

	int br = diff_integral_img.at<int>(br_y, br_x);
	int tl = diff_integral_img.at<int>(tl_y, tl_x);
	int tr = diff_integral_img.at<int>(tl_y, br_x);
	int bl = diff_integral_img.at<int>(br_y, tl_x);
	motion = br + tl - tr - bl;

	return (motion > MOTION_THRESHOLD);
}

void MoFREAKUtilities::computeMoFREAKFromFile(std::string video_filename, std::string mofreak_filename, bool clear_features_after_computation)
{
	std::string debug_filename = video_filename;
	// ignore the first frames because we can't compute the frame difference with them.
	const int GAP_FOR_FRAME_DIFFERENCE = 5;

	cv::VideoCapture capture;
	capture.open(video_filename);

	if (!capture.isOpened())
	{
		cout << "Could not open file: " << video_filename << endl;
	}

	cv::Mat current_frame;
	cv::Mat prev_frame;

	std::queue<cv::Mat> frame_queue;
	for (unsigned int i = 0; i < GAP_FOR_FRAME_DIFFERENCE; ++i)
	{
		capture >> prev_frame; // ignore first 'GAP_FOR_FRAME_DIFFERENCE' frames.  Read them in and carry on.
		cv::cvtColor(prev_frame, prev_frame, CV_BGR2GRAY);
		frame_queue.push(prev_frame.clone());
	}
	prev_frame = frame_queue.front();
	frame_queue.pop();

	unsigned int frame_num = GAP_FOR_FRAME_DIFFERENCE - 1;
	
	while (true)
	{
		capture >> current_frame;
		if (current_frame.empty())	
		{
			break;
		}
		cv::cvtColor(current_frame ,current_frame, CV_BGR2GRAY);

		// compute the difference image for use in later computations.
		cv::Mat diff_img(current_frame.rows, current_frame.cols, CV_8U);
		cv::absdiff(current_frame, prev_frame, diff_img);

		vector<cv::KeyPoint> keypoints, diff_keypoints;
		cv::Mat descriptors;

		// detect all keypoints.		
		cv::Ptr<cv::BRISK> ptrBrisk = cv::BRISK::create(30);
		ptrBrisk->detect(diff_img, keypoints);		

		// extract the FREAK descriptors efficiently over the whole frame
		// For now, we are just computing the motion FREAK!  It seems to be giving better results.
		cv::Ptr<cv::xfeatures2d::FREAK> ptrFreak = cv::xfeatures2d::FREAK::create();
		ptrFreak->compute(diff_img, keypoints, descriptors);

		// for each detected keypoint
		vector<cv::KeyPoint> current_frame_keypts;
		unsigned char *pointer_to_descriptor_row = 0;
		unsigned int keypoint_row = 0;
		for (auto keypt = keypoints.begin(); keypt != keypoints.end(); ++keypt)
		{
			pointer_to_descriptor_row = descriptors.ptr<unsigned char>(keypoint_row);

			// only take points with sufficient motion.
			int motion = 0;
			
			if (sufficientMotion(current_frame, prev_frame, keypt->pt.x, keypt->pt.y, keypt->size))
			{
				//cout << "feature: motion bytes: " << NUMBER_OF_BYTES_FOR_MOTION << endl;
				//cout << "feature: app bytes: " << NUMBER_OF_BYTES_FOR_APPEARANCE << endl;
				MoFREAKFeature ftr(NUMBER_OF_BYTES_FOR_MOTION, NUMBER_OF_BYTES_FOR_APPEARANCE);
				ftr.frame_number = frame_num;
				ftr.scale = keypt->size;
				ftr.x = keypt->pt.x;
				ftr.y = keypt->pt.y;

				for (unsigned i = 0; i < NUMBER_OF_BYTES_FOR_APPEARANCE; ++i)
				{
					ftr.appearance[i] = pointer_to_descriptor_row[i];
				}

				// MIP
				vector<unsigned int> motion_desc;
				extractMotionByMotionInterchangePatterns(current_frame, prev_frame, motion_desc, keypt->size, keypt->pt.x, keypt->pt.y);

				for (unsigned i = 0; i < NUMBER_OF_BYTES_FOR_MOTION; ++i)
				{
					ftr.motion[i] = motion_desc[i];
				}

				// gather metadata.
				int action, person, video_number;
				readMetadata(video_filename, action, video_number, person);

				ftr.action = action;
				ftr.video_number = video_number;
				ftr.person = person;

				// these parameters aren't useful right now.
				ftr.motion_x = 0;
				ftr.motion_y = 0;

				features.push_back(ftr);
				current_frame_keypts.push_back(*keypt);
			}
			keypoint_row++;
		} // at this point, gathered all the mofreak pts from the frame.

		frame_queue.push(current_frame.clone());
		prev_frame = frame_queue.front();
		frame_queue.pop();
		frame_num++;
	}

	// in the end, print the mofreak file and reset the features for a new file.
	cout << "Writing this mofreak file: " << mofreak_filename << endl;
	writeMoFREAKFeaturesToFile(mofreak_filename);

	if (clear_features_after_computation)
		features.clear();

}
