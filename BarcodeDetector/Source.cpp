#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace cv;

enum GrayscaleMethods {
	GRAY_AVG = 1,
	GRAY_WEIGHT = 2
};

enum ThresholdMethods {
	THRESH_MOD = 1
};

// Matrices used
Mat src, gray, sobel, gradient, blurred, thresh, closed, dilated, eroded;

int threshold_value = 0;

// read the image
Mat ReadImage(string imageName) {
	Mat img = imread(imageName, IMREAD_COLOR);
	if (img.empty()) {
		cout << "Cannot read image: " << imageName << std::endl;
		system("EXIT");
	}
	return img;
}

// Grayscale function
Mat grayscale(Mat src, int method) {
	Mat output(src.rows, src.cols, CV_8UC1);
	// basic grayscale
	if (method == GRAY_AVG) {
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				//cout << "(" << to_string(i) << ", " << to_string(j) << ")" << to_string(img.at<Vec3b>(i, j)[0]) << " " << to_string(img.at<Vec3b>(i, j)[1]) << " " << to_string(img.at<Vec3b>(i, j)[2]) << endl;
				int origBlue = src.at<Vec3b>(i, j)[0]; //blue
				int origGreen = src.at<Vec3b>(i, j)[1]; //green
				int origRed = src.at<Vec3b>(i, j)[2]; //red
				int grayscale = (origBlue + origGreen + origRed) / 3;
				output.at<uchar>(i, j) = grayscale;
			}
		}
	}
	// clearer grayscale
	else if (method == GRAY_WEIGHT) {
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				//cout << "(" << to_string(i) << ", " << to_string(j) << ")" << to_string(img.at<Vec3b>(i, j)[0]) << " " << to_string(img.at<Vec3b>(i, j)[1]) << " " << to_string(img.at<Vec3b>(i, j)[2]) << endl;
				int origBlue = src.at<Vec3b>(i, j)[0]; //blue
				int origGreen = src.at<Vec3b>(i, j)[1]; //green
				int origRed = src.at<Vec3b>(i, j)[2]; //red
				int grayscale = (0.3 * origRed) + (0.59 * origGreen) + (0.11 * origBlue);
				output.at<uchar>(i, j) = grayscale;
			}
		}
	}

	return output;
}

// Threshold function
Mat thresholding(Mat sobel, int method) {
	Mat output = Mat(sobel.rows, sobel.cols, CV_8U);
	if (method == THRESH_MOD) {
		if (sobel.channels() != 1) {
			cout << "The input image must be single-channeled!" << endl;
			system("EXIT");
		}
		double minVal = 0, maxVal = 0;
		for (int i = 0; i < sobel.rows; i++) {
			for (int j = 0; j < sobel.cols; j++) {
				//Mat kernel = Mat::zeros(9, 9, output.type());
				// Implement logic to fill the 9x9 kernel with
				// values from the gray Mat, respecting boundaries.

				//Scalar avg_intensity = mean(kernel);
				//minMaxLoc(kernel, &minVal, &maxVal);

				if (sobel.at<uchar>(i, j) <= 255 && sobel.at<uchar>(i, j) > threshold_value) {
					output.at<uchar>(i, j) = 255;
				}
				else {
					output.at<uchar>(i, j) = 0;
				}
			}
		}
	}
	return output;
}

void thresholding_call(int, void*) {
	threshold_value = (int)threshold_value;
	thresh = Mat(blurred.rows, blurred.cols, CV_8U);
	thresh = thresholding(blurred, THRESH_MOD);
	imshow("Thresholded Image", thresh);
}

// Sobel Edge Detection function
Mat SobelDetect(Mat gray) {
	int dx[3][3] = { {1, 0, -1},{2, 0, -2},{1, 0, -1 } };
	int dy[3][3] = { {1, 2, 1},{0, 0, 0},{-1, -2, -1} };

	Mat output = Mat(gray.rows, gray.cols, CV_8U);
	Mat kernel = Mat(3, 3, CV_8U);

	int max = -200, min = 2000;

	for (int i = 1; i < gray.rows - 2; i++) {
		for (int j = 1; j < gray.cols - 2; j++) {
			// apply kernel in X and Y directions
			int sumX = 0;
			int sumY = 0;
			uchar ker;
			for (int m = -1; m <= 1; m++) {
				for (int n = -1; n <= 1; n++) {
					// get the (i,j) pixel value
					kernel.at<uchar>(m + 1, n + 1) = gray.at<uchar>(i + m, j + n);
					sumX += kernel.at<uchar>(m + 1, n + 1) * dx[m + 1][n + 1];
					sumY += kernel.at<uchar>(m + 1, n + 1) * dy[m + 1][n + 1];
				}
			}
			int sum = abs(sumX) + abs(sumY);
			//cout << sum << endl;
			output.at<uchar>(i, j) = (sum > 255) ? 255 : sum;
			//output2.at<uchar>(i, j) = kernel.at<uchar>(i, j);
		}
	}
	return output;
}

// Blur function 9x9 kernel
Mat blurImage(Mat gradient) {
	Mat output(gradient.rows, gradient.cols, CV_8U);
	int total = 0;
	//blur
	for (int i = 0; i < gradient.rows; i++) {
		for (int j = 0; j < gradient.cols; j++) {
			int ksize = 9;
			total = 0;
			for (int x = -ksize / 2; x <= ksize / 2; x++) {
				for (int y = -ksize / 2; y <= ksize / 2; y++) {
					int tx = i + x;
					int ty = j + y;
					if (tx > 0 && tx < gradient.rows && ty >= 0 && ty < gradient.cols) {
						total += gradient.at<uchar>(tx, ty);
					}
				}
			}
			output.at<uchar>(i, j) = total / ksize / ksize;
		}
	}
	return output;
}

// Erode function
Mat erodeImage(Mat closed) {
	Mat output(closed.rows, closed.cols, CV_8U);
	// dilate
	for (int i = 1; i < closed.rows; i++) {
		for (int j = 1; j < closed.cols; j++) {
			if (closed.at<uchar>(i, j) == 0) {
				if (i > 0 && closed.at<uchar>(i - 1, j) == 255) {
					closed.at<uchar>(i - 1, j) = 1;
				}
				if (j > 0 && closed.at<uchar>(i, j - 1) == 255) {
					closed.at<uchar>(i, j - 1) = 1;
				}
				if (i + 1 < closed.rows && closed.at<uchar>(i + 1, j) == 255) {
					closed.at<uchar>(i + 1, j) = 1;
				}
				if (j + 1 < closed.cols && closed.at<uchar>(i, j + 1) == 255) {
					closed.at<uchar>(i, j + 1) = 1;
				}
				if (i > 0 && j > 0 && closed.at<uchar>(i - 1, j - 1) == 255) {
					closed.at<uchar>(i - 1, j - 1) = 1;
				}
				if (i > 0 && j + 1 < closed.cols && closed.at<uchar>(i - 1, j + 1) == 255) {
					closed.at<uchar>(i - 1, j + 1) = 1;
				}
				if (i + 1 < closed.rows && j > 0 && closed.at<uchar>(i + 1, j - 1) == 255) {
					closed.at<uchar>(i + 1, j - 1) = 1;
				}
				if (i + 1 < closed.rows && j + 1 < closed.cols && closed.at<uchar>(i + 1, j + 1) == 255) {
					closed.at<uchar>(i + 1, j + 1) = 1;
				}
			}
		}
	}
	for (int i = 0; i < closed.rows; i++) {
		for (int j = 0; j < closed.cols; j++) {
			if (closed.at<uchar>(i, j) == 1) {
				closed.at<uchar>(i, j) = 0;
			}
			output.at<uchar>(i, j) = closed.at<uchar>(i, j);
		}
	}
	return output;
}

// Dilate function
Mat dilateImage(Mat eroded) {
	Mat output(eroded.rows, eroded.cols, CV_8U);
	// dilate
	for (int i = 1; i < eroded.rows; i++) {
		for (int j = 1; j < eroded.cols; j++) {
			if (eroded.at<uchar>(i, j) == 255) {
				if (i > 0 && eroded.at<uchar>(i - 1, j) == 0) {
					eroded.at<uchar>(i - 1, j) = 254;
				}
				if (j > 0 && eroded.at<uchar>(i, j - 1) == 0) {
					eroded.at<uchar>(i, j - 1) = 254;
				}
				if (i + 1 < eroded.rows && eroded.at<uchar>(i + 1, j) == 0) {
					eroded.at<uchar>(i + 1, j) = 254;
				}
				if (j + 1 < eroded.cols && eroded.at<uchar>(i, j + 1) == 0) {
					eroded.at<uchar>(i, j + 1) = 254;
				}
				if (i > 0 && j > 0 && eroded.at<uchar>(i - 1, j - 1) == 0) {
					eroded.at<uchar>(i - 1, j - 1) = 254;
				}
				if (i > 0 && j + 1 < eroded.cols && eroded.at<uchar>(i - 1, j + 1) == 0) {
					eroded.at<uchar>(i - 1, j + 1) = 254;
				}
				if (i + 1 < eroded.rows && j > 0 && eroded.at<uchar>(i + 1, j - 1) == 0) {
					eroded.at<uchar>(i + 1, j - 1) = 254;
				}
				if (i + 1 < eroded.rows && j + 1 < eroded.cols && eroded.at<uchar>(i + 1, j + 1) == 0) {
					eroded.at<uchar>(i + 1, j + 1) = 254;
				}
			}
		}
	}
	for (int i = 0; i < eroded.rows; i++) {
		for (int j = 0; j < eroded.cols; j++) {
			if (eroded.at<uchar>(i, j) == 254) {
				eroded.at<uchar>(i, j) = 255;
			}
			output.at<uchar>(i, j) = eroded.at<uchar>(i, j);
		}
	}
	return output;
}

// rectangular morph
Mat closeContours(Mat thresh, int pixel_spacing) {
	Mat output(thresh.rows, thresh.cols, CV_8U);
	for (int i = 0; i < thresh.rows; i++) {
		for (int j = 0; j < thresh.cols; j++) {
			if (thresh.at<uchar>(i, j) == 255) {
				for (int a = 0; a < pixel_spacing + 1; a++) {
					if (thresh.at<uchar>(i, j + a) == 255) {
						for (int b = 0; b < a; b++) {
							thresh.at<uchar>(i, j + b) = 255;
						}
					}
				}

			}
		}
	}
	for (int i = 0; i < thresh.rows; i++) {
		for (int j = 0; j < thresh.cols; j++) {
			output.at<uchar>(i, j) = thresh.at<uchar>(i, j);
		}
	}
	return output;
}

// find all contours
queue<int> find_contours(Mat dilated) {
	int max_x = 0, max_y = 0, min_x = dilated.cols, min_y = dilated.rows;
	//Mat output = Mat(dilated.rows, dilated.cols, CV_8UC3);
	queue<int> output;
	vector<Point> all_contours;

	// find all white px and store in vector
	for (int i = 0; i < dilated.rows; i++) {
		for (int j = 0; j < dilated.cols; j++) {
			if (dilated.at<uchar>(i, j) == 255) {
				all_contours.push_back(Point(j, i));
			}
		}
	}
	// search for max and min x,y
	for (int cont = 0; cont < all_contours.size(); cont++) {
		if (all_contours.at(cont).x > max_x) {
			max_x = all_contours.at(cont).x;
		}
		if (all_contours.at(cont).y > max_y) {
			max_y = all_contours.at(cont).y;
		}
		if (all_contours.at(cont).x < min_x) {
			min_x = all_contours.at(cont).x;
		}
		if (all_contours.at(cont).y < min_y) {
			min_y = all_contours.at(cont).y;
		}
	}

	// detect in min and max x,y for lines more than 80% white
	//top to bottom
	for (int i = min_y; i <= max_y; i++) {
		float count = 0.0;
		for (int j = min_x; j < max_x; j++) {
			//if white
			if (dilated.at<uchar>(j, i) == 255) {
				count++;
			}
		}
		float percentage = (count / (max_x - min_x)) * 100;
		if (percentage < 80.0) {
			break;
		}
		else {
			min_y++;
		}
	}

	//bottom to top
	for (int i = max_y; i >= min_y; i--) {
		float count = 0.0;
		for (int j = min_x; j < max_x; j++) {
			//if white
			if (dilated.at<uchar>(j, i) == 255) {
				count++;
			}
		}
		float percentage = (count / (max_x - min_x)) * 100;
		if (percentage < 80.0) {
			break;
		}
		else {
			max_y--;
		}
	}
	output.push(max_x);
	output.push(min_x);
	output.push(max_y);
	output.push(min_y);
	return output;
}

// compare contours
bool compareContourAreas(vector<Point>contour1, vector<Point>contour2) {
	double i = fabs(contourArea(Mat(contour1)));
	double j = fabs(contourArea(Mat(contour2)));
	return (i < j);
}

// point vector to mat function
void vector_Point_to_Mat(std::vector<Point>& v_point, Mat& mat)
{
	mat = Mat(v_point, true);
}


// main function
int main() {
	String imageName;
	cout << "Enter the image file name (ex. barcode.jpg): " << endl;
	cin >> imageName;
	// read the image
	src = ReadImage(imageName);
	// show original image
	namedWindow("Original Image", WINDOW_NORMAL);
	imshow("Original Image", src);
	waitKey(0);
	destroyWindow("Original Image");
	// grayscale the image
	Mat gray = grayscale(src.clone(), GRAY_WEIGHT);
	namedWindow("Grayscale", WINDOW_NORMAL);
	imshow("Grayscale", gray);
	waitKey(0);
	destroyWindow("Grayscale");
	// Sobel Edge Detection
	sobel = Mat(src.rows, src.cols, CV_32F);
	sobel = SobelDetect(gray);
	namedWindow("Sobel Edge Detection", WINDOW_NORMAL);
	imshow("Sobel Edge Detection", sobel);
	waitKey(0);
	destroyWindow("Sobel Edge Detection");
	// Gradient
	convertScaleAbs(sobel, gradient);
	namedWindow("Gradient", WINDOW_NORMAL);
	imshow("Gradient", gradient);
	waitKey(0);
	destroyWindow("Gradient");
	// blur the image
	blurred = blurImage(gradient);
	namedWindow("Blurred Image", WINDOW_NORMAL);
	imshow("Blurred Image", blurred);
	waitKey(0);
	destroyWindow("Blurred Image");
	namedWindow("Thresholded Image", WINDOW_NORMAL);
	// create trackbar
	const char* trackbar_value = "Value";
	createTrackbar(trackbar_value, "Thresholded Image", &threshold_value, 254, thresholding_call);
	thresholding_call(0, 0);
	thresh = Mat(blurred.rows, blurred.cols, CV_8U);
	// threshold the image
	thresh = thresholding(blurred, THRESH_MOD);
	//imshow("Thresholded Image", thresh);
	waitKey(0);
	destroyWindow("Thresholded Image");
	// constructing a closing kernel
	//Mat kernel = getStructuringElement(MORPH_RECT, Size(21, 7));
	// applying the kernel to the thresholded image
	closed = closeContours(thresh, 100);
	//morphologyEx(thresh, closed, MORPH_CLOSE, kernel);
	namedWindow("Closed Contoured Image", WINDOW_NORMAL);
	imshow("Closed Contoured Image", closed);
	waitKey(0);
	destroyWindow("Closed Contoured Image");

	// perform erosions
	eroded = Mat(closed.rows, closed.cols, CV_8U);
	eroded = erodeImage(closed);
	for (int i = 0; i < 3; i++) {
		eroded = erodeImage(eroded);
	}
	namedWindow("Eroded Image", WINDOW_NORMAL);
	imshow("Eroded Image", eroded);
	waitKey(0);
	destroyWindow("Eroded Image");

	// perform dilations
	dilated = Mat(eroded.rows, eroded.cols, CV_8U);
	dilated = dilateImage(eroded);
	for (int i = 0; i < 3; i++) {
		dilated = dilateImage(dilated);
	}
	namedWindow("Dilated Image", WINDOW_NORMAL);
	imshow("Dilated Image", dilated);
	waitKey(0);
	destroyWindow("Dilated Image");

	// make the output the same size as the image by equating them
	queue<int>minmax = find_contours(dilated);
	int max_x, min_x, max_y, min_y;
	max_x = minmax.front();
	minmax.pop();
	min_x = minmax.front();
	minmax.pop();
	max_y = minmax.front();
	minmax.pop();
	min_y = minmax.front();
	minmax.pop();

	Rect box = Rect(min_x, min_y, max_x - min_x, max_y - min_y);
	rectangle(src, box, Scalar(255, 0, 0), 1, 8, 0);

	namedWindow("Scanned Image", WINDOW_NORMAL);
	imshow("Scanned Image", src);
	waitKey(0);
	return 0;
}