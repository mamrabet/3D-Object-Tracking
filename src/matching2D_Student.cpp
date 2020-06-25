#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
//https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
	double t = (double)cv::getTickCount();
    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
    t = (double)cv::getTickCount();
	matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    matcher->match(descSource, descRef, matches);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << " FLANN match in " << 1000 * t / 1.0 << " ms" << endl;
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
	t = (double)cv::getTickCount();
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << " NN match in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
    t = (double)cv::getTickCount();
      std::vector< std::vector<cv::DMatch> > knn_matches;
      matcher->knnMatch( descSource, descRef, knn_matches, 2 );
      //-- Filter matches using the Lowe's ratio test
      const float ratio_thresh = 0.8f;
     
      for (size_t i = 0; i < knn_matches.size(); i++)
      {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
           matches.push_back(knn_matches[i][0]);
        }
      }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << " KNN match in " << 1000 * t / 1.0 << " ms" << endl;
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
// https://docs.opencv.org/3.4.9/d3/df6/namespacecv_1_1xfeatures2d.html
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
  else if (descriptorType == "BRIEF")
  {
    int bytes = 32;                
    bool use_orientation = false;  

    extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, use_orientation);
  }
  else if (descriptorType == "ORB")
  {
    int   nfeatures = 500;      
    float scaleFactor = 1.2f;  
    int   nlevels = 8;          
    int   edgeThreshold = 31;   
    int   firstLevel = 0;       
    int   WTA_K = 2;            
    auto  scoreType = cv::ORB::HARRIS_SCORE;  
    int   patchSize = 31;       
    int   fastThreshold = 20;   

    extractor = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold,
                                firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
  }
  else if (descriptorType == "FREAK")
  {
    bool orientationNormalized = true;  
    bool scaleNormalized = true;        
    float patternScale = 22.0f;        
    int nOctaves = 4;                  
    const std::vector<int>& selectedPairs = std::vector<int>();

    extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale,
                                               nOctaves, selectedPairs);
  }
  else if (descriptorType == "AKAZE")
  {
 
    auto  descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
    int   descriptor_size = 0;        
    int   descriptor_channels = 3;     
    float threshold = 0.001f;          
    int   nOctaves = 4;                
    int   nOctaveLayers = 4;          
    auto  diffusivity = cv::KAZE::DIFF_PM_G2; 
 
    extractor = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels,
                                  threshold, nOctaves, nOctaveLayers, diffusivity);
  }
  else if (descriptorType == "SIFT")
  {
    int nfeatures = 0;  
    int nOctaveLayers = 3; 
    
    double contrastThreshold = 0.04;
    double edgeThreshold = 10;  
    double sigma = 1.6;  

    extractor = cv::SiftFeatureDetector::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
  }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
 
    // Apply corner detection
    double t = (double)cv::getTickCount();
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // visualize results
    string windowName = "Harris Corner Detector Response Matrix";
    cv::namedWindow(windowName, 4);
    cv::imshow(windowName, dst_norm_scaled);
    cv::waitKey(0);

    // STUDENTS NEET TO ENTER THIS CODE (C3.2 Atom 4)

    // Look for prominent corners and instantiate keypoints
    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            { // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                      // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }     // eof loop over rows

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis){
  double t = (double)cv::getTickCount();
  cv::Ptr<cv::FeatureDetector> detector;
  
  if (detectorType == "FAST")
  {  
        

          int threshold = 30; // difference between intensity of the central pixel and pixels of a circle around this pixel
          bool bNMS = true;   // perform non-maxima suppression on keypoints
          cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8
          detector = cv::FastFeatureDetector::create(threshold, bNMS, type);



  }
  else if (detectorType == "BRISK"){
    int threshold = 30;       
    int octaves = 3;           
    float patternScale = 1.0f; 
    detector = cv::BRISK::create(threshold, octaves, patternScale);
    
  }
  else if (detectorType == "ORB")
  {
    int   nfeatures = 500;     
    float scaleFactor = 1.2f;  
    int   nlevels = 8;         
    int   edgeThreshold = 31;  
    int   firstLevel = 0;      
    int   WTA_K = 2;           
    auto  scoreType = cv::ORB::HARRIS_SCORE; 
    int   patchSize = 31;      
    int   fastThreshold = 20;  
    detector = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold,
                               firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
  }
  else if (detectorType == "AKAZE")
  {

    auto  descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
    int   descriptor_size = 0;        
    int   descriptor_channels = 3;   
    float threshold = 0.001f;        
    int   nOctaves = 4;             
    int   nOctaveLayers = 4;          
    auto  diffusivity = cv::KAZE::DIFF_PM_G2; 
 
    detector = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels,
                                 threshold, nOctaves, nOctaveLayers, diffusivity);
  }
  else if (detectorType == "SIFT")
  {
    int nfeatures = 0; 
    int nOctaveLayers = 3; 
   
    double contrastThreshold = 0.04;
    double edgeThreshold = 10; 
    double sigma = 1.6; 

    detector = cv::SiftFeatureDetector::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
  }
  
  
          
          t = (double)cv::getTickCount();
          detector->detect(img, keypoints);
          t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
          //cout << detectorType <<" with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
  
}
