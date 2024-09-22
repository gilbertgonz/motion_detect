#include <opencv2/opencv.hpp>

// Constants
const int THRESHOLD_VALUE = 25;
const int BBOX_OFFSET = 20;
const cv::Size BLUR_SIZE(3, 3);

cv::Rect detect_motion(cv::Mat& gray_prev_frame, cv::Mat& gray_frame) {
    cv::Mat diff_frame, thresh_frame;

    // Compute abs diff and apply threshold
    cv::absdiff(gray_prev_frame, gray_frame, diff_frame);
    cv::threshold(diff_frame, thresh_frame, THRESHOLD_VALUE, 255, cv::THRESH_BINARY);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh_frame, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Rect bbox;
    for (const auto& contour : contours) {
        cv::Rect contour_box = cv::boundingRect(contour);        
        
        // New bbox
        bbox = cv::Rect(
            contour_box.x - BBOX_OFFSET / 2,
            contour_box.y - BBOX_OFFSET / 2,
            contour_box.width + BBOX_OFFSET,
            contour_box.height + BBOX_OFFSET
        );       
    }
    return bbox;
}

int main() {
    std::string vid_file = "/assets/vid1.mp4";
    cv::VideoCapture cap(vid_file);

    cv::Mat gray_prev_frame, gray_frame;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // Convert current frame to grayscale and blur it
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray_frame, gray_frame, BLUR_SIZE, 0);

        if (!gray_prev_frame.empty()) {
            // Compute bbox
            cv::Rect bbox = detect_motion(gray_prev_frame, gray_frame);

            // Draw bbox
            cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);
            
            // Show
            cv::imshow("Moving Object", frame);
            cv::waitKey(5);
        }

        // Update prev_frame
        gray_prev_frame = gray_frame.clone();

    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
