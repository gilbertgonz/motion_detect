#include <opencv2/opencv.hpp>

// Constants
const int MASK_THRESHOLD = 25;
const float PMS_THRESHOLD = 0.35;
const cv::Size BLUR_SIZE(3, 3);
const cv::Size DILATION_SIZE(7, 7);
const bool SAVE_IMAGES = true;

struct Box {
    int x, y, w, h;
    int area;
};

std::vector<Box> detect_motion_clusters(cv::Mat& gray_prev_frame, cv::Mat& gray_frame) {
    cv::Mat diff_frame, thresh_frame, dilated_frame;

    // Compute abs diff and apply threshold
    cv::absdiff(gray_prev_frame, gray_frame, diff_frame);
    cv::threshold(diff_frame, thresh_frame, MASK_THRESHOLD, 255, cv::THRESH_BINARY);

    // Dilate thresholded image
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, DILATION_SIZE);
    cv::dilate(thresh_frame, dilated_frame, element);

    // Find connected components (clusters)
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(dilated_frame, labels, stats, centroids, 8, CV_32S);

    std::vector<Box> bounding_boxes;
    for (int i = 1; i < num_labels; i++) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        Box box{
            stats.at<int>(i, cv::CC_STAT_LEFT),
            stats.at<int>(i, cv::CC_STAT_TOP),
            stats.at<int>(i, cv::CC_STAT_WIDTH),
            stats.at<int>(i, cv::CC_STAT_HEIGHT),
            area
        };
        bounding_boxes.push_back(box);
    }
    return bounding_boxes;
}

// reference: https://medium.com/@itberrios6/introduction-to-motion-detection-part-1-e031b0bb9bb2
void non_max_suppression(std::vector<Box>& boxes, float thresh) {
    // Remove bboxes that have high Intersection Over Union (IoU)
    for (size_t i = 0; i < boxes.size(); i++) {
        for (size_t j = i + 1; j < boxes.size(); ) {
            float intersection_width = std::max(0, std::min(boxes[i].w + boxes[i].x, boxes[j].w + boxes[j].x) - std::max(boxes[i].x, boxes[j].x));
            float intersection_height = std::max(0, std::min(boxes[i].h + boxes[i].y, boxes[j].w + boxes[j].y) - std::max(boxes[i].y, boxes[j].y));
            float intersection = intersection_width * intersection_height;
            float total_union = (boxes[i].area + boxes[j].area) - intersection;

            float iou = 0.0f;
            if (total_union > 0) {
                iou = intersection / total_union;
            }

            // Check IoU threshold
            if (iou > thresh) {
                if (boxes[i].area > boxes[j].area) {
                    boxes.erase(boxes.begin() + j);
                } else {
                    boxes.erase(boxes.begin() + i);
                    break; // Break since box i was removed
                }
            } else {
                j++; // Only increment j if no box was removed
            }
        }
    }
}

int main() {
    std::string vid_file = "/assets/vid1.mp4";
    cv::VideoCapture cap(vid_file);

    cv::Mat gray_prev_frame, gray_frame;

    int frame_count = 0;
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // Convert current frame to grayscale and blur it
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray_frame, gray_frame, BLUR_SIZE, 0);

        if (!gray_prev_frame.empty()) {
            // Compute contours
            std::vector<Box> boxes = detect_motion_clusters(gray_prev_frame, gray_frame);

            // Apply non-maximal suppression to reduce noisy bboxs
            non_max_suppression(boxes, PMS_THRESHOLD);

            cv::Rect bbox;
            for (const auto& box : boxes) {
                cv::Rect bbox(box.x, box.y, box.w, box.h);
                // Draw bbox
                cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);
            }

            // Show
            cv::imshow("Result", frame);
            cv::waitKey(30);

            if (SAVE_IMAGES) {
                std::stringstream filename;
                filename << "/imgs/frame_" << std::setw(4) << std::setfill('0') << frame_count << ".jpg";
                cv::imwrite(filename.str(), frame);
                frame_count++;
            }
        }

        // Update prev_frame
        gray_prev_frame = gray_frame.clone();
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
