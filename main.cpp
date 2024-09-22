#include <opencv2/opencv.hpp>

// TODO: use Box struct, fix NMS logic (maybe use opencv's?: https://docs.opencv.org/4.x/d6/d0f/group__dnn.html#ga9d118d70a1659af729d01b10233213ee)

// Constants
const int THRESHOLD_VALUE = 25;
const int BBOX_OFFSET = 20;
const int MIN_AREA = 30;
const cv::Size BLUR_SIZE(3, 3);

struct Box {
    int x, y, width, height, area;
};

std::vector<std::vector<int>> detect_motion(cv::Mat& gray_prev_frame, cv::Mat& gray_frame) {
    cv::Mat diff_frame, thresh_frame;

    // Compute abs diff and apply threshold
    cv::absdiff(gray_prev_frame, gray_frame, diff_frame);
    cv::threshold(diff_frame, thresh_frame, THRESHOLD_VALUE, 255, cv::THRESH_BINARY);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh_frame, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<int>> contour_data;
    for (const auto& contour : contours) {
        cv::Rect contour_box = cv::boundingRect(contour);    

        int area = contour_box.width * contour_box.height;
        std::vector<int> data = {
            contour_box.x,
            contour_box.y,
            contour_box.width,
            contour_box.height,
            area
        };
        contour_data.push_back(data);             
    }
    return contour_data;
}

// reference: https://medium.com/@itberrios6/introduction-to-motion-detection-part-1-e031b0bb9bb2
void non_max_suppression(std::vector<std::vector<int>>& boxes, int& thresh) {
    // Remove bboxes that have high Intersection Over Union (IoU)
    for (size_t i = 0; i < boxes.size(); i++) {
        for (size_t j = i + 1; j < boxes.size(); ) { // dont increment j when erasing
            // boxes[i] = {x, y, w, h, area}

            // Remove boxes that are

            float intersection_width = std::max(0, std::min((boxes[i][2] + boxes[i][0]), (boxes[j][2] + boxes[j][0])) - std::max(boxes[i][0], boxes[j][0]));
            float intersection_height = std::max(0, std::min((boxes[i][3] + boxes[i][1]), (boxes[j][3] + boxes[j][0])) - std::max(boxes[i][1], boxes[j][1]));
            float intersection = intersection_width * intersection_height;

            float total_area_i = boxes[i][4];
            float total_area_j = boxes[j][4];
            float total_union = total_area_i + total_area_j - intersection;

            float iou = 0.0f;
            if (total_union > 0) {
                iou = intersection / total_union;
            }

            // if (iou > 0.6) {
            //     std::cout << "iou " << iou << "\n" << std::flush;
            // }
            
            if (iou > thresh) {
                if (boxes[i][4] > boxes[j][4]) {
                    boxes.erase(boxes.begin() + j);
                } else {
                    boxes.erase(boxes.begin() + i);
                    break; // break since i is removed
                }
            } else {
                j++; // only increment j if no box was removed
            }
        }
    }
}

int main() {
    std::string vid_file = "/assets/vid2.mp4";
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
            // Compute contours
            std::vector<std::vector<int>> boxes = detect_motion(gray_prev_frame, gray_frame);

            // Apply non-maximal suppression to reduce noisy bboxs
            int thresh = 0.1;
            non_max_suppression(boxes, thresh);

            cv::Rect bbox;
            for (const auto& box : boxes) {
                bbox = cv::Rect(
                        box[0],
                        box[1],
                        box[2],
                        box[3]
                );
                // Draw bbox
                cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);
            }

            // Show
            cv::imshow("Result", frame);
            cv::waitKey(50);
        }

        // Update prev_frame
        gray_prev_frame = gray_frame.clone();

    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
