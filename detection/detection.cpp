#include <opencv2/calib3d.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kInputSize = 640;
constexpr float kConfidenceThreshold = 0.5f;
constexpr float kNmsThreshold = 0.5f;

struct Detection {
    std::vector<cv::Point2f> corners;
    float confidence = 0.0f;
    int color = 0;
    int pattern = 0;
    cv::Rect2f box;
};

struct Letterbox {
    cv::Mat image;
    float scale = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;
};

struct Pose {
    bool valid = false;
    std::vector<cv::Point3f> corners;
    double reprojection_error = 0.0;
};

float sigmoid(float value) {
    if (value >= 0.0f) {
        return 1.0f / (1.0f + std::exp(-value));
    }
    const float e = std::exp(value);
    return e / (1.0f + e);
}

Letterbox make_letterbox(const cv::Mat& frame) {
    const float scale = std::min(
        static_cast<float>(kInputSize) / static_cast<float>(frame.cols),
        static_cast<float>(kInputSize) / static_cast<float>(frame.rows));
    const int width = static_cast<int>(std::round(frame.cols * scale));
    const int height = static_cast<int>(std::round(frame.rows * scale));

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);
    cv::Mat output(kInputSize, kInputSize, CV_8UC3, cv::Scalar(114, 114, 114));
    const int left = (kInputSize - width) / 2;
    const int top = (kInputSize - height) / 2;
    resized.copyTo(output(cv::Rect(left, top, width, height)));
    return {output, scale, static_cast<float>(left), static_cast<float>(top)};
}

float intersection_over_union(const cv::Rect2f& a, const cv::Rect2f& b) {
    const cv::Rect2f intersection = a & b;
    const float union_area = a.area() + b.area() - intersection.area();
    return union_area > 0.0f ? intersection.area() / union_area : 0.0f;
}

std::vector<Detection> decode(const cv::Mat& output, const Letterbox& letterbox,
                              const cv::Size& original_size) {
    if (output.total() % 22 != 0) {
        throw std::runtime_error("model output does not contain 22 values per prediction");
    }

    cv::Mat float_output;
    output.convertTo(float_output, CV_32F);
    cv::Mat rows = float_output.reshape(1, static_cast<int>(float_output.total() / 22));
    std::vector<Detection> candidates;

    for (int row = 0; row < rows.rows; ++row) {
        const float confidence = sigmoid(rows.at<float>(row, 8));
        if (!std::isfinite(confidence) || confidence < kConfidenceThreshold) {
            continue;
        }

        Detection detection;
        detection.confidence = confidence;
        detection.color = 0;
        detection.pattern = 0;
        for (int i = 1; i < 4; ++i) {
            if (rows.at<float>(row, 9 + i) > rows.at<float>(row, 9 + detection.color)) {
                detection.color = i;
            }
        }
        for (int i = 1; i < 9; ++i) {
            if (rows.at<float>(row, 13 + i) > rows.at<float>(row, 13 + detection.pattern)) {
                detection.pattern = i;
            }
        }

        detection.corners.reserve(4);
        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float max_y = std::numeric_limits<float>::lowest();
        for (int corner = 0; corner < 4; ++corner) {
            const float x = (rows.at<float>(row, corner * 2) - letterbox.pad_x) / letterbox.scale;
            const float y = (rows.at<float>(row, corner * 2 + 1) - letterbox.pad_y) / letterbox.scale;
            detection.corners.emplace_back(x, y);
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
        }
        detection.box = cv::Rect2f(
            std::max(0.0f, min_x), std::max(0.0f, min_y),
            std::min(static_cast<float>(original_size.width), max_x) -
                std::max(0.0f, min_x),
            std::min(static_cast<float>(original_size.height), max_y) -
                std::max(0.0f, min_y));
        if (detection.box.width > 0.0f && detection.box.height > 0.0f) {
            candidates.push_back(std::move(detection));
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });
    std::vector<Detection> kept;
    for (const Detection& candidate : candidates) {
        bool duplicate = false;
        for (const Detection& existing : kept) {
            if (intersection_over_union(candidate.box, existing.box) > kNmsThreshold) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            kept.push_back(candidate);
        }
    }
    return kept;
}

Pose estimate_pose(const Detection& detection, const cv::Mat& camera_matrix,
                   const cv::Mat& distortion) {
    const bool large = detection.pattern == 1 || detection.pattern == 8;
    const float width = large ? 0.230f : 0.135f;
    const float height = 0.056f;
    const std::vector<cv::Point3f> object_points{
        {-width / 2.0f, -height / 2.0f, 0.0f},
        {-width / 2.0f, height / 2.0f, 0.0f},
        {width / 2.0f, height / 2.0f, 0.0f},
        {width / 2.0f, -height / 2.0f, 0.0f}};

    cv::Mat rotation;
    cv::Mat translation;
    if (!cv::solvePnP(object_points, detection.corners, camera_matrix, distortion,
                      rotation, translation, false, cv::SOLVEPNP_IPPE)) {
        return {};
    }

    cv::Mat rotation_matrix;
    cv::Rodrigues(rotation, rotation_matrix);
    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_points, rotation, translation, camera_matrix, distortion, projected);
    double squared_error = 0.0;
    std::vector<cv::Point3f> camera_points;
    camera_points.reserve(object_points.size());
    for (size_t i = 0; i < object_points.size(); ++i) {
        const double x = rotation_matrix.at<double>(0, 0) * object_points[i].x +
                         rotation_matrix.at<double>(0, 1) * object_points[i].y +
                         rotation_matrix.at<double>(0, 2) * object_points[i].z +
                         translation.at<double>(0);
        const double y = rotation_matrix.at<double>(1, 0) * object_points[i].x +
                         rotation_matrix.at<double>(1, 1) * object_points[i].y +
                         rotation_matrix.at<double>(1, 2) * object_points[i].z +
                         translation.at<double>(1);
        const double z = rotation_matrix.at<double>(2, 0) * object_points[i].x +
                         rotation_matrix.at<double>(2, 1) * object_points[i].y +
                         rotation_matrix.at<double>(2, 2) * object_points[i].z +
                         translation.at<double>(2);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || z <= 0.0) {
            return {};
        }
        camera_points.emplace_back(static_cast<float>(x), static_cast<float>(y),
                                   static_cast<float>(z));
        const double dx = projected[i].x - detection.corners[i].x;
        const double dy = projected[i].y - detection.corners[i].y;
        if (!std::isfinite(projected[i].x) || !std::isfinite(projected[i].y)) {
            return {};
        }
        squared_error += dx * dx + dy * dy;
    }

    Pose pose;
    pose.valid = true;
    pose.corners = std::move(camera_points);
    pose.reprojection_error = std::sqrt(squared_error / 4.0);
    if (!std::isfinite(pose.reprojection_error)) {
        return {};
    }
    return pose;
}

std::string color_name(int index) {
    static const std::vector<std::string> names{"blue", "red", "neutral", "purple"};
    return names.at(static_cast<size_t>(index));
}

std::string pattern_name(int index) {
    static const std::vector<std::string> names{
        "sentry", "1", "2", "3", "4", "5", "outpost", "base_small", "base_big"};
    return names.at(static_cast<size_t>(index));
}

void write_point(std::ostream& output, const cv::Point2f& point) {
    output << '[' << point.x << ',' << point.y << ']';
}

void write_point(std::ostream& output, const cv::Point3f& point) {
    output << '[' << point.x << ',' << point.y << ',' << point.z << ']';
}

void write_json_frame(std::ostream& output, int frame_index,
                      const std::vector<Detection>& detections,
                      const cv::Mat& camera_matrix, const cv::Mat& distortion) {
    output << "{\"frame_index\":" << frame_index
           << ",\"timestamp_s\":" << (static_cast<double>(frame_index) / 30.0)
           << ",\"detections\":[";
    for (size_t i = 0; i < detections.size(); ++i) {
        if (i != 0) {
            output << ',';
        }
        const Detection& detection = detections[i];
        const Pose pose = estimate_pose(detection, camera_matrix, distortion);
        output << "{\"color\":\"" << color_name(detection.color)
               << "\",\"pattern\":\"" << pattern_name(detection.pattern)
               << "\",\"confidence\":" << detection.confidence << ",\"corners_px\":[";
        for (size_t corner = 0; corner < detection.corners.size(); ++corner) {
            if (corner != 0) {
                output << ',';
            }
            write_point(output, detection.corners[corner]);
        }
        output << "],\"corners_camera_m\":";
        if (!pose.valid) {
            output << "null,\"reprojection_error_px\":null}";
            continue;
        }
        output << '[';
        for (size_t corner = 0; corner < pose.corners.size(); ++corner) {
            if (corner != 0) {
                output << ',';
            }
            write_point(output, pose.corners[corner]);
        }
        output << "],\"reprojection_error_px\":" << pose.reprojection_error << '}';
    }
    output << "]}\n";
}

std::string required_argument(int& index, int argc, char** argv, const std::string& option) {
    if (index + 1 >= argc) {
        throw std::runtime_error("missing value for " + option);
    }
    return argv[++index];
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string model_path;
        std::string video_path;
        std::string calibration_path;
        std::string output_path;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--model") {
                model_path = required_argument(i, argc, argv, option);
            } else if (option == "--video") {
                video_path = required_argument(i, argc, argv, option);
            } else if (option == "--calibration") {
                calibration_path = required_argument(i, argc, argv, option);
            } else if (option == "--output") {
                output_path = required_argument(i, argc, argv, option);
            } else {
                throw std::runtime_error("unknown argument: " + option);
            }
        }
        if (model_path.empty() || video_path.empty() || calibration_path.empty() ||
            output_path.empty()) {
            throw std::runtime_error(
                "usage: detection --model MODEL --video VIDEO --calibration YAML --output JSONL");
        }

        cv::FileStorage calibration(calibration_path, cv::FileStorage::READ);
        if (!calibration.isOpened()) {
            throw std::runtime_error("cannot open calibration file: " + calibration_path);
        }
        cv::Mat camera_matrix;
        cv::Mat distortion;
        calibration["camera_matrix"] >> camera_matrix;
        calibration["distort_coeffs"] >> distortion;
        if (camera_matrix.empty() || distortion.empty()) {
            throw std::runtime_error("calibration is missing camera_matrix or distort_coeffs");
        }
        camera_matrix.convertTo(camera_matrix, CV_64F);
        distortion.convertTo(distortion, CV_64F);

        cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);
        if (net.empty()) {
            throw std::runtime_error("cannot load ONNX model: " + model_path);
        }
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        cv::VideoCapture video(video_path);
        if (!video.isOpened()) {
            throw std::runtime_error("cannot open video: " + video_path);
        }
        std::ofstream output(output_path, std::ios::binary);
        if (!output) {
            throw std::runtime_error("cannot open output file: " + output_path);
        }
        output << std::setprecision(9);

        cv::Mat frame;
        int frame_index = 0;
        while (video.read(frame)) {
            const Letterbox letterbox = make_letterbox(frame);
            cv::Mat rgb;
            cv::cvtColor(letterbox.image, rgb, cv::COLOR_BGR2RGB);
            cv::Mat input = cv::dnn::blobFromImage(
                rgb, 1.0 / 255.0, cv::Size(kInputSize, kInputSize),
                cv::Scalar(), false, false, CV_32F);
            net.setInput(input, "images");
            const cv::Mat inference = net.forward("output");
            const std::vector<Detection> detections = decode(inference, letterbox, frame.size());
            write_json_frame(output, frame_index, detections, camera_matrix, distortion);
            ++frame_index;
        }
        if (frame_index == 0) {
            throw std::runtime_error("video contains no frames");
        }
        return 0;
    } catch (const cv::Exception& error) {
        std::cerr << "OpenCV error: " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}