#pragma once

#include <vector>
#include <string>
#include <opencv2/core.hpp>

namespace rtdet {

// 检测框
struct BBox {
    float x1, y1, x2, y2;  // 左上角和右下角坐标（原图尺度）
    float confidence;
    int class_id;

    float area() const { return (x2 - x1) * (y2 - y1); }
};

// 一帧的检测结果
struct FrameResult {
    cv::Mat frame;                // 原始帧
    std::vector<BBox> detections; // 检测结果
    double preprocess_ms = 0;     // 预处理耗时
    double inference_ms = 0;      // 推理耗时
    double postprocess_ms = 0;    // 后处理耗时
};

// COCO 80类标签（简化版，只列常用的）
inline const std::vector<std::string>& coco_labels() {
    static const std::vector<std::string> labels = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };
    return labels;
}

} // namespace rtdet
