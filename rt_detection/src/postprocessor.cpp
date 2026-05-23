#include "postprocessor.h"
#include "preprocessor.h"  // for LetterboxInfo
#include <algorithm>
#include <numeric>
#include <cmath>

namespace rtdet {

float Postprocessor::compute_iou(const BBox& a, const BBox& b) {
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);

    float inter_area = std::max(0.0f, inter_x2 - inter_x1) *
                       std::max(0.0f, inter_y2 - inter_y1);

    float union_area = a.area() + b.area() - inter_area;
    return (union_area > 0) ? inter_area / union_area : 0.0f;
}

std::vector<BBox> Postprocessor::nms(std::vector<BBox>& boxes, float iou_threshold) {
    // 按置信度降序排序
    std::sort(boxes.begin(), boxes.end(),
              [](const BBox& a, const BBox& b) { return a.confidence > b.confidence; });

    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<BBox> result;

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(boxes[i]);

        // 抑制与当前框 IoU 过高的后续框
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (!suppressed[j] && boxes[i].class_id == boxes[j].class_id) {
                if (compute_iou(boxes[i], boxes[j]) > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }
    return result;
}

std::vector<BBox> Postprocessor::process(const std::vector<float>& raw_output,
                                          const LetterboxInfo& info,
                                          int orig_w, int orig_h) const {
    // YOLOv8 输出格式: (1, 84, 8400)
    // 84 = 4 (xywh) + 80 (class scores)
    // 8400 个候选框
    constexpr int num_classes = 80;
    constexpr int num_boxes = 8400;
    // data_per_box = 84 (4 xywh + 80 class scores)

    // 原始数据是 (84, 8400)，需要按列读取
    std::vector<BBox> candidates;
    candidates.reserve(256); // 预分配，减少动态扩容

    for (int i = 0; i < num_boxes; ++i) {
        // 提取 xywh（中心点格式）
        float cx = raw_output[0 * num_boxes + i];
        float cy = raw_output[1 * num_boxes + i];
        float w  = raw_output[2 * num_boxes + i];
        float h  = raw_output[3 * num_boxes + i];

        // 找到最大类别分数
        int best_class = 0;
        float best_score = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            float score = raw_output[(4 + c) * num_boxes + i];
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }

        if (best_score < conf_threshold_) continue;

        // xywh → xyxy（模型输出尺度，640x640）
        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;
        float x2 = cx + w / 2.0f;
        float y2 = cy + h / 2.0f;

        // 映射回原图坐标：去掉 padding，除以缩放比例
        x1 = (x1 - info.pad_w) / info.scale;
        y1 = (y1 - info.pad_h) / info.scale;
        x2 = (x2 - info.pad_w) / info.scale;
        y2 = (y2 - info.pad_h) / info.scale;

        // 裁剪到原图范围
        x1 = std::clamp(x1, 0.0f, static_cast<float>(orig_w));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(orig_h));
        x2 = std::clamp(x2, 0.0f, static_cast<float>(orig_w));
        y2 = std::clamp(y2, 0.0f, static_cast<float>(orig_h));

        candidates.push_back({x1, y1, x2, y2, best_score, best_class});
    }

    // 执行 NMS
    return nms(candidates, iou_threshold_);
}

} // namespace rtdet
