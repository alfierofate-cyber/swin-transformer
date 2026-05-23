#pragma once

#include "types.h"
#include <vector>

namespace rtdet {

struct LetterboxInfo;

// NMS 后处理器
// 面试亮点：在端侧场景中，NMS 通常会交给 NPU 或用 CUDA/Metal 实现以避免 CPU 瓶颈
class Postprocessor {
public:
    Postprocessor(float conf_threshold = 0.25f, float iou_threshold = 0.45f)
        : conf_threshold_(conf_threshold), iou_threshold_(iou_threshold) {}

    // 解析 YOLOv8 输出并执行 NMS
    // raw_output: 模型原始输出 (1, 84, 8400) → 转置为 (8400, 84)
    // info: Letterbox 信息，用于将坐标映射回原图
    // orig_w, orig_h: 原图尺寸
    std::vector<BBox> process(const std::vector<float>& raw_output,
                              const LetterboxInfo& info,
                              int orig_w, int orig_h) const;

private:
    float conf_threshold_;
    float iou_threshold_;

    // 计算两个框的 IoU（Intersection over Union）
    static float compute_iou(const BBox& a, const BBox& b);

    // 标准 NMS：按置信度排序 → 逐框过滤高 IoU 的重叠框
    static std::vector<BBox> nms(std::vector<BBox>& boxes, float iou_threshold);
};

} // namespace rtdet
