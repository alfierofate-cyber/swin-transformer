#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace rtdet {

// Letterbox 缩放参数
struct LetterboxInfo {
    float scale;        // 缩放比例
    int pad_w, pad_h;   // 填充像素数
    int new_w, new_h;   // 缩放后尺寸（填充前）
};

// 高性能预处理器
// 面试重点：模板零开销抽象 + 算子融合 + 可选 OpenMP 并行
class Preprocessor {
public:
    explicit Preprocessor(int target_size = 640) : target_size_(target_size) {}

    // 完整预处理流水线：Letterbox + BGR2RGB + Normalize + HWC2CHW
    // 返回 CHW 格式的 float 数据，可直接送入推理引擎
    std::vector<float> process(const cv::Mat& frame, LetterboxInfo& info) const;

private:
    int target_size_;

    // Letterbox 缩放：保持纵横比，填充灰色边框
    cv::Mat letterbox(const cv::Mat& img, LetterboxInfo& info) const;

    // 算子融合版本：将 BGR2RGB + /255.0 归一化 + HWC转CHW 合并在一次遍历中
    // 面试亮点：减少对图像内存的反复读写，模拟 TensorRT 算子融合思想
    std::vector<float> fused_normalize_and_transpose(const cv::Mat& img) const;
};

// 模板封装的通用缩放算子（零开销抽象示例）
// 面试重点：C++ 模板在编译期展开，运行时无额外开销
template <int TargetH, int TargetW>
struct StaticResize {
    static cv::Mat apply(const cv::Mat& img) {
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(TargetW, TargetH));
        return resized;
    }
};

// 特化常用尺寸
using Resize640 = StaticResize<640, 640>;
using Resize320 = StaticResize<320, 320>;

} // namespace rtdet
