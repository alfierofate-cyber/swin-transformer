#include "preprocessor.h"
#include <algorithm>

namespace rtdet {

std::vector<float> Preprocessor::process(const cv::Mat& frame, LetterboxInfo& info) const {
    // Step 1: Letterbox 缩放
    cv::Mat letterboxed = letterbox(frame, info);

    // Step 2: 算子融合 — BGR2RGB + 归一化 + HWC2CHW 一次遍历完成
    return fused_normalize_and_transpose(letterboxed);
}

cv::Mat Preprocessor::letterbox(const cv::Mat& img, LetterboxInfo& info) const {
    int img_h = img.rows;
    int img_w = img.cols;

    // 计算缩放比例（保持纵横比）
    float scale = std::min(
        static_cast<float>(target_size_) / img_h,
        static_cast<float>(target_size_) / img_w
    );
    info.scale = scale;
    info.new_w = static_cast<int>(img_w * scale);
    info.new_h = static_cast<int>(img_h * scale);

    // 计算填充量（居中填充）
    info.pad_w = (target_size_ - info.new_w) / 2;
    info.pad_h = (target_size_ - info.new_h) / 2;

    // 缩放
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(info.new_w, info.new_h));

    // 创建灰色画布并将缩放后的图像贴在中央
    cv::Mat canvas(target_size_, target_size_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(info.pad_w, info.pad_h, info.new_w, info.new_h)));

    return canvas;
}

std::vector<float> Preprocessor::fused_normalize_and_transpose(const cv::Mat& img) const {
    // 算子融合：一次遍历完成 BGR→RGB + /255.0 归一化 + HWC→CHW 转换
    // 面试亮点：减少内存读写次数，模拟 TensorRT 的算子融合优化
    const int channels = 3;
    const int height = img.rows;
    const int width = img.cols;
    const int hw = height * width;

    std::vector<float> chw_data(channels * hw);

    // 指向 R、G、B 三个通道在 CHW 数据中的起始位置
    float* r_plane = chw_data.data();
    float* g_plane = chw_data.data() + hw;
    float* b_plane = chw_data.data() + 2 * hw;

    // 可选：#pragma omp parallel for 加速（需要链接 OpenMP）
    for (int i = 0; i < height; ++i) {
        const uint8_t* row = img.ptr<uint8_t>(i);
        for (int j = 0; j < width; ++j) {
            const int idx = i * width + j;
            const int pixel_offset = j * 3;
            // OpenCV 默认 BGR 顺序，这里转为 RGB 并归一化
            b_plane[idx] = row[pixel_offset + 0] / 255.0f;
            g_plane[idx] = row[pixel_offset + 1] / 255.0f;
            r_plane[idx] = row[pixel_offset + 2] / 255.0f;
        }
    }

    return chw_data;
}

} // namespace rtdet
