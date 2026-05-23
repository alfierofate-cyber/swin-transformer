#pragma once

#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

namespace rtdet {

// ONNX Runtime 推理引擎封装
// 面试重点：unique_ptr 管理 Session 生命周期 + 零拷贝输入 + 移动语义
class Detector {
public:
    Detector() = default;
    ~Detector() = default;

    // 禁止拷贝（Session 不可拷贝），允许移动
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;
    Detector(Detector&&) = default;
    Detector& operator=(Detector&&) = default;

    // 初始化：加载 ONNX 模型
    // num_threads: 推理线程数，Apple Silicon 建议 2-4
    void init(const std::string& model_path, int num_threads = 2);

    // 执行推理
    // 接收移动语义传入的预处理数据，避免深拷贝
    // 返回模型原始输出（交给 Postprocessor 解析）
    std::vector<float> infer(std::vector<float>&& input_data);

    // 获取模型输入尺寸
    int input_height() const { return input_h_; }
    int input_width() const { return input_w_; }

    bool is_ready() const { return session_ != nullptr; }

private:
    // 独占所有权管理 ONNX Runtime 对象
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_ = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    // 模型 I/O 元信息
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_name_ptrs_;
    std::vector<const char*> output_name_ptrs_;
    int input_h_ = 640;
    int input_w_ = 640;
    int input_c_ = 3;
};

} // namespace rtdet
