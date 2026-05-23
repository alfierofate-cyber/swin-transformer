#include "detector.h"
#include <stdexcept>
#include <iostream>

namespace rtdet {

void Detector::init(const std::string& model_path, int num_threads) {
    // 创建 ONNX Runtime 运行环境
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YoloDetector");

    // 配置 Session 选项
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(num_threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 加载模型 — unique_ptr 管理生命周期，出作用域自动释放
    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

    // 获取输入节点信息
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_inputs = session_->GetInputCount();
    for (size_t i = 0; i < num_inputs; ++i) {
        auto name = session_->GetInputNameAllocated(i, allocator);
        input_names_.emplace_back(name.get());

        // 获取输入维度
        auto type_info = session_->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        if (shape.size() == 4) { // NCHW
            input_c_ = static_cast<int>(shape[1]);
            input_h_ = static_cast<int>(shape[2]);
            input_w_ = static_cast<int>(shape[3]);
        }
    }

    // 获取输出节点信息
    size_t num_outputs = session_->GetOutputCount();
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name = session_->GetOutputNameAllocated(i, allocator);
        output_names_.emplace_back(name.get());
    }

    // 构建 C 字符串指针数组（ONNX Runtime API 需要）
    input_name_ptrs_.clear();
    for (auto& n : input_names_) input_name_ptrs_.push_back(n.c_str());
    output_name_ptrs_.clear();
    for (auto& n : output_names_) output_name_ptrs_.push_back(n.c_str());

    std::cout << "[Detector] Model loaded: " << model_path << "\n";
    std::cout << "[Detector] Input: " << input_c_ << "x" << input_h_ << "x" << input_w_ << "\n";
    std::cout << "[Detector] Outputs: " << num_outputs << "\n";
}

std::vector<float> Detector::infer(std::vector<float>&& input_data) {
    if (!session_) {
        throw std::runtime_error("Detector not initialized");
    }

    // 构建输入 Tensor — 零拷贝，直接包装已有内存
    std::vector<int64_t> input_shape = {1, input_c_, input_h_, input_w_};
    size_t input_count = input_data.size();

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        input_data.data(),
        input_count,
        input_shape.data(),
        input_shape.size()
    );

    // 执行推理
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_name_ptrs_.data(),
        &input_tensor,
        1,
        output_name_ptrs_.data(),
        output_name_ptrs_.size()
    );

    // 提取输出数据
    auto& output = output_tensors[0];
    auto type_info = output.GetTensorTypeAndShapeInfo();
    auto shape = type_info.GetShape();

    size_t total = 1;
    for (auto s : shape) total *= s;

    const float* output_data = output.GetTensorData<float>();
    return std::vector<float>(output_data, output_data + total);
}

} // namespace rtdet
