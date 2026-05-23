// 端侧高性能目标检测流水线
// 多线程架构：视频解码线程(Producer) → 线程安全队列 → 推理线程(Consumer)

#include "types.h"
#include "frame_queue.h"
#include "preprocessor.h"
#include "detector.h"
#include "postprocessor.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace rtdet;
using Clock = std::chrono::high_resolution_clock;

// 计时辅助
inline double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// 绘制检测框
void draw_detections(cv::Mat& frame, const std::vector<BBox>& boxes) {
    const auto& labels = coco_labels();
    for (const auto& box : boxes) {
        cv::rectangle(frame,
                      cv::Point(static_cast<int>(box.x1), static_cast<int>(box.y1)),
                      cv::Point(static_cast<int>(box.x2), static_cast<int>(box.y2)),
                      cv::Scalar(0, 255, 0), 2);

        std::string label = (box.class_id < static_cast<int>(labels.size()))
                            ? labels[box.class_id] : "unknown";
        std::ostringstream oss;
        oss << label << " " << std::fixed << std::setprecision(2) << box.confidence;

        int baseline = 0;
        auto text_size = cv::getTextSize(oss.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(frame,
                      cv::Point(static_cast<int>(box.x1), static_cast<int>(box.y1) - text_size.height - 4),
                      cv::Point(static_cast<int>(box.x1) + text_size.width, static_cast<int>(box.y1)),
                      cv::Scalar(0, 255, 0), -1);
        cv::putText(frame, oss.str(),
                    cv::Point(static_cast<int>(box.x1), static_cast<int>(box.y1) - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

// ========== 生产者线程：视频解码 ==========
void producer_thread(const std::string& source,
                     ThreadSafeQueue<cv::Mat>& queue,
                     std::atomic<bool>& running) {
    cv::VideoCapture cap;

    // 尝试作为摄像头 ID 打开，否则作为文件路径
    if (source == "0" || source == "1") {
        cap.open(std::stoi(source));
    } else {
        cap.open(source);
    }

    if (!cap.isOpened()) {
        std::cerr << "[Producer] Failed to open source: " << source << "\n";
        running = false;
        return;
    }

    std::cout << "[Producer] Video source opened: "
              << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
              << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << " @ "
              << cap.get(cv::CAP_PROP_FPS) << " FPS\n";

    while (running) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::cout << "[Producer] End of video stream\n";
            running = false;
            break;
        }
        // 移动语义入队，避免深拷贝
        queue.push(std::move(frame));
    }

    queue.stop(); // 通知消费者退出
}

// ========== 消费者线程：预处理 + 推理 + 后处理 ==========
void consumer_thread(ThreadSafeQueue<cv::Mat>& queue,
                     ThreadSafeQueue<FrameResult>& result_queue,
                     Detector& detector,
                     std::atomic<bool>& running) {
    Preprocessor preprocessor(detector.input_height());
    Postprocessor postprocessor(0.25f, 0.45f);

    while (running || !queue.empty()) {
        auto maybe_frame = queue.wait_and_pop();
        if (!maybe_frame.has_value()) break;

        cv::Mat frame = std::move(maybe_frame.value());
        FrameResult result;
        result.frame = frame.clone(); // 保留一份用于绘制

        // 预处理
        auto t0 = Clock::now();
        LetterboxInfo info{};
        auto input_data = preprocessor.process(frame, info);
        result.preprocess_ms = elapsed_ms(t0);

        // 推理 — 移动语义传递数据
        auto t1 = Clock::now();
        auto raw_output = detector.infer(std::move(input_data));
        result.inference_ms = elapsed_ms(t1);

        // 后处理
        auto t2 = Clock::now();
        result.detections = postprocessor.process(raw_output, info, frame.cols, frame.rows);
        result.postprocess_ms = elapsed_ms(t2);

        result_queue.push(std::move(result));
    }

    result_queue.stop();
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " <model.onnx> <video_source>\n"
              << "  model.onnx   : Path to YOLOv8 ONNX model\n"
              << "  video_source : Video file path, or camera ID (0, 1)\n"
              << "\nExample:\n"
              << "  " << program << " models/yolov8n.onnx 0          # webcam\n"
              << "  " << program << " models/yolov8n.onnx test.mp4   # video file\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string video_source = argv[2];

    // 初始化推理引擎
    Detector detector;
    try {
        detector.init(model_path, 4); // Apple Silicon �� 4 线程
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << "\n";
        return 1;
    }

    // 创建线程安全队列
    ThreadSafeQueue<cv::Mat> frame_queue(8);        // 解码 → 推理
    ThreadSafeQueue<FrameResult> result_queue(4);    // 推理 → 显示

    std::atomic<bool> running{true};

    // 启动生产者和消费者线程
    std::thread producer(producer_thread, std::ref(video_source),
                         std::ref(frame_queue), std::ref(running));
    std::thread consumer(consumer_thread, std::ref(frame_queue),
                         std::ref(result_queue), std::ref(detector), std::ref(running));

    // 主线程：显示结果
    int frame_count = 0;
    double total_time = 0;

    while (true) {
        auto maybe_result = result_queue.wait_and_pop();
        if (!maybe_result.has_value()) break;

        auto& result = maybe_result.value();
        draw_detections(result.frame, result.detections);

        double frame_total = result.preprocess_ms + result.inference_ms + result.postprocess_ms;
        total_time += frame_total;
        frame_count++;

        // 打印性能信息
        std::cout << "\r[Frame " << frame_count << "] "
                  << "Pre: " << std::fixed << std::setprecision(1) << result.preprocess_ms << "ms | "
                  << "Infer: " << result.inference_ms << "ms | "
                  << "Post: " << result.postprocess_ms << "ms | "
                  << "Total: " << frame_total << "ms | "
                  << "Detections: " << result.detections.size()
                  << std::flush;

        // 显示画面
        cv::imshow("RT Detection Pipeline", result.frame);
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q') { // ESC 或 q 退出
            running = false;
            frame_queue.stop();
            break;
        }
    }

    // 等待线程结束
    if (producer.joinable()) producer.join();
    if (consumer.joinable()) consumer.join();
    cv::destroyAllWindows();

    // 打印汇总统计
    if (frame_count > 0) {
        std::cout << "\n\n=== Performance Summary ===\n";
        std::cout << "Total frames: " << frame_count << "\n";
        std::cout << "Avg latency: " << std::fixed << std::setprecision(1)
                  << total_time / frame_count << " ms/frame\n";
        std::cout << "Avg FPS: " << std::setprecision(1)
                  << 1000.0 * frame_count / total_time << "\n";
    }

    return 0;
}
