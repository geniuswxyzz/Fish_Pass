#pragma once
#include "Algorithm.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <vector>
#include <string>

namespace AVSAnalyzer {

    // 全局TensorRT日志器（线程安全）
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) { // 仅输出警告及以上
                printf("[TRT] %s\n", msg);
            }
        }
    };

    extern Logger gLogger; // 只声明不定义，在一个.cpp文件里定义

    // COCO类别名称（可从配置读取）
    extern const std::vector<std::string> class_names; // 放namespace里，.cpp定义

    class FinshPass : public Algorithm {
    private:
        // TensorRT核心对象
        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::IBuilder* builder = nullptr;
        nvinfer1::INetworkDefinition* network = nullptr;
        nvinfer1::IBuilderConfig* config_mode = nullptr;
        nvinfer1::IOptimizationProfile* profile = nullptr;
        nvonnxparser::IParser* parser = nullptr;
        nvinfer1::ICudaEngine* engine = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;

        // CUDA资源
        cudaStream_t stream = nullptr;
        void* gpu_buffers[2] = { nullptr, nullptr };

        // 模型动态参数
        int INPUT_H = 0, INPUT_W = 0;
        int OUTPUT_ROWS = 0, OUTPUT_COLS = 0;
        size_t inputSize = 0, outputSize = 0;
        std::string input_tensor_name = "images";
        std::string output_tensor_name = "output0";
        int num_classes = 0;

        // 线程安全锁
        std::mutex infer_mutex;
        std::vector<float> host_output;
        bool use_fp16 = false;

        void cleanUpResources();

    public:
        FinshPass(const std::string& model_path);
        ~FinshPass() override;
        bool objectDetect(cv::Mat& image, std::vector<DetectObject>& detects) override;
    };
}