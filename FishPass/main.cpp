/**
 * 项目名：FishPass Integrated System (Ultimate Edition)
 * 功能：3路RTSP拼接 -> 去畸变 -> AI检测 -> 计数入库 -> OSD实时显示 -> 结果推流
 */
/*
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp> 
#include <opencv2/cudaimgproc.hpp> 
#include <opencv2/cudaarithm.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <cstdio>
#include <ctime>
#include <chrono>
#include <filesystem>
#define NOMINMAX
#include <windows.h>

 // ================== 核心头文件引入 ==================
#include "DBHelper.h"       // 数据库功能
#include "PoolConfig.h"     // 池室配置
#include "FinshPass.h"      // TensorRT 检测模型
#include "FinshPrecent.h"   // 业务计数逻辑
// ==============================================================

using namespace std;
namespace fs = std::filesystem;

static fs::path GetExecutableDir() {
    char buffer[MAX_PATH] = {};
    DWORD len = GetModuleFileNameA(nullptr, buffer, MAX_PATH);
    if (len == 0 || len == MAX_PATH) {
        return fs::current_path();
    }
    return fs::path(buffer).parent_path();
}

static std::string ResolveExistingPath(const std::vector<fs::path>& candidates, const std::string& fallback) {
    for (const auto& candidate : candidates) {
        if (fs::exists(candidate)) {
            return candidate.string();
        }
    }
    return fallback;
}

// 全局原子标志位：防止查询时日志干扰
std::atomic<bool> g_is_querying(false);

// --- [配置部分] 加载池室检测区域 ---
// 注意：这里的坐标必须对应拼接后的 3840x1080 分辨率画面
std::vector<PoolConfig> LoadPoolConfigs() {
    std::vector<PoolConfig> configs;

    // 池室 2
    configs.emplace_back(
        1, "Pool_Channel_2",
        cv::Point(2180, 441), cv::Point(2180, 564), // 入口线
        cv::Point(2485, 441), cv::Point(2485, 564), // 出口线
        0, 0
    );

    // 池室 1
    configs.emplace_back(
        2, "Pool_Channel_1",
        cv::Point(1402, 465), cv::Point(1402, 588),
        cv::Point(1640, 465), cv::Point(1640, 588),
        0, 0
    );

    return configs;
}

// --- [查询部分] 后台多线程查询交互 ---
void QueryThreadFunc(std::vector<PoolConfig> pool_configs) {
    std::string start_time, end_time;
    std::cout << "\n>>> [后台模式] 进入查询系统 (视频拼接与检测仍在极速运行...) <<<" << std::endl;

    if (std::cin.rdbuf()->in_avail() > 0) std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.clear();

    std::cout << "请输入开始时间 (如 2026-01-24 00:00:00): " << std::endl;
    std::getline(std::cin, start_time);
    if (start_time.empty()) std::getline(std::cin, start_time);
    if (start_time.empty()) start_time = "2026-01-24 00:00:00";

    std::cout << "请输入结束时间 (如 2026-01-24 23:59:59): " << std::endl;
    std::getline(std::cin, end_time);
    if (end_time.empty()) end_time = "2026-01-24 23:59:59";

    std::cout << "正在查询数据库 [" << start_time << " ~ " << end_time << "] ..." << std::endl;

    for (const auto& cfg : pool_configs) {
        PoolStats stats = DBHelper::GetInstance().QueryPassRateByTime(cfg.id, start_time, end_time);
        std::cout << "--------------------------------" << std::endl;
        std::cout << "池室: " << cfg.name << " (ID:" << cfg.id << ")" << std::endl;
        std::cout << "  - [分母] 起点净流量: " << stats.start_net_flow << std::endl;
        std::cout << "  - [分子] 终点净流量: " << stats.end_net_flow << std::endl;
        if (stats.is_valid) printf("  - ★ 通过率: %.2f%%\n", stats.pass_rate * 100.0);
        else std::cout << "  - ★ 通过率: -- (分母为0)" << std::endl;
    }

    std::cout << "\n是否查看详细流水? (y/n): ";
    std::string choice;
    std::getline(std::cin, choice);
    if (choice == "y" || choice == "Y") {
        for (const auto& cfg : pool_configs) {
            DBHelper::GetInstance().QueryLogsByTime(cfg.id, start_time, end_time);
        }
    }

    std::cout << "查询完毕！按回车键退出..." << std::endl;
    std::string temp;
    std::getline(std::cin, temp);
    g_is_querying = false;
    std::cout << ">>> 已退出查询，恢复日志 <<<" << std::endl;
}

// --- [拼接部分] 结构体与类定义 ---
struct StreamGpuBuffer {
    cv::cuda::GpuMat raw_gpu;
    cv::cuda::GpuMat resized_gpu;
    cv::cuda::GpuMat undistorted_gpu;
    cv::cuda::GpuMat final_gpu;
    cv::cuda::GpuMat map_x;
    cv::cuda::GpuMat map_y;
    cv::cuda::Stream stream;
};

struct StreamConfig {
    std::string url;
    int target_w, target_h;
    cv::Mat K, D;
    cv::Rect crop_rect;
    cv::Size final_size;
    cv::Rect canvas_roi;
};

class RTSPStream {
public:
    RTSPStream(std::string url) : url_(url), running_(true), has_frame_(false) {
        worker_ = std::thread(&RTSPStream::update, this);
    }
    ~RTSPStream() { stop(); if (worker_.joinable()) worker_.join(); }
    void stop() { running_ = false; }
    bool getFrame(cv::Mat& out_frame) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!has_frame_) return false;
        last_frame_.copyTo(out_frame);
        return true;
    }
private:
    void update() {
        cv::VideoCapture cap;
        cap.open(url_, cv::CAP_FFMPEG);
        cv::Mat temp;
        while (running_) {
            if (cap.isOpened()) {
                if (cap.read(temp) && !temp.empty()) {
                    std::lock_guard<std::mutex> lock(mtx_);
                    temp.copyTo(last_frame_);
                    has_frame_ = true;
                }
                else {
                    cap.release();
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                }
            }
            else {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                cap.open(url_, cv::CAP_FFMPEG);
            }
        }
    }
    std::string url_;
    std::thread worker_;
    std::atomic<bool> running_;
    std::mutex mtx_;
    cv::Mat last_frame_;
    bool has_frame_;
};

class AsyncWriter {
public:
    AsyncWriter(const std::string& cmd) : running_(true) {
        pipe_ = _popen(cmd.c_str(), "wb");
        worker_ = std::thread(&AsyncWriter::process_queue, this);
    }
    ~AsyncWriter() { running_ = false; if (worker_.joinable()) worker_.join(); if (pipe_) _pclose(pipe_); }
    void push(const cv::Mat& frame) {
        if (!running_) return;
        std::lock_guard<std::mutex> lock(mtx_);
        if (queue_.size() < 4) queue_.push(frame.clone());
    }
private:
    void process_queue() {
        while (running_) {
            cv::Mat frame;
            bool has_data = false;
            {
                std::lock_guard<std::mutex> lock(mtx_);
                if (!queue_.empty()) { frame = queue_.front(); queue_.pop(); has_data = true; }
            }
            if (has_data) fwrite(frame.data, 1, frame.total() * frame.elemSize(), pipe_);
            else std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    FILE* pipe_;
    std::thread worker_;
    std::atomic<bool> running_;
    std::mutex mtx_;
    std::queue<cv::Mat> queue_;
};

std::vector<StreamConfig> get_configs() {
    std::vector<StreamConfig> configs;
    {
        StreamConfig cfg;
        cfg.url = "rtsp://127.0.0.1:9994/live/1";
        float k[] = { 3632.08770f, 0, 932.18005f, 0, 3736.83206f, 530.14428f, 0, 0, 1 };
        float d[] = { -1.2936f, -9.3598f, 0.0183f, 0.0860f, 59.2869f };
        cfg.K = cv::Mat(3, 3, CV_32F, k).clone(); cfg.D = cv::Mat(1, 5, CV_32F, d).clone();
        cfg.crop_rect = cv::Rect(0, 0, 1920, 824);
        cfg.final_size = cv::Size(1920, 1080);
        cfg.canvas_roi = cv::Rect(0, 0, 1920, 1080);
        configs.push_back(cfg);
    }
    {
        StreamConfig cfg;
        cfg.url = "rtsp://127.0.0.1:9994/live/2";
        float k[] = { 3641.99471f, 0, 977.13310f, 0, 7186.56110f, 539.32540f, 0, 0, 1 };
        float d[] = { -2.0241f, -5.6109f, -0.0280f, -0.0056f, 53.5588f };
        cfg.K = cv::Mat(3, 3, CV_32F, k).clone(); cfg.D = cv::Mat(1, 5, CV_32F, d).clone();
        cfg.crop_rect = cv::Rect(589, 116, 850, 824);
        cfg.final_size = cv::Size(960, 1080);
        cfg.canvas_roi = cv::Rect(1920, 0, 960, 1080);
        configs.push_back(cfg);
    }
    {
        StreamConfig cfg;
        cfg.url = "rtsp://127.0.0.1:9994/live/3";
        float k[] = { 1480.34081f, 0, 972.90160f, 0, 1483.11981f, 558.28559f, 0, 0, 1 };
        float d[] = { -0.6412f, 0.4056f, 0.0038f, 0.00002f, -0.1189f };
        cfg.K = cv::Mat(3, 3, CV_32F, k).clone(); cfg.D = cv::Mat(1, 5, CV_32F, d).clone();
        cfg.crop_rect = cv::Rect(406, 43, 1104, 824);
        cfg.final_size = cv::Size(960, 1080);
        cfg.canvas_roi = cv::Rect(2880, 0, 960, 1080);
        configs.push_back(cfg);
    }
    return configs;
}

int main() {
    _putenv_s("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000");
    const fs::path exe_dir = GetExecutableDir();
    const fs::path repo_root = exe_dir.parent_path().parent_path();

    // 1. CUDA 初始化
    try {
        if (cv::cuda::getCudaEnabledDeviceCount() == 0) return -1;
        cv::cuda::setDevice(0);
    }
    catch (...) { return -1; }

    // 2. 数据库连接
    std::cout << "[Init] 连接数据库..." << std::endl;
    if (DBHelper::GetInstance().Connect("127.0.0.1", "root", "123456", "fish_db")) {
        std::cout << "数据库连接成功！" << std::endl;
    }
    else {
        std::cerr << "警告：数据库连接失败！" << std::endl;
    }

    // 3. 模型与算法初始化
    std::cout << "[Init] 初始化 AI 模型..." << std::endl;
    const std::string model_path = ResolveExistingPath({
        exe_dir / "model" / "fishmonitor_lxf_12_9.engine",
        exe_dir.parent_path() / "model" / "fishmonitor_lxf_12_9.engine",
        repo_root / "model" / "fishmonitor_lxf_12_9.engine"
        }, "model\\fishmonitor_lxf_12_9.engine");
    AVSAnalyzer::FinshPass* fish_detector = new AVSAnalyzer::FinshPass(model_path);
    std::cout << "模型加载成功！" << std::endl;

    std::vector<PoolConfig> pool_configs = LoadPoolConfigs();
    std::vector<AVSAnalyzer::FinshPrecent*> calculators;
    for (const auto& cfg : pool_configs) {
        calculators.push_back(new AVSAnalyzer::FinshPrecent(
            cfg.id, cfg.entry_p1, cfg.entry_p2, cfg.exit_p1, cfg.exit_p2, cfg.direction_in, cfg.direction_out
        ));
    }

    // 4. 拼接推流初始化
    auto configs = get_configs();
    std::string ffmpeg_path = ResolveExistingPath({
        exe_dir / "ffmpeg.exe",
        exe_dir.parent_path() / "ffmpeg.exe",
        repo_root / "3rdparty" / "ffmpeg" / "bin" / "ffmpeg.exe"
        }, "ffmpeg");
    
    // [Stream 1] OSD 处理后的流
    std::string cmd = "\"" + ffmpeg_path + "\" -y -f rawvideo -pix_fmt bgr24 -s 3840x1080 -framerate 30 -i - "
        "-vf format=yuv420p "
        "-c:v hevc_nvenc -preset p1 -tune ull -b:v 6M -maxrate 8M -bufsize 4M -g 60 "
        "-f rtsp -rtsp_transport tcp rtsp://127.0.0.1:9994/live/output";
    AsyncWriter writer(cmd);

    // [Stream 2] 纯净流 (无OSD)
    std::string cmd_clean = "\"" + ffmpeg_path + "\" -y -f rawvideo -pix_fmt bgr24 -s 3840x1080 -framerate 30 -i - "
        "-vf format=yuv420p "
        "-c:v hevc_nvenc -preset p1 -tune ull -b:v 6M -maxrate 8M -bufsize 4M -g 60 "
        "-f rtsp -rtsp_transport tcp rtsp://127.0.0.1:9994/live/output_origin";
    AsyncWriter writer_clean(cmd_clean);

    std::vector<std::unique_ptr<RTSPStream>> streams;
    for (const auto& cfg : configs) streams.push_back(std::unique_ptr<RTSPStream>(new RTSPStream(cfg.url)));

    std::vector<StreamGpuBuffer> gpu_buffers(3);
    for (int i = 0; i < 3; i++) {
        const auto& cfg = configs[i];
        cv::Mat map1, map2;
        cv::Mat new_K = cv::getOptimalNewCameraMatrix(cfg.K, cfg.D, cv::Size(1920, 1080), 0);
        cv::initUndistortRectifyMap(cfg.K, cfg.D, cv::Mat(), new_K, cv::Size(1920, 1080), CV_32FC1, map1, map2);
        gpu_buffers[i].map_x.upload(map1);
        gpu_buffers[i].map_y.upload(map2);
    }

    cv::cuda::HostMem pinned_mem(1080, 3840, CV_8UC3, cv::cuda::HostMem::PAGE_LOCKED);
    cv::cuda::GpuMat canvas_gpu(1080, 3840, CV_8UC3);

    std::cout << "[System Ready] 系统运行中 (GPU拼接 + AI检测 + DB统计 + OSD显示)..." << std::endl;

    const double target_frame_time_ms = 1000.0 / 30.0;
    std::vector<cv::Mat> current_frames(3);
    for (int i = 0; i < 3; i++) current_frames[i] = cv::Mat::zeros(1080, 1920, CV_8UC3);

    // ================== OSD 实时显示缓存与时间初始化 ==================
    std::vector<PoolStats> g_realtime_stats(pool_configs.size());
    auto last_query_time = std::chrono::steady_clock::now() - std::chrono::seconds(5); // 确保第一帧立即查一次

    // [New] 热力图初始化 (32F 用于累积计数)
    cv::Mat g_heatmap = cv::Mat::zeros(1080, 3840, CV_32FC1);

    // 获取当天的起始时间 (今天零点)
    time_t t = time(nullptr);
#pragma warning(disable : 4996) // 禁用 localtime 安全警告 (或者改用 localtime_s)
    tm* now_tm = localtime(&t);
    char start_time_buf[32];
    sprintf(start_time_buf, "%04d-%02d-%02d 00:00:00", now_tm->tm_year + 1900, now_tm->tm_mon + 1, now_tm->tm_mday);
    std::string today_start = start_time_buf;
    std::string future_end = "2099-12-31 23:59:59";
    // ==================================================================

    // ================= MAIN LOOP =================
    while (true) {
        auto loop_start = std::chrono::high_resolution_clock::now();

        // A. 获取源视频
        for (int i = 0; i < 3; i++) streams[i]->getFrame(current_frames[i]);

        // B. GPU 拼接处理
        for (int i = 0; i < 3; i++) {
            auto& buf = gpu_buffers[i];
            const auto& cfg = configs[i];
            auto& st = buf.stream;
            if (current_frames[i].empty()) continue;

            buf.raw_gpu.upload(current_frames[i], st);

            if (buf.raw_gpu.cols != 1920) cv::cuda::resize(buf.raw_gpu, buf.resized_gpu, cv::Size(1920, 1080), 0, 0, cv::INTER_LINEAR, st);
            else buf.raw_gpu.copyTo(buf.resized_gpu, st);

            cv::cuda::remap(buf.resized_gpu, buf.undistorted_gpu, buf.map_x, buf.map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(), st);
            if (i == 0) {
                cv::cuda::remap(buf.undistorted_gpu, buf.resized_gpu, buf.map_x, buf.map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(), st);
                cv::cuda::GpuMat temp = buf.undistorted_gpu; buf.undistorted_gpu = buf.resized_gpu; buf.resized_gpu = temp;
            }

            cv::cuda::GpuMat cropped = buf.undistorted_gpu(cfg.crop_rect);
            if (cropped.size() != cfg.final_size) cv::cuda::resize(cropped, buf.final_gpu, cfg.final_size, 0, 0, cv::INTER_LINEAR, st);
            else buf.final_gpu = cropped;

            buf.final_gpu.copyTo(canvas_gpu(cfg.canvas_roi), st);
        }

        // C. 下载到 CPU 内存
        for (int i = 0; i < 3; i++) gpu_buffers[i].stream.waitForCompletion();
        canvas_gpu.download(pinned_mem);
        cv::Mat canvas_cpu = pinned_mem.createMatHeader();

        // D. 绘制白色遮罩
        cv::rectangle(canvas_cpu, cv::Rect(0, 0, 3840, 239), cv::Scalar(255, 255, 255), -1);
        cv::rectangle(canvas_cpu, cv::Rect(0, 850, 3840, 330), cv::Scalar(255, 255, 255), -1);
        // 额外局部遮罩：覆盖左下和右下红圈区域
        cv::rectangle(canvas_cpu, cv::Rect(0, 760, 820, 190), cv::Scalar(255, 255, 255), -1);      // 左下
        cv::rectangle(canvas_cpu, cv::Rect(2860, 620, 980, 330), cv::Scalar(255, 255, 255), -1);    // 右下
        cv::rectangle(canvas_cpu, cv::Rect(2860, 239, 3840, 39), cv::Scalar(255, 255, 255), -1);
        if (!canvas_cpu.empty()) {
            // E. AI 检测与入库
            std::vector<AVSAnalyzer::DetectObject> fish_detects;
            bool detect_ok = fish_detector->objectDetect(canvas_cpu, fish_detects);

            // [New] 热力图更新与推流 (替换原纯净流)
            if (detect_ok) {
                // 1. 累积热力值 (点半径增大至 3)
                // 优化：降低打点频率，每5帧打一次，防止堆积过快
                static int heatmap_frame_count = 0;
                heatmap_frame_count++;
                
                if (heatmap_frame_count % 5 == 0) {
                    for (const auto& fish : fish_detects) {
                        float cx = (fish.x1 + fish.x2) / 2.0f;
                        float cy = (fish.y1 + fish.y2) / 2.0f;
                        // radius=3 (用户要求稍大), value=1.0
                        cv::circle(g_heatmap, cv::Point((int)cx, (int)cy), 3, cv::Scalar(1.0), -1);
                    }
                }

                // 2. 生成热力图可视化
                cv::Mat heatmap_norm, heatmap_color, heatmap_overlay;
                
                // [Change] 调整缩放比例: 1 hit = 40 (绿色), ~6 hits = 240 (红色)
                g_heatmap.convertTo(heatmap_norm, CV_8U, 100.0);
                
                // [New] 高斯模糊：让点之间融合，形成真正的“密度”感
                cv::GaussianBlur(heatmap_norm, heatmap_norm, cv::Size(9, 9), 0);
                
                // [Fix] 直接创建 256x1 的 LUT，避免 reshape 问题
                cv::Mat lut(256, 1, CV_8UC3);
                for (int i = 0; i < 256; i++) {
                    if (i == 0) {
                        lut.at<cv::Vec3b>(i, 0) = cv::Vec3b(0, 0, 0); // 背景
                    }
                    else if (i < 50) {
                        // 绿 -> 黄 (Red 增加, Green 保持 255)
                        int r = (int)(i * 30.0);
                        lut.at<cv::Vec3b>(i, 0) = cv::Vec3b(0, 255, r);
                    }
                    else {
                        // 黄 -> 红 (Green 减少, Red 保持 255)
                        int g = (int)(255 - (i - 128) * 200.0);
                        lut.at<cv::Vec3b>(i, 0) = cv::Vec3b(0, g, 255);
                    }
                }

                // 应用自定义颜色映射
                try {
                    cv::applyColorMap(heatmap_norm, heatmap_color, lut);
                }
                catch (const cv::Exception& e) {
                    std::cerr << "[Error] applyColorMap failed: " << e.what() << std::endl;
                    // Fallback to built-in colormap
                    cv::applyColorMap(heatmap_norm, heatmap_color, cv::COLORMAP_HOT);
                }

                // 3. 叠加到原图
                heatmap_overlay = canvas_cpu.clone();
                // 只叠加非黑色的部分 (mask)
                cv::Mat mask;
                cv::cvtColor(heatmap_color, mask, cv::COLOR_BGR2GRAY);
                cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY);
                heatmap_color.copyTo(heatmap_overlay, mask);

                // 4. 推送热力图流
                writer_clean.push(heatmap_overlay);
            }
            else {
                // 如果没有检测或检测失败，推送上一帧的热力图或者原图
                // 这里简单处理：如果没有检测结果，仍然生成热力图(静态)并推送
                // 但为了流畅性，我们只在 detect_ok 时更新 g_heatmap，
                // 若 detect_ok=false (极少见)，可选择不推或推上一帧。
                // 鉴于 canvas_cpu 始终有值，我们仍然执行热力图渲染逻辑(虽然没有新点增加)
                
                // (代码复用逻辑优化：将热力图生成移出 if(detect_ok) 块？
                //  不，因为 fish_detects 依赖 detect_ok。
                //  如果 detect_ok 失败，我们依然可以渲染旧的 heatmap)
                
                // 简单起见，仅在 detect_ok 时更新。若要持续推流，需重构。
                // 考虑到 objectDetect 几乎总是返回 true (除非 tensorrt 崩溃)，此处逻辑尚可。
            }

            if (detect_ok) {
                cv::Mat frame_before_draw = canvas_cpu.clone();
                const float capture_conf_threshold = 0.55f; // 仅用于抓拍，不影响通过率统计
                const int capture_min_bbox_area = 900;      // 过滤小杂点（约 30x30）
                const bool capture_score_weighted = true;   // 按置信度加权随机抽样
                // 1. 计数与绘制检测框
                for (size_t i = 0; i < calculators.size(); i++) {
                    AVSAnalyzer::FinshPrecent* calc = calculators[i];
                    const PoolConfig& cfg = pool_configs[i];

                    // 核心计数更新
                    calc->countPrecent(fish_detects, canvas_cpu);

                    // 随机抓图：从当前池子的检测结果中抽取，默认每 10 秒最多抓 1 张
                    int pool_center_x = (cfg.entry_p1.x + cfg.entry_p2.x + cfg.exit_p1.x + cfg.exit_p2.x) / 4;
                    cv::Rect capture_roi;
                    if (pool_center_x < 1920) capture_roi = cv::Rect(0, 0, 1920, 1080);
                    else if (pool_center_x < 2880) capture_roi = cv::Rect(1920, 0, 960, 1080);
                    else capture_roi = cv::Rect(2880, 0, 960, 1080);
                    calc->TryCaptureRandomFish(
                        fish_detects, frame_before_draw, capture_roi, 10,
                        capture_conf_threshold, capture_min_bbox_area, capture_score_weighted
                    );

                    // 绘制区域线
                    cv::line(canvas_cpu, cfg.entry_p1, cfg.entry_p2, cv::Scalar(0, 0, 255), 2);
                    cv::line(canvas_cpu, cfg.exit_p1, cfg.exit_p2, cv::Scalar(255, 0, 0), 2);
                }

                // 绘制鱼的检测框
                for (auto& fish : fish_detects) {
                    cv::rectangle(canvas_cpu, cv::Point(fish.x1, fish.y1), cv::Point(fish.x2, fish.y2), cv::Scalar(0, 255, 0), 2);
                }
            }
        }

        // G. 推流 (OSD画面)
        writer.push(canvas_cpu);

        // H. 精准控帧 (30FPS)
        while (true) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(now - loop_start).count();
            if (elapsed >= target_frame_time_ms) break;
            std::this_thread::yield();
        }
    }

    // 资源清理
    delete fish_detector;
    for (auto* calc : calculators) delete calc;

    return 0;
}
*/
/**
 * 项目名：FishPass Integrated System (Ultimate Edition)
 * 功能：3路RTSP拼接 -> 去畸变 -> AI检测 -> 计数入库 -> OSD实时显示 -> 结果推流
 */

#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp> 
#include <opencv2/cudaimgproc.hpp> 
#include <opencv2/cudaarithm.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <cstdio>
#include <ctime>
#include <chrono>
#include <filesystem>
#define NOMINMAX
#include <windows.h>


 // ================== 核心头文件引入 ==================
#include "DBHelper.h"       // 数据库功能
#include "PoolConfig.h"     // 池室配置
#include "FinshPass.h"      // TensorRT 检测模型
#include "FinshPrecent.h"   // 业务计数逻辑
// ==============================================================

using namespace std;
namespace fs = std::filesystem;

// ================== [模式切换配置] ==================
// 开关：true 代表单流模式，false 代表原版三流拼接模式
const bool USE_SINGLE_STREAM = false;
// 单流模式下的 RTSP 测试地址
const std::string SINGLE_STREAM_URL = "rtsp://127.0.0.1:9994/live/liu";

// ====================================================

static fs::path GetExecutableDir() {
    char buffer[MAX_PATH] = {};
    DWORD len = GetModuleFileNameA(nullptr, buffer, MAX_PATH);
    if (len == 0 || len == MAX_PATH) {
        return fs::current_path();
    }
    return fs::path(buffer).parent_path();
}

static std::string ResolveExistingPath(const std::vector<fs::path>& candidates, const std::string& fallback) {
    for (const auto& candidate : candidates) {
        if (fs::exists(candidate)) {
            return candidate.string();
        }
    }
    return fallback;
}

// 全局原子标志位：防止查询时日志干扰
std::atomic<bool> g_is_querying(false);

// --- [配置部分] 加载池室检测区域 ---
std::vector<PoolConfig> LoadPoolConfigs() {
    std::vector<PoolConfig> configs;
   
    // 池室 1
    configs.emplace_back(
        2, "Pool_Channel_1",
        cv::Point(1400, 469), cv::Point(1400, 569), // 入口线 (线 4)
        cv::Point(1515, 469), cv::Point(1515, 569), // 出口线 (线 5)
        0, 0
    );
    // 池室 2
    configs.emplace_back(
        1, "Pool_Channel_2",
        cv::Point(2178, 458), cv::Point(2178, 578), // 入口线 (线 2)
        cv::Point(2488, 458), cv::Point(2488, 578), // 出口线 (线 3)
        0, 0
    );
    return configs;
}

// --- [查询部分] 后台多线程查询交互 ---
void QueryThreadFunc(std::vector<PoolConfig> pool_configs) {
    std::string start_time, end_time;
    std::cout << "\n>>> [后台模式] 进入查询系统 (视频拼接与检测仍在极速运行...) <<<" << std::endl;

    if (std::cin.rdbuf()->in_avail() > 0) std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.clear();

    std::cout << "请输入开始时间 (如 2026-01-24 00:00:00): " << std::endl;
    std::getline(std::cin, start_time);
    if (start_time.empty()) start_time = "2026-01-24 00:00:00";

    std::cout << "请输入结束时间 (如 2026-01-24 23:59:59): " << std::endl;
    std::getline(std::cin, end_time);
    if (end_time.empty()) end_time = "2026-01-24 23:59:59";

    std::cout << "正在查询数据库 [" << start_time << " ~ " << end_time << "] ..." << std::endl;

    for (const auto& cfg : pool_configs) {
        PoolStats stats = DBHelper::GetInstance().QueryPassRateByTime(cfg.id, start_time, end_time);
        std::cout << "--------------------------------" << std::endl;
        std::cout << "池室: " << cfg.name << " (ID:" << cfg.id << ")" << std::endl;
        std::cout << "  - [分母] 起点净流量: " << stats.start_net_flow << std::endl;
        std::cout << "  - [分子] 终点净流量: " << stats.end_net_flow << std::endl;
        if (stats.is_valid) printf("  - ★ 通过率: %.2f%%\n", stats.pass_rate * 100.0);
        else std::cout << "  - ★ 通过率: -- (分母为0)" << std::endl;
    }

    std::cout << "\n是否查看详细流水? (y/n): ";
    std::string choice;
    std::getline(std::cin, choice);
    if (choice == "y" || choice == "Y") {
        for (const auto& cfg : pool_configs) {
            DBHelper::GetInstance().QueryLogsByTime(cfg.id, start_time, end_time);
        }
    }

    std::cout << "查询完毕！按回车键退出..." << std::endl;
    std::string temp;
    std::getline(std::cin, temp);
    g_is_querying = false;
    std::cout << ">>> 已退出查询，恢复日志 <<<" << std::endl;
}

// --- [拼接部分] 结构体与类定义 ---
struct StreamGpuBuffer {
    cv::cuda::GpuMat raw_gpu;
    cv::cuda::GpuMat resized_gpu;
    cv::cuda::GpuMat undistorted_gpu;
    cv::cuda::GpuMat final_gpu;
    cv::cuda::GpuMat map_x;
    cv::cuda::GpuMat map_y;
    cv::cuda::Stream stream;
};

struct StreamConfig {
    std::string url;
    int target_w, target_h;
    cv::Mat K, D;
    cv::Rect crop_rect;
    cv::Size final_size;
    cv::Rect canvas_roi;
};

class RTSPStream {
public:
    RTSPStream(std::string url) : url_(url), running_(true), has_frame_(false) {
        worker_ = std::thread(&RTSPStream::update, this);
    }
    ~RTSPStream() { stop(); if (worker_.joinable()) worker_.join(); }
    void stop() { running_ = false; }
    bool getFrame(cv::Mat& out_frame) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!has_frame_) return false;
        last_frame_.copyTo(out_frame);
        return true;
    }
private:
    void update() {
        cv::VideoCapture cap;
        cap.open(url_, cv::CAP_FFMPEG);
        cv::Mat temp;
        while (running_) {
            if (cap.isOpened()) {
                if (cap.read(temp) && !temp.empty()) {
                    std::lock_guard<std::mutex> lock(mtx_);
                    temp.copyTo(last_frame_);
                    has_frame_ = true;
                }
                else {
                    cap.release();
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                }
            }
            else {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                cap.open(url_, cv::CAP_FFMPEG);
            }
        }
    }
    std::string url_;
    std::thread worker_;
    std::atomic<bool> running_;
    std::mutex mtx_;
    cv::Mat last_frame_;
    bool has_frame_;
};

class AsyncWriter {
public:
    AsyncWriter(const std::string& cmd) : running_(true) {
        pipe_ = _popen(cmd.c_str(), "wb");
        worker_ = std::thread(&AsyncWriter::process_queue, this);
    }
    ~AsyncWriter() { running_ = false; if (worker_.joinable()) worker_.join(); if (pipe_) _pclose(pipe_); }
    void push(const cv::Mat& frame) {
        if (!running_) return;
        std::lock_guard<std::mutex> lock(mtx_);
        if (queue_.size() < 4) queue_.push(frame.clone());
    }
private:
    void process_queue() {
        while (running_) {
            cv::Mat frame;
            bool has_data = false;
            {
                std::lock_guard<std::mutex> lock(mtx_);
                if (!queue_.empty()) { frame = queue_.front(); queue_.pop(); has_data = true; }
            }
            if (has_data) fwrite(frame.data, 1, frame.total() * frame.elemSize(), pipe_);
            else std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    FILE* pipe_;
    std::thread worker_;
    std::atomic<bool> running_;
    std::mutex mtx_;
    std::queue<cv::Mat> queue_;
};

std::vector<StreamConfig> get_configs() {
    std::vector<StreamConfig> configs;
    {
        StreamConfig cfg;
        cfg.url = "rtsp://127.0.0.1:9994/live/fish1";
        float k[] = { 3632.08770f, 0, 932.18005f, 0, 3736.83206f, 530.14428f, 0, 0, 1 };
        float d[] = { -1.2936f, -9.3598f, 0.0183f, 0.0860f, 59.2869f };
        cfg.K = cv::Mat(3, 3, CV_32F, k).clone(); cfg.D = cv::Mat(1, 5, CV_32F, d).clone();
        cfg.crop_rect = cv::Rect(0, 0, 1920, 824);
        cfg.final_size = cv::Size(1920, 1080);
        cfg.canvas_roi = cv::Rect(0, 0, 1920, 1080);
        configs.push_back(cfg);
    }
    {
        StreamConfig cfg;
        cfg.url = "rtsp://127.0.0.1:9994/live/fish2";
        float k[] = { 3641.99471f, 0, 977.13310f, 0, 7186.56110f, 539.32540f, 0, 0, 1 };
        float d[] = { -2.0241f, -5.6109f, -0.0280f, -0.0056f, 53.5588f };
        cfg.K = cv::Mat(3, 3, CV_32F, k).clone(); cfg.D = cv::Mat(1, 5, CV_32F, d).clone();
        cfg.crop_rect = cv::Rect(589, 116, 850, 824);
        cfg.final_size = cv::Size(960, 1080);
        cfg.canvas_roi = cv::Rect(1920, 0, 960, 1080);
        configs.push_back(cfg);
    }
    {
        StreamConfig cfg;
        cfg.url = "rtsp://127.0.0.1:9994/live/fish3";
        float k[] = { 1480.34081f, 0, 972.90160f, 0, 1483.11981f, 558.28559f, 0, 0, 1 };
        float d[] = { -0.6412f, 0.4056f, 0.0038f, 0.00002f, -0.1189f };
        cfg.K = cv::Mat(3, 3, CV_32F, k).clone(); cfg.D = cv::Mat(1, 5, CV_32F, d).clone();
        cfg.crop_rect = cv::Rect(406, 43, 1104, 824);
        cfg.final_size = cv::Size(960, 1080);
        cfg.canvas_roi = cv::Rect(2880, 0, 960, 1080);
        configs.push_back(cfg);
    }
    return configs;
}

int main() {
    _putenv_s("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000");
    const fs::path exe_dir = GetExecutableDir();
    const fs::path repo_root = exe_dir.parent_path().parent_path();

    // 1. CUDA 初始化
    try {
        if (cv::cuda::getCudaEnabledDeviceCount() == 0) return -1;
        cv::cuda::setDevice(0);
    }
    catch (...) { return -1; }

    // 2. 数据库连接
    std::cout << "[Init] 连接数据库..." << std::endl;
    if (DBHelper::GetInstance().Connect("127.0.0.1", "root", "123456", "fish_db")) {
        std::cout << "数据库连接成功！" << std::endl;
    }
    else {
        std::cerr << "警告：数据库连接失败！" << std::endl;
    }

    // 3. 模型与算法初始化
    std::cout << "[Init] 初始化 AI 模型..." << std::endl;
    const std::string model_path = ResolveExistingPath({
        exe_dir / "model" / "fishmonitor_lxf_12_9.engine",
        exe_dir.parent_path() / "model" / "fishmonitor_lxf_12_9.engine",
        repo_root / "model" / "fishmonitor_lxf_12_9.engine"
        }, "model\\fishmonitor_lxf_12_9.engine");
    AVSAnalyzer::FinshPass* fish_detector = new AVSAnalyzer::FinshPass(model_path);
    std::cout << "模型加载成功！" << std::endl;

    std::vector<PoolConfig> pool_configs = LoadPoolConfigs();
    std::vector<AVSAnalyzer::FinshPrecent*> calculators;
    for (const auto& cfg : pool_configs) {
        calculators.push_back(new AVSAnalyzer::FinshPrecent(
            cfg.id, cfg.entry_p1, cfg.entry_p2, cfg.exit_p1, cfg.exit_p2, cfg.direction_in, cfg.direction_out
        ));
    }

    // 4. 拼接推流初始化
    auto configs = get_configs();
    std::string ffmpeg_path = ResolveExistingPath({
        exe_dir / "ffmpeg.exe",
        exe_dir.parent_path() / "ffmpeg.exe",
        repo_root / "3rdparty" / "ffmpeg" / "bin" / "ffmpeg.exe"
        }, "ffmpeg");

    // [Stream 1] OSD 处理后的流
    std::string cmd = "\"" + ffmpeg_path + "\" -y -f rawvideo -pix_fmt bgr24 -s 3840x1080 -framerate 30 -i - "
        "-vf format=yuv420p "
        "-c:v hevc_nvenc -preset p1 -tune ull -b:v 6M -maxrate 8M -bufsize 4M -g 60 "
        "-f rtsp -rtsp_transport tcp rtsp://127.0.0.1:9994/live/output";
    AsyncWriter writer(cmd);

    // [Stream 2] 纯净流 (无OSD)
    std::string cmd_clean = "\"" + ffmpeg_path + "\" -y -f rawvideo -pix_fmt bgr24 -s 3840x1080 -framerate 30 -i - "
        "-vf format=yuv420p "
        "-c:v hevc_nvenc -preset p1 -tune ull -b:v 6M -maxrate 8M -bufsize 4M -g 60 "
        "-f rtsp -rtsp_transport tcp rtsp://127.0.0.1:9994/live/output_origin";
    AsyncWriter writer_clean(cmd_clean);

    // ==========================================
    // 5. 初始化视频流读取 (支持单流/多流快速切换)
    // ==========================================
    std::vector<std::unique_ptr<RTSPStream>> streams;
    if (USE_SINGLE_STREAM) {
        streams.push_back(std::unique_ptr<RTSPStream>(new RTSPStream(SINGLE_STREAM_URL)));
    }
    else {
        for (const auto& cfg : configs) streams.push_back(std::unique_ptr<RTSPStream>(new RTSPStream(cfg.url)));
    }

    // CUDA 缓冲区与内存初始化
    std::vector<StreamGpuBuffer> gpu_buffers(3);
    cv::cuda::HostMem pinned_mem(1080, 3840, CV_8UC3, cv::cuda::HostMem::PAGE_LOCKED);
    cv::cuda::GpuMat canvas_gpu(1080, 3840, CV_8UC3);

    // 仅在多流模式下初始化拼接对应的 map 参数
    if (!USE_SINGLE_STREAM) {
        for (int i = 0; i < 3; i++) {
            const auto& cfg = configs[i];
            cv::Mat map1, map2;
            cv::Mat new_K = cv::getOptimalNewCameraMatrix(cfg.K, cfg.D, cv::Size(1920, 1080), 0);
            cv::initUndistortRectifyMap(cfg.K, cfg.D, cv::Mat(), new_K, cv::Size(1920, 1080), CV_32FC1, map1, map2);
            gpu_buffers[i].map_x.upload(map1);
            gpu_buffers[i].map_y.upload(map2);
        }
    }

    std::cout << "[System Ready] 系统运行中 (当前模式: "
        << (USE_SINGLE_STREAM ? "单流测试" : "GPU多流拼接") << ")..." << std::endl;

    const double target_frame_time_ms = 1000.0 / 30.0;
    std::vector<cv::Mat> current_frames(3);
    for (int i = 0; i < 3; i++) current_frames[i] = cv::Mat::zeros(1080, 1920, CV_8UC3);

    // OSD 实时显示缓存与时间初始化
    std::vector<PoolStats> g_realtime_stats(pool_configs.size());
    auto last_query_time = std::chrono::steady_clock::now() - std::chrono::seconds(5);

    // 热力图初始化
    cv::Mat g_heatmap = cv::Mat::zeros(1080, 3840, CV_32FC1);

    time_t t = time(nullptr);
#pragma warning(disable : 4996)
    tm* now_tm = localtime(&t);
    char start_time_buf[32];
    sprintf(start_time_buf, "%04d-%02d-%02d 00:00:00", now_tm->tm_year + 1900, now_tm->tm_mon + 1, now_tm->tm_mday);
    std::string today_start = start_time_buf;
    std::string future_end = "2099-12-31 23:59:59";

    // ================= MAIN LOOP =================
    while (true) {
        auto loop_start = std::chrono::high_resolution_clock::now();
        cv::Mat canvas_cpu;

        if (USE_SINGLE_STREAM) {
            // ---------------- [单流模式] ----------------
            cv::Mat single_frame;
            streams[0]->getFrame(single_frame);
            if (single_frame.empty()) continue;

            // 安全处理：如果尺寸不是 3840x1080，进行等比例缩放 + 填充 (Letterbox)
            if (single_frame.cols != 3840 || single_frame.rows != 1080) {
                // 创建一个纯黑的 3840x1080 画布
                canvas_cpu = cv::Mat::zeros(1080, 3840, CV_8UC3);

                // 计算等比例缩放系数
                float scale = std::min(3840.0f / single_frame.cols, 1080.0f / single_frame.rows);
                int new_w = static_cast<int>(single_frame.cols * scale);
                int new_h = static_cast<int>(single_frame.rows * scale);

                // 进行等比例缩放
                cv::Mat resized_frame;
                cv::resize(single_frame, resized_frame, cv::Size(new_w, new_h));

                // 计算居中放置的偏移量
                int x_offset = (3840 - new_w) / 2;
                int y_offset = (1080 - new_h) / 2;

                // 将缩放后的画面贴到 3840x1080 画布的中央
                resized_frame.copyTo(canvas_cpu(cv::Rect(x_offset, y_offset, new_w, new_h)));
            }
            else {
                canvas_cpu = single_frame.clone();
            }
        }
        else {
            // -------------- [多流拼接模式] --------------
            // -------------- [多流拼接模式] --------------
            // A. 获取源视频
            for (int i = 0; i < 3; i++) streams[i]->getFrame(current_frames[i]);

            // B. GPU 拼接处理
            for (int i = 0; i < 3; i++) {
                auto& buf = gpu_buffers[i];
                const auto& cfg = configs[i];
                auto& st = buf.stream;
                if (current_frames[i].empty()) continue;

                buf.raw_gpu.upload(current_frames[i], st);

                if (buf.raw_gpu.cols != 1920) cv::cuda::resize(buf.raw_gpu, buf.resized_gpu, cv::Size(1920, 1080), 0, 0, cv::INTER_LINEAR, st);
                else buf.raw_gpu.copyTo(buf.resized_gpu, st);

                cv::cuda::remap(buf.resized_gpu, buf.undistorted_gpu, buf.map_x, buf.map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(), st);
                if (i == 0) {
                    cv::cuda::remap(buf.undistorted_gpu, buf.resized_gpu, buf.map_x, buf.map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(), st);
                    cv::cuda::GpuMat temp = buf.undistorted_gpu; buf.undistorted_gpu = buf.resized_gpu; buf.resized_gpu = temp;
                }

                cv::cuda::GpuMat cropped = buf.undistorted_gpu(cfg.crop_rect);
                if (cropped.size() != cfg.final_size) cv::cuda::resize(cropped, buf.final_gpu, cfg.final_size, 0, 0, cv::INTER_LINEAR, st);
                else buf.final_gpu = cropped;

                buf.final_gpu.copyTo(canvas_gpu(cfg.canvas_roi), st);
            }

            // C. 下载到 CPU 内存
            for (int i = 0; i < 3; i++) gpu_buffers[i].stream.waitForCompletion();
            canvas_gpu.download(pinned_mem);
            canvas_cpu = pinned_mem.createMatHeader().clone(); // 克隆以防内存被覆写
        }

        // D. 绘制白色遮罩 (安全绘制闭包：防止切换流后因为尺寸引发越界断言)
        auto drawSafeMask = [&](cv::Rect r) {
            cv::Rect imgRect(0, 0, canvas_cpu.cols, canvas_cpu.rows);
            cv::Rect intersect = r & imgRect;
            if (intersect.width > 0 && intersect.height > 0) {
                cv::rectangle(canvas_cpu, intersect, cv::Scalar(255, 255, 255), -1);
            }
            };

        drawSafeMask(cv::Rect(0, 0, 3840, 239));
        drawSafeMask(cv::Rect(0, 850, 3840, 330));
        drawSafeMask(cv::Rect(0, 760, 820, 190));
        drawSafeMask(cv::Rect(2860, 620, 980, 330));
        drawSafeMask(cv::Rect(2860, 239, 3840, 39));

        if (!canvas_cpu.empty()) {
            // E. AI 检测与入库
            std::vector<AVSAnalyzer::DetectObject> fish_detects;
            bool detect_ok = fish_detector->objectDetect(canvas_cpu, fish_detects);

            // 热力图更新与推流
            if (detect_ok) {
                static int heatmap_frame_count = 0;
                heatmap_frame_count++;

                if (heatmap_frame_count % 5 == 0) {
                    for (const auto& fish : fish_detects) {
                        float cx = (fish.x1 + fish.x2) / 2.0f;
                        float cy = (fish.y1 + fish.y2) / 2.0f;
                        cv::circle(g_heatmap, cv::Point((int)cx, (int)cy), 3, cv::Scalar(1.0), -1);
                    }
                }

                cv::Mat heatmap_norm, heatmap_color, heatmap_overlay;
                g_heatmap.convertTo(heatmap_norm, CV_8U, 100.0);
                cv::GaussianBlur(heatmap_norm, heatmap_norm, cv::Size(9, 9), 0);

                cv::Mat lut(256, 1, CV_8UC3);
                for (int i = 0; i < 256; i++) {
                    if (i == 0) {
                        lut.at<cv::Vec3b>(i, 0) = cv::Vec3b(0, 0, 0);
                    }
                    else if (i < 50) {
                        int r = (int)(i * 30.0);
                        lut.at<cv::Vec3b>(i, 0) = cv::Vec3b(0, 255, r);
                    }
                    else {
                        int g = (int)(255 - (i - 128) * 200.0);
                        lut.at<cv::Vec3b>(i, 0) = cv::Vec3b(0, g, 255);
                    }
                }

                try {
                    cv::applyColorMap(heatmap_norm, heatmap_color, lut);
                }
                catch (const cv::Exception& e) {
                    std::cerr << "[Error] applyColorMap failed: " << e.what() << std::endl;
                    cv::applyColorMap(heatmap_norm, heatmap_color, cv::COLORMAP_HOT);
                }

                heatmap_overlay = canvas_cpu.clone();
                cv::Mat mask;
                cv::cvtColor(heatmap_color, mask, cv::COLOR_BGR2GRAY);
                cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY);
                heatmap_color.copyTo(heatmap_overlay, mask);

                writer_clean.push(heatmap_overlay);
            }

            if (detect_ok) {
                cv::Mat frame_before_draw = canvas_cpu.clone();
                const float capture_conf_threshold = 0.55f;
                const int capture_min_bbox_area = 900;
                const bool capture_score_weighted = true;

                for (size_t i = 0; i < calculators.size(); i++) {
                    AVSAnalyzer::FinshPrecent* calc = calculators[i];
                    const PoolConfig& cfg = pool_configs[i];

                    // 1. 获取通过率和平均滞留时间 (原 countPrecent 返回的就是通过率 0.0~1.0)
                    double pass_rate = calc->countPrecent(fish_detects, canvas_cpu);
                    double avg_time = calc->getAveragePassageTime();

                    int pool_center_x = (cfg.entry_p1.x + cfg.entry_p2.x + cfg.exit_p1.x + cfg.exit_p2.x) / 4;
                    cv::Rect capture_roi;
                    if (pool_center_x < 1920) capture_roi = cv::Rect(0, 0, 1920, 1080);
                    else if (pool_center_x < 2880) capture_roi = cv::Rect(1920, 0, 960, 1080);
                    else capture_roi = cv::Rect(2880, 0, 960, 1080);

                    calc->TryCaptureRandomFish(
                        fish_detects, frame_before_draw, capture_roi, 10,
                        capture_conf_threshold, capture_min_bbox_area, capture_score_weighted
                    );

                    cv::line(canvas_cpu, cfg.entry_p1, cfg.entry_p2, cv::Scalar(0, 0, 255), 2);
                    cv::line(canvas_cpu, cfg.exit_p1, cfg.exit_p2, cv::Scalar(255, 0, 0), 2);

                    // ================= [新增] 绘制 OSD 统计信息 =================
                    char osd_text[256];
                    // 格式化输出文本
                    std::sprintf(osd_text, "%s Pass: %.1f%%, Avg Time: %.1fs",
                        cfg.name.c_str(), pass_rate * 100.0, avg_time);

                    // 直接按序号分配左右位置：第一个池室在左，第二个池室在右
                    cv::Point text_pos;
                    if (i == 0) {
                        text_pos = cv::Point(80, 80);         // 左上角
                    }
                    else {
                        text_pos = cv::Point(3840 - 1000, 80); // 右上角 (往左多挪一点，防文字太长出界)
                    }

                    // 计算文字尺寸，以绘制背景黑框(提高可读性)
                    int baseline = 0;
                    double font_scale = 1.2;
                    int thickness = 3;
                    cv::Size text_size = cv::getTextSize(osd_text, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);

                    cv::Rect text_bg(text_pos.x - 10, text_pos.y - text_size.height - 10,
                        text_size.width + 20, text_size.height + 20);

                    // 安全截断背景框，防止越界崩溃
                    text_bg &= cv::Rect(0, 0, canvas_cpu.cols, canvas_cpu.rows);
                    if (text_bg.area() > 0) {
                        // 绘制纯黑背景块
                        cv::rectangle(canvas_cpu, text_bg, cv::Scalar(0, 0, 0), -1);
                    }

                    // 绘制绿色文字
                    cv::putText(canvas_cpu, osd_text, text_pos, cv::FONT_HERSHEY_SIMPLEX,
                        font_scale, cv::Scalar(0, 255, 0), thickness, cv::LINE_AA);
                    // ============================================================
                }

                for (auto& fish : fish_detects) {
                    cv::rectangle(canvas_cpu, cv::Point(fish.x1, fish.y1), cv::Point(fish.x2, fish.y2), cv::Scalar(0, 255, 0), 2);
                }
            }
        }

        // G. 推流 (OSD画面)
        writer.push(canvas_cpu);
        
        // 保存为当前目录下的 output_record.mp4，30帧，分辨率为你画布的大小
      
        // H. 精准控帧 (30FPS)
        while (true) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(now - loop_start).count();
            if (elapsed >= target_frame_time_ms) break;
            std::this_thread::yield();
        }
    }

    delete fish_detector;
    for (auto* calc : calculators) delete calc;

    return 0;
}