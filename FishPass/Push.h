#pragma once 
#include <queue>
#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <string>
#include "opencv2/opencv.hpp"

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
}

namespace AVSAnalyzer {
    class Push {
    public:
        Push();
        ~Push();

        bool connect(std::string Push_Url); // 连接流媒体服务
        bool reConnect(); // 重新连接流媒体服务
        bool closeConnect(); // 关闭流媒体服务连接
        std::atomic<int> mConnectCount{ 0 }; // 重新连接次数（非静态，类内初始化）

        void addVideoFrame(cv::Mat& frame); // 储存推流帧的队列
        int getVideoFrameQSize(); // 获取队列帧的大小

        static void encodeVideoThread(void* arg); // 编码视频帧并推流（仅这个需要static）
        void handleEncodeVideo(); // 将视频编码并且推流

        // 启动内部线程开始拉流解码
        void startThread();

        // 停止线程并清理
        void stopThread();

    private:
        // -------------------------- 全部改为非静态成员（归属于类内） --------------------------
        // FFmpeg核心上下文
        AVFormatContext* mFmtCtx = nullptr; // 类内直接初始化
        AVStream* mVideoStream = nullptr;
        int mVideoIndex = -1;
        AVCodecContext* mVideoCodecCtx = nullptr; // 非静态
        SwsContext* mSwsCtx = nullptr; // 非静态
        std::thread mEncodeThread; // 非静态
        std::atomic<bool> mIsRunning{ false }; // 原子变量类内初始化（C++11+支持）
        std::atomic<bool> mIsConnected{ false };
        std::string mPushUrl = ""; // 非静态
        int mLastWidth = 0;
        int mLastHeight = 0;

        // 视频帧队列
        std::queue<cv::Mat> mVideoFrameQ; // 储存视频帧的队列
        std::mutex mVideoFrameQMutex; // 互斥锁
        std::condition_variable mVideoFrameCV; // 条件变量（非静态）

        // 私有方法
        bool getVideoFrame(cv::Mat& frame); // 获取推流帧数据
        void clearVideoFrameQueue(); // 清空视频队列
        bool opencv_bgr24ToYuv420p(unsigned char* bgrBuf, int width, int height, unsigned char* yuvBuf);
    };
}