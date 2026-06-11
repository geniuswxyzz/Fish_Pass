#pragma once
#pragma once

#include <string>
#include <thread>
#include <mutex>
#include <atomic>

#include <opencv2/opencv.hpp>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

namespace AVSAnalyzer {
    class FFmpegPull {
    public:
        FFmpegPull();
        ~FFmpegPull();

        // 初始化（打开 url，准备解码器等）
        // 返回 0 表示成功
        int init(const std::string& url);

        // 视频参数
        int video_width;
        int video_height;

        // 启动内部线程开始拉流解码（异步）
        void startThread();

        // 停止线程并清理
        void stopThread();

        // 主线程获取最新一帧（BGR），返回 true 表示获取到
        bool getFrame(cv::Mat& out);

        // 可选：查询是否正在运行
        bool isRunning() const { return running.load(); }

    private:
        // 线程函数（内部）
        void pullLoop();

        // 初始化codec ctx
        int init_codec(AVCodecContext** codec_ctx, AVStream* stream);

        // 解码一个包并把最新帧写入共享缓冲
        int decode_packet(AVPacket* pkt, AVFrame* frame);

        // SwsContext（如果分辨率或格式改变）
        bool ensure_swsctx(int src_w, int src_h, AVPixelFormat src_fmt);

    private:
        // FFmpeg
        AVFormatContext* fmt_ctx = nullptr;
        AVCodecContext* v_codec_ctx = nullptr;
        int video_stream_idx = -1;

        // sws
        SwsContext* sws_ctx = nullptr;
        int sws_src_w = 0, sws_src_h = 0;
        AVPixelFormat sws_src_fmt = AV_PIX_FMT_NONE;

        // Threading
        std::thread pull_thread;
        std::atomic<bool> running{ false };

        // Shared latest frame
        cv::Mat latest_frame;
        std::mutex frame_mutex;

        // store url for potential restart
        std::string source_url;
    };
}
