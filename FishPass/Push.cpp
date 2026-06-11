#include "Push.h"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>

// FFmpeg 相关头文件（必须extern "C"包裹）
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <libavutil/time.h>
}

namespace AVSAnalyzer {
    // -------------------------- 构造函数（你原有代码，优化） --------------------------
    Push::Push() {
        // 静态成员已在类内初始化，这里只需初始化FFmpeg网络模块
        avformat_network_init();
    }

    // -------------------------- 析构函数（修正FFmpeg版本问题，补充资源释放） --------------------------
    Push::~Push() {
        // 1. 先停止编码线程
        stopThread();

        // 2. 关闭推流连接
        closeConnect();

        // 3. 释放FFmpeg核心资源（修正codec->codecpar问题）
        if (mFmtCtx) {
            // 推流场景：写入流尾，保证接收端正常解析
            if (mFmtCtx->oformat) {
                av_write_trailer(mFmtCtx);
            }
            avformat_close_input(&mFmtCtx);
            avformat_free_context(mFmtCtx);
            mFmtCtx = nullptr;
        }

        // 4. 释放编码器上下文（单独管理，替代mVideoStream->codec）
        if (mVideoCodecCtx) {
            avcodec_close(mVideoCodecCtx);
            avcodec_free_context(&mVideoCodecCtx);
            mVideoCodecCtx = nullptr;
        }

        // 5. 释放格式转换上下文（BGR→YUV）
        if (mSwsCtx) {
            sws_freeContext(mSwsCtx);
            mSwsCtx = nullptr;
        }

        // 6. 清空帧队列
        clearVideoFrameQueue();

        // 7. 置空辅助变量
        mVideoStream = nullptr;
        mVideoIndex = -1;

        // 清理FFmpeg网络模块
        avformat_network_deinit();
    }

    // -------------------------- 核心：连接推流服务器 --------------------------
    bool Push::connect(std::string Push_Url) {
        // 保存推流URL到类内成员
        mPushUrl = Push_Url;

        // 先关闭旧连接
        if (mIsConnected) {
            closeConnect();
        }

        try {
            // 1. 分配输出格式上下文（自动识别推流协议：RTSP/RTMP/HTTP）
            int ret = avformat_alloc_output_context2(&mFmtCtx, nullptr, "rtsp", Push_Url.c_str());
            if (ret < 0 || !mFmtCtx) {
                char err_buf[AV_ERROR_MAX_STRING_SIZE] = { 0 };
                av_strerror(ret, err_buf, sizeof(err_buf));
                std::cerr << "分配输出上下文失败：" << err_buf << std::endl;
                return false;
            }

            // 2. 创建视频流
            mVideoStream = avformat_new_stream(mFmtCtx, nullptr);
            if (!mVideoStream) {
                std::cerr << "创建视频流失败" << std::endl;
                avformat_free_context(mFmtCtx);
                mFmtCtx = nullptr;
                return false;
            }
            mVideoIndex = mVideoStream->index;

            // 3. 初始化H264编码器（核心：单独管理AVCodecContext）
            const AVCodec* h264_codec = avcodec_find_encoder(AV_CODEC_ID_H264);
            if (!h264_codec) {
                std::cerr << "未找到H264编码器" << std::endl;
                avformat_free_context(mFmtCtx);
                mFmtCtx = nullptr;
                mVideoStream = nullptr;
                return false;
            }

            // 4. 初始化编码器上下文
            mVideoCodecCtx = avcodec_alloc_context3(h264_codec);
            if (!mVideoCodecCtx) {
                std::cerr << "分配编码器上下文失败" << std::endl;
                avformat_free_context(mFmtCtx);
                mFmtCtx = nullptr;
                mVideoStream = nullptr;
                return false;
            }

            // 5. 配置编码器参数（需与输入帧匹配，这里先默认配置）
            mVideoCodecCtx->codec_id = AV_CODEC_ID_H264;
            mVideoCodecCtx->codec_type = AVMEDIA_TYPE_VIDEO;
            mVideoCodecCtx->pix_fmt = AV_PIX_FMT_YUV420P; // H264标准格式
            mVideoCodecCtx->width = 1920; // 默认宽度，实际需替换为帧的宽度
            mVideoCodecCtx->height = 1080; // 默认高度，实际需替换为帧的高度
            mVideoCodecCtx->bit_rate = 2000000; // 码率2Mbps
            mVideoCodecCtx->gop_size = 25; // 关键帧间隔
            mVideoCodecCtx->time_base = { 1, 25 }; // 时间基：25fps
            mVideoCodecCtx->framerate = { 25, 1 };
            mVideoCodecCtx->max_b_frames = 0; // 无B帧，降低延迟
            mVideoCodecCtx->thread_count = 4; // 编码线程数

            // 6. 设置H264编码参数（零延迟、快速编码）
            AVDictionary* codec_opts = nullptr;
            av_dict_set(&codec_opts, "preset", "ultrafast", 0);
            av_dict_set(&codec_opts, "tune", "zerolatency", 0);
            av_dict_set(&codec_opts, "profile", "baseline", 0);

            // 7. 打开编码器
            ret = avcodec_open2(mVideoCodecCtx, h264_codec, &codec_opts);
            av_dict_free(&codec_opts);
            if (ret < 0) {
                char err_buf[AV_ERROR_MAX_STRING_SIZE] = { 0 };
                av_strerror(ret, err_buf, sizeof(err_buf));
                std::cerr << "打开H264编码器失败：" << err_buf << std::endl;
                avcodec_free_context(&mVideoCodecCtx);
                avformat_free_context(mFmtCtx);
                mFmtCtx = nullptr;
                mVideoStream = nullptr;
                return false;
            }

            // 8. 将编码器参数复制到视频流（新版FFmpeg用codecpar）
            ret = avcodec_parameters_from_context(mVideoStream->codecpar, mVideoCodecCtx);
            if (ret < 0) {
                char err_buf[AV_ERROR_MAX_STRING_SIZE] = { 0 };
                av_strerror(ret, err_buf, sizeof(err_buf));
                std::cerr << "复制编码器参数失败：" << err_buf << std::endl;
                avcodec_close(mVideoCodecCtx);
                avcodec_free_context(&mVideoCodecCtx);
                avformat_free_context(mFmtCtx);
                mFmtCtx = nullptr;
                mVideoStream = nullptr;
                return false;
            }

            // 9. 打开推流URL并写入头信息
            if (!(mFmtCtx->oformat->flags & AVFMT_NOFILE)) {
                ret = avio_open(&mFmtCtx->pb, Push_Url.c_str(), AVIO_FLAG_WRITE);
                if (ret < 0) {
                    char err_buf[AV_ERROR_MAX_STRING_SIZE] = { 0 };
                    av_strerror(ret, err_buf, sizeof(err_buf));
                    std::cerr << "打开推流URL失败：" << err_buf << std::endl;
                    avcodec_close(mVideoCodecCtx);
                    avcodec_free_context(&mVideoCodecCtx);
                    avformat_free_context(mFmtCtx);
                    mFmtCtx = nullptr;
                    mVideoStream = nullptr;
                    return false;
                }
            }

            ret = avformat_write_header(mFmtCtx, nullptr);
            if (ret < 0) {
                char err_buf[AV_ERROR_MAX_STRING_SIZE] = { 0 };
                av_strerror(ret, err_buf, sizeof(err_buf));
                std::cerr << "写入流头失败：" << err_buf << std::endl;
                if (!(mFmtCtx->oformat->flags & AVFMT_NOFILE)) {
                    avio_close(mFmtCtx->pb);
                }
                avcodec_close(mVideoCodecCtx);
                avcodec_free_context(&mVideoCodecCtx);
                avformat_free_context(mFmtCtx);
                mFmtCtx = nullptr;
                mVideoStream = nullptr;
                return false;
            }

            // 10. 标记连接成功
            mIsConnected = true;
            mConnectCount++;
            std::cout << "推流连接成功！URL：" << Push_Url << "，重连次数：" << mConnectCount << std::endl;
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "连接推流服务器异常：" << e.what() << std::endl;
            closeConnect();
            return false;
        }
    }

    // -------------------------- 重新连接推流服务器 --------------------------
    bool Push::reConnect() {
        if (mPushUrl.empty()) {
            std::cerr << "推流URL为空，无法重连" << std::endl;
            return false;
        }
        std::cout << "尝试重新连接推流服务器... 当前重连次数：" << mConnectCount + 1 << std::endl;
        return connect(mPushUrl);
    }

    // -------------------------- 关闭推流连接 --------------------------
    bool Push::closeConnect() {
        if (!mIsConnected) {
            return true;
        }

        // 1. 写入流尾
        if (mFmtCtx && mFmtCtx->oformat) {
            av_write_trailer(mFmtCtx);
        }

        // 2. 关闭IO
        if (mFmtCtx && !(mFmtCtx->oformat->flags & AVFMT_NOFILE)) {
            avio_close(mFmtCtx->pb);
        }

        // 3. 释放编码器
        if (mVideoCodecCtx) {
            avcodec_close(mVideoCodecCtx);
            avcodec_free_context(&mVideoCodecCtx);
            mVideoCodecCtx = nullptr;
        }

        // 4. 释放格式上下文
        if (mFmtCtx) {
            avformat_free_context(mFmtCtx);
            mFmtCtx = nullptr;
        }

        // 5. 释放格式转换上下文
        if (mSwsCtx) {
            sws_freeContext(mSwsCtx);
            mSwsCtx = nullptr;
        }

        // 6. 标记连接断开
        mIsConnected = false;
        mVideoStream = nullptr;
        mVideoIndex = -1;
        std::cout << "推流连接已关闭" << std::endl;
        return true;
    }

    // -------------------------- 添加视频帧到队列（生产端） --------------------------
    void Push::addVideoFrame(cv::Mat& frame) {
        if (!frame.empty() && mIsRunning) {
            std::lock_guard<std::mutex> lock(mVideoFrameQMutex);
            // 限制队列大小，避免内存溢出
            if (mVideoFrameQ.size() > 00) {
                mVideoFrameQ.pop();
                std::cerr << "帧队列已满，丢弃最早帧" << std::endl;
            }
            mVideoFrameQ.push(frame.clone()); // 克隆帧，避免原数据被修改
            mVideoFrameCV.notify_one(); // 唤醒消费线程
        }
    }

    // -------------------------- 获取队列帧数量 --------------------------
    int Push::getVideoFrameQSize() {
        std::lock_guard<std::mutex> lock(mVideoFrameQMutex);
        return mVideoFrameQ.size();
    }

    // -------------------------- 获取视频帧（消费端） --------------------------
    bool Push::getVideoFrame(cv::Mat& frame) {
        std::unique_lock<std::mutex> lock(mVideoFrameQMutex);
        // 队列为空且线程运行中，等待帧
        while (mIsRunning && mVideoFrameQ.empty()) {
            mVideoFrameCV.wait_for(lock, std::chrono::milliseconds(10));
        }
        if (!mIsRunning || mVideoFrameQ.empty()) {
            return false;
        }
        frame = mVideoFrameQ.front();
        mVideoFrameQ.pop();
        return true;
    }

    // -------------------------- 清空帧队列 --------------------------
    void Push::clearVideoFrameQueue() {
        std::lock_guard<std::mutex> lock(mVideoFrameQMutex);
        while (!mVideoFrameQ.empty()) {
            mVideoFrameQ.pop();
        }
    }

    // -------------------------- BGR24转YUV420P（编码前置处理） --------------------------
    bool Push::opencv_bgr24ToYuv420p(unsigned char* bgrBuf, int width, int height, unsigned char* yuvBuf) {
        // 1. 参数校验：指针非空+YUV缓冲区大小足够（YUV420P = width*height*3/2字节）
        if (!bgrBuf || !yuvBuf || width <= 0 || height <= 0) {
            return false;
        }
        int yuvSize = width * height * 3 / 2;
        if (yuvSize <= 0) return false;

        // 2. 封装裸BGR指针为OpenCV的cv::Mat（BGR24格式）
        // CV_8UC3：8位无符号char + 3通道（BGR顺序）；width*3是行步长（无内存对齐时）
        cv::Mat bgrMat(height, width, CV_8UC3, bgrBuf, width * 3);

        // 3. BGR转YUV420P（I420布局，与你原手写代码的Y→U→V顺序一致）
        cv::Mat yuvMat;
        cv::cvtColor(bgrMat, yuvMat, cv::COLOR_BGR2YUV_I420);

        // 4. 将OpenCV的YUV数据拷贝到你的目标缓冲区frame_yuv420p_buff
        memcpy(yuvBuf, yuvMat.data, yuvSize);

        return true;
    }

    // -------------------------- 启动编码推流线程 --------------------------
    void Push::startThread() {
        if (!mIsRunning && mIsConnected) {
            mIsRunning = true;
            // 启动编码线程（静态函数作为入口，传递this指针）
            mEncodeThread = std::thread(encodeVideoThread, this);
            std::cout << "编码推流线程已启动" << std::endl;
        }
        else if (!mIsConnected) {
            std::cerr << "推流未连接，无法启动线程" << std::endl;
        }
    }

    // -------------------------- 停止编码推流线程 --------------------------
    void Push::stopThread() {
        if (mIsRunning) {
            mIsRunning = false;
            mVideoFrameCV.notify_one(); // 唤醒等待的线程
            // 等待线程结束
            if (mEncodeThread.joinable()) {
                mEncodeThread.join();
            }
            std::cout << "编码推流线程已停止" << std::endl;
        }
    }

    // -------------------------- 静态线程入口函数（仅这个需要static） --------------------------
    void Push::encodeVideoThread(void* arg) {
        Push* push = static_cast<Push*>(arg);
        if (push) {
            push->handleEncodeVideo(); // 调用非静态的核心逻辑
        }
        else {
            std::cerr << "线程入口参数为空" << std::endl;
        }
    }

    // -------------------------- 核心：编码并推流视频帧 --------------------------
    void Push::handleEncodeVideo() {
        cv::Mat frame;
        AVPacket* pkt = av_packet_alloc();
        AVFrame* yuv_frame = av_frame_alloc();
        if (!pkt || !yuv_frame) {
            std::cerr << "分配AVPacket/AVFrame失败" << std::endl;
            return;
        }

        while (mIsRunning) {
            // 1. 获取队列中的帧
            if (!getVideoFrame(frame) || frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            // 2. 检查推流连接状态，断开则尝试重连
            if (!mIsConnected) {
                if (mConnectCount < 5) { // 最多重连5次
                    reConnect();
                }
                else {
                    std::cerr << "重连次数达到上限，停止推流" << std::endl;
                    mIsRunning = false;
                    break;
                }
                continue;
            }

            int width = frame.cols;
            int height = frame.rows;

            // 3. 调整编码器参数（匹配帧分辨率）
            if (mVideoCodecCtx->width != width || mVideoCodecCtx->height != height) {
                mVideoCodecCtx->width = width;
                mVideoCodecCtx->height = height;
                avcodec_close(mVideoCodecCtx);
                AVDictionary* codec_opts = nullptr;
                av_dict_set(&codec_opts, "preset", "ultrafast", 0);
                av_dict_set(&codec_opts, "tune", "zerolatency", 0);
                avcodec_open2(mVideoCodecCtx, avcodec_find_encoder(AV_CODEC_ID_H264), &codec_opts);
                av_dict_free(&codec_opts);
            }

            // 4. BGR转YUV420P
            int yuv_size = width * height * 3 / 2; // YUV420P总大小
            unsigned char* yuv_buf = new unsigned char[yuv_size];
            if (!opencv_bgr24ToYuv420p(frame.data, width, height, yuv_buf)) {
                delete[] yuv_buf;
                continue;
            }

            // 5. 填充YUV帧数据
            yuv_frame->width = width;
            yuv_frame->height = height;
            yuv_frame->format = AV_PIX_FMT_YUV420P;
            av_frame_get_buffer(yuv_frame, 0);

            int y_size = width * height;

            const uint8_t* src_data[4] = { yuv_buf, yuv_buf + y_size, yuv_buf + y_size + y_size / 4, nullptr };
            int src_linesizes[4] = { width, width / 2, width / 2, 0 };
            av_image_copy(yuv_frame->data, yuv_frame->linesize,
                src_data, src_linesizes,
                AV_PIX_FMT_YUV420P, width, height);

            // 6. 设置帧时间戳（避免音视频不同步）
            static int64_t pts = 0;
            yuv_frame->pts = pts++;
            av_frame_make_writable(yuv_frame);

            // 7. 编码YUV帧为H264包
            int ret = avcodec_send_frame(mVideoCodecCtx, yuv_frame);
            if (ret < 0) {
                char err_buf[AV_ERROR_MAX_STRING_SIZE] = { 0 };
                av_strerror(ret, err_buf, sizeof(err_buf));
                std::cerr << "发送帧到编码器失败：" << err_buf << std::endl;
                delete[] yuv_buf;
                continue;
            }

            // 8. 获取编码后的数据包并推流
            while (ret >= 0) {
                ret = avcodec_receive_packet(mVideoCodecCtx, pkt);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                }
                else if (ret < 0) {
                    char err_buf[AV_ERROR_MAX_STRING_SIZE] = { 0 };
                    av_strerror(ret, err_buf, sizeof(err_buf));
                    std::cerr << "接收编码数据包失败：" << err_buf << std::endl;
                    break;
                }

                // 9. 调整数据包时间基
                av_packet_rescale_ts(pkt, mVideoCodecCtx->time_base, mVideoStream->time_base);
                pkt->stream_index = mVideoIndex;

                // 10. 发送数据包到推流服务器
                ret = av_interleaved_write_frame(mFmtCtx, pkt);
                if (ret < 0) {
                    char err_buf[AV_ERROR_MAX_STRING_SIZE] = { 0 };
                    av_strerror(ret, err_buf, sizeof(err_buf));
                    std::cerr << "推流发送数据包失败：" << err_buf << "，尝试重连" << std::endl;
                    mIsConnected = false;
                    break;
                }

                av_packet_unref(pkt); // 释放数据包引用
            }

            // 11. 释放YUV缓存
            delete[] yuv_buf;
        }

        // 12. 释放资源
        av_packet_free(&pkt);
        av_frame_free(&yuv_frame);
        std::cout << "编码推流线程退出" << std::endl;
    }
}