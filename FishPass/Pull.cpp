#include "Pull.h"
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

extern "C" {
#include <libavutil/error.h>
}

namespace AVSAnalyzer {
    // ----------------- Helper: format error -----------------
    static std::string fferr(int err) {
        char buf[256];
        av_strerror(err, buf, sizeof(buf));
        return std::string(buf);
    }

    // ----------------- Constructor / Destructor -----------------
    FFmpegPull::FFmpegPull() {
        fmt_ctx = nullptr;
        v_codec_ctx = nullptr;
        video_stream_idx = -1;
        sws_ctx = nullptr;
    }

    FFmpegPull::~FFmpegPull() {
        stopThread();

        if (sws_ctx) {
            sws_freeContext(sws_ctx);
            sws_ctx = nullptr;
        }
        if (v_codec_ctx) {
            avcodec_free_context(&v_codec_ctx);
            v_codec_ctx = nullptr;
        }
        if (fmt_ctx) {
            avformat_close_input(&fmt_ctx);
            fmt_ctx = nullptr;
        }
    }

    // ----------------- init -----------------
    int FFmpegPull::init(const std::string& url) {
        source_url = url;

        avformat_network_init();

        AVDictionary* opts = nullptr;
        // For RTSP streams you might set transport to tcp:
        av_dict_set(&opts, "rtsp_transport", "tcp", 0);
        av_dict_set(&opts, "stimeout", "5000000", 0); // microseconds

        int ret = avformat_open_input(&fmt_ctx, url.c_str(), nullptr, &opts);
        av_dict_free(&opts);
        if (ret < 0) {
            std::cerr << "avformat_open_input failed: " << fferr(ret) << std::endl;
            return -1;
        }

        ret = avformat_find_stream_info(fmt_ctx, nullptr);
        if (ret < 0) {
            std::cerr << "avformat_find_stream_info failed: " << fferr(ret) << std::endl;
            avformat_close_input(&fmt_ctx);
            fmt_ctx = nullptr;
            return -1;
        }

        video_stream_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (video_stream_idx < 0) {
            std::cerr << "no video stream found" << std::endl;
            avformat_close_input(&fmt_ctx);
            fmt_ctx = nullptr;
            return -1;
        }

        int r = init_codec(&v_codec_ctx, fmt_ctx->streams[video_stream_idx]);
        if (r < 0) {
            std::cerr << "init codec failed" << std::endl;
            avformat_close_input(&fmt_ctx);
            fmt_ctx = nullptr;
            return -1;
        }

        if (v_codec_ctx) {
            video_width = v_codec_ctx->width;
            video_height = v_codec_ctx->height;
        }

        // success
        std::cout << "FFmpegPull init success" << std::endl;
        return 0;
    }

    // ----------------- init codec -----------------
    int FFmpegPull::init_codec(AVCodecContext** codec_ctx, AVStream* stream) {
        if (!stream) return -1;
        const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
        if (!codec) {
            std::cerr << "decoder not found" << std::endl;
            return -1;
        }
        *codec_ctx = avcodec_alloc_context3(codec);
        if (!*codec_ctx) {
            std::cerr << "avcodec_alloc_context3 failed" << std::endl;
            return -1;
        }
        int ret = avcodec_parameters_to_context(*codec_ctx, stream->codecpar);
        if (ret < 0) {
            std::cerr << "avcodec_parameters_to_context failed: " << fferr(ret) << std::endl;
            avcodec_free_context(codec_ctx);
            return -1;
        }
        ret = avcodec_open2(*codec_ctx, codec, nullptr);
        if (ret < 0) {
            std::cerr << "avcodec_open2 failed: " << fferr(ret) << std::endl;
            avcodec_free_context(codec_ctx);
            return -1;
        }
        return 0;
    }

    // ----------------- ensure sws context (recreate if needed) -----------------
    bool FFmpegPull::ensure_swsctx(int src_w, int src_h, AVPixelFormat src_fmt) {
        if (sws_ctx && src_w == sws_src_w && src_h == sws_src_h && src_fmt == sws_src_fmt) {
            return true;
        }
        if (sws_ctx) {
            sws_freeContext(sws_ctx);
            sws_ctx = nullptr;
        }
        sws_ctx = sws_getContext(src_w, src_h, src_fmt,
            src_w, src_h, AV_PIX_FMT_BGR24,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws_ctx) {
            std::cerr << "sws_getContext failed" << std::endl;
            return false;
        }
        sws_src_w = src_w;
        sws_src_h = src_h;
        sws_src_fmt = src_fmt;
        return true;
    }

    // ----------------- decode_packet -----------------
    int FFmpegPull::decode_packet(AVPacket* pkt, AVFrame* frame) {
        if (!v_codec_ctx || !frame) return -1;

        int ret = avcodec_send_packet(v_codec_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) return 0;
        if (ret < 0) {
            std::cerr << "avcodec_send_packet error: " << fferr(ret) << std::endl;
            return ret;
        }

        ret = avcodec_receive_frame(v_codec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) return 0;
        if (ret < 0) {
            std::cerr << "avcodec_receive_frame error: " << fferr(ret) << std::endl;
            return ret;
        }

        int width = frame->width;
        int height = frame->height;
        AVPixelFormat src_fmt = (AVPixelFormat)frame->format;

        // map deprecated yuvj formats to normal ones
        if (src_fmt == AV_PIX_FMT_YUVJ420P) src_fmt = AV_PIX_FMT_YUV420P;
        if (src_fmt == AV_PIX_FMT_YUVJ422P) src_fmt = AV_PIX_FMT_YUV422P;
        if (src_fmt == AV_PIX_FMT_YUVJ444P) src_fmt = AV_PIX_FMT_YUV444P;

        if (!ensure_swsctx(width, height, src_fmt)) {
            return -1;
        }

        // prepare destination Mat
        cv::Mat bgr(height, width, CV_8UC3);
        uint8_t* dst_data[1] = { bgr.data };
        int dst_linesize[1] = { static_cast<int>(bgr.step) };

        sws_scale(sws_ctx, frame->data, frame->linesize, 0, height, dst_data, dst_linesize);

        // store latest frame (thread-safe)
        {
            std::lock_guard<std::mutex> lk(frame_mutex);
            latest_frame = bgr.clone(); // clone to be safe
        }

        return 0;
    }

    // ----------------- pullLoop (thread) -----------------
    void FFmpegPull::pullLoop() {
        if (!fmt_ctx || video_stream_idx < 0 || !v_codec_ctx) {
            running.store(false);
            return;
        }

        AVPacket* pkt = av_packet_alloc();
        AVFrame* frame = av_frame_alloc();
        if (!pkt || !frame) {
            if (pkt) av_packet_free(&pkt);
            if (frame) av_frame_free(&frame);
            running.store(false);
            return;
        }

        while (running.load()) {
            int ret = av_read_frame(fmt_ctx, pkt);
            if (ret < 0) {
                // if EOF or error, break loop
                if (ret == AVERROR_EOF) {
                    break;
                }
                else {
                    std::cerr << "av_read_frame error: " << fferr(ret) << std::endl;
                    // wait a little then continue or break depending on policy
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }
            }

            if (pkt->stream_index == video_stream_idx) {
                decode_packet(pkt, frame);
            }
            av_packet_unref(pkt);
        }

        // flush decoder
        avcodec_send_packet(v_codec_ctx, nullptr);
        while (true) {
            int r = avcodec_receive_frame(v_codec_ctx, frame);
            if (r == AVERROR_EOF || r == AVERROR(EAGAIN)) break;
            if (r < 0) break;
            decode_packet(nullptr, frame); // handle remaining frames (note: decode_packet expects pkt but we can reuse logic)
        }

        av_packet_free(&pkt);
        av_frame_free(&frame);

        running.store(false);
    }

    // ----------------- startThread / stopThread / getFrame -----------------
    void FFmpegPull::startThread() {
        if (running.load()) return;
        running.store(true);
        pull_thread = std::thread(&FFmpegPull::pullLoop, this);
    }

    void FFmpegPull::stopThread() {
        if (!running.load()) return;
        running.store(false);
        if (pull_thread.joinable()) pull_thread.join();
    }

    bool FFmpegPull::getFrame(cv::Mat& out) {
        std::lock_guard<std::mutex> lk(frame_mutex);
        if (latest_frame.empty()) return false;
        out = latest_frame.clone();
        return true;
    }
}
