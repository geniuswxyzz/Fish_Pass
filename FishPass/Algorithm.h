#pragma once
#ifndef ANALYZER_ALGORITHM_H
#define ANALYZER_ALGORITHM_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>    //opencv header file

namespace AVSAnalyzer {
    //class Config;
    const std::string yolov8_path; // 模型路径

    static std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 0, 255) ,
        cv::Scalar(0, 255, 0) ,
        cv::Scalar(255, 0, 0) ,
        cv::Scalar(255, 100, 50) ,
        cv::Scalar(50, 100, 255) ,
        cv::Scalar(255, 50, 100)
    };

    cv::Mat static letterbox(const cv::Mat& source)
    {
        int col = source.cols;
        int row = source.rows;
        int _max = MAX(col, row);
        cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
        source.copyTo(result(cv::Rect(0, 0, col, row)));
        return result;
    };

    // YOLOv8模型标定的上下还有矩阵
    struct DetectObject
    {
        // 矩形左上角
        int x1;
        int y1;
        // 矩形右下角
        int x2;
        int y2;
        float score; // 置信度分数
        std::string class_name; // 类别名字

        int frame_nums = 0; // 消失帧数计算，默认值为0
        int weight = 1; // 此帧权重计算，越低越代表重要，越高越代表不重要
    };

    class Algorithm
    {
    public:
        Algorithm() = delete;
        Algorithm(const std::string& yolov8_path); // 参数传输
        virtual ~Algorithm();
    public:
        virtual bool objectDetect(cv::Mat& image, std::vector<DetectObject>& detects) = 0;
    protected:
        const std::string model_path;

    };

}
#endif //ANALYZER_ALGORITHM_H

