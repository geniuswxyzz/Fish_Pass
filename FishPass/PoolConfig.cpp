#pragma once
#include <string>
#include <opencv2/opencv.hpp>

// 池室配置结构体：对应数据库 fish_pool_config 表的一行
struct PoolConfig {
    int id;               // 数据库中的 ID
    std::string name;     // 池室名称

    // 几何信息
    cv::Point entry_p1, entry_p2; // 起点线
    cv::Point exit_p1, exit_p2;   // 终点线

    // 业务方向 (保留兼容)
    int direction_in;
    int direction_out;

    // 构造函数
    PoolConfig(int _id, std::string _name,
        cv::Point en1, cv::Point en2, cv::Point ex1, cv::Point ex2,
        int d_in, int d_out)
        : id(_id), name(_name),
        entry_p1(en1), entry_p2(en2), exit_p1(ex1), exit_p2(ex2),
        direction_in(d_in), direction_out(d_out) {
    }
};