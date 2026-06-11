#pragma once
// ����Ŀ���ڼ�����Ⱥͨ���İٷ���
#include<vector>
#include"opencv2/opencv.hpp"
#include"Algorithm.h"
#include <cmath>
#include <queue>
#include <ctime>

namespace AVSAnalyzer {
	class FinshPrecent{
	public:
		//// ���캯��Ϊ�����������Ϣ
		//FinshPrecent(cv::Point P1_1, cv::Point P1_2,cv::Point P2_1, cv::Point P2_2, int ff_in, int ff_out,
		//	int oushi_thr=50,int noFind_oushi_thr=100,int frame_num=59, int weigth_conti=2,
		//	int frame_correct=30, int max_iter=30); // ������
		FinshPrecent(int id, // <--- ����������ID
			cv::Point P1_1, cv::Point P1_2, cv::Point P2_1, cv::Point P2_2, int ff_in, int ff_out,
			int oushi_thr = 50, int noFind_oushi_thr = 100, int frame_num = 59, int weigth_conti = 2,
			int frame_correct = 30, int max_iter = 30);

		~FinshPrecent(); // ������
		
		// �������ֵ
		// �����������
		cv::Point P_left1;
		cv::Point P_left2;
		// ������������
		cv::Point P_right1;
		cv::Point P_right2;
		// �������ڷ����ж�
		int ff_in; // �����ڷ����жϣ�0Ϊ����(X����)��1Ϊ����(X��С)��2Ϊ����(Y��С)��3Ϊ����(Y����)
		int ff_out; // ������ڷ����жϣ���ff_inͬ��

		// ����
		int oushi_thr; //  ŷʽ������ֵ
		int noFind_oushi_thr; // �ڶ��β���ŷʽ������ֵ
		int frame_num ; // ֡��ֵ ����30֡ȥ���õ�
		int weigth_conti; // ����֡��Ȩ��ֵ
		int frame_correct; // ÿ��20֡����һ��
		int max_iter; // �����㷨��������

		// ���ĺ��������ڼ������յ�ͨ����
		double countPrecent(std::vector<AVSAnalyzer::DetectObject> fish_detects, cv::Mat& frame);

		// 从当前检测到的鱼里随机抓图（按时间间隔节流）
		void TryCaptureRandomFish(const std::vector<AVSAnalyzer::DetectObject>& fish_detects,
			const cv::Mat& frame, const cv::Rect& capture_roi, int interval_seconds = 10,
			float capture_conf_threshold = 0.55f, int min_bbox_area = 900, bool score_weighted_random = true);

		// �����������
		void FinshPrecentClear();

		// 实时平均通过时间累加器 (记录Unix时间戳，单位：秒)
		double Sum_Start_Up = 0.0;
		double Sum_Finish_Down = 0.0;
		double Sum_Finish_Up = 0.0;
		double Sum_Start_Down = 0.0;
		
		// 记录已经成功计算通过的鱼的滞留时间总和及数量
		double total_valid_time = 0.0;
		int valid_passages = 0;
		
		// 记录每次跨入事件的时间戳队列，用于更精准地匹配跨出事件
		std::queue<time_t> up_enter_times;
		std::queue<time_t> down_enter_times;

		// 获取实时平均通过时间
		double getAveragePassageTime();

		int ShangShu1=0; // ͨ������
		int Jianghe1 = 0; // ������
		int ShangShu2 =0;
		int Jianghe2 = 0;

		// 2. ������Ա��������¼��ǰʵ�����������ݿ�ID
		int pool_id;

	private:
		// �����㷨����ȡ��С����
		void auction_matching(const std::vector<std::vector<double>>& cost, int rows, int cols, double thr, std::vector<int>& match_idx);

		// �ж�����ֱ���Ƿ��ཻ
		bool IS_cross(const cv::Point& P1_1, const cv::Point& P1_2, const cv::Point& P2_1, const cv::Point& P2_2);

		// ��������������Ⱥ�������仯
		bool Fish_In_Percentage(const cv::Point P1_1, const cv::Point P1_2,
			const cv::Point P3_1, const cv::Point P3_2, const int ff_in);

		// ��������������Ⱥ�����仯
		bool Fish_out_Percentage(const cv::Point P2_1, const cv::Point P2_2,
			const cv::Point P3_1, const cv::Point P3_2, const int ff_out);

		// 3. ����˽�и�������������ִ�����ݿ�������
		// location_type: 0=���, 1=�յ�
		// flow_type: 0=����, 1=����
		void insert_pass_log(int location_type, int flow_type);


		std::vector<DetectObject> oder_fish_detect;// ��¼֮ǰ�洢�ĵ�����
		std::vector<AVSAnalyzer::DetectObject> noFind_fish_detect; // ��¼��ʧ�������
		time_t last_random_capture_time = 0;
		
		size_t number = 0; // ��ǰ����
	};
}
