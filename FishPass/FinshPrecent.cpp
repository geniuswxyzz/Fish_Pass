#include<iostream>
#include "FinshPrecent.h"
#include "DBHelper.h" // <--- 1. 锟斤拷锟斤拷锟斤拷锟捷匡拷锟斤拷锟斤拷
#include <ctime>
#include <cstdio>
#include <random>
#include <algorithm>

const double INF_M = 1e9;
const double eps = 1e-6;

namespace AVSAnalyzer {
	/*FinshPrecent::FinshPrecent(cv::Point P1_1, cv::Point P1_2, cv::Point P2_1, cv::Point P2_2, int ff_in, int ff_out,
		int oushi_thr, int noFind_oushi_thr, int frame_num, int weigth_conti,
		int frame_correct, int max_iter) {*/
	FinshPrecent::FinshPrecent(int id, // <--- 锟斤拷锟斤拷锟斤拷锟斤拷
		cv::Point P1_1, cv::Point P1_2, cv::Point P2_1, cv::Point P2_2, int ff_in, int ff_out,
		int oushi_thr, int noFind_oushi_thr, int frame_num, int weigth_conti,
		int frame_correct, int max_iter) {
		this->pool_id = id; // <--- 锟斤拷锟斤拷ID
        // 原锟叫革拷值锟竭硷拷
        this->P_left1 = P1_1;
        this->P_left2 = P1_2;
        this->P_right1 = P2_1;
        this->P_right2 = P2_2;
        this->ff_in = ff_in;
        this->ff_out = ff_out;
        this->oushi_thr = oushi_thr;
        this->noFind_oushi_thr = noFind_oushi_thr;
        this->frame_num = frame_num;
        this->weigth_conti = weigth_conti;
        this->frame_correct = frame_correct;
		this->max_iter = max_iter;
	}
	
    // 锟斤拷锟斤拷锟斤拷锟斤拷实锟斤拷
    FinshPrecent::~FinshPrecent() {
		// 锟斤拷实锟街ｏ拷锟斤拷锟叫讹拷态锟斤拷源锟斤拷锟节达拷锟酵放ｏ拷
		FinshPrecentClear();
    }

	// 2. 实锟斤拷实锟绞的诧拷锟斤拷锟竭硷拷
	void FinshPrecent::insert_pass_log(int location_type, int flow_type) {
		// 直锟接碉拷锟矫碉拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷
		// location_type: 0=锟斤拷锟? 1=锟秸碉拷
		// flow_type: 0=锟斤拷锟斤拷, 1=锟斤拷锟斤拷
		DBHelper::GetInstance().InsertPassageLog(this->pool_id, location_type, flow_type);

	//	// 锟斤拷印锟斤拷锟斤拷锟斤拷息 (锟斤拷锟斤拷锟斤拷原锟斤拷锟侥匡拷锟斤拷台锟斤拷锟斤拷)
	//	printf("[Log] Pool:%d Loc:%s Flow:%s\n",
	//		this->pool_id,
	//		(location_type == 0 ? "Start" : "End"),
	//		(flow_type == 0 ? "Up" : "Down"));
	}


	void FinshPrecent::FinshPrecentClear() {
		oder_fish_detect.clear();
		noFind_fish_detect.clear();
		Sum_Start_Up = 0.0;
		Sum_Finish_Down = 0.0;
		Sum_Finish_Up = 0.0;
		Sum_Start_Down = 0.0;
		total_valid_time = 0.0;
		valid_passages = 0;
		std::queue<time_t> empty_up;
		std::swap(up_enter_times, empty_up);
		std::queue<time_t> empty_down;
		std::swap(down_enter_times, empty_down);
		last_random_capture_time = 0;
	}

	void FinshPrecent::TryCaptureRandomFish(const std::vector<AVSAnalyzer::DetectObject>& fish_detects,
		const cv::Mat& frame, const cv::Rect& capture_roi, int interval_seconds,
		float capture_conf_threshold, int min_bbox_area, bool score_weighted_random) {
		if (frame.empty() || fish_detects.empty()) return;

		time_t now = std::time(nullptr);
		if (interval_seconds > 0 && last_random_capture_time > 0) {
			if (std::difftime(now, last_random_capture_time) < interval_seconds) return;
		}

		cv::Rect frame_rect(0, 0, frame.cols, frame.rows);
		cv::Rect roi = capture_roi & frame_rect;
		if (roi.width <= 0 || roi.height <= 0) return;

		std::vector<cv::Rect> candidates;
		std::vector<double> weights;
		candidates.reserve(fish_detects.size());
		weights.reserve(fish_detects.size());
		for (const auto& fish : fish_detects) {
			int w = fish.x2 - fish.x1;
			int h = fish.y2 - fish.y1;
			if (w <= 0 || h <= 0) continue;
			int area = w * h;
			if (min_bbox_area > 0 && area < min_bbox_area) continue;
			if (capture_conf_threshold > 0.0f && fish.score < capture_conf_threshold) continue;
			int cx = fish.x1 + w / 2;
			int cy = fish.y1 + h / 2;
			if (!roi.contains(cv::Point(cx, cy))) continue;
			candidates.emplace_back(fish.x1, fish.y1, w, h);
			if (score_weighted_random) {
				double score_weight = std::max(0.001f, fish.score - capture_conf_threshold + 0.001f);
				weights.push_back(score_weight);
			}
			else {
				weights.push_back(1.0);
			}
		}
		if (candidates.empty()) return;

		static thread_local std::mt19937 rng(std::random_device{}());
		std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
		cv::Rect fish_rect = candidates[dist(rng)];

		const int padding = 30;
		cv::Rect padded_rect;
		padded_rect.x = std::max(0, fish_rect.x - padding);
		padded_rect.y = std::max(0, fish_rect.y - padding);
		padded_rect.width = std::min(frame.cols - padded_rect.x, fish_rect.width + padding * 2);
		padded_rect.height = std::min(frame.rows - padded_rect.y, fish_rect.height + padding * 2);
		if (padded_rect.area() <= 0) return;

		cv::Mat crop = frame(padded_rect).clone();
		char filename[256];
		std::snprintf(filename, sizeof(filename), "crops/pool%d_time%lld_x%d_y%d.jpg",
			this->pool_id, (long long)now, padded_rect.x, padded_rect.y);
		if (cv::imwrite(filename, crop)) {
			last_random_capture_time = now;
		}
	}

	// 锟斤拷锟斤拷锟姐法
	void FinshPrecent::auction_matching(const std::vector<std::vector<double>>& cost, int rows, int cols, double thr, std::vector<int>& match_idx)
	{
		std::vector<int> match_to(cols, -1);
		std::vector<double> price(cols, 0.0);
		match_idx.assign(rows, -1);

		for (int iter = 0; iter < max_iter; ++iter)
		{
			std::vector<int> unmatched;
			for (int i = 0; i < rows; ++i)
			{
				if (match_idx[i] == -1)
					unmatched.push_back(i);
			}
			if (unmatched.empty()) break;

			for (int i : unmatched)
			{
				double min_cost = INF_M;
				double second_min = INF_M;
				int best_j = -1;

				for (int j = 0; j < cols; ++j)
				{
					if (cost[i][j] > thr) continue;
					double val = cost[i][j] - price[j];
					if (val < min_cost - eps)
					{
						second_min = min_cost;
						min_cost = val;
						best_j = j;
					}
					else if (val < second_min - eps)
					{
						second_min = val;
					}
				}
				if (best_j == -1) continue;

				double bid = min_cost - second_min + eps;
				price[best_j] += bid;

				if (match_to[best_j] != -1)
				{
					int prev_i = match_to[best_j];
					match_idx[prev_i] = -1;
				}
				match_idx[i] = best_j;
				match_to[best_j] = i;
			}
		}
	}

	// 锟叫讹拷锟斤拷锟斤拷锟竭讹拷锟角凤拷锟洁交
	bool FinshPrecent::IS_cross(const cv::Point& P1_1, const cv::Point& P1_2, const cv::Point& P2_1, const cv::Point& P2_2) {
		// 锟斤拷锟斤拷锟脚筹拷实锟斤拷
		// P1_1锟斤拷P1_2为一锟斤拷直锟竭ｏ拷P2_1锟斤拷P2_2为一锟斤拷直锟斤拷
		int min_x1 = std::min(P1_1.x, P1_2.x);
		int max_x1 = std::max(P1_1.x, P1_2.x);
		int min_y1 = std::min(P1_1.y, P1_2.y);
		int max_y1 = std::max(P1_1.y, P1_2.y);

		int min_x2 = std::min(P2_1.x, P2_2.x);
		int max_x2 = std::max(P2_1.x, P2_2.x);
		int min_y2 = std::min(P2_1.y, P2_2.y);
		int max_y2 = std::max(P2_1.y, P2_2.y);
		// 锟斤拷围锟叫诧拷锟洁交 锟斤拷 锟竭段诧拷锟洁交
		if (max_x1 < min_x2 || max_x2 < min_x1 || max_y1 < min_y2 || max_y2 < min_y1) {
			return false;
		}

		// 锟斤拷锟斤拷实锟斤拷
		auto crossProduct = [](const cv::Point& Q, const cv::Point& P, const cv::Point& R) {
			return (P.x - Q.x) * (R.y - Q.y) - (P.y - Q.y) * (R.x - Q.x); };

		double cross1 = crossProduct(P1_1, P1_2, P2_1);
		double cross2 = crossProduct(P1_1, P1_2, P2_2);
		double cross3 = crossProduct(P2_1, P2_2, P1_1);
		double cross4 = crossProduct(P2_1, P2_2, P1_2);

		// 锟斤拷锟斤拷锟竭讹拷锟洁互锟斤拷锟斤拷 -> 锟洁交
		bool condition1 = (cross1 * cross2) <= 1e-6; // 锟斤拷锟姐精锟斤拷锟捷达拷锟斤拷锟斤拷锟斤拷0值锟叫讹拷锟斤拷锟?
		bool condition2 = (cross3 * cross4) <= 1e-6;

		return condition1 && condition2;
	}

	// 3. 锟斤拷写 Fish_In_Percentage (锟斤拷锟斤拷卸锟?
	bool FinshPrecent::Fish_In_Percentage(const cv::Point P1_1, const cv::Point P1_2,
		const cv::Point P3_1, const cv::Point P3_2, const int ff_in) {

		bool flag = false;
		if (IS_cross(P1_1, P1_2, P3_1, P3_2)) { // 锟斤拷锟斤拷锟斤拷越
			flag = true;
			int delta_x = P3_2.x - P3_1.x; // X锟斤拷位锟斤拷

			// 统一锟叫讹拷锟竭硷拷锟斤拷X锟斤拷锟斤拷为锟斤拷锟斤拷(0)锟斤拷X锟斤拷小为锟斤拷锟斤拷(1)
			// 锟斤拷锟斤拷锟斤拷写锟街憋拷锟斤拷锟斤拷锟斤拷锟诫保锟斤拷原锟叫碉拷 switch(ff_in) 锟竭硷拷锟斤拷锟节此达拷锟斤拷锟斤拷 insert_pass_log

			if (delta_x > 0) { // 锟斤拷锟斤拷
				ShangShu1++;
				time_t t = std::time(nullptr);
				Sum_Start_Up += t;
				up_enter_times.push(t);
				insert_pass_log(0, 0); // 0:锟斤拷锟? 0:锟斤拷锟斤拷
			}
			else { // 锟斤拷锟斤拷
				Jianghe1++;
				time_t t = std::time(nullptr);
				Sum_Start_Down += t;
				down_enter_times.push(t);
				insert_pass_log(0, 1); // 0:锟斤拷锟? 1:锟斤拷锟斤拷
			}

		}
		return flag;
	}

	/*double FinshPrecent::getAveragePassageTime() {
		if (valid_passages <= 0) {
			return 0.0;
		}
		return total_valid_time / valid_passages;
	}*/
	double FinshPrecent::getAveragePassageTime() {
		// 1. 获取入池室总次数和总时间
		int total_in_count = ShangShu1 + Jianghe1;
		double total_in_time = Sum_Start_Up + Sum_Start_Down;

		// 2. 获取出池室总次数和总时间
		int total_out_count = ShangShu2 + Jianghe2;
		double total_out_time = Sum_Finish_Up + Sum_Finish_Down;

		// 3. 容错：如果还没有入池或出池记录，防除零，直接返回 0
		if (total_in_count <= 0 || total_out_count <= 0) {
			return 0.0;
		}

		// 4. 采用你的新逻辑公式：
		// 入池室平均时间 = 入池室时间和 / 入池室次数
		double avg_entry_time = total_in_time / total_in_count;

		// 出池室平均时间 = 出池室时间和 / 出池室次数
		double avg_exit_time = total_out_time / total_out_count;

		// 通过时间 = 出池室平均时间 - 入池室平均时间
		double avg_passage_time = avg_exit_time - avg_entry_time;

		// 5. 兜底保护：极其偶发的情况下（比如刚开始统计时数据极少），防止出现负数
		return avg_passage_time > 0 ? avg_passage_time : 0.0;
	}

	// 4. 锟斤拷写 Fish_out_Percentage (锟秸碉拷锟叫讹拷)
	bool FinshPrecent::Fish_out_Percentage(const cv::Point P2_1, const cv::Point P2_2,
		const cv::Point P3_1, const cv::Point P3_2, const int ff_out) {

		bool flag = false;
		if (IS_cross(P2_1, P2_2, P3_1, P3_2)) {
			flag = true;
			int delta_x = P3_2.x - P3_1.x;

			if (delta_x > 0) { // 锟斤拷锟斤拷
				ShangShu2++;
				time_t current_time = std::time(nullptr);
				Sum_Finish_Up += current_time;
				if (!up_enter_times.empty()) {
					time_t entry_time = up_enter_times.front();
					up_enter_times.pop();
					double duration = std::difftime(current_time, entry_time);
					if (duration >= 0) {
						total_valid_time += duration;
						valid_passages++;
					}
				}
				insert_pass_log(1, 0); // 1:锟秸碉拷, 0:锟斤拷锟斤拷
			}
			else { // 锟斤拷锟斤拷
				Jianghe2++;
				time_t current_time = std::time(nullptr);
				Sum_Finish_Down += current_time;
				if (!down_enter_times.empty()) {
					time_t entry_time = down_enter_times.front();
					down_enter_times.pop();
					double duration = std::difftime(current_time, entry_time);
					if (duration >= 0) {
						total_valid_time += duration;
						valid_passages++;
					}
				}
				insert_pass_log(1, 1); // 1:锟秸碉拷, 1:锟斤拷锟斤拷
			}
		}
		return flag;
	}

	// 锟斤拷锟侥猴拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷群通锟斤拷锟斤拷
	double FinshPrecent::countPrecent(std::vector<AVSAnalyzer::DetectObject> fish_detects, cv::Mat& frame) {
		// 欧式锟斤拷锟斤拷
		auto oushi = [](float x1, float y1, float x2, float y2)->float { return (std::sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))); };

		// 锟斤拷一锟轿革拷锟斤拷通锟斤拷欧式锟斤拷锟斤拷锟斤拷锟矫匡拷锟侥匡拷锟街拷锟侥达拷锟桔撅拷锟斤拷
		int rows = oder_fish_detect.size();
		//printf("rows锟斤拷%d\n", rows);
		int cols = fish_detects.size();
		//printf("cols锟斤拷%d\n", cols);
		std::vector<std::vector<double >> frist_oushi(rows, std::vector<double>(cols, 0.0)); // 锟斤拷锟桔撅拷锟斤拷
		// 锟斤拷锟斤拷锟斤拷劬锟斤拷锟?
		for (size_t i = 0; i < rows; i++) {
			int oder_center_x = (oder_fish_detect[i].x1 + oder_fish_detect[i].x2) / 2;
			int oder_center_y = (oder_fish_detect[i].y1 + oder_fish_detect[i].y2) / 2;
			for (size_t j = 0; j < cols; j++) {
				int new_center_x = (fish_detects[j].x1 + fish_detects[j].x2) / 2;
				int new_center_y = (fish_detects[j].y1 + fish_detects[j].y2) / 2;
				frist_oushi[i][j] = oushi(oder_center_x, oder_center_y, new_center_x, new_center_y);
			}
		}

		std::vector<int> match_idx(rows, -1);  // 匹锟斤拷锟斤拷锟斤拷锟介，锟斤拷式锟斤拷锟斤拷
		std::vector<bool> used_cols(cols, false); // 锟斤拷占锟矫憋拷锟斤拷锟斤拷椋拷锟绞斤拷锟斤拷锟?
		auction_matching(frist_oushi, rows, cols, oushi_thr, match_idx);

		// 锟斤拷锟斤拷used_cols锟斤拷锟斤拷锟斤拷原锟斤拷锟斤拷锟竭硷拷一锟铰ｏ拷锟斤拷锟捷猴拷锟斤拷
		for (int i = 0; i < rows; i++) {
			if (match_idx[i] != -1 && match_idx[i] < cols) {
				used_cols[match_idx[i]] = true;
			}
		}
		// 锟斤拷锟狡轨迹锟斤拷,锟斤拷锟斤拷帧锟斤拷锟斤拷锟斤拷通锟斤拷锟斤拷锟斤拷
		for (size_t i = 0; i < rows; i++) {
			if (match_idx[i] != -1 && match_idx[i] < (int)fish_detects.size()) {
				int o_cx = (oder_fish_detect[i].x1 + oder_fish_detect[i].x2) / 2;
				int o_cy = (oder_fish_detect[i].y1 + oder_fish_detect[i].y2) / 2;
				int c_cx = (fish_detects[match_idx[i]].x1 + fish_detects[match_idx[i]].x2) / 2;
				int c_cy = (fish_detects[match_idx[i]].y1 + fish_detects[match_idx[i]].y2) / 2;
				cv::line(frame, cv::Point(o_cx, o_cy), cv::Point(c_cx, c_cy), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
				
				// 锟斤拷锟斤拷锟斤拷群锟斤拷锟斤拷锟戒化 (绗竴姝ュ尮閰嶄篃闇€瑕佹娴嬭繃绾匡紝淇鍙湪閲嶈瘑鍒椂妫€娴嬬殑娼滃湪閬楁紡)
				Fish_In_Percentage(P_left1, P_left2, cv::Point(o_cx, o_cy), cv::Point(c_cx, c_cy), ff_in);
				Fish_out_Percentage(P_right1, P_right2, cv::Point(o_cx, o_cy), cv::Point(c_cx, c_cy), ff_out);
			}
		}

		// 锟节讹拷锟轿革拷锟铰ｏ拷锟斤拷原锟饺碉拷锟斤拷锟斤拷帧锟斤拷锟斤拷锟睫筹拷锟斤拷母锟斤拷锟?
		int oder_rows = noFind_fish_detect.size();
		std::vector<std::vector<double >> noFind_frist_oushi(oder_rows, std::vector<double>(cols, 0.0)); // 锟斤拷锟桔撅拷锟斤拷
		for (size_t i = 0; i < oder_rows; i++) { // 欧式锟斤拷锟斤拷
			int oder_center_x = (noFind_fish_detect[i].x1 + noFind_fish_detect[i].x2) / 2;
			int oder_center_y = (noFind_fish_detect[i].y1 + noFind_fish_detect[i].y2) / 2;
			int weight_oder = noFind_fish_detect[i].weight;
			for (size_t j = 0; j < cols; j++) {
				int new_center_x = (fish_detects[j].x1 + fish_detects[j].x2) / 2;
				int new_center_y = (fish_detects[j].y1 + fish_detects[j].y2) / 2;
				noFind_frist_oushi[i][j] = weight_oder * oushi(oder_center_x, oder_center_y, new_center_x, new_center_y);
				if (used_cols[j] == true) { // 锟斤拷前锟斤拷锟窖撅拷锟斤拷匹锟斤拷
					noFind_frist_oushi[i][j] = INF_M;
				}

			}
		}
		// 锟斤拷锟斤拷
		std::vector<int> noFind_match_idx(oder_rows, -1);  // 匹锟斤拷锟斤拷锟斤拷锟介，锟斤拷式锟斤拷锟斤拷
		auction_matching(noFind_frist_oushi, oder_rows, cols, noFind_oushi_thr, noFind_match_idx);

		// 锟斤拷锟狡轨迹锟斤拷
		for (size_t i = 0; i < oder_rows; i++) {
			if (noFind_match_idx[i] != -1 && noFind_match_idx[i] < (int)fish_detects.size()) {
				int o_cx = (noFind_fish_detect[i].x1 + noFind_fish_detect[i].x2) / 2;
				int o_cy = (noFind_fish_detect[i].y1 + noFind_fish_detect[i].y2) / 2;
				int c_cx = (fish_detects[noFind_match_idx[i]].x1 + fish_detects[noFind_match_idx[i]].x2) / 2;
				int c_cy = (fish_detects[noFind_match_idx[i]].y1 + fish_detects[noFind_match_idx[i]].y2) / 2;
				cv::line(frame, cv::Point(o_cx, o_cy), cv::Point(c_cx, c_cy), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
				
				// 锟斤拷锟斤拷锟斤拷群锟斤拷锟斤拷锟戒化
				Fish_In_Percentage(P_left1, P_left2, cv::Point(o_cx, o_cy), cv::Point(c_cx, c_cy), ff_in);
				Fish_out_Percentage(P_right1, P_right2,cv::Point(o_cx, o_cy), cv::Point(c_cx, c_cy), ff_out);
			}
		}

		// ================================= 锟斤拷锟捷革拷锟斤拷 ============================= 
		// 原始锟斤拷锟斤拷
		for (int i = oder_rows - 1; i >= 0; i--) {
			if (noFind_match_idx[i] == -1) { // 锟斤拷锟斤拷锟矫伙拷锟斤拷业锟?
				if (noFind_fish_detect[i].frame_nums >= frame_num) { // 锟斤拷锟斤拷十帧锟斤拷直锟斤拷删锟斤拷
					noFind_fish_detect.erase(noFind_fish_detect.begin() + i);
				}
				else {
					if (noFind_fish_detect[i].weight != 1) { // 锟斤拷锟斤拷锟斤拷锟街?
						noFind_fish_detect[i].weight--;
					}
					noFind_fish_detect[i].frame_nums++; // 锟斤拷锟斤拷帧锟斤拷+1
				}
			}
			else { // 锟斤拷锟斤拷锟斤拷匹锟戒，删锟斤拷锟斤拷锟斤拷
				noFind_fish_detect.erase(noFind_fish_detect.begin() + i);
			}
		}
		// 锟斤拷原始锟斤拷锟斤拷映锟戒到锟斤拷一帧锟斤拷锟斤拷锟斤拷锟斤拷
		for (size_t i = 0; i < rows; i++) {
			if (match_idx[i] == -1) { // 锟斤拷录锟斤拷时锟斤拷位锟斤拷
				oder_fish_detect[i].weight = weigth_conti;// 锟斤拷锟斤拷锟揭恢★拷锟斤拷锟揭恢★拷锟斤拷锟斤拷锟街★拷锟斤拷糯锟斤拷锟揭恢★拷锟饺ㄖ?
				noFind_fish_detect.push_back(oder_fish_detect[i]);
			}
		}
		// 锟斤拷一帧锟斤拷锟斤拷
		oder_fish_detect = fish_detects;

		// 锟斤拷锟斤拷锟斤拷锟?
		// 锟斤拷锟斤拷,每锟斤拷3帧锟斤拷锟斤拷一锟轿斤拷锟斤拷
		if (number == frame_correct) {
			int Fish_nums = 0;
			number = 0;

			// 锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷
			// P_left1:锟斤拷锟斤拷锟斤拷辖锟斤拷锟斤拷锟?
			// P_right2:锟斤拷锟斤拷锟斤拷辖锟斤拷锟斤拷锟?
			cv::Rect YUD_Rect(P_left1, P_right2);
			// 锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟饺猴拷锟斤拷锟斤拷锟?
			for (size_t i = 0; i < fish_detects.size(); i++) {
				int fish_x = fish_detects[i].x1;
				int fish_y = fish_detects[i].y1;
				int fish_width = fish_detects[i].x2 - fish_detects[i].x1;
				int fish_height = fish_detects[i].y2 - fish_detects[i].y1;
				// 锟斤拷止锟斤拷锟斤拷为锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟剿筹拷锟斤拷耍锟?
				fish_width = std::max(0, fish_width);
				fish_height = std::max(0, fish_height);
				cv::Rect Fish_Rect(fish_x, fish_y, fish_width, fish_height);
				// 锟斤拷锟斤拷每锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷IOU值
				cv::Rect intersection_rect = YUD_Rect & Fish_Rect;
				// 锟斤拷锟姐交锟斤拷锟斤拷锟?
				float intersection_area = intersection_rect.width * intersection_rect.height;
				// 锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟轿革拷锟皆碉拷锟斤拷锟?
				float rect1_area = YUD_Rect.width * YUD_Rect.height;
				float rect2_area = Fish_Rect.width * Fish_Rect.height;
				float union_area = rect1_area + rect2_area - intersection_area;
				if (union_area <= 0)
				{
					continue;
				}
				float iou = intersection_area / union_area;
				const float eps = 1e-6; // 锟斤拷锟斤拷锟捷达拷
				bool is_fish_in_yud = (std::fabs(intersection_area - rect2_area) < eps);
				if (is_fish_in_yud) {
					Fish_nums++;
				}
			}
			// 绉婚櫎寮哄埗璁℃暟琛ュ伩閫昏緫锛屼弗鏍间緷璧栬繃绾挎娴?
			/*
			if (ShangShu1 - Jianghe1 - ShangShu2 + Jianghe2 > Fish_nums) {
				int diff = (ShangShu1 - Jianghe1 - ShangShu2 + Jianghe2) - Fish_nums;
				ShangShu2 += diff;
				Sum_Finish_Up += diff * std::time(nullptr); // 鍚屾琛ュ叏缁堢偣绾跨寮€鏃堕棿
			}
			else if (ShangShu1 - Jianghe1 - ShangShu2 + Jianghe2 < Fish_nums) {
				int diff = Fish_nums - (ShangShu1 - Jianghe1 - ShangShu2 + Jianghe2);
				ShangShu1 += diff;
				Sum_Start_Up += diff * std::time(nullptr); // 鍚屾琛ュ叏璧风偣绾胯繘鍏ユ椂闂?
			}
			*/
		}
		else {
			number++;
		}

		// 始 
		double taotuo_precent = 0.0; // 始未值
		if (ShangShu1 - Jianghe1 != 0) {
			taotuo_precent = static_cast<double>(ShangShu2 - Jianghe2) / static_cast<double>(ShangShu1 - Jianghe1);
		}
		// 通食100%强100%
		if (taotuo_precent >= 1) {
			taotuo_precent = 1;
			/* 绉婚櫎寮哄埗淇敼 ShangShu1 鐨勯€昏緫
			int diff = (ShangShu2 - Jianghe2 + Jianghe1) - ShangShu1;
			if (diff > 0) {
				ShangShu1 += diff;
				Sum_Start_Up += diff * std::time(nullptr); // 鍚屾琛ュ叏璧风偣绾胯繘鍏ユ椂闂?
			}
			*/
		}
		return taotuo_precent;
	}
}
