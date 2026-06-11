#include "FinshPass.h"
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <mutex> 

using namespace nvinfer1;

namespace AVSAnalyzer {
	const std::vector<std::string> class_names = {
	"fish"};
	Logger gLogger;
	// 资源清理（使用 delete）
	void FinshPass::cleanUpResources() {
		if (parser) { delete parser; parser = nullptr; }
		if (context) { delete context; context = nullptr; }
		if (engine) { delete engine; engine = nullptr; }
		if (config_mode) { delete config_mode; config_mode = nullptr; }
		if (network) { delete network; network = nullptr; }
		if (builder) { delete builder; builder = nullptr; }

		if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
		if (gpu_buffers[0]) { cudaFree(gpu_buffers[0]); gpu_buffers[0] = nullptr; }
		if (gpu_buffers[1]) { cudaFree(gpu_buffers[1]); gpu_buffers[1] = nullptr; }
	}

	// 构造函数
	FinshPass::FinshPass(const std::string& model_path)
		: Algorithm(model_path), // 继承父类model_path
		builder(nullptr), network(nullptr), config_mode(nullptr), parser(nullptr),
		engine(nullptr), context(nullptr), profile(nullptr), stream(nullptr), num_classes(0) {

		gpu_buffers[0] = nullptr;
		gpu_buffers[1] = nullptr;

		//// ----------------------- onnx 模型导入 --------------------
		//try {
		//	// ... (Steps 1-5 保持不变)
		//	builder = createInferBuilder(gLogger);
		//	if (!builder) throw std::runtime_error("创建IBuilder失败");
		//	uint32_t explicit_batch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
		//	network = builder->createNetworkV2(explicit_batch);
		//	if (!network) throw std::runtime_error("创建INetworkDefinition失败");
		//	config_mode = builder->createBuilderConfig();
		//	if (!config_mode) throw std::runtime_error("创建IBuilderConfig失败");
		//	//config_mode->setFlag(BuilderFlag::kFP16);
		//	config_mode->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30); // 分配内存 1G
		//	profile = builder->createOptimizationProfile();
		//	if (!profile) throw std::runtime_error("创建IOptimizationProfile失败");
		//	parser = nvonnxparser::createParser(*network, gLogger);
		//	if (!parser) throw std::runtime_error("创建ONNX Parser失败");
		//	std::string onnx_path = model_path; // 模型路径
		//	if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(ILogger::Severity::kINFO))) {
		//		throw std::runtime_error("解析ONNX模型失败");
		//	}

		//	// -------- Step 6. 动态获取输入维度 --------
		//	auto input_tensor = network->getInput(0);
		//	if (!input_tensor) throw std::runtime_error("获取输入张量失败");
		//	input_tensor_name = input_tensor->getName();
		//	Dims input_dims = input_tensor->getDimensions();
		//	if (input_dims.nbDims != 4) throw std::runtime_error("输入张量维度错误（需NCHW）");
		//	INPUT_H = input_dims.d[2];
		//	INPUT_W = input_dims.d[3];

		//	// -------- Step 7. 设置优化配置文件维度 --------
		//	Dims4 min_dims{ 1, 3, INPUT_H, INPUT_W };
		//	profile->setDimensions(input_tensor_name.c_str(), OptProfileSelector::kMIN, min_dims);
		//	profile->setDimensions(input_tensor_name.c_str(), OptProfileSelector::kOPT, min_dims);
		//	profile->setDimensions(input_tensor_name.c_str(), OptProfileSelector::kMAX, min_dims);
		//	if (config_mode->addOptimizationProfile(profile) == -1) {
		//		throw std::runtime_error("添加优化配置文件失败");
		//	}

		//	// -------- Step 8. 构建Engine --------
		//	engine = builder->buildEngineWithConfig(*network, *config_mode);
		//	if (!engine) throw std::runtime_error("构建ICudaEngine失败");
		//	std::ofstream file("D:\\Big_Fish\\FishPass\\model\\fishmonitor_lxf_12_9.engine", std::ios::binary); // engine 模型保存位置
		//	auto serialized = engine->serialize();
		//	file.write((const char*)serialized->data(), serialized->size());
		//	if (!file) { printf("engine文件保存失败！\n"); }
		//	delete serialized;

		//	// -------- Step 9. 创建执行上下文 --------
		//	context = engine->createExecutionContext();
		//	if (!context) throw std::runtime_error("创建IExecutionContext失败");

		//	// -------- Step 10. 动态获取输出维度，并计算类别数 (使用 INetworkDefinition) --------

		//	// 1. 从 network 获取输出张量
		//	auto output_tensor = network->getOutput(0);
		//	if (!output_tensor) throw std::runtime_error("获取输出张量失败");

		//	output_tensor_name = output_tensor->getName();
		//	// 直接从网络定义中获取维度，因为在构建时维度已经确定
		//	Dims output_dims = output_tensor->getDimensions();

		//	if (output_dims.nbDims != 3) throw std::runtime_error("输出张量维度错误（需NCx）");
		//	OUTPUT_ROWS = output_dims.d[1]; // Boxes count
		//	OUTPUT_COLS = output_dims.d[2]; // Data per box (4 + 1 + C)

		//	// 计算类别数
		//	num_classes = OUTPUT_COLS - 5;
		//	if (num_classes <= 0) throw std::runtime_error("类别数计算错误或模型输出格式错误");

		//	// -------- Step 11. 分配CUDA内存 --------
		//	inputSize = 3LL * INPUT_H * INPUT_W * sizeof(float);
		//	outputSize = (long long)OUTPUT_ROWS * OUTPUT_COLS * sizeof(float);

		//	if (cudaMalloc(&gpu_buffers[0], inputSize) != cudaSuccess) {
		//		throw std::runtime_error("分配输入GPU内存失败");
		//	}
		//	if (cudaMalloc(&gpu_buffers[1], outputSize) != cudaSuccess) {
		//		throw std::runtime_error("分配输出GPU内存失败");
		//	}
		//	if (cudaStreamCreate(&stream) != cudaSuccess) {
		//		throw std::runtime_error("创建CUDA流失败");
		//	}

		//	//LOGI("模型加载成功！输入尺寸：%dx%d，输出维度：行=%d, 列=%d，类别数=%d", INPUT_W, INPUT_H, OUTPUT_ROWS, OUTPUT_COLS, num_classes);
		//	std::cout << "模型加载成功！！！" << "输入尺寸" << INPUT_W << "," << INPUT_W << "，输入维度:行" << OUTPUT_ROWS << "，列=" << OUTPUT_COLS << "，类别数=" << num_classes;
		//}
		//catch (const std::exception& e) {
		//	//LOGE("模型初始化失败：%s", e.what());
		//	std::cerr << "模型初始化失败！！！" << std::endl;
		//	cleanUpResources();
		//	throw;
		//}
		//std::cout << "TensorRT Version: " << NV_TENSORRT_VERSION << std::endl;
		// -------------------------- .engine -----------------------
		try {
			// 1. 加载 .engine 模型文件
			std::ifstream file(model_path, std::ios::binary);
			if (!file) throw std::runtime_error("无法打开.engine文件");
			std::vector<char> engine_data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

			runtime = createInferRuntime(gLogger);
			if (!runtime) throw std::runtime_error("创建IRuntime失败");

			engine = runtime->deserializeCudaEngine(engine_data.data(),
				engine_data.size());
			if (!engine) {
				throw std::runtime_error("TensorRT 反序列化失败：engine 文件不兼容！");
			}

			context = engine->createExecutionContext();
			if (!context) throw std::runtime_error("创建IExecutionContext失败");

			// 2. 获取输入输出tensor属性
			// 输入
			int n_io = engine->getNbIOTensors();
			for (int i = 0; i < n_io; ++i) {
				const char* tensorName = engine->getIOTensorName(i);
				TensorIOMode mode = engine->getTensorIOMode(tensorName);
				if (mode == TensorIOMode::kINPUT)
					input_tensor_name = tensorName;
				if (mode == TensorIOMode::kOUTPUT)
					output_tensor_name = tensorName;
			}
			// 拿输入输出维度
			Dims input_dims = engine->getTensorShape(input_tensor_name.c_str());
			if (input_dims.nbDims != 4) throw std::runtime_error("输入维度错误（需NCHW）");
			INPUT_H = input_dims.d[2];
			INPUT_W = input_dims.d[3];
			std::cout << "模型张量:  " << "INPUT_W:" << INPUT_W << " INPUT_H:" << INPUT_H << std::endl;
			Dims output_dims = engine->getTensorShape(output_tensor_name.c_str());
			if (output_dims.nbDims != 3) throw std::runtime_error("输出维度错误（需N,Cx）");
			OUTPUT_COLS = output_dims.d[1];
			OUTPUT_ROWS = output_dims.d[2];
			std::cout << "模型张量:  " << "OUTPUT_COLS:" << OUTPUT_COLS << " OUTPUT_ROWS:" << OUTPUT_ROWS << std::endl;
			num_classes = OUTPUT_COLS - 5;                
			if (num_classes <= -1) throw std::runtime_error("类别数无效");

			inputSize = 3LL * INPUT_H * INPUT_W * sizeof(float);
			outputSize = (long long)OUTPUT_ROWS * OUTPUT_COLS * sizeof(float);

			if (cudaMalloc(&gpu_buffers[0], inputSize) != cudaSuccess)
				throw std::runtime_error("分配输入GPU内存失败");
			if (cudaMalloc(&gpu_buffers[1], outputSize) != cudaSuccess)
				throw std::runtime_error("分配输出GPU内存失败");
			if (cudaStreamCreate(&stream) != cudaSuccess)
				throw std::runtime_error("创建CUDA流失败");

			printf("engine模型加载成功！输入: %dx%d, 输出: 行%d, 列%d, 类别数%d", INPUT_W, INPUT_H, OUTPUT_ROWS, OUTPUT_COLS, num_classes);
		}
		catch (const std::exception& e) {
			printf("engine初始化失败: %s", e.what());
			cleanUpResources();
			throw;
		}
	}

	// ... (析构函数和 objectDetect 保持不变)

	FinshPass::~FinshPass() {
		cleanUpResources();
		std::cerr << "资源释放完成" << std::endl;
	}

	bool FinshPass::objectDetect(cv::Mat& image, std::vector<DetectObject>& detects) {
		if (num_classes <= -1) {
			return false;
		}

		std::lock_guard<std::mutex> lock(infer_mutex);
		detects.clear();

		if (image.empty() || image.cols <= 0 || image.rows <= 0) {
			std::cerr << "#出现错误# 图像为空!!!! " << std::endl;
			return false;
		}
		if (image.channels() != 3) {
			std::cerr << "#出现错误# 图像不为3通道!!! " << std::endl;
			return false;
		}
		int src_w = image.cols;
		int src_h = image.rows;

		// -------- 图像预处理（Letterbox + 归一化） --------
		float scale = std::min(static_cast<float>(INPUT_W) / src_w, static_cast<float>(INPUT_H) / src_h);
		int new_w = static_cast<int>(src_w * scale);
		int new_h = static_cast<int>(src_h * scale);
		int pad_left = (INPUT_W - new_w) / 2;
		int pad_top = (INPUT_H - new_h) / 2;
		//printf("scale::%.2f\n", scale);
		cv::Mat resized_img, letterbox_img;
		cv::resize(image, resized_img, cv::Size(new_w, new_h));
		cv::copyMakeBorder(resized_img, letterbox_img, pad_top, INPUT_H - new_h - pad_top,
			pad_left, INPUT_W - new_w - pad_left, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

		// 归一化 + 转CHW 重要：设置 swapRB=true，将 BGR 转换为 RGB
		cv::Mat blob = cv::dnn::blobFromImage(letterbox_img, 1.0 / 255.0, letterbox_img.size(), cv::Scalar(), true, false);

		// -------- 数据拷贝到GPU --------
		if (cudaMemcpyAsync(gpu_buffers[0], blob.ptr<float>(), inputSize, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
			std::cerr << "输入数据拷贝到GPU失败" << std::endl;
			return false;
		}

		// -------- 设置张量地址并执行推理 --------
		if (!context || !gpu_buffers[0] || !gpu_buffers[1]) {
			std::cerr << "TensorRT 上下文或GPU缓冲区无效" << std::endl;
			return false;
		}

		context->setInputTensorAddress(input_tensor_name.c_str(), gpu_buffers[0]);
		context->setTensorAddress(output_tensor_name.c_str(), gpu_buffers[1]);

		if (!context->enqueueV3(stream)) {
			std::cerr << "TensorRT推理执行失败" << std::endl;
			cudaStreamSynchronize(stream);
			return false;
		}

		// -------- 结果拷贝到CPU --------
		std::vector<float> host_output(OUTPUT_ROWS * OUTPUT_COLS);
		if (cudaMemcpyAsync(host_output.data(), gpu_buffers[1], outputSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
			std::cerr << "输出数据拷贝到CPU失败" << std::endl;
			return false;
		}
		if (cudaStreamSynchronize(stream) != cudaSuccess) {
			std::cerr << "数据不匹配！！！" << std::endl;
			return false;
		}

		std::vector<float> transposed(OUTPUT_ROWS * OUTPUT_COLS);

		// -------- 后处理（解码检测框） --------
		std::vector<cv::Rect> boxes;
		std::vector<float> confidences;
		std::vector<int> class_ids;

		float conf_threshold = 0.4;
		float nms_threshold = 0.45;

		float* ptr = host_output.data();
		float* cx_ptr = ptr + 0 * OUTPUT_ROWS;
		float* cy_ptr = ptr + 1 * OUTPUT_ROWS;
		float* w_ptr = ptr + 2 * OUTPUT_ROWS;
		float* h_ptr = ptr + 3 * OUTPUT_ROWS;
		float* score_ptr = ptr + 4 * OUTPUT_ROWS;

		auto sigmoid = [](float x)->float { return 1.f / (1.f + expf(-x)); };
		//std::cout << "开始推理！！！" << std::endl;
		//const int num_anchors = 8400;
		//std::cout << "OUTPUT_ROWS：" << OUTPUT_ROWS << "，OUTPUT_COLS：" << OUTPUT_COLS << std::endl;
		for (int i = 0; i < OUTPUT_ROWS; ++i) {
			//float* row = host_output.data() + i * OUTPUT_COLS;
			//float* row = &host_output[i * OUTPUT_COLS];
			
			float score = score_ptr[i];
			//std::cout << "cx：" << cx << "，cy：" << cy << "，w：" << w << "，h：" << h << "，score：" << score << std::endl;

			if (score < conf_threshold) continue;
			float cx = cx_ptr[i];
			float cy = cy_ptr[i];
			float w = w_ptr[i];
			float h = h_ptr[i];
			int class_id = 0;
			// 解码检测框
			float x1 = cx - w / 2.0f;
			float y1 = cy - h / 2.0f;
			float x2 = cx + w / 2.0f;
			float y2 = cy + h / 2.0f;

			float x1_o = (x1 - pad_left) / scale;
			float y1_o = (y1 - pad_top) / scale;
			float x2_o = (x2 - pad_left) / scale;
			float y2_o = (y2 - pad_top) / scale;

			x1_o = std::max(0.0f, std::min(x1_o, (float)src_w - 1.0f));
			y1_o = std::max(0.0f, std::min(y1_o, (float)src_h - 1.0f));
			x2_o = std::max(0.0f, std::min(x2_o, (float)src_w - 1.0f));
			y2_o = std::max(0.0f, std::min(y2_o, (float)src_h - 1.0f));

			boxes.emplace_back(cv::Point((int)x1_o, (int)y1_o), cv::Point((int)x2_o, (int)y2_o));
			confidences.push_back(score);
			class_ids.push_back(class_id);
		}

		// -------- NMS非极大值抑制 -------
		std::vector<int> indices;
		cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);

		// -------- 构建检测结果 --------
		for (int idx : indices) {
			DetectObject detect;
			detect.score = confidences[idx];
			detect.class_name = (class_ids[idx] >= 0 && class_ids[idx] < class_names.size())
				? class_names[class_ids[idx]] : "unknown";

			detect.x1 = boxes[idx].x;
			detect.y1 = boxes[idx].y;
			detect.x2 = boxes[idx].x + boxes[idx].width;
			detect.y2 = boxes[idx].y + boxes[idx].height;
			detects.push_back(detect);
		}

		return true;
	}
}