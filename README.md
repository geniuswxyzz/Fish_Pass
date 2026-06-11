# FishPass

FishPass 是一个鱼道智能监测与通过率统计系统。项目主体是 Windows Visual Studio C++ 程序，负责接入多路 RTSP 视频流、进行画面拼接与畸变校正、调用 TensorRT 模型检测鱼体、统计通过情况并写入 MySQL；同时提供一个轻量 WebDashboard，用于查询和展示数据库中的监测流水。

## 功能概览

- 多路 RTSP 视频接入与实时帧读取
- OpenCV CUDA 画面处理、裁剪、去畸变与拼接
- TensorRT `.engine` 模型推理，识别鱼体目标
- 按池室入口线、出口线统计鱼类通过情况
- MySQL 记录通过流水，并支持按时间段查询通过率
- FFmpeg 输出处理后的视频流
- Flask + ECharts 数据看板，展示流水、方向统计和通过率

## 目录结构

```text
FishPass/
|-- FishPass.sln                 # Visual Studio 解决方案
|-- FishPass/                    # C++ 主程序源码
|   |-- main.cpp                  # RTSP、拼接、检测、推流主流程
|   |-- FinshPass.*               # TensorRT 推理封装
|   |-- FinshPrecent.*            # 通过率与业务计数逻辑
|   |-- DBHelper.*                # MySQL 数据库访问
|   |-- PoolConfig.*              # 池室区域配置
|   |-- Pull.* / Push.*           # 视频流拉取与推送相关代码
|   `-- FishPass.vcxproj          # C++ 工程文件
|-- WebDashboard/                # Web 数据看板
|   |-- server.py                 # Flask API 服务
|   |-- index.html                # 前端页面
|   |-- app.js                    # 数据请求、表格、图表逻辑
|   `-- style.css                 # 页面样式
`-- .gitignore
```

## 不上传到 Git 的内容

本仓库建议只上传代码和必要的小体积工程文件。以下内容不建议直接提交到 Git：

- `3rdparty/` 第三方 SDK、库文件和运行时文件
- `model/` 模型目录
- `*.engine`、`*.onnx` 等模型文件
- `x64/`、`Release/`、`Debug/` 等编译产物
- `FishPass/crops/` 运行时截图
- `*.mp4`、`*.avi` 等运行输出视频
- `.vs/`、`.vscode/` 等本机 IDE 配置
- Python 虚拟环境、缓存和本地环境变量文件

如果这些文件已经被 Git 跟踪，单独修改 `.gitignore` 不会自动移除它们，需要再执行 `git rm --cached <path>` 将其从索引中移除。

## 编译环境

推荐环境：

- Windows x64
- Visual Studio 2022
- MSVC toolset `v143`
- Windows SDK `10.0`
- C++17
- CUDA 12.5
- TensorRT 10
- OpenCV 4.x，当前工程中使用 OpenCV 4.13.0 相关库名
- FFmpeg
- MySQL Connector/C++ 8.4
- MySQL Server

当前工程文件中包含部分本机绝对路径，例如 OpenCV 和 MySQL Connector/C++ 路径。换机器后需要在 Visual Studio 中按本机安装位置调整：

- `C/C++ -> Additional Include Directories`
- `Linker -> General -> Additional Library Directories`
- `Linker -> Input -> Additional Dependencies`

## 第三方依赖目录建议

建议将大体积依赖放在仓库根目录的 `3rdparty/` 下，但不要提交到 Git：

```text
3rdparty/
|-- CUDA125/
|-- TensorRT/
|-- ffmpeg/
`-- opencv/
```

工程中目前有相对路径引用：

- `..\3rdparty\ffmpeg\include`
- `..\3rdparty\CUDA125\include`
- `..\3rdparty\TensorRT\include`
- `..\3rdparty\ffmpeg\x64\lib`
- `..\3rdparty\CUDA125\lib\x64`
- `..\3rdparty\TensorRT\lib`
- `..\3rdparty\opencv\x64\vc16\lib`

## 模型文件

C++ 程序会优先查找以下位置的 TensorRT engine 文件：

```text
model/fishmonitor_lxf_12_9.engine
```

模型文件体积通常较大，不建议直接提交到 Git 仓库。可以通过以下方式管理：

- 放入 GitHub Release 附件
- 放入内部文件服务器或对象存储
- 使用 Git LFS
- 在 README 或部署文档中说明下载地址和放置位置

## 数据库配置

C++ 主程序和 WebDashboard 都会连接 MySQL。

C++ 侧配置位置：

```text
FishPass/main.cpp
```

当前示例连接：

```cpp
DBHelper::GetInstance().Connect("127.0.0.1", "root", "123456", "fish_db")
```

WebDashboard 配置位置：

```text
WebDashboard/server.py
```

当前示例连接：

```python
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "123",
    "database": "fish_db",
}
```

请根据实际部署环境修改数据库账号、密码和库名。建议后续改为读取环境变量，避免把真实密码写入代码。

程序使用的数据表为：

```text
fish_passage_log
```

主要字段包括：

- `id`
- `pool_id`
- `location_type`
- `flow_type`
- `pass_time`

## 视频流配置

RTSP 输入和输出地址主要配置在：

```text
FishPass/main.cpp
```

当前代码中使用了本地 RTSP 地址，例如：

```text
rtsp://127.0.0.1:9994/live/fish1
rtsp://127.0.0.1:9994/live/fish2
rtsp://127.0.0.1:9994/live/fish3
rtsp://127.0.0.1:9994/live/output
rtsp://127.0.0.1:9994/live/output_origin
```

部署时需要保证 RTSP 服务可用，并按现场摄像头或流媒体服务地址修改这些配置。

## 编译步骤

1. 克隆仓库。
2. 准备 CUDA、TensorRT、OpenCV、FFmpeg、MySQL Connector/C++ 等依赖。
3. 按本机路径调整 `FishPass/FishPass.vcxproj` 中的 include/lib 配置。
4. 准备 TensorRT `.engine` 模型文件，并放到程序可找到的位置。
5. 打开 `FishPass.sln`。
6. 选择 `Release | x64`。
7. 编译并运行。

## 运行 WebDashboard

进入 `WebDashboard/`：

```powershell
cd WebDashboard
python -m pip install flask flask-cors mysql-connector-python
python server.py
```

服务默认监听：

```text
http://127.0.0.1:5000
```

前端页面会请求：

```text
http://127.0.0.1:5000/api/logs
```

可以直接打开 `WebDashboard/index.html` 查看页面。页面图表依赖 ECharts CDN，离线环境需要改成本地引入。

## 上传代码前建议

提交前建议确认：

```powershell
git status --short
git check-ignore -v 3rdparty/ model/ FishPass/crops/ FishPass/output_record.mp4
```

如果模型、视频、截图或编译产物已经出现在 `git status` 的待提交列表中，先从 Git 索引移除它们，再提交代码。
