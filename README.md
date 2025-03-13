# README

## 简介

本项目旨在研究并实现一种基于 Deep Reinforcement Learning (DRL) 的自适应码率 (ABR) 算法，以提升基于 DASH 协议的视频播放质量（QoE）。项目主要分为两个阶段：

1. **数据采集阶段**：搭建服务器并部署 DASH 视频流，记录网络带宽、缓冲状态、重缓冲事件等关键指标，并使用 Exelizer 工具进行结构化处理。
2. **DRL 模型开发与部署阶段**：基于采集的数据训练 DRL 模型，替换传统 ABR 策略并评估其在各种网络条件下的表现。

---

## 目录

1. [环境准备](#环境准备)  
2. [系统架构](#系统架构)  
3. [第一阶段：数据采集](#第一阶段数据采集)  
   - [1. 启动 Nginx 服务器](#1-启动-nginx-服务器)  
   - [2. 部署多码率 DASH 内容](#2-部署多码率-dash-内容)  
   - [3. DASH 客户端与日志记录](#3-dash-客户端与日志记录)  
   - [4. 使用 Exelizer 工具收集数据](#4-使用-exelizer-工具收集数据)  
4. [第二阶段：DRL 模型开发与部署](#第二阶段drl-模型开发与部署)  
   - [1. 定义 DRL 相关要素](#1-定义-drl-相关要素)  
   - [2. 训练 DRL 模型](#2-训练-drl-模型)  
   - [3. 将 DRL 策略整合到 DASH 客户端](#3-将-drl-策略整合到-dash-客户端)  
5. [测试与评估](#测试与评估)  
6. [常见问题](#常见问题)  
7. [许可证](#许可证)

---

## 环境准备

- **操作系统**: Ubuntu 或其他 Linux 发行版（Windows/MacOS 也可，但需相应调整安装方式）
- **编程环境**: 
  - Python 3 (建议使用 Anaconda/Miniconda 管理依赖)
  - Node.js / npm（如需自定义 DASH.js 前端逻辑）
- **服务器**: Nginx
- **网络仿真工具 (可选)**: `tc` / `netem` / Mininet，用于模拟网络带宽、时延、丢包等
- **深度学习/强化学习库**: TensorFlow 或 PyTorch
- **日志与可视化**: Exelizer

---

## 系统架构

1. **Nginx 服务器**：用于部署多码率视频内容。  
2. **DASH.js 客户端**：在网页中播放 DASH 流，并记录实时网络信息（带宽/吞吐量、缓冲区、码率选择等）。  
3. **Exelizer**：对从 DASH.js 客户端采集到的日志进行解析和可视化，形成训练用数据集。  
4. **DRL 模型**：基于深度强化学习算法 (如 DQN, PPO 或 A3C)，学习并输出最优的自适应码率策略。  
5. **可选网络仿真/测试**：使用 `tc` 或 Mininet 等工具以不同的网络状态测试系统性能。

---

## 第一阶段：数据采集

### 1. 启动 Nginx 服务器

1. **安装 Nginx**
   ```bash
   sudo apt-get update
   sudo apt-get install nginx
   ```
2. **测试 Nginx 是否运行**  
   访问 http://127.0.0.1 或者 http://<服务器IP>，若能看到默认欢迎页，说明 Nginx 已正常启动。

3. **目录结构（示例）**  
   - 你可以将项目相关文件放在 `/var/www/html/dash_content/` (或在 Nginx 配置文件中自定义)。  
   - 默认 Nginx 配置文件一般位于 `/etc/nginx/sites-available/default`。如需自定义根目录，可编辑其中的 `root` 路径。

### 2. 部署多码率 DASH 内容

1. **准备视频文件**  
   - 使用 ffmpeg 生成多码率、分段化的 DASH 视频（例如 240p, 360p, 720p, 1080p…），并生成相应的 `.mpd` 文件。
   - 将生成的文件 (含 `.mpd` 和分段文件) 放置于 Nginx 指定的根目录下，示例：
     ```
     /var/www/html/dash_content/
       ├── video_240p.mp4
       ├── video_360p.mp4
       ├── video_720p.mp4
       ├── video_1080p.mp4
       ├── manifest.mpd
       └── ...
     ```
2. **验证可访问性**  
   - 在浏览器中打开 http://<服务器IP>/dash_content/manifest.mpd，检查文件能否正常访问。

### 3. DASH 客户端与日志记录

1. **获取/修改 DASH.js**
   - 下载或克隆 [DASH.js](https://github.com/Dash-Industry-Forum/dash.js) 源码。  
   - 编译或使用官方预编译文件（`dist/dash.all.debug.js` / `dist/dash.all.min.js`）。
2. **自定义日志逻辑**  
   - 在 `MediaPlayer` 或 `AbrController` 中添加日志输出代码，记录：
     - 实时带宽 / 吞吐量  
     - 当前缓冲区大小  
     - 选择的码率 / 分辨率  
     - 重缓冲事件及时长  
   - 可将日志通过控制台输出、或通过 AJAX/WebSocket 提交到后端，以便后续汇总分析。
3. **前端页面示例**  
   - 新建 `index.html` 并引入 dash.js、manifest.mpd 等，示例如下：
     ```html
     <!DOCTYPE html>
     <html>
     <head>
       <meta charset="UTF-8">
       <title>DASH.js Test</title>
       <script src="dash.all.min.js"></script>
     </head>
     <body>
       <video id="videoPlayer" controls autoplay></video>
       <script>
         const url = 'http://<服务器IP>/dash_content/manifest.mpd';
         const player = dashjs.MediaPlayer().create();
         player.initialize(document.getElementById("videoPlayer"), url, true);

         // 示例：监听事件，输出日志
         player.on(dashjs.MediaPlayer.events['QUALITY_CHANGE_REQUESTED'], function(e){
           console.log('Quality change requested:', e);
           // 也可将 e.detail 发送到后端
         });
       </script>
     </body>
     </html>
     ```

### 4. 使用 Exelizer 工具收集数据

1. **安装 Exelizer**  
   - 具体安装方式参考官方文档（若支持 npm，可 `npm install -g exelizer`）。
2. **日志处理**  
   - 将从客户端获取的日志（含时间戳、带宽、缓冲区、码率信息等）输入 Exelizer。  
   - 生成结构化数据 (CSV、JSON 等)，包括 `timestamp, throughput, buffer_level, chosen_bitrate, rebuffering_duration, …`。
3. **结果验证**  
   - 确认生成的数据包含你在 DRL 训练阶段所需的特征：  
     - 状态特征 (带宽、缓存…)，以及  
     - 对应的 QoE 指标 (重缓冲、质量切换等)。

---

## 第二阶段：DRL 模型开发与部署

### 1. 定义 DRL 相关要素

1. **状态空间 (State)**  
   - 当前网络带宽、网络时延、缓冲区占用量、上一次播放的码率等。  
2. **动作空间 (Action)**  
   - 若有 5 个码率层级，动作空间可设为 {0, 1, 2, 3, 4}，对应不同清晰度/比特率。  
3. **奖励函数 (Reward)**  
   ```text
   R = w1 * bitrate - w2 * rebuffering - w3 * quality_switches
   ```
   - 根据实际需求设置 w1, w2, w3，平衡清晰度与流畅度。

### 2. 训练 DRL 模型

1. **选择框架**: TensorFlow 或 PyTorch。  
2. **构建训练脚本**:
   - 从 Exelizer 输出的数据集中获取状态、动作、奖励 (或在在线仿真环境中交互)。  
   - 使用 DQN、PPO 或 A3C 等算法进行反复迭代训练，更新网络参数。
3. **训练超参数** (示例):
   - 学习率: 0.001  
   - 折扣因子: 0.99  
   - batch size, replay buffer size（若 DQN）等。  
4. **模型输出**:
   - 最终得到一个策略网络，用于在给定状态时输出最优/近似最优的动作（码率）。

### 3. 将 DRL 策略整合到 DASH 客户端

1. **推理过程**:
   - 当客户端准备请求新的视频分段时，将状态输入 DRL 模型，得到最优码率的索引。  
2. **替换原有 ABR**:
   - 在 DASH.js 中移除/屏蔽默认的 ABR 逻辑 (如 ThroughputRule、BufferOccupancyRule)，改为使用 DRL 策略。  
3. **在线测试**:
   - 观察日志和实际播放表现，确认 DRL 策略能否根据网络变化实时调整码率。

---

## 测试与评估

1. **QoE 指标**  
   - 平均码率、重缓冲比率、质量切换次数。  
2. **网络效率**  
   - 吞吐量利用率、对网络变化的响应度。  
3. **基准算法对比**  
   - 传统 GA (Genetic Algorithm) 或 BOLA、MPC 等开源/自带的 ABR 策略。  
   - 比较不同算法在相同网络条件下的结果，验证 DRL 的改进程度。  
4. **Exelizer 报告**  
   - 使用 Exelizer 分析播放日志并可视化曲线 (码率随时间、缓冲区随时间)，便于直观比较。

---

## 常见问题

1. **如何模拟不稳定网络？**  
   - 使用 `tc`/`netem` 注入限速、丢包、抖动等；或使用 Mininet 构建更复杂的拓扑。  
2. **训练时间过长？**  
   - 可先在小规模数据集或短视频切片测试，逐步扩大规模；或使用 GPU 加速。  
3. **无法正常生成 `.mpd` 文件？**  
   - 请确认 ffmpeg 命令参数，或参考 [Shaka Packager](https://github.com/shaka-project/shaka-packager) 等工具生成多码率 DASH 内容。  
4. **Nginx 配置错误导致无法访问？**  
   - 检查 `nginx.conf` 与站点配置文件（如 `/etc/nginx/sites-available/default`），确保 `root` 路径正确且 `.mpd` 文件允许访问。

---

## 许可证

本项目使用 [MIT License](https://opensource.org/licenses/MIT)（或其他许可证）进行开源分发。  
如需引用本项目成果或二次开发，请遵循相应许可证协议并保留原作者信息。

---

**祝您复现顺利！** 若在使用过程中遇到任何问题，欢迎提出 Issue 或 Pull Request，一起完善本项目。