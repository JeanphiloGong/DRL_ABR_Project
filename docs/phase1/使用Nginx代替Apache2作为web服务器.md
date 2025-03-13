# **如何使用 Nginx 代替 Apache2？**
### **步骤 1：安装 Nginx**
如果你还没有安装 Nginx，可以使用以下命令：
```bash
sudo apt update
sudo apt install nginx
```
安装完成后，启动 Nginx：
```bash
sudo systemctl start nginx
```
并设置开机自启：
```bash
sudo systemctl enable nginx
```
检查 Nginx 是否运行：
```bash
sudo systemctl status nginx
```
如果 Nginx 正常运行，你可以在浏览器中访问 `http://your_server_ip/`，应该会看到 Nginx 的默认欢迎页面。

---

### **步骤 2：配置 Nginx 以支持 DASH 流媒体**
你需要修改 Nginx 配置文件，让它正确处理 DASH 视频文件。

1. **打开 Nginx 配置文件**
```bash
sudo nano /etc/nginx/sites-available/default
```
或者：
```bash
sudo nano /etc/nginx/nginx.conf
```

2. **添加 DASH 视频流的配置**
在 `server {}` 块中，添加如下配置：
```nginx
server {
    listen 80;
    server_name your_server_ip;  # 或者你的域名

    root /var/www/html/dash_content;  # 存放 DASH 视频文件的目录
    index index.html;

    location / {
        try_files $uri $uri/ =404;
    }

    # 允许 CORS，防止 DASH.js 播放器跨域问题
    location /dash_content/ {
        add_header Access-Control-Allow-Origin *;
        add_header Access-Control-Allow-Methods 'GET, OPTIONS';
        add_header Access-Control-Allow-Headers 'Range';

        types {
            application/dash+xml mpd;
            video/mp4 mp4;
            video/webm webm;
        }
    }
}
```

3. **测试 Nginx 配置**
保存文件后，运行：
```bash
sudo nginx -t
```
如果没有错误，重新加载 Nginx：
```bash
sudo systemctl reload nginx
```

---

### **步骤 3：准备 DASH 视频内容**
你需要将多码率的视频文件上传到 `/var/www/html/dash_content/` 目录。可以使用 `ffmpeg` 进行转码：
```bash
ffmpeg -i input.mp4 -c:v libx264 -b:v 500k -vf scale=640:360 360p.mp4
ffmpeg -i input.mp4 -c:v libx264 -b:v 1000k -vf scale=1280:720 720p.mp4
```
然后使用 `MP4Box` 生成 MPD（DASH 清单）：
```bash
MP4Box -dash 4000 -frag 4000 -rap -segment-name segment_ -url-template -out manifest.mpd 360p.mp4 720p.mp4
```
将 `manifest.mpd` 和视频分片放入 `/var/www/html/dash_content/` 目录。

---

### **步骤 4：测试 DASH.js 播放**
在你的前端 HTML 文件中，加载 DASH.js 并播放视频：
```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.dashjs.org/latest/dash.all.min.js"></script>
</head>
<body>
    <video id="videoPlayer" controls></video>
    <script>
        var url = "http://your_server_ip/dash_content/manifest.mpd";
        var player = dashjs.MediaPlayer().create();
        player.initialize(document.querySelector("#videoPlayer"), url, true);
    </script>
</body>
</html>
```
用浏览器访问 `http://your_server_ip/index.html`，检查 DASH.js 是否能够正确播放 DASH 流。

---

### **Nginx vs. Apache2 对比**
| **对比项**   | **Nginx** | **Apache2** |
|-------------|---------|---------|
| **性能** | 更高并发处理能力，适用于流媒体 | 多进程架构，在高流量时性能较低 |
| **易用性** | 需要手动配置 `.conf` 文件 | `.htaccess` 易用，但性能不及 Nginx |
| **静态文件** | 处理静态文件更快（DASH 视频适用） | 适用于动态内容（PHP、Python） |
| **流媒体支持** | 适用于 DASH/HLS 传输 | 需要额外模块支持 |

如果你的服务器 **主要用于流媒体（DASH 视频）**，推荐使用 **Nginx**，它的静态资源处理速度比 Apache2 更快，占用资源更少。

---

### **总结**
✅ **可以使用 Nginx 代替 Apache2**  
✅ **Nginx 适用于 DASH 流媒体传输，性能更优**  
✅ **修改 Nginx 配置，添加 CORS 支持，确保 DASH.js 正常播放**  
✅ **使用 ffmpeg + MP4Box 生成多码率 DASH 视频流**  
✅ **在前端 HTML 中使用 DASH.js 播放 DASH 视频**  
