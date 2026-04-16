# 一键启动说明

## 启动后端分析系统

在项目根目录执行：

```bash
python start_backend.py
```

如果电脑上 `python` 不是 Python 3，可以改用：

```bash
python3 start_backend.py
```

脚本会自动完成：

- 创建 `web_app/venv` 虚拟环境
- 使用国内 pip 镜像安装 `web_app/requirements.txt`
- 创建默认演示账号
- 启动 Flask 后端分析系统

启动完成后打开：

```text
http://127.0.0.1:5001
```

默认账号：

```text
123456 / 123456
```

停止服务：在启动窗口按 `Ctrl+C`。

## 常用参数

换端口：

```bash
python start_backend.py --port 5002
```

强制重新安装依赖：

```bash
python start_backend.py --reinstall
```

只安装依赖，不启动：

```bash
python start_backend.py --install-only
```

指定 pip 镜像：

```bash
python start_backend.py --mirror https://pypi.tuna.tsinghua.edu.cn/simple
```

## Python 版本

建议使用 Python 3.10 或 3.11。Python 3.9 及以上可以尝试运行；如果依赖安装失败，优先换成 Python 3.10/3.11。
