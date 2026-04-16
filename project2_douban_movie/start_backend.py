#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
WEB_APP_DIR = PROJECT_ROOT / "web_app"
REQUIREMENTS_FILE = WEB_APP_DIR / "requirements.txt"
VENV_DIR = WEB_APP_DIR / "venv"
STAMP_FILE = VENV_DIR / ".requirements.sha256"

DEFAULT_MIRRORS = [
    ("Tsinghua", "https://pypi.tuna.tsinghua.edu.cn/simple"),
    ("Aliyun", "https://mirrors.aliyun.com/pypi/simple"),
    ("Tencent", "https://mirrors.cloud.tencent.com/pypi/simple"),
]


def print_step(message: str) -> None:
    print(f"\n==> {message}", flush=True)


def run_command(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    print("+ " + " ".join(command), flush=True)
    process = subprocess.run(command, cwd=str(cwd) if cwd else None, env=env)
    return process.returncode


def require_supported_python() -> None:
    version = sys.version_info
    if version < (3, 9):
        raise SystemExit("需要 Python 3.9 或更高版本，建议使用 Python 3.10/3.11。")
    if version >= (3, 12):
        print("提示：当前 Python 版本较新。如依赖安装失败，建议换 Python 3.10 或 3.11 后重试。")


def venv_python() -> Path:
    if platform.system().lower().startswith("win"):
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def ensure_project_files() -> None:
    required_files = [
        WEB_APP_DIR / "app.py",
        REQUIREMENTS_FILE,
        PROJECT_ROOT / "thesis_analysis" / "data" / "douban_movies.xlsx",
        PROJECT_ROOT / "thesis_analysis" / "data" / "experiment_results.json",
        PROJECT_ROOT / "thesis_analysis" / "data" / "experiment_results_improved.json",
    ]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required_files if not path.exists()]
    if missing:
        raise SystemExit("项目文件不完整，缺少：\n" + "\n".join(f"- {item}" for item in missing))


def ensure_venv() -> Path:
    python_path = venv_python()
    if python_path.exists():
        return python_path

    if VENV_DIR.exists():
        print_step("发现不完整虚拟环境，正在重新创建")
        shutil.rmtree(VENV_DIR)

    print_step("创建虚拟环境")
    code = run_command([sys.executable, "-m", "venv", str(VENV_DIR)], cwd=PROJECT_ROOT)
    if code != 0:
        raise SystemExit("虚拟环境创建失败，请确认 Python 自带 venv 模块可用。")
    return python_path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mirror_candidates(custom_mirror: str | None) -> list[tuple[str, str]]:
    if custom_mirror:
        return [("Custom", custom_mirror)]
    return DEFAULT_MIRRORS


def trusted_host(index_url: str) -> str | None:
    host = index_url.replace("https://", "").replace("http://", "").split("/", 1)[0]
    return host or None


def pip_install(python_path: Path, index_url: str) -> bool:
    env = os.environ.copy()
    env.setdefault("PIP_DEFAULT_TIMEOUT", "120")
    host = trusted_host(index_url)
    base_args = [str(python_path), "-m", "pip", "install", "-i", index_url]
    if host:
        base_args.extend(["--trusted-host", host])

    upgrade_code = run_command(base_args + ["-U", "pip", "setuptools", "wheel"], cwd=PROJECT_ROOT, env=env)
    if upgrade_code != 0:
        return False

    install_code = run_command(base_args + ["-r", str(REQUIREMENTS_FILE)], cwd=PROJECT_ROOT, env=env)
    return install_code == 0


def ensure_dependencies(python_path: Path, args: argparse.Namespace) -> None:
    current_hash = file_sha256(REQUIREMENTS_FILE)
    if not args.reinstall and STAMP_FILE.exists() and STAMP_FILE.read_text().strip() == current_hash:
        print_step("依赖已安装，跳过安装步骤")
        return

    print_step("安装后端依赖")
    errors: list[str] = []
    for name, index_url in mirror_candidates(args.mirror):
        print(f"正在使用 {name} 镜像：{index_url}", flush=True)
        if pip_install(python_path, index_url):
            STAMP_FILE.write_text(current_hash)
            print("依赖安装完成。", flush=True)
            return
        errors.append(f"{name}: {index_url}")
        print(f"{name} 镜像安装失败，准备尝试下一个镜像。", flush=True)

    raise SystemExit("依赖安装失败，已尝试：\n" + "\n".join(f"- {item}" for item in errors))


def start_server(python_path: Path, args: argparse.Namespace) -> int:
    env = os.environ.copy()
    env["APP_PORT"] = str(args.port)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("FLASK_DEBUG", "1")

    if not args.no_demo_user:
        env["CREATE_DEMO_USER"] = "1"
        env["DEMO_USERNAME"] = args.demo_user
        env["DEMO_PASSWORD"] = args.demo_password
        env["DEMO_EMAIL"] = args.demo_email

    print_step("启动豆瓣电影后端分析系统")
    print(f"访问地址：http://127.0.0.1:{args.port}", flush=True)
    if not args.no_demo_user:
        print(f"默认账号：{args.demo_user} / {args.demo_password}", flush=True)
    print("首次启动会加载 Excel 数据并训练模型，等待看到 Running on 后再打开页面。")
    print("停止服务：在这个窗口按 Ctrl+C。", flush=True)

    try:
        return subprocess.run([str(python_path), "app.py"], cwd=str(WEB_APP_DIR), env=env).returncode
    except KeyboardInterrupt:
        print("\n服务已停止。")
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键安装依赖并启动豆瓣电影后端分析系统")
    parser.add_argument("--port", type=int, default=5001, help="启动端口，默认 5001")
    parser.add_argument("--mirror", default=None, help="自定义 pip 镜像地址")
    parser.add_argument("--reinstall", action="store_true", help="强制重新安装依赖")
    parser.add_argument("--no-install", action="store_true", help="跳过依赖安装，直接启动")
    parser.add_argument("--install-only", action="store_true", help="只安装依赖，不启动服务")
    parser.add_argument("--no-demo-user", action="store_true", help="不自动创建默认演示账号")
    parser.add_argument("--demo-user", default="123456", help="默认演示账号用户名")
    parser.add_argument("--demo-password", default="123456", help="默认演示账号密码")
    parser.add_argument("--demo-email", default="demo@example.local", help="默认演示账号邮箱")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_supported_python()
    ensure_project_files()
    python_path = ensure_venv()

    if not args.no_install:
        ensure_dependencies(python_path, args)
    if args.install_only:
        print_step("安装完成，未启动服务")
        return 0
    return start_server(python_path, args)


if __name__ == "__main__":
    raise SystemExit(main())
