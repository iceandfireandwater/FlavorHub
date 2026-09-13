"""
启动 GustoBot 聊天系统

同时启动后端 API 服务器和前端 Vite 开发服务器
"""
import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web"
sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_env_hint() -> None:
    if os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return
    print("⚠️  警告: 未设置 LLM_API_KEY (或 OPENAI_API_KEY) 环境变量")
    print("   请先复制并编辑 .env：cp .env.example .env")


def start_backend() -> subprocess.Popen:
    """启动后端 FastAPI 服务器"""
    print("🚀 启动后端服务器...")
    _ensure_env_hint()

    # 启动 uvicorn
    cmd = [
        sys.executable, "-m", "uvicorn",
        "gustobot.main:application",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8000"
    ]
    return subprocess.Popen(cmd, cwd=PROJECT_ROOT)


def start_frontend() -> subprocess.Popen:
    """启动前端 Vite 开发服务器"""
    print("🌐 启动前端 Vite 服务器...")

    if not WEB_ROOT.exists():
        raise RuntimeError(f"web 目录不存在: {WEB_ROOT}")

    vite_port = os.getenv("VITE_PORT", "5173")

    print(f"WEB_ROOT: {WEB_ROOT}")
    print(f"Vite 服务器端口: {vite_port}")

    # Install deps if needed
    if not (WEB_ROOT / "node_modules").exists():
        print("📦 安装前端依赖 (npm install)...")
        subprocess.run(["npm", "install"], cwd=WEB_ROOT, check=True)

    cmd = ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", str(vite_port)]
    return subprocess.Popen(cmd, cwd=WEB_ROOT)


def open_browser() -> None:
    """打开浏览器"""
    time.sleep(3)  # 等待服务器启动
    vite_port = os.getenv("VITE_PORT", "5173")
    webbrowser.open(f"http://localhost:{vite_port}/")

def main():
    """主函数"""
    print("="*60)
    print("🤖 GustoBot 智能菜谱助手")
    print("="*60)
    print("\n正在启动服务...\n")

    backend_proc = start_backend()
    time.sleep(1)
    frontend_proc = start_frontend()

    open_browser()

    print("\n" + "="*60)
    print("✅ 服务已启动!")
    print("\n访问地址:")
    vite_port = os.getenv("VITE_PORT", "5173")
    print(f"  • 前端界面: http://localhost:{vite_port}/")
    print("  • API 文档: http://localhost:8000/docs")
    print("\n使用说明:")
    print("  1. 在浏览器中打开前端界面")
    print("  2. 可以选择全屏模式或右下角小部件")
    print("  3. 输入问题，系统会自动路由并回复")
    print("\n按 Ctrl+C 停止服务")
    print("="*60)

    try:
        while True:
            if backend_proc.poll() is not None:
                raise RuntimeError("后端进程已退出")
            if frontend_proc.poll() is not None:
                raise RuntimeError("前端进程已退出")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 正在停止服务...")
        for proc in (frontend_proc, backend_proc):
            try:
                proc.terminate()
            except Exception:
                pass
        for proc in (frontend_proc, backend_proc):
            try:
                proc.wait(timeout=10)
            except Exception:
                pass
        sys.exit(0)

if __name__ == "__main__":
    main()
