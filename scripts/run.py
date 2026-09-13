#!/usr/bin/env python
"""
GustoBot 简化启动脚本

提供命令行界面快速启动和管理服务
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

def check_requirements():
    """检查基本要求"""
    print("🔍 检查系统要求...")

    # 检查 Python 版本
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 或更高版本是必需的")
        return False
    print(f"✅ Python 版本: {sys.version}")

    # 检查 .env 文件
    env_file = Path('.env')
    if not env_file.exists():
        print("⚠️  .env 文件不存在")
        if input("是否创建示例 .env 文件? (y/n): ").lower() == 'y':
            create_env_example()
    else:
        print("✅ .env 文件存在")

    # 检查依赖
    try:
        import fastapi
        import uvicorn
        print("✅ 核心依赖已安装")
    except ImportError:
        print("❌ 缺少核心依赖，请运行: pip install -r requirements.txt")
        return False

    return True

def create_env_example():
    """创建示例 .env 文件"""
    env_content = """# OpenAI API 配置（必需）
#
# 推荐使用 LLM_* 变量（项目内部统一配置），也兼容 OPENAI_* 变量名。
# 任选其一即可。
LLM_PROVIDER=openai
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

# 兼容旧变量名（可选）
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

# 数据库配置
DATABASE_URL=sqlite:///./gustobot.db
REDIS_URL=redis://localhost:6379

# Milvus 配置（可选）
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Neo4j 配置（可选）
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# 文件上传配置
FILE_UPLOAD_MAX_MB=50

# 应用配置
DEBUG=false
HOST=0.0.0.0
PORT=8000
"""

    with open('.env', 'w') as f:
        f.write(env_content)

    print("✅ 已创建 .env 文件，请编辑后重新运行")

def start_server(mode='dev'):
    """启动服务器"""
    print(f"\n🚀 启动 GustoBot 服务器 (模式: {mode})")

    if mode == 'dev':
        cmd = [
            sys.executable, "-m", "uvicorn",
            "gustobot.main:application",
            "--reload",
            "--host", "0.0.0.0",
            "--port", "8000"
        ]
        print("📝 开发模式: 代码热重载已启用")
    else:
        cmd = [
            sys.executable, "-m", "uvicorn",
            "gustobot.main:application",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--workers", "4"
        ]
        print("🏭 生产模式: 多进程模式")

    print("🌐 服务地址: http://localhost:8000")
    print("📚 API 文档: http://localhost:8000/docs")
    print("\n按 Ctrl+C 停止服务\n")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n👋 服务已停止")

def start_docker():
    """使用 Docker 启动"""
    print("\n🐳 使用 Docker 启动服务")

    if not Path('docker-compose.yml').exists():
        print("❌ docker-compose.yml 文件不存在")
        return

    try:
        # 构建镜像
        print("📦 构建 Docker 镜像...")
        subprocess.run(["docker-compose", "build"], check=True)

        # 启动服务
        print("🚀 启动服务...")
        subprocess.run(["docker-compose", "up"], check=True)

    except subprocess.CalledProcessError as e:
        print(f"❌ Docker 启动失败: {e}")
    except KeyboardInterrupt:
        print("\n\n👋 正在停止服务...")
        subprocess.run(["docker-compose", "down"])

def test_api():
    """测试 API"""
    print("\n🧪 测试 API 连接...")

    import asyncio
    import httpx

    async def test():
        async with httpx.AsyncClient(timeout=15.0, trust_env=False) as session:
            try:
                # 测试健康检查
                resp = await session.get("http://localhost:8000/health")
                resp.raise_for_status()
                data = resp.json()
                print("✅ 健康检查通过")
                print(f"   版本: {data.get('version')}")

                # 测试聊天 API
                test_data = {
                    "message": "你好",
                    "session_id": "test",
                    "user_id": "tester"
                }

                resp = await session.post("http://localhost:8000/api/v1/chat/", json=test_data)
                resp.raise_for_status()
                data = resp.json()
                print("✅ 聊天 API 测试通过")
                print(f"   路由: {data.get('route')}")
                print(f"   回复: {data.get('message', '')[:50]}...")

            except httpx.HTTPError as e:
                print(f"❌ 连接失败: {e}")
                print("   请确保服务正在运行 (python run.py start)")

    asyncio.run(test())

def main():
    parser = argparse.ArgumentParser(
        description="GustoBot 智能菜谱助手管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py start       # 开发模式启动
  python run.py start prod  # 生产模式启动
  python run.py docker      # Docker 启动
  python run.py test        # 测试 API
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 启动命令
    start_parser = subparsers.add_parser('start', help='启动服务器')
    start_parser.add_argument(
        'mode',
        nargs='?',
        default='dev',
        choices=['dev', 'prod'],
        help='启动模式 (默认: dev)'
    )

    # Docker 命令
    subparsers.add_parser('docker', help='使用 Docker 启动')

    # 测试命令
    subparsers.add_parser('test', help='测试 API')

    # 检查命令
    subparsers.add_parser('check', help='检查系统要求')

    args = parser.parse_args()

    # 显示欢迎信息
    print("""
    ╔════════════════════════════════════════╗
    ║          GustoBot 智能菜谱助手           ║
    ║                                        ║
    ║  🤖 多智能体架构                      ║
    ║  🔍 自动路由查询                      ║
    ║  📚 知识库 + 图谱 + 统计               ║
    ╚════════════════════════════════════════╝
    """)

    if not args.command:
        parser.print_help()
        return

    # 执行命令
    if args.command == 'start':
        if check_requirements():
            start_server(args.mode)

    elif args.command == 'docker':
        start_docker()

    elif args.command == 'test':
        test_api()

    elif args.command == 'check':
        check_requirements()

if __name__ == '__main__':
    main()
