"""
GustoBot FastAPI application entry point.
"""

import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from gustobot.application.services.lightrag_service import get_lightrag_service
from gustobot.config import settings
from gustobot.infrastructure.core import configure_logging
from gustobot.infrastructure.core.database import Base, engine
from gustobot.interfaces.http import knowledge_router, lightrag_router
from gustobot.interfaces.http.knowledge_router import get_neo4j_qa_service
from gustobot.interfaces.http.v1 import api_router as api_v1_router

# Configure logging before app creation
configure_logging(debug=settings.DEBUG)

# Application instance --------------------------------------------------------
application = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Smart culinary assistant powered by a multi-agent architecture",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global middleware -----------------------------------------------------------
application.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，方便开发和测试
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers --------------------------------------------------------------------
application.include_router(knowledge_router.router, prefix=settings.API_V1_PREFIX)
application.include_router(lightrag_router.router, prefix=settings.API_V1_PREFIX)
application.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)

# Static files ---------------------------------------------------------------
# 创建上传目录
uploads_dir = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
os.makedirs(uploads_dir, exist_ok=True)

# 挂载静态文件
application.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

@application.get("/")
async def root_redirect():
    """Redirect to API docs.

    The web UI is a separate Vite app under `web/` and is typically run via `npm run dev`.
    """
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


@application.get("/favicon.ico")
async def favicon():
    """Return an empty favicon to avoid noisy 404s in dev."""
    from fastapi.responses import Response
    return Response(content=b"", media_type="image/x-icon")


@application.get("/api")
async def root() -> dict:
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
    }


@application.get("/health")
async def health_check() -> dict:
    return {"status": "healthy", "version": settings.APP_VERSION}


@application.on_event("startup")
async def startup_event() -> None:
    logger.info("Starting {} v{}", settings.APP_NAME, settings.APP_VERSION)
    logger.info("Debug mode: {}", settings.DEBUG)
    logger.info("API docs available at http://{}:{}/docs", settings.HOST, settings.PORT)

    # Auto-create database tables on startup
    import gustobot.infrastructure.persistence.db.models  # noqa: F401 - Import all models to register them with Base
    logger.info("Creating database tables if not exist...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready")

    # LangGraph checkpoint 连接池（持久化会话上下文）
    from gustobot.application.agents.lg_builder import prepare_checkpointer
    await prepare_checkpointer()


@application.on_event("shutdown")
async def shutdown_event() -> None:
    logger.info("Shutting down {}", settings.APP_NAME)

    # Ensure Neo4j connections are closed gracefully
    try:
        service = get_neo4j_qa_service()
        service.close()
        get_neo4j_qa_service.cache_clear()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Failed to close Neo4j service cleanly: {exc}")

    # Cleanup LightRAG resources
    try:
        lightrag_service = get_lightrag_service()
        await lightrag_service.cleanup()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Failed to cleanup LightRAG service: {exc}")


@application.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception on {} {}: {}", request.method, request.url, exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc) if settings.DEBUG else "An unexpected error occurred",
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        "gustobot.main:application",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )

# Backwards-compatible alias for tooling expecting `app` symbol
app = application
