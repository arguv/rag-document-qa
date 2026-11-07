import logging

from fastapi import FastAPI
from app.api.routes import router
from app.api.security import BoundaryMiddleware


def create_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')
    application = FastAPI(
        title='RAG Document Q&A API',
        description='Document question answering using RAG',
        version='0.2.0',
        docs_url='/swagger',
        redoc_url=None,
        openapi_url='/json.json',
    )

    @application.get('/health')
    def health() -> dict[str, str]:
        return {'status': 'ok', 'service': 'rag-document-qa'}

    application.include_router(router)
    application.add_middleware(BoundaryMiddleware)
    return application


app = create_app()
