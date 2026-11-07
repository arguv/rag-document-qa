"""Indexes for filtered semantic retrieval."""
from alembic import op

revision = '0002_search_indexes'
down_revision = '0001_vector_storage'
branch_labels = None
depends_on = None


def upgrade():
    op.create_index('ix_documents_embedding_model', 'documents', ['embedding_model'])
    op.execute('CREATE INDEX ix_document_chunks_embedding_hnsw ON document_chunks USING hnsw (embedding vector_cosine_ops)')


def downgrade():
    op.drop_index('ix_document_chunks_embedding_hnsw', table_name='document_chunks')
    op.drop_index('ix_documents_embedding_model', table_name='documents')
