"""Add topic modeling and language support

Revision ID: add_topic_and_language_support
Revises: 
Create Date: 2024-02-21

This migration adds support for:
- Topic modeling with BERTopic
- Language detection and translation
- Semantic search via embeddings
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_topic_and_language_support'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create documents table
    op.create_table(
        'documents',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('filename', sa.String()),
        sa.Column('filepath', sa.String(), unique=True),
        sa.Column('title', sa.String()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'))
    )
    
    # Create topics table
    op.create_table(
        'topics',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('keywords', sa.JSON()),
        sa.Column('parent_topic_id', sa.Integer(), nullable=True),
        sa.Column('description', sa.Text()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['parent_topic_id'], ['topics.id'], ondelete='SET NULL')
    )
    
    # Create highlights table
    op.create_table(
        'highlights',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('document_id', sa.Integer(), sa.ForeignKey('documents.id')),
        sa.Column('text', sa.String()),
        sa.Column('original_language', sa.String(10)),
        sa.Column('translated_text', sa.Text(), nullable=True),
        sa.Column('embedding', sa.LargeBinary(), nullable=True),
        sa.Column('page_number', sa.Integer()),
        sa.Column('rect_x0', sa.Float()),
        sa.Column('rect_y0', sa.Float()),
        sa.Column('rect_x1', sa.Float()),
        sa.Column('rect_y1', sa.Float()),
        sa.Column('topic_id', sa.Integer(), nullable=True),
        sa.Column('topic_confidence', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['topic_id'], ['topics.id'], ondelete='SET NULL')
    )
    
    # Create watched_directories table
    op.create_table(
        'watched_directories',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('path', sa.String(500), nullable=False, unique=True),
        sa.Column('last_scan', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'))
    )

def downgrade():
    # Drop tables in reverse order
    op.drop_table('highlights')
    op.drop_table('topics')
    op.drop_table('documents')
    op.drop_table('watched_directories') 