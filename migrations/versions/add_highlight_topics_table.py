"""add highlight topics table

Revision ID: 2024_02_22_add_highlight_topics
Revises: add_topic_and_language_support
Create Date: 2024-02-22 10:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '2024_02_22_add_highlight_topics'
down_revision = 'add_topic_and_language_support'
branch_labels = None
depends_on = None

def upgrade():
    # Create highlight_topics table
    op.create_table(
        'highlight_topics',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('highlight_id', sa.Integer(), nullable=False),
        sa.Column('topic_id', sa.Integer(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('is_primary', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['highlight_id'], ['highlights.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['topic_id'], ['topics.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('highlight_id', 'topic_id', name='uq_highlight_topic')
    )
    
    # Create index for faster lookups
    op.create_index('idx_highlight_topics_highlight_id', 'highlight_topics', ['highlight_id'])
    op.create_index('idx_highlight_topics_topic_id', 'highlight_topics', ['topic_id'])
    op.create_index('idx_highlight_topics_is_primary', 'highlight_topics', ['is_primary'])

def downgrade():
    # Drop highlight_topics table and its indexes
    op.drop_table('highlight_topics') 