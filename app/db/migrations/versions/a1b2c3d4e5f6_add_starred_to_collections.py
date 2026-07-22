"""add starred column to collections

Revision ID: a1b2c3d4e5f6
Revises: ca17b542ea8a
Create Date: 2026-07-22 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "ca17b542ea8a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("collections", sa.Column("starred", sa.Boolean(), server_default=sa.text("false"), nullable=False))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("collections", "starred")
