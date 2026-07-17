from django.db import migrations


def add_pgvector_search_column(apps, schema_editor):
    if schema_editor.connection.vendor != "postgresql":
        return

    with schema_editor.connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cursor.execute(
            """
            ALTER TABLE ai_engine_objectembedding
            ADD COLUMN IF NOT EXISTS embedding_vector vector(512)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS ai_obj_emb_vector_hnsw_idx
            ON ai_engine_objectembedding
            USING hnsw (embedding_vector vector_cosine_ops)
            """
        )


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0001_cloud_database_extensions"),
        ("ai_engine", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(add_pgvector_search_column, migrations.RunPython.noop),
    ]
