# Databricks notebook source
# MAGIC %md
# MAGIC # Summarize Page Contents
# MAGIC
# MAGIC This notebook summarizes the content of high-quality pages using an LLM:
# MAGIC 1. Filter pages with quality_score = 1 from the gold layer
# MAGIC 2. Concatenate metadata (company_name, document_type, year) to page content
# MAGIC 3. Summarize using ai_query() with configured LLM and prompt
# MAGIC 4. Write to platinum layer table
# MAGIC
# MAGIC **Input:** `sec_docs_gold_pages_with_quality_scores` (gold layer)
# MAGIC **Output:** `sec_docs_platinum` (platinum layer)

# COMMAND ----------

# MAGIC %pip install pyyaml --quiet
# MAGIC %restart_python

# COMMAND ----------

import os
import yaml
from pyspark.sql.functions import *
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Configuration

# COMMAND ----------

config_path = "config.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

catalog = config["global"]["catalog"]
schema = config["global"]["schema"]

# Gold layer config (input)
quality_scored_table = "docs_gold_pages_with_quality_scores"

# Platinum layer config (output)
platinum_config = config["ingestion"]["platinum_layer"]
summarization_llm = platinum_config["summarization_llm"]
summarization_prompt = platinum_config["summarization_prompt"]
platinum_table = platinum_config["platinum_table"]

file_description_path = config["usecase1"]["file_description_path"]
file_description_path_volume = f"/Volumes/{catalog}/{schema}/{file_description_path}"

print(f"Configuration loaded.")
print(f"  Input:                        {catalog}.{schema}.{quality_scored_table}")
print(f"  Output:                       {catalog}.{schema}.{platinum_table}")
print(f"  Summarization LLM:            {summarization_llm}")
print(f"  file_description_path_volume: {file_description_path_volume}")
print(f"  Summarization prompt:         {summarization_prompt}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Spark Session

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Source Table and Add UUIDs

# COMMAND ----------

quality_scored_table_path = f"{catalog}.{schema}.{quality_scored_table}"
quality_df = spark.table(quality_scored_table_path)

# Add UUID to each row before any filtering or processing
quality_df = quality_df.withColumn("chunk_id", F.expr("uuid()"))

total_pages = quality_df.count()
print(f"Total pages in source table: {total_pages}")
print(f"Added unique chunk_id (UUID) to each row")

display(quality_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Purpose CSV and Join File Descriptions

# COMMAND ----------

purpose_df = spark.read.option("header", "true").option("multiLine", "true").csv(file_description_path_volume)
display(purpose_df)

# COMMAND ----------

quality_df = quality_df.join(purpose_df, on="file_name", how="left")
print(f"Joined file_description from usecase1-purpose.csv")
display(quality_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filter Quality Pages

# COMMAND ----------

high_quality_df = quality_df.filter(F.col("quality_score") == 1)

high_quality_count = high_quality_df.count()

print(f"High-quality pages (score=1): {high_quality_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Content with Metadata Context
# MAGIC
# MAGIC Concatenate company_name, document_type, and year to page_content
# MAGIC to provide context for the summarization.

# COMMAND ----------

# Create enriched content with metadata context
enriched_df = high_quality_df.withColumn(
    "enriched_content",
    F.concat(
        F.lit("\nDocument Type: "), F.col("document_type"),
        F.lit("\nFile Description: "), F.coalesce(F.col("file_description"), F.lit("")),
        F.lit("\n\nPage Content:\n"), F.col("page_content")
    )
)

display(enriched_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summarize Pages with ai_query()

# COMMAND ----------

# Create temp view for SQL query
enriched_df.createOrReplaceTempView("enriched_pages")

# Escape single quotes in the prompt for SQL
escaped_prompt = summarization_prompt.replace("'", "''")

# Build the summarization query using ai_query
summarization_query = f"""
SELECT
    chunk_id,
    file_name,
    file_path,
    document_type,
    file_description,
    page_id,
    page_content,
    image_uri,
    enriched_content,
    element_count,
    element_types,
    ai_query(
        '{summarization_llm}',
        CONCAT('{escaped_prompt}', '\\n\\n', enriched_content)
    ) AS page_summary,
    current_timestamp() AS summarized_at
FROM enriched_pages
"""

summarized_df = spark.sql(summarization_query)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Platinum Layer Table

# COMMAND ----------

final_df = summarized_df.selectExpr(
    "chunk_id",
    "file_name",
    "file_path",
    "document_type",
    "file_description",
    "page_id",
    "concat(page_content, '\\n\\n','summary','\\n', page_summary) AS page_content_final",
    "page_content",
    "page_summary",
    "image_uri",
    "element_count",
    "element_types",
    "summarized_at"
)

platinum_table_path = f"{catalog}.{schema}.{platinum_table}"

final_df.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(platinum_table_path)

print(f"Summarized pages written to {platinum_table_path}")

# COMMAND ----------

result_df = spark.table(platinum_table_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Statistics

# COMMAND ----------

total_summarized = result_df.count()
unique_docs = result_df.select("file_name").distinct().count()

print("=" * 50)
print("SUMMARIZATION SUMMARY")
print("=" * 50)
print(f"Documents processed: {unique_docs}")
print(f"Pages summarized: {total_summarized}")

# COMMAND ----------

# Summary length statistics
summary_stats = result_df.agg(
    F.avg(F.length("page_summary")).alias("avg_summary_length"),
    F.min(F.length("page_summary")).alias("min_summary_length"),
    F.max(F.length("page_summary")).alias("max_summary_length")
).collect()[0]

print(f"\nSummary length statistics:")
print(f"  Average: {summary_stats['avg_summary_length']:.0f} characters")
print(f"  Min: {summary_stats['min_summary_length']} characters")
print(f"  Max: {summary_stats['max_summary_length']} characters")

# COMMAND ----------

# Content compression ratio
compression_stats = result_df.agg(
    F.avg(F.length("page_content")).alias("avg_content_length"),
    F.avg(F.length("page_summary")).alias("avg_summary_length")
).collect()[0]

compression_ratio = compression_stats["avg_content_length"] / compression_stats["avg_summary_length"]
print(f"\nCompression ratio: {compression_ratio:.1f}x")
print(f"  (Average content: {compression_stats['avg_content_length']:.0f} chars -> Summary: {compression_stats['avg_summary_length']:.0f} chars)")
