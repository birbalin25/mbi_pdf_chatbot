# Databricks notebook source
# MAGIC %md
# MAGIC # Extract File Metadata using AI Functions
# MAGIC
# MAGIC This notebook extracts metadata from SEC filing PDFs using Databricks AI functions:
# MAGIC - **ai_extract**: Extract the year from the file path
# MAGIC - **ai_classify**: Classify the company name based on file path using labels from the entities table
# MAGIC - **ai_classify**: Classify the document type as 10k, 8k, 10q, or Earnings Report

# COMMAND ----------

# DBTITLE 1,Install pyyaml
# MAGIC %pip install pyyaml --quiet
# MAGIC %restart_python

# COMMAND ----------

import os
import yaml

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Configuration

# COMMAND ----------

config_path = "config.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

catalog = config["global"]["catalog"]
schema = config["global"]["schema"]
source_volume_path = config["ingestion"]["source_volume_path"]
bronze_table = config["ingestion"]["bronze_layer"]["bronze_table"]
llm = config["ingestion"]["bronze_layer"]["metadata_extraction_llm"]

source_volume_path = f"/Volumes/{catalog}/{schema}/{source_volume_path}"

print(f"Catalog:             {catalog}")
print(f"Schema:              {schema}")
print(f"Source volume path:  {source_volume_path}")
print(f"Bronze table:        {catalog}.{schema}.{bronze_table}")
print(f"llm:                 {llm}")
print(f"source_volume_path:  {source_volume_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## List Files from Volume

# COMMAND ----------

from pyspark.sql import functions as F

# List all PDF files in the volume
files_df = spark.createDataFrame(
    [(f.name, f.path, f.size) for f in dbutils.fs.ls(source_volume_path) if f.name.endswith('.pdf')],
    ["file_name", "file_path", "file_size"]
)

files_df.createOrReplaceTempView("pdf_files")

print(f"Found {files_df.count()} PDF files")

display(files_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract document_type
# MAGIC

# COMMAND ----------

metadata_query = f"""
SELECT
    file_name,
    file_path,
    file_size,
    regexp_extract(split(file_path, '/')[size(split(file_path, '/')) - 1], '^([^\\-]*\\-[^\\-]*)', 1) AS document_type
FROM pdf_files
"""

metadata_df = spark.sql(metadata_query)

display(metadata_df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Bronze Table

# COMMAND ----------

bronze_df = metadata_df.select(
    F.col("file_name"),
    F.col("file_path"),
    F.col("file_size"),
    F.col("document_type"),
    F.current_timestamp().alias("ingested_at")
)

bronze_table_path = f"{catalog}.{schema}.{bronze_table}"

bronze_df.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(bronze_table_path)

print(f"Written {bronze_df.count()} records to {bronze_table_path}")

# COMMAND ----------

result_df = table(bronze_table_path)
display(result_df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Statistics

# COMMAND ----------

print("Documents by Type:")
display(result_df.groupBy("document_type").count().orderBy(F.desc("count")))
