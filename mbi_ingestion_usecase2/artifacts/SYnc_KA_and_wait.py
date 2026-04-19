# Databricks notebook source
# MAGIC %pip install --upgrade databricks-sdk
# MAGIC %restart_python

# COMMAND ----------

# from databricks.sdk import WorkspaceClient
# w = WorkspaceClient()
# KA_ID = "f1eb283a-2b17-40ec-9259-5aabcfb67737"
# ka_name = f"knowledge-assistants/{KA_ID}"
# w.knowledge_assistants.sync_knowledge_sources(name=ka_name)
# print("Triggered KA knowledge source sync.")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.knowledgeassistants import KnowledgeSourceState
import time

def sync_and_wait_for_ka_sources(
    ka_id: str,
    timeout_seconds: int = 1800,      # 30 minutes
    poll_interval_seconds: int = 10,  # 10s between checks
):
    """
    Trigger KA knowledge source sync and block until it finishes or fails.

    ka_id should be the bare ID from the UI URL, e.g. "b92dd584-1d6d-4d32-8e30-c58a32e4bc66".
    """
    w = WorkspaceClient()

    ka_name = f"knowledge-assistants/{ka_id}"

    w.knowledge_assistants.sync_knowledge_sources(name=ka_name)

    start = time.time()
    while True:
        sources = list(w.knowledge_assistants.list_knowledge_sources(parent=ka_name))

        states = {src.display_name: src.state for src in sources}
        print("Current knowledge source states:", states)

        failed = [s for s in sources if s.state == KnowledgeSourceState.FAILED_UPDATE]
        if failed:
            failed_names = ", ".join(s.display_name for s in failed)
            raise RuntimeError(f"KA sync failed for sources: {failed_names}")

        if all(s.state != KnowledgeSourceState.UPDATING for s in sources):
            print("Knowledge Assistant sync completed successfully.")
            return

        if time.time() - start > timeout_seconds:
            raise TimeoutError(
                f"Timed out waiting for KA sync after {timeout_seconds} seconds. "
                f"Last states: {states}"
            )

        time.sleep(poll_interval_seconds)

sync_and_wait_for_ka_sources("f1eb283a-2b17-40ec-9259-5aabcfb67737")

# COMMAND ----------


