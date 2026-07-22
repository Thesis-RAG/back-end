"""Idempotent OpenFGA bootstrap script.

Run with: python -m app.fga.setup
"""
import sys
import time

sys.path.insert(0, ".")

from app.fga.client import fga_client


# Ensure the FGA store/model exist and print the IDs for diagnostics.
def main():
    last_error = None
    for attempt in range(30):
        try:
            # The compose dependency guarantees start order, not readiness.
            fga_client.list_stores()
            store_id, model_id = fga_client.ensure_configured()
            print(f"OPENFGA_STORE_ID={store_id}")
            print(f"OPENFGA_MODEL_ID={model_id}")
            return
        except Exception as exc:
            last_error = exc
            print(f"Waiting for OpenFGA ({attempt + 1}/30): {exc}", flush=True)
            time.sleep(2)
    raise RuntimeError("OpenFGA bootstrap failed") from last_error

if __name__ == "__main__":
    main()
