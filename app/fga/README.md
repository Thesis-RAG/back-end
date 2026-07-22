## Production

`docker-compose.prod.yml` now starts an idempotent `openfga-bootstrap` service.
It waits for OpenFGA, creates or reuses the `rag-enterprise` store, uploads or
reuses the authorization model, and persists both IDs in the shared
`openfga_config` volume. `OPENFGA_STORE_ID` and `OPENFGA_MODEL_ID` can therefore
remain empty in `.env`; no manual copy is required.

From the backend directory, run:

```bash
docker compose -f docker-compose.prod.yml up -d --build --force-recreate openfga-bootstrap api worker caddy
```

The API startup then reconciles the initial organization hierarchy, positions,
admin assignment, OpenFGA parent links, memberships, and MinIO buckets.

## Local OpenFGA API

The same idempotent script can be run manually when OpenFGA is available at the
configured `OPENFGA_URL`:

```bash
python -m app.fga.setup
```

# To check the store that was created, browse: http://localhost:8080/stores
# Result example: {"stores":[{"id":"01KMTFH0653Q23BT8R9BCA4GQN","name":"rag-enterprise","created_at":"2026-03-28T15:03:14.380041Z","updated_at":"2026-03-28T15:03:14.380041Z","deleted_at":null}],"continuation_token":""}

# To view schemas by UI, browse: https://play.fga.dev/sandbox/?fga_api_host=localhost%3A8080&fga_api_scheme=http&store=<enter-store-id-here>
# Example: https://play.fga.dev/sandbox/?fga_api_host=localhost%3A8080&fga_api_scheme=http&store=01KMTFH0653Q23BT8R9BCA4GQN
