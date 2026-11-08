#!/usr/bin/env bash

set -euo pipefail

DB_USER="${DB_USER:-checker}"
DB_NAME="${DB_NAME:-factchecks}"
RAILWAY_DB_SERVICE="${RAILWAY_DB_SERVICE:-Postgres}"
DRY_RUN="${DRY_RUN:-0}"

cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

command -v docker  >/dev/null || die "docker not found"
command -v railway >/dev/null || die "railway CLI not found: https://docs.railway.com/guides/cli"

docker compose ps --status running --format '{{.Service}}' 2>/dev/null | grep -qx postgres \
  || die "the local postgres container is not running. Start it with: docker compose up -d postgres"

say "Local index"
LOCAL_ROWS=$(docker compose exec -T postgres \
  psql -U "$DB_USER" -d "$DB_NAME" -tAc "select count(*) from fact_checks;" | tr -d '[:space:]')
[ "$LOCAL_ROWS" -gt 0 ] 2>/dev/null \
  || die "the local fact_checks table is empty. Harvest first: see ml/src/data/harvest_factchecks.py"
echo "fact_checks rows: $LOCAL_ROWS"
docker compose exec -T postgres psql -U "$DB_USER" -d "$DB_NAME" -tAc \
  "select 'harvested up to ' || max(published_at)::date from fact_checks;"

say "Checking the deployed publishers table"
REMOTE_PUBLISHERS=$(printf 'select count(*) from publishers;\n' \
  | railway connect "$RAILWAY_DB_SERVICE" 2>/dev/null \
  | grep -oE '^\s*[0-9]+\s*$' | head -1 | tr -d '[:space:]' || true)

if [ -z "${REMOTE_PUBLISHERS:-}" ]; then
  echo "WARNING: could not read the publishers count (the tunnel output was not parseable)."
  echo "         Continuing: the load is idempotent, so a FK failure costs nothing but a retry."
elif [ "$REMOTE_PUBLISHERS" -eq 0 ]; then
  die "the deployed publishers table is empty, so every insert would fail its foreign key.
     Apply the schema first:  railway connect $RAILWAY_DB_SERVICE < infra/postgres/schema.sql"
else
  echo "deployed publishers: $REMOTE_PUBLISHERS"
fi

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT
LOAD_SQL="$WORKDIR/load-fact-checks.sql"

say "Dumping"
{
  printf '\\set ON_ERROR_STOP on\n'
  docker compose exec -T postgres pg_dump -U "$DB_USER" -d "$DB_NAME" \
    --data-only --no-owner --no-privileges \
    --inserts --rows-per-insert=500 --on-conflict-do-nothing \
    --table=fact_checks
  printf '\nSELECT count(*) AS deployed_fact_checks FROM public.fact_checks;\n'
} > "$LOAD_SQL"

echo "$(wc -l < "$LOAD_SQL") lines, $(du -h "$LOAD_SQL" | cut -f1)"

if [ "$DRY_RUN" != "0" ]; then
  say "DRY_RUN=1, nothing was loaded"
  echo "SQL kept at: $LOAD_SQL"
  trap - EXIT
  exit 0
fi

say "Loading into Railway ($RAILWAY_DB_SERVICE)"
railway connect "$RAILWAY_DB_SERVICE" < "$LOAD_SQL"

cat <<'NEXT'

== Done, but the backend is still serving the OLD index

db_factcheck.py caches the index at module level, so the running process will keep reporting
`disabled (no fact_checks rows)` until it restarts. Two steps remain:

  railway redeploy --service backend --yes
  railway ssh --service backend python -c "from app.api import db_factcheck as d; print(d.status())"

The second must report an enabled index with the row count above. If it still says `disabled`,
the restart did not happen - not the load.
NEXT
