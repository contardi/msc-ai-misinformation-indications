CREATE TABLE IF NOT EXISTS publishers (
    slug            TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    domains         TEXT[] NOT NULL DEFAULT '{}',
    mechanism       TEXT NOT NULL DEFAULT 'none',
    emits_jsonld    BOOLEAN NOT NULL DEFAULT FALSE,
    active          BOOLEAN NOT NULL DEFAULT TRUE,
    notes           TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS fact_checks (
    url             TEXT PRIMARY KEY,
    claim           TEXT NOT NULL,
    claim_is_title  BOOLEAN NOT NULL DEFAULT FALSE,
    summary         TEXT,
    rating          TEXT NOT NULL DEFAULT '',
    verdict         TEXT NOT NULL DEFAULT 'unknown'
                    CHECK (verdict IN ('true', 'partial', 'false', 'unknown')),
    publisher_slug  TEXT REFERENCES publishers(slug) ON DELETE SET NULL,
    publisher       TEXT NOT NULL DEFAULT '',
    published_at    TIMESTAMPTZ,
    source          TEXT NOT NULL DEFAULT '',
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS fact_checks_published_idx ON fact_checks (published_at DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS fact_checks_publisher_idx ON fact_checks (publisher_slug);

CREATE TABLE IF NOT EXISTS harvest_runs (
    id              BIGSERIAL PRIMARY KEY,
    publisher_slug  TEXT NOT NULL REFERENCES publishers(slug) ON DELETE CASCADE,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at     TIMESTAMPTZ,
    newest_item_at  TIMESTAMPTZ,
    items_seen      INTEGER NOT NULL DEFAULT 0,
    items_added     INTEGER NOT NULL DEFAULT 0,
    http_status     INTEGER,
    error           TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS harvest_runs_publisher_idx
    ON harvest_runs (publisher_slug, started_at DESC);

CREATE TABLE IF NOT EXISTS result_feedback (
    id              BIGSERIAL PRIMARY KEY,
    reason          TEXT NOT NULL CHECK (reason IN (
                        'evidence_irrelevant', 'evidence_missing',
                        'score_too_high', 'score_too_low',
                        'explanation_unclear', 'other')),
    band            TEXT CHECK (band IN ('low', 'inconclusive', 'high')),
    had_evidence    BOOLEAN NOT NULL,
    locale          TEXT NOT NULL CHECK (locale IN ('pt-BR', 'en-GB')),
    day             DATE NOT NULL DEFAULT CURRENT_DATE
);

CREATE INDEX IF NOT EXISTS result_feedback_day_idx ON result_feedback (day DESC);


INSERT INTO publishers (slug, name, domains, mechanism, emits_jsonld, notes) VALUES
    ('aosfatos',  'Aos Fatos',        ARRAY['aosfatos.org'],            'google_api', TRUE,
     'Emits ClaimReview JSON-LD. Also blocks AI crawlers in robots.txt, so prefer the sanctioned Google API.'),
    ('boatos',    'Boatos.org',       ARRAY['boatos.org'],              'wordpress',  FALSE,
     'Largest back catalogue (13,570 posts). No on-page JSON-LD, but the EXCERPT opens with their own verdict word (Boato/Bulo/Rumor/Hoax, 99.8% of posts) and then states the claim affirmatively - the title states its negation. Runs Spanish and English editions under /espanol/ and /english/: 38% of a 200-post sample. The harvester filters those by URL path; the bundled DataFeed corpus was already clean because it was filtered by .br TLD. No AI restrictions.'),
    ('lupa',      'Agência Lupa',     ARRAY['lupa.uol.com.br','lupa.news'], 'wordpress', FALSE,
     'Content-Signal: search=yes,ai-train=no,use=reference. Indexing and linking permitted; training is not.'),
    ('comprova',  'UOL Comprova',     ARRAY['projetocomprova.com.br','/comprova'], 'wordpress', FALSE,
     'Custom post type: /wp-json/wp/v2/publicacoes. robots.txt disallows /feed/, so use REST.'),
    ('efarsas',   'e-farsas',         ARRAY['e-farsas.com'],            'rss',        FALSE,
     'REST endpoint returned 503; RSS is the fallback. No AI restrictions.'),
    ('estadao',   'Estadão Verifica', ARRAY['estadao.com.br/estadao-verifica'], 'google_api', FALSE,
     'Blocks AI crawlers. Reach via the Google API.'),
    ('afp',       'AFP Checamos',     ARRAY['checamos.afp.com'],        'google_api', FALSE,
     'Returned 403 to a direct fetch. Reach via the Google API.'),
    ('g1',        'G1 Fato ou Fake',  ARRAY['g1.globo.com/fato-ou-fake','globo.com/fato-ou-fake'], 'rss', FALSE,
     'Added 2026-11-08 because G1 was listed in the per-checker panel while being unreachable: zero rows in the bundled corpus, zero in the harvest, and the live Google API returned nothing for claims G1 had published days earlier. robots.txt does not disallow /fato-ou-fake/, lists no AI-crawler rules and publishes sitemaps; no Content-Signal header is sent. Official feed: https://g1.globo.com/rss/g1/fato-ou-fake/ - 100 items, no pagination, so the index accumulates across scheduled runs rather than backfilling. Titles carry the verdict (#FAKE 65, #FATO 9, #NAOEBEMASSIM 2 of 100), which is why the claim and the rating both come from metadata. The remaining 24 are debate roundups ("Veja o que e #FATO ou #FAKE no debate...") carrying many claims and no single verdict; the harvester refuses them via a verdict allow-list, so a format change is skipped rather than mis-indexed.')
ON CONFLICT (slug) DO NOTHING;

ALTER TABLE fact_checks ADD COLUMN IF NOT EXISTS summary TEXT;
