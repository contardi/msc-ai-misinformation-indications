# Indícios de desinformação

An explainable platform for surfacing indicators of political misinformation in Brazilian
Portuguese. Artefact of an MSc Artificial Intelligence dissertation, University of Essex Online
(2026).

**Live:** https://indicios-desinformacao.contardi.com.br

The platform takes a Brazilian Portuguese text and returns, in this order: published fact-checks
that match it, measured wording indicators, and (outside the public build) a classifier probability
with a plain-language explanation. It reports likelihoods and evidence, never a true or false
verdict, and nothing a user submits is stored.

## Repository layout

```
backend/app/        FastAPI service
  api/              routes; evidence retrieval (live API, bundled ClaimReview, harvested cache)
  models/           classifier loading, inference, closed-form SHAP attribution
  explanation/      deterministic plain-language template, wording indicators
  schemas/          Pydantic contract
backend/tests/      pytest suite (uses a toy model, no dataset needed)
frontend/src/       React, TypeScript, Tailwind CSS v4; Vitest suite alongside components
ml/src/data/        corpus preparation and the fact-check harvester
ml/src/training/    TF-IDF baseline and transformer fine-tuning
ml/src/evaluation/  every analysis reported in the dissertation
ml/configs/         versioned hyperparameters and seeds
ml/results/         result files cited in the dissertation
ml/artifacts/       the served model (TF-IDF + logistic regression)
infra/              Dockerfiles, Postgres schema, deployment variable templates
scripts/            labelling helpers, accessibility audit, index loader
```

## Running it

Requires Docker with Compose v2.

```bash
cp .env.example .env                 # optional; every key may stay empty
docker compose up --build            # backend on :8000, frontend on :5173
```

A lighter build without the transformer and media dependencies:

```bash
INSTALL_TRANSFORMER=0 INSTALL_MEDIA=0 docker compose up --build
```

Example request:

```bash
curl -s localhost:8000/analyse -H 'Content-Type: application/json' \
  -d '{"text": "Urgente! Compartilhe antes que apaguem essa denúncia."}'
```

The harvested fact-check cache starts empty. Populate it with the harvester in
`ml/src/data/harvest_factchecks.py`; without it that source disables itself and the rest of the
platform works.

## Tests

```bash
docker compose run --rm -v "$PWD/backend:/app" --no-deps backend python -m pytest -q
docker compose --profile tools run --rm ml-tests
docker compose run --rm --no-deps frontend npm test
docker compose run --rm --no-deps frontend npm run build
```

## Reproducing the analyses

```bash
docker compose run --rm trainer      # fetches Fake.Br and trains the served model
docker compose --profile tools run --rm analysis python ml_src/evaluation/<script>.py
```

Seeds and hyperparameters live in `ml/configs/`. Datasets and transformer weights are not in this
repository; the preparation scripts in `ml/src/data/` fetch the public corpora below.

## Data and licences

| Source | Use | Terms |
| --- | --- | --- |
| Fake.Br (Monteiro et al., 2018), github.com/roneysco/Fake.br-Corpus | training | released by the authors for research use |
| FakeRecogna 2.0 (Garcia et al., 2024), HuggingFace `recogna-nlp/fakerecogna2-extrativa` | cross-corpus evaluation | MIT |
| ClaimReview DataFeed (Google / Data Commons) | evidence retrieval; `data/processed/claimreview_br.csv` | CC BY 4.0; per-markup licence in `sdLicense` |
| Google Fact Check Tools API | live evidence retrieval | Google API Terms of Service |
| Publishers' official feeds (Boatos.org, Agência Lupa, UOL Comprova, G1 Fato ou Fake) | harvested evidence cache, not included here | each publisher's terms; retrieval and linking only |

Only structured fact-check metadata (claim, rating, publisher, URL, date) is redistributed here.
No model is trained on fact-checkers' content.

## Licence

Code: MIT, see `LICENSE`. Data keeps the terms of its source, listed above.
