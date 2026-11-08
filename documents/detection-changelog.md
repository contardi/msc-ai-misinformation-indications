# Detection Changelog

This changelog summarises the main implementation changes and technical decisions made during development. It focuses on code, system behaviour and data-pipeline changes rather than experiments or academic discussion.


## Development history

| Date | Changes |
|---|---|
| 28 June | Implemented the initial TF-IDF and Logistic Regression classifier. Added text preprocessing, text-size normalisation and Portuguese stopword handling, and adjusted model regularisation. |
| 28 June | Added the fact-check retrieval layer and an evidence panel showing the status of individual checks. Kept retrieved evidence separate from the classifier score. |
| 1 July | Added training and inference pipelines for BERTugues and BERTimbau. Retained TF-IDF as the default model following comparative evaluation. |
| 27–28 July | Added cross-corpus evaluation, ranking diagnostics and probability-calibration analysis. The TF-IDF model remained the production classifier. |
| 21 August | Added scripts for ensemble evaluation, learning-curve analysis and automated fact-check retrieval evaluation. These additions did not change the production classifier. |
| 24 August | Updated dataset preprocessing to preserve publication dates, titles, URLs and categories. Added date parsing and evaluation scripts for temporal drift and topic bias. |
| 25 August | Integrated live Google Fact Check API results with local ClaimReview data. Corrected verdict handling so that unknown verdicts are not treated as false. Added a text-length baseline/control to the evaluation workflow. |
| 26 August | Added publisher harvesting scripts and PostgreSQL storage for retrieved fact-checks. Implemented URL-based deduplication, upserts, incremental harvesting and filtering. Expanded the retrieval corpus to 24,397 records. |
| 26 August | Added safeguards to identify identical experiment outputs and cases where a source contributes no records. |
| 25 September | Implemented the result cascade in `frontend/src/components/ResultCascade.tsx`. The interface presents published fact-checks, wording indicators and classifier output as distinct stages, with guidance about the limits of the available evidence. |
| 5 October | Separated pre-specified and exploratory political-direction evaluation outputs into distinct files. |
| 7 October | Added the deployed fact-check index to PostgreSQL. Updated retrieval so the checker-status panel is derived from the full deduplicated result set, while the displayed evidence list remains capped. |
| 17 October | Improved local fact-check matching with accent-insensitive tokenisation and matching stopwords. Updated the monitored publisher list to include `agencialupa.org`. |
| 17 October | Fixed the frontend Docker build context and dynamic port configuration for deployment. Updated the Nginx configuration to preserve SPA route handling and added tests for the public-build disclaimer. |
| 7 November | Added the G1 Fato ou Fake RSS harvester. Updated verdict normalisation to handle `fato` and `não é bem assim`. |
| 7 November | Added publisher feed descriptions to `fact_checks.summary` and included summaries in the retrieval index through `INDEX_SUMMARY`. |
| 7 November | Refreshed the harvested fact-check database with records from Boatos, Comprova, Lupa and G1, bringing the corpus to 29,132 records. |
| 8 November | Corrected the retrieval evaluation configuration to use summary indexing and regenerated the corresponding evaluation outputs. |

## Current implementation notes

- **Classifier:** TF-IDF with Logistic Regression remains the default production classifier.
- **Retrieval:** Fact-check evidence is retrieved from the local PostgreSQL index and, where configured, the live Google Fact Check API.
- **Evidence presentation:** Retrieved fact-checks, wording indicators and classifier output are presented separately rather than being merged into a single score.
- **Data ingestion:** Publisher harvesting supports incremental updates and URL-based deduplication. Feed summaries can be included in the retrieval index.
- **Deployment:** The frontend uses a Docker build context rooted at `frontend/`, with Nginx configured to use the deployment-provided port.
