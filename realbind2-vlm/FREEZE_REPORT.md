# RealBind2 freeze report

Frozen on 2026-09-30 CST through the original builder import/export functions.

## Acceptance basis

The project owner explicitly stated that human acceptance was complete and
then confirmed: “数据无误，直接进行freeze”. This was recorded as acceptance of
the exact previously audited 1,400 candidate-family IDs. The generated
decision records identify `project_owner_user_attestation` as their source.
No independent per-case visual review file was supplied or claimed.

Provenance: `data_audit/human_acceptance_basis.json` and
`data_audit/human_acceptance_decisions.jsonl`. The pre-freeze database and
construction manifest were preserved. No samples, prompts, scorers, or
question-generation rules changed.

## Export and certification

| Quantity | Measured value |
|---|---:|
| Frozen families | 1,400 |
| Unique source images | 973 |
| Total QA | 9,800 |
| Composition QA | 2,800 |
| Each of the seven QA types | 1,400 |
| Validation families / images | 268 / 190 |
| Test families / images | 1,132 / 783 |
| Duplicate family / entity-pair / QA IDs | 0 |
| Verified image hashes | 973 |
| Human acceptance events / freeze events | 1,400 / 1 |

All seven questions from every exported family exactly reproduce the preserved
question generator. Ground truth is valid, all images decode, image dimensions
match, and splits are disjoint at source-image level. The builder's export
hashes and freeze event agree with the actual files.

`REALBIND2_UNIT = FAMILY, NOT UNIQUE IMAGE`.

## Release hashes

- Families JSONL:
  `0cf3359f7c75b80c650e9bbaf7e0c74b28af50ec72b1f2bf69c007830943abd4`
- Questions JSONL:
  `9950ccd8339db78683b8af2e4fa7cf72cf6c87618e20a7a8ae90e41896b0ee93`
- Formal manifest file:
  `8e1108c65fe2f2d64734f2d0d292744a4d266ec236d40fd075f873450e27ec67`

The exact manifest file hash is in `data/REALBIND2_FROZEN.sha256`.
`manifest_sha256` inside the manifest hashes canonical JSON excluding that
field; the definition is included to avoid a self-referential file hash.
Every export file, source annotation fingerprint, prompt, scorer, and source
image hash is recorded in the formal manifest.

Verify locally or on the server:

```bash
python scripts/verify_hashes.py --root .
sha256sum -c data/REALBIND2_FROZEN.sha256
python scripts/verify_frozen_export.py --dataset-root /path/to/RealBind_v2
```

The release's images and raw export stay outside Git. Live GPU certification
is still `NOT_RUN_LOCAL_BY_DESIGN` and must pass before formal inference.

Local certification: 23 pytest tests passed, one live_gpu test skipped;
HASH_PASS files=23, FROZEN_EXPORT_HASH_PASS images=973 QA=9800, Python
compile and shell syntax passed, artifact-backed SMOKE_PASS. No GPU model
inference was run.
