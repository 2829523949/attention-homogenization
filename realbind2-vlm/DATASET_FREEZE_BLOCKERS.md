# RealBind2 freeze blockers

> Update, 2026-09-30: the owner explicitly confirmed acceptance of the audited
> 1,400-family set. The original builder imported those decisions and froze
> 1,400 families / 973 images / 9,800 QA / 2,800 composition. Exact export and
> image-hash certification passed. See FREEZE_REPORT.md. Prior unfrozen
> findings below are retained as historical evidence and are superseded.


## What exists

- A hash-identified construction database with 1,400 `AUTO_VALID` families.
- 973 unique source images; 546 contribute one family and 427 contribute two.
- A complete deterministic seven-QA schema for every selected family: 9,800
  QA rows and 2,800 composition rows in the current snapshot.
- Source image, dimensions, ground truth, dependency, and identifier integrity
  checks passing for all 1,400 families.
- The frozen builder protocol, human audit interface, and export code.

## What is missing

- Human accept/reject decisions: `HUMAN_ACCEPTED=0`.
- Resolution of the referent uniqueness gate for all 1,400 current families.
- A builder-generated frozen export (`families.jsonl`, `questions.jsonl`,
  `construction_stats.json`, split files, and release image paths).
- A `DATASET_FROZEN` event and final export hashes.

## Evidence required to freeze

1. Import explicit audit decisions produced by the existing audit interface.
2. Export only `HUMAN_ACCEPTED` families with the preserved builder.
3. Run the builder validator and the repository dataset audit on that export.
4. Generate the final manifest and `data/REALBIND2_FROZEN.sha256` from the
   exported records. Counts must be measured from that export; they must not be
   copied from the candidate snapshot.

## Certification boundary

Certified now: construction snapshot provenance, family/image relationship,
identifier uniqueness, source image integrity, seven-QA schema, and current
snapshot counts.

Not certified: release membership, human acceptance, frozen splits, or a final
RealBind2 benchmark export.

Status: `DATASET_NOT_FROZEN`. No script in this package automatically converts
`AUTO_VALID` to `HUMAN_ACCEPTED` or `FROZEN`.

