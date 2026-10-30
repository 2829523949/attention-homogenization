`paths.yaml` must be edited before GPU runs. `release_root` must point to the
frozen RealBind2 release containing `src/repair/strong_integrated/` and
`configs/*.yaml`. The model adapters, question templates, scorer, R4 protocol,
and paired statistics are imported from that release. `reference_hashes.json`
records the implementation hashes used while preparing this runner.
