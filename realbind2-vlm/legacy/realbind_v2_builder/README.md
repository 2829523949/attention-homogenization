# RealBind-v2 construction

The builder is model-free. Metadata mining reads the existing VAW train/validation/test annotation files plus Visual Genome `objects.json` and `image_data.json`; it does not download the VG image archive. The frozen v1.1 image IDs are excluded.

```bash
python build_realbind_v2.py mine
python build_realbind_v2.py mine --refresh-metadata
python build_realbind_v2.py status
python build_realbind_v2.py download-process --max-working-gb 10 --min-free-gb 10 --target-families 1400
python build_realbind_v2.py audit
python build_realbind_v2.py audit --import-decisions audit_decisions.jsonl
python build_realbind_v2.py gc --dry-run
python build_realbind_v2.py export
```

`download-process` first completes or resumes a 50-image dry run. On pass it continues in the same invocation until 1,400 `AUTO_VALID` families, disk/network stop, or exhaustion. It can be rerun safely. Image rejection is image-level: every metadata-selected family for that image is evaluated, the rejection state and event are committed and fsynced, and only then is its temporary image deleted. An image is retained if any family passes.

All candidate referents carry `HUMAN_UNIQUENESS_CHECK_REQUIRED`; they are not final benchmark referents until a person reviews the full image and overlays. Export includes only `HUMAN_ACCEPTED` families. The HTML audit page records annotator alias, decision, standardized reject reason, timestamp, and note through the imported decision JSONL.

The construction database is `state/construction_state.sqlite`; event history, image state, family state, retry count, hashes, and human decisions are stored there. The protocol frozen during mining is `realbind_v2_protocol.yaml`.
