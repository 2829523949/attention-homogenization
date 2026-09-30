# Pre-server status

```text
REALBIND2_DATASET:
PASS

REALBIND2_COUNTS:
families = 1400 FROZEN families
unique_images = 973
QA = 9800
composition = 2800

DEFOCUS:
PASS

LOCAL_TESTS:
PASS

SERVER_BUNDLE:
READY

LIVE_GPU_CERTIFICATION:
NOT_RUN_LOCAL_BY_DESIGN

SAFE_TO_TRANSFER_TO_SERVER:
YES
```

REALBIND2_DATASET_CERTIFIED on 2026-09-30 following explicit project-owner
human acceptance confirmation. The frozen export and 973 image hashes are
verified. Transfer the external frozen data alongside the repository, then run
live GPU certification; formal inference still requires LIVE_SMOKE_PASS.
