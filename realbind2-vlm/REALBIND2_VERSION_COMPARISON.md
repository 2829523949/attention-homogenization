# RealBind2 version comparison

Audit date: 2026-09-30 CST.

No competing frozen, final, release, certified, or accepted RealBind2 export
was found in the current repository, its parent workspace, the construction
workspace, historical outputs, or the searched local data volumes.

## Located states

| State | Records | Size | Modified (CST) | SHA256 | Interpretation |
|---|---:|---:|---|---|---|
| Metadata candidate manifest | 18,294 unique candidates | 23,481,566 B | 2026-09-26 23:44 | `940a6f2b16899b8b2833bc244f29fc6659421da2a1b7ea46e71abd6ef7ec9c7a` | Preselection input |
| Construction SQLite | 1,400 `AUTO_VALID`; 11,902 `METADATA_VALID` | 102,039,552 B | 2026-09-26 23:44 | `3e1981cdf2cd4a81c714954069e7e2d1f6722936d7f2c6f93e7b2240ddc030c0` | Current authoritative working state |
| Human audit HTML | 1,400 review cards | 3,435,057 B | 2026-09-26 23:51 | `4cf0ffbea5f9e72324706d43155b1072c9a680ab2100736184ab39b13a304720` | Pending manual gate |
| Frozen export | 0 files | 0 B | absent | absent | Required release not produced |

All 1,400 current `AUTO_VALID` family IDs occur in the 18,294-record metadata
candidate manifest. The package manifest contains exactly those same 1,400
family IDs. This is provenance agreement, not freeze evidence.

No version was selected by timestamp. The builder contract requires explicit
human decisions and export, so none of the located intermediate states can be
renamed or treated as the final release.

