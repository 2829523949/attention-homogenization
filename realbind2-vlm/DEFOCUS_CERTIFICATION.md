# DEFOCUS certification

Audit date: 2026-09-30 CST.

Status: `DEFOCUS_CERTIFIED` for the recovered historical pair-defocus image
transform. This certifies image-transform equivalence. Its historical formal
experiment used evaluator/oracle family boxes; it is not evidence that ground
truth boxes are permitted in runtime repair.

## Historical source and evidence

| Artifact | SHA256 |
|---|---|
| `legacy/defocus_history/qwen35_dah_oracle_eval.py` | `f6c47e15190603172074e66253307303ffbdc0ab04b8a41dfdcb932e84935680` |
| `legacy/defocus_history/qwen35_dah_oracle_v1/config.json` | `ff173c5cd44187a353034ff701581b4c42102e56afce09ff3db760b56376e00a` |
| `legacy/defocus_history/qwen35_dah_oracle_v1/image_transform_audit.json` | `ee1093101ccb23499050eb571d91a02e2b7c7f45a6dd4f4cd4ca2b8a09ce1e15` |
| Recovered implementation `src/failure/defocus_frozen.py` | `22f24ba28151092968824567566df8aaa896028bbd95b5e57e63f370f81baecd` |
| Regression fixture manifest | `d002595bf7305fe540dcb8448892005766d192642fe3adfdaff6cc8f3157f052` |

The historical transform audit covers 100 families and 300 derived images and
records deterministic, resolution-preserving generation. Five retained
input/output pairs were copied as regression fixtures. The recovered
implementation reproduces all five expected PNG files both pixel-for-pixel and
byte-for-byte.

## Frozen protocol

1. Target support is the union of the A and B target boxes.
2. The support uses bounding boxes, not segmentation masks.
3. Each box is input as `[x, y, width, height]`, expanded by 10% in both axes,
   and clipped at the original image boundary.
4. Original pixels inside the union are preserved, with a soft feather at the
   boundary.
5. The background is Pillow `GaussianBlur` with radius
   `0.025 * min(original_width, original_height)` pixels.
6. Each box mask is feathered with Gaussian radius
   `0.010 * min(original_width, original_height)` pixels.
7. There is no transform-side resize. Model processor resize occurs later.
8. Output is RGB PNG.
9. Encoding uses `optimize=False`, `compress_level=6`; JPEG quality is not
   applicable.
10. The transform is deterministic for fixed Pillow behavior and input bytes.

The frozen protocol hash list is `data/DEFOCUS_PROTOCOL.sha256`.

## Historical REMOVE == FOCUS bug

A separate repair-view branch contained a real bug: 22 of 122 inspected
`remove`/`focus` output pairs were byte-identical. Historical change notes say
`remove.py` was repaired by replacing target-preserving `auto_defocus` with
target-region neutral-fill suppression. The preserved current harness tests
assert that REMOVE and FOCUS differ.

That defect concerned repair-view semantics. It does not change the recovered
formal DaH pair-defocus transform above, and the buggy output is not used as a
certification fixture.

