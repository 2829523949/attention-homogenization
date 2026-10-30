# RealBind2 facts-only vs image-plus-facts 500-image cohort

- Selected cases/images: **500 / 500**
- Dataset split: `test`
- Selection seed: `realbind2-facts-image-500-v1-20261030`
- Target-side balance: `{'REL2C_LEFT': 250, 'REL2C_RIGHT': 250}`
- Left-is-A balance: `{False: 255, True: 245}`
- Same-category cases: `44`
- Previous v1.1 pilot image overlap: `0`

## Selection rule

1. Start from the frozen RealBind2 release and use split `test`.
2. Exclude families whose two referent strings are identical after whitespace/case normalization.
3. Group eligible families by source image and choose at most one family per image.
4. Rank images by SHA256 of `seed:image:<image_id>` and take the first 500.
5. Balance `REL2C_LEFT` / `REL2C_RIGHT` as 250 / 250 by selection rank.
6. No model prediction is used in selection.

## Condition contract

Both conditions receive byte-identical fact text and question text. `facts_only` has no image;
`image_plus_facts` prepends the original full image as the only additional content block.
No crop, bbox, mark, arrow, or object label is added.

## Files

- `cases.jsonl`: one self-contained case per image (facts, prompt, hash, metadata).
- `manifest.json`: dataset/protocol hashes and condition contract.
- `selection_audit.json`: exclusion, balance, and integrity audit.
- `selected_images.csv`: compact case/image index.
