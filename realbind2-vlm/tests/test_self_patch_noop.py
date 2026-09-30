from src.mechanism.residual_patch import PatchSpec, assert_self_patch_preserved, self_patch


def test_self_patch_is_strict_noop():
    before = (1.0, 2.0, 3.0)
    spec = PatchSpec("case", "case", 12, (1,), "forward")
    after = self_patch(before, spec)
    assert after is before
    assert_self_patch_preserved(before, after)

