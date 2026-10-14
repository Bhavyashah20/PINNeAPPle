"""hstack_gifs: animations side by side in one GIF."""
import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageSequence  # noqa: E402

from pinneapple_tools.visualization.dashboard.static import hstack_gifs  # noqa: E402

RED, GREEN, BLUE, YELLOW, CYAN = (220, 30, 30), (30, 200, 40), (30, 40, 220), (230, 220, 20), (20, 210, 210)


def _gif(path, colors, size):
    frames = [Image.new("RGB", size, c) for c in colors]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    return path


def _read(path):
    with Image.open(path) as im:
        return [np.asarray(f.convert("RGB")) for f in ImageSequence.Iterator(im)]


def _old_stitch(left, right, out, fps=10):
    """The helper of examples/physics_ensemble/compare_lookahead_gif.py this function replaces."""
    a = [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(left))]
    b = [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(right))]
    frames = []
    for fa, fb in zip(a, b, strict=True):
        im = Image.new("RGB", (fa.width + fb.width, max(fa.height, fb.height)), "white")
        im.paste(fa, (0, 0))
        im.paste(fb, (fa.width, 0))
        frames.append(im.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return out


def test_same_frame_count_matches_the_old_example_helper(tmp_path):
    a = _gif(tmp_path / "a.gif", [RED, GREEN, BLUE], (8, 6))
    b = _gif(tmp_path / "b.gif", [YELLOW, CYAN, RED], (10, 4))
    new = _read(hstack_gifs([a, b], tmp_path / "new.gif"))
    old = _read(_old_stitch(a, b, tmp_path / "old.gif"))
    assert len(new) == len(old) == 3
    for f, g in zip(new, old, strict=True):
        np.testing.assert_array_equal(f, g)
    assert new[0].shape == (6, 18, 3)                                  # height of the tallest, widths added
    np.testing.assert_array_equal(new[1][:, :8], np.broadcast_to(GREEN, (6, 8, 3)))
    np.testing.assert_array_equal(new[1][:4, 8:], np.broadcast_to(CYAN, (4, 10, 3)))
    np.testing.assert_array_equal(new[1][4:, 8:], 255)                  # shorter input padded with the background


def test_different_frame_counts(tmp_path):
    a = _gif(tmp_path / "a.gif", [RED, GREEN, BLUE], (8, 6))
    b = _gif(tmp_path / "b.gif", [YELLOW, CYAN, RED, GREEN, BLUE], (10, 4))
    with pytest.raises(ValueError, match="frame counts"):
        hstack_gifs([a, b], tmp_path / "x.gif")                         # default, as the old helper: refuse
    held = _read(hstack_gifs([a, b], tmp_path / "hold.gif", mismatch="hold"))
    looped = _read(hstack_gifs([a, b], tmp_path / "loop.gif", mismatch="loop"))
    assert len(held) == len(looped) == 5
    for k in (3, 4):
        np.testing.assert_array_equal(held[k][:, :8], np.broadcast_to(BLUE, (6, 8, 3)))      # last frame held
    np.testing.assert_array_equal(looped[3][:, :8], np.broadcast_to(RED, (6, 8, 3)))         # starts again
    np.testing.assert_array_equal(looped[4][:, :8], np.broadcast_to(GREEN, (6, 8, 3)))


def test_frame_arrays_as_input(tmp_path):
    left = [np.full((5, 4, 3), c, np.uint8) for c in (RED, GREEN)]
    right = [np.full((5, 6, 3), c, np.uint8) for c in (BLUE, YELLOW)]
    out = _read(hstack_gifs([left, right, left], tmp_path / "three.gif", fps=5))
    assert len(out) == 2 and out[0].shape == (5, 14, 3)
    np.testing.assert_array_equal(out[1][:, 4:10], np.broadcast_to(YELLOW, (5, 6, 3)))
