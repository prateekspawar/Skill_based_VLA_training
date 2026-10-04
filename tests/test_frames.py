import math

import torch

from so101_lang.robots import frames
from so101_lang.robots.so101 import SO101Adapter


def random_quat(n):
    q = torch.randn(n, 4, dtype=torch.float64)
    return q / q.norm(dim=-1, keepdim=True)


def test_quat_matrix_roundtrip():
    q = random_quat(1000)
    q2 = frames.matrix_to_quat(frames.quat_to_matrix(q))
    assert torch.allclose(q2.abs(), q.abs(), atol=1e-6) or torch.allclose(
        (q2 * q).sum(-1).abs(), torch.ones(1000, dtype=torch.float64), atol=1e-6
    )


def test_rot6d_roundtrip():
    m = frames.quat_to_matrix(random_quat(100))
    assert torch.allclose(frames.matrix_from_rot6d(frames.rot6d_from_matrix(m)), m, atol=1e-6)


def test_adapter_roundtrip():
    ad = SO101Adapter()
    n = 50
    head = frames.make_transform(torch.randn(n, 3, dtype=torch.float64), frames.quat_to_matrix(random_quat(n)))
    link_pos, link_quat = torch.randn(n, 3, dtype=torch.float64) * 0.2, random_quat(n)
    joint = torch.rand(n, dtype=torch.float64) * 1.745
    state = ad.state(link_pos, link_quat, joint, head)
    cmd = ad.ik_command(state, head)
    assert torch.allclose(cmd[:, :3], link_pos, atol=1e-6)
    assert torch.allclose((cmd[:, 3:7] * link_quat).sum(-1).abs(), torch.ones(n, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(cmd[:, 7], 1 - joint / 1.745, atol=1e-6)


if __name__ == "__main__":
    for f in [test_quat_matrix_roundtrip, test_rot6d_roundtrip, test_adapter_roundtrip]:
        f()
        print("ok", f.__name__)
