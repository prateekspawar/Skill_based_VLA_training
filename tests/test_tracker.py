import numpy as np

from so101_lang.perception.tracker import TargetTracker


def test_static_noise_and_outlier():
    rng = np.random.default_rng(0)
    tr = TargetTracker()
    truth = np.array([0.5, 0.1, -0.2])
    for i in range(10):
        tr.update(truth + rng.normal(0, 0.01, 3), 0.9, 0.5 * i)
    assert np.linalg.norm(tr.position - truth) < 0.01
    # A single wrong detection 20 cm away is rejected.
    assert tr.update(truth + np.array([0.2, 0, 0]), 0.9, 5.5) == "outlier"
    assert np.linalg.norm(tr.position - truth) < 0.015
    # Low-confidence detections are ignored.
    assert tr.update(truth + 1.0, 0.01, 6.0) == "skipped"


def test_push_is_reacquired():
    tr = TargetTracker()
    a, b = np.array([0.5, 0.1, -0.2]), np.array([0.55, 0.13, -0.2])  # pushed ~6 cm
    for i in range(6):
        tr.update(a, 0.9, 0.5 * i)
    states = [tr.update(b + np.random.default_rng(i).normal(0, 0.003, 3), 0.9, 3 + 0.5 * i) for i in range(3)]
    assert "reacquired" in states or "accepted" in states
    assert np.linalg.norm(tr.position - b) < 0.015


if __name__ == "__main__":
    test_static_noise_and_outlier()
    test_push_is_reacquired()
    print("ok")
