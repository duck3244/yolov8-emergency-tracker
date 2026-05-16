"""Tracker(Hungarian) 단위 테스트."""
import pytest

from tracker import Tracker


def test_assigns_new_ids_on_first_frame():
    t = Tracker(distance_threshold=50)
    results = t.update([[10, 10, 30, 30], [200, 200, 220, 220]])
    ids = sorted(r[4] for r in results)
    assert ids == [0, 1]


def test_id_stable_across_frames_with_small_motion():
    t = Tracker(distance_threshold=50)
    r1 = t.update([[100, 100, 150, 200], [300, 300, 350, 400]])
    r2 = t.update([[105, 105, 155, 205], [295, 295, 345, 395]])
    assert {o[4] for o in r1} == {o[4] for o in r2}


def test_object_beyond_threshold_gets_new_id():
    t = Tracker(distance_threshold=20)
    r1 = t.update([[100, 100, 120, 120]])
    # Move beyond threshold — should be treated as a brand-new object
    r2 = t.update([[300, 300, 320, 320]])
    assert r1[0][4] != r2[0][4]


def test_disappearance_pruning():
    t = Tracker(distance_threshold=50, max_disappeared=3)
    t.update([[10, 10, 30, 30]])
    for _ in range(5):
        t.update([])
    assert len(t.center_points) == 0


def test_hungarian_resolves_id_swap_between_close_objects():
    """가까이 있는 두 객체가 서로 교차할 때 greedy는 잘못 매칭하지만,
    Hungarian은 비용 합이 최소가 되는 매칭을 찾는다."""
    t = Tracker(distance_threshold=200)
    # 초기: A=(100,100), B=(150,100)
    r1 = t.update([[90, 90, 110, 110], [140, 90, 160, 110]])
    id_a, id_b = r1[0][4], r1[1][4]
    # 다음 프레임: A는 약간 오른쪽으로, B는 약간 왼쪽으로 — 서로 가까워지지만 교차 안 함
    r2 = t.update([[110, 90, 130, 110], [130, 90, 150, 110]])
    # 결과에서 A는 첫 번째 박스(왼쪽), B는 두 번째 박스(오른쪽)에 그대로 유지
    mapping = {tuple(r[:4]): r[4] for r in r2}
    assert mapping[(110, 90, 130, 110)] == id_a
    assert mapping[(130, 90, 150, 110)] == id_b
