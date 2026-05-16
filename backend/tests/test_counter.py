"""AreaCounter state-machine 카운팅 단위 테스트."""
import time

import pytest

from counter import AreaCounter


@pytest.fixture
def areas():
    entrance = [[100, 100], [200, 100], [200, 200], [100, 200]]
    exit_a = [[300, 100], [400, 100], [400, 200], [300, 200]]
    return entrance, exit_a


def _box_at(cx, cy, obj_id):
    return [cx - 10, cy - 10, cx + 10, cy + 10, obj_id]


def test_no_count_without_transition(areas):
    entrance, exit_a = areas
    c = AreaCounter(entrance, exit_a, min_residence_time=0)
    # entrance 안에 머무는 것만으로는 카운트되면 안 된다
    c.update([_box_at(150, 150, 1)])
    c.update([_box_at(150, 150, 1)])
    assert c.get_counts()['entered'] == 0
    assert c.get_counts()['exited'] == 0


def test_entrance_to_exit_counts_as_entered(areas):
    entrance, exit_a = areas
    c = AreaCounter(entrance, exit_a, min_residence_time=0)
    c.update([_box_at(150, 150, 1)])  # entrance
    c.update([_box_at(350, 150, 1)])  # exit
    counts = c.get_counts()
    assert counts['entered'] == 1
    assert counts['exited'] == 0
    assert counts['current_inside'] == 1


def test_exit_to_entrance_after_entered_counts_as_exited(areas):
    entrance, exit_a = areas
    c = AreaCounter(entrance, exit_a, min_residence_time=0)
    c.update([_box_at(150, 150, 1)])  # entrance
    c.update([_box_at(350, 150, 1)])  # exit  (= entered)
    c.update([_box_at(150, 150, 1)])  # entrance again (= exited)
    counts = c.get_counts()
    assert counts['entered'] == 1
    assert counts['exited'] == 1
    assert counts['current_inside'] == 0


def test_exit_to_entrance_without_prior_entry_does_not_count(areas):
    entrance, exit_a = areas
    c = AreaCounter(entrance, exit_a, min_residence_time=0)
    c.update([_box_at(350, 150, 1)])  # starts in exit
    c.update([_box_at(150, 150, 1)])  # moves to entrance
    counts = c.get_counts()
    # exit→entrance 자체는 입장으로 카운트하지 않음 (방향성 보존)
    assert counts['entered'] == 0
    assert counts['exited'] == 0


def test_min_residence_time_filters_fast_crossings(areas):
    entrance, exit_a = areas
    # 매우 큰 임계값으로 노이즈성 짧은 통과를 차단
    c = AreaCounter(entrance, exit_a, min_residence_time=10.0)
    c.update([_box_at(150, 150, 1)])
    c.update([_box_at(350, 150, 1)])
    assert c.get_counts()['entered'] == 0


def test_reset_clears_state(areas):
    entrance, exit_a = areas
    c = AreaCounter(entrance, exit_a, min_residence_time=0)
    c.update([_box_at(150, 150, 1)])
    c.update([_box_at(350, 150, 1)])
    assert c.get_counts()['entered'] == 1
    c.reset_counts()
    assert c.get_counts() == {
        'entered': 0, 'exited': 0, 'current_inside': 0, 'area_name': c.area_name
    }
    assert isinstance(c.entered_ids, set)
    assert isinstance(c.exited_ids, set)


def test_idempotent_when_same_object_passes_again(areas):
    """같은 obj_id가 다시 entrance→exit 해도 한 번만 카운트된다."""
    entrance, exit_a = areas
    c = AreaCounter(entrance, exit_a, min_residence_time=0)
    c.update([_box_at(150, 150, 1)])
    c.update([_box_at(350, 150, 1)])
    # 다시 entrance, 다시 exit
    c.update([_box_at(150, 150, 1)])  # 이건 "exited"로 카운트됨 (entered→exited)
    c.update([_box_at(350, 150, 1)])  # entered가 이미 있으니 추가 카운트 없음
    counts = c.get_counts()
    assert counts['entered'] == 1
    assert counts['exited'] == 1
