import random

import numpy as np
import pytest

from app.simulation.constants import SENSOR_RANGE
from app.simulation.engine import Fleet, decide, move, score, sense
from app.simulation.geometry import boxes_hit_walls, car_corners, cast_rays
from app.simulation.road import Road


def straight_walls():
    starts = np.array([[-100.0, 500.0], [100.0, 500.0]])
    ends = np.array([[-100.0, -500.0], [100.0, -500.0]])
    return starts, ends


def test_rays_measure_distance_to_walls():
    distances = cast_rays(np.array([[0.0, 0.0]]), np.array([[0.0, 90.0, 270.0, 45.0]]), SENSOR_RANGE, *straight_walls())
    front, right, left, front_right = distances[0]
    assert front == pytest.approx(SENSOR_RANGE)
    assert right == pytest.approx(100.0)
    assert left == pytest.approx(100.0)
    assert front_right == pytest.approx(100.0 * np.sqrt(2))


def test_rays_follow_car_rotation():
    distances = cast_rays(np.array([[50.0, 0.0]]), np.array([[90.0, 270.0]]), SENSOR_RANGE, *straight_walls())
    assert distances[0].tolist() == pytest.approx([50.0, 150.0])


def test_collision_detects_car_over_the_line():
    walls = straight_walls()
    inside = car_corners(np.array([0.0]), np.array([0.0]), np.array([0.0]), 100, 44)
    touching = car_corners(np.array([85.0]), np.array([0.0]), np.array([0.0]), 100, 44)
    sideways = car_corners(np.array([0.0]), np.array([0.0]), np.array([90.0]), 100, 44)
    assert not boxes_hit_walls(inside, *walls)[0]
    assert boxes_hit_walls(touching, *walls)[0]
    assert not boxes_hit_walls(sideways, *walls)[0]


def test_decide_uses_threshold_and_opposites():
    outputs = np.array([[0.9, 0.2, 0.7, 0.6], [0.4, 0.3, 0.1, 0.9], [0.8, 0.9, 0.9, 0.9]])
    d = decide(outputs)
    assert d.accelerate.tolist() == [True, False, False]
    assert d.brake.tolist() == [False, False, True]
    assert d.turn_left.tolist() == [True, False, False]
    assert d.turn_right.tolist() == [False, True, False]


def test_move_and_score_reward_forward_progress():
    road = Road(random.Random(3), 200, 0, 6000)
    fleet = Fleet.launch([1, 2])
    fleet.rotation[1] = 180
    previous = fleet.y.copy()
    d = decide(np.array([[0.9, 0.0, 0.0, 0.0], [0.9, 0.0, 0.0, 0.0]]))
    move(fleet, d)
    checks = score(fleet, previous, road, 0.0, tick=20)
    assert fleet.y[0] < 0
    assert fleet.fitness[0] > 0
    assert checks[2, 1]
    assert fleet.fitness[1] == -1


def test_sense_returns_nine_normalized_inputs():
    road = Road(random.Random(5), 200, 300, 6000)
    inputs, distances = sense(Fleet.launch([1, 2, 3]), road)
    assert inputs.shape == (3, 9)
    assert ((inputs >= 0) & (inputs <= 1)).all()
    assert distances[0, 2] == pytest.approx(100.0, abs=1)


def test_road_is_reproducible_and_bounded():
    a = Road(random.Random(42), 200, 300, 6000)
    b = Road(random.Random(42), 200, 300, 6000)
    assert a.left == b.left
    a.extend_to(-20000)
    assert a.end_y >= a.finish_y - 1000


def test_road_edges_never_step_backward():
    for seed in range(20):
        road = Road(random.Random(seed), 160, 450, 12000)
        road.extend_to(-20000)
        for side in road.arrays():
            ys = side[:, 1]
            assert (np.diff(ys) <= 1e-9).all()
            xs_jumps = np.abs(np.diff(side[:, 0]))[np.diff(ys) == 0]
            assert (xs_jumps < 1e-9).all()
