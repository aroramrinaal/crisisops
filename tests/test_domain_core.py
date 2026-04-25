"""Tests for the pure Python Crisisops domain core."""

from server.grader import EasyGrader
from server.reward import compute_step_reward
from server.rules import compute_optimal_plan
from server.scenario_generator import generate_scenario


def test_generate_scenario_is_deterministic():
    first = generate_scenario("easy", 42)
    second = generate_scenario("easy", 42)

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert first[2] == second[2]
    assert first[3] == second[3]
    assert first[4] == second[4]


def test_optimal_plan_grades_inside_validation_clamp():
    zones, units, reports, hidden_truth, optimal_plan = generate_scenario("easy", 42)

    score = EasyGrader().grade(
        optimal_plan,
        {
            **hidden_truth,
            "zones": zones,
            "units": units,
            "reports": reports,
        },
    )

    assert 0.01 <= score <= 0.99
    assert score > 0.7


def test_reward_prefers_correct_false_alarm_flag():
    _, _, _, hidden_truth, optimal_plan = generate_scenario("easy", 42)
    false_alarm = next(
        action for action in optimal_plan if action.root.type == "flag_false_alarm"
    )

    reward = compute_step_reward(false_alarm, {}, {}, hidden_truth)

    assert reward > 0


def test_rules_compute_plan_without_hand_authored_answers():
    zones, units, reports, hidden_truth, _ = generate_scenario("medium", 7)
    plan = compute_optimal_plan(
        {
            "zones": zones,
            "units": units,
            "reports": reports,
            "hidden_truth": hidden_truth,
        }
    )

    assert plan
    assert plan[-1].root.type == "publish_sitrep"
