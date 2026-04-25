"""Tests for the pure Python Crisisops domain core."""

from server.grader import EasyGrader, ExpertGrader
from server.crisisops_environment import CrisisopsEnvironment
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


def test_scenario_tiers_follow_plan_counts_and_streaming():
    _, easy_units, easy_reports, easy_hidden, _ = generate_scenario("easy", 42)
    _, expert_units, expert_reports, expert_hidden, _ = generate_scenario("expert", 42)

    assert len(easy_reports) == 3
    assert len(easy_hidden["true_report_ids"]) == 2
    assert len(easy_hidden["false_report_ids"]) == 1
    assert len(easy_units) == 2
    assert {report.reveal_at_step for report in easy_reports} == {0}

    assert len(expert_reports) == 15
    initial_expert_true_reports = [
        report_id
        for report_id in expert_hidden["true_report_ids"]
        if report_id not in expert_hidden["event_reports_by_id"]
    ]
    assert len(initial_expert_true_reports) == 10
    assert len(expert_hidden["false_report_ids"]) == 5
    assert len(expert_units) == 8
    assert max(report.reveal_at_step for report in expert_reports) <= 20
    assert all(unit.shared_pool for unit in expert_units)
    assert expert_hidden["district_comms_degraded_from_step"] == 8
    assert "report-mid-1" in expert_hidden["true_report_ids"]
    assert "report-mid-1" in expert_hidden["sitrep_truth"]["incidents_confirmed"]


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


def test_expert_optimal_plan_satisfies_timeliness_and_event_truth():
    _, _, _, hidden_truth, optimal_plan = generate_scenario("expert", 42)

    actions = [action.root.model_dump(mode="json") for action in optimal_plan]
    breakdown = ExpertGrader().subscores(actions, hidden_truth)

    assert ExpertGrader().grade(optimal_plan, hidden_truth) > 0.95
    assert breakdown.timeliness == 1.0
    assert breakdown.sitrep_quality == 1.0
    assert "report-mid-1" in actions[-1]["payload"]["incidents_confirmed"]


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


def test_environment_terminal_score_lands_in_observation_metadata():
    env = CrisisopsEnvironment()
    env.reset(task_id="single_zone_response", seed=42, episode_id="episode-test")
    terminal = env.step(env.optimal_plan[-1])

    assert terminal.done is True
    assert "terminal_score" in terminal.metadata
    assert 0.01 <= terminal.metadata["terminal_score"] <= 0.99
    assert env.state.current_task_id == "single_zone_response"
