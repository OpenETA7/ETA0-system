import pytest

from deploy.audit_mug_progress import PlacementReview, audit, demo_samples


@pytest.mark.parametrize("partial", [(True, False), (False, True)])
def test_partial_is_status_until_finish_or_repeated_attempts(partial):
    rule = PlacementReview()
    assert rule.observe(1, partial, ["open", "close", "open"]) == []
    assert rule.observe(101, partial, ["hold"]) == []
    assert rule.observe(102, partial, ["open", "close", "open"]) == []
    assert rule.observe(103, partial, ["close", "open"]) == []
    assert rule.observe(112, partial, []) == []
    assert rule.observe(113, partial, []) == ["partial_after_two_releases"]
    assert rule.observe(114, partial, [], finish=True) == []
    new_rule = PlacementReview()
    assert new_rule.observe(1, partial, [], finish=True) == ["partial_finish"]


def test_missing_and_changed_partial_reset_evidence():
    rule = PlacementReview()
    for tick, flags in enumerate(([False, True], [False, True], [None, True], [True, False])):
        assert rule.observe(tick, flags, ["open", "close", "open"], [None, True]) == []
    assert rule.observe(5, [True, False], ["open", "close", "open"]) == []
    assert rule.observe(50, [True, False], []) == []
    with pytest.raises(ValueError, match="advance"):
        rule.observe(50, [True, False], [])


def test_sustained_wrong_support_applies_to_same_mug_after_release():
    rule = PlacementReview()
    goals = [False, False]
    for tick, crossed in ((1, [True, False]), (11, [False, True]), (21, [True, False])):
        assert rule.observe(tick, goals, ["open"], crossed) == []
    assert rule.observe(31, None, [], [True, False]) == []
    assert rule.observe(41, goals, [], [True, False]) == []
    assert rule.observe(51, goals, ["close"], [True, False]) == []
    assert rule.observe(61, goals, ["open"], [True, False]) == []
    assert rule.observe(70, goals, ["hold"], [True, False]) == []
    assert rule.observe(71, goals, [], [True, False]) == ["white_on_wrong_plate"]
    assert rule.observe(81, goals, [], [True, False]) == []
    assert PlacementReview().observe(1, goals, [], finish=True) == []


def test_sampling_uses_only_elapsed_commands_and_stops_at_success():
    rows = [
        dict(flags=flags, cmd=cmd, cross=[False, False])
        for flags, cmd in [
            ([False, False], None),
            ([True, False], -1),
            ([True, False], 1),
            ([True, False], -1),
            ([True, True], -1),
            ([True, False], 1),
        ]
    ]
    samples = list(demo_samples(rows, 2))
    assert [s["commands"] for s in samples] == [[], ["open", "close"], ["open"], ["close"]]
    result = audit(samples)
    assert (result["first_partial"], result["first_success"], result["alerts"]) == (2, 4, [])


def test_unknown_empty_receipt_never_establishes_success():
    rows = [dict(flags=[], tick=0, action=0, commands=[])]
    assert audit(rows)["first_success"] is None
