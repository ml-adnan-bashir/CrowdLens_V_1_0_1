from crowdlens.events import BehaviorMonitor, WeaponEventMemory, _contains
from crowdlens.models import AssignedPerson, ObjectObservation


def track(x=10, sid="P0001", match="ongoing"):
    return AssignedPerson(1, sid, (x, 10, x + 20, 90), .95, match)


def monitor(**kw):
    defaults = dict(zones={"entry": [[0, 0], [.5, 0], [.5, .5], [0, .5]]},
                    motion_threshold=1., min_motion_frames=2, prolonged_seconds=2.,
                    absence_reset_seconds=1., motion_cooldown_seconds=3., zone_cooldown_seconds=2.)
    defaults.update(kw)
    return BehaviorMonitor(**defaults)


def test_point_in_polygon():
    square = [[0, 0], [.5, 0], [.5, .5], [0, .5]]
    assert _contains((.2, .2), square)
    assert not _contains((.9, .2), square)


def test_zone_entry_is_not_emitted_every_frame():
    m = monitor()
    first = m.observe([track()], 1, fps=10., width=100, height=200)
    second = m.observe([track()], 2, fps=10., width=100, height=200)
    assert [e.event_type for e in first] == ["zone_entry"]
    assert not [e for e in second if e.event_type == "zone_entry"]


def test_rapid_motion_requires_persistence():
    m = monitor(zones={})
    m.observe([track(0)], 1, fps=10., width=300, height=100)
    one = m.observe([track(30)], 2, fps=10., width=300, height=100)
    two = m.observe([track(60)], 3, fps=10., width=300, height=100)
    assert not one
    assert [e.event_type for e in two] == ["rapid_motion"]


def test_prolonged_presence_once():
    m = monitor(zones={}, motion_threshold=100.)
    found = []
    for frame in range(1, 8):
        found.extend(m.observe([track()], frame, fps=2., width=100, height=100))
    assert [e.event_type for e in found] == ["prolonged_presence"]


def test_weapon_event_is_review_hypothesis_not_possession():
    memory = WeaponEventMemory(3.)
    weapon = ObjectObservation("handgun", (15, 20, 22, 29), .81)
    first = memory.observe([weapon], [track()], 1, fps=10.)
    repeat = memory.observe([weapon], [track()], 2, fps=10.)
    assert len(first) == 1 and not repeat
    assert first[0].event_type == "possible_weapon"
    assert first[0].person_id is None
    assert first[0].details["nearby_anonymous_tracks"] == ["P0001"]
    assert first[0].details["proximity_is_not_possession"] is True

