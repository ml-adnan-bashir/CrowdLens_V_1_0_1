import numpy as np
import pytest

from crowdlens.identity import IdentityGallery
from crowdlens.models import PersonObservation


def person(local_id, feature):
    return PersonObservation(local_id, (10, 10, 40, 90), .99, np.array(feature, dtype=np.float32))


def test_ongoing_track_keeps_same_id_even_when_appearance_drifts():
    gallery = IdentityGallery(similarity=.98)
    first = gallery.update([person(7, [1, 0, 0])], frame=1)[0]
    second = gallery.update([person(7, [0, 1, 0])], frame=2)[0]
    assert first.stable_id == second.stable_id == "P0001"
    assert second.match_kind == "ongoing"


def test_exit_and_return_can_reacquire_old_anonymous_id():
    gallery = IdentityGallery(similarity=.94, ambiguity_margin=.03, max_gap_frames=10)
    first = gallery.update([person(1, [1, 0, 0])], 1)[0]
    gallery.update([], 2)
    returned = gallery.update([person(81, [.99, .04, 0])], 3)[0]
    assert returned.stable_id == first.stable_id
    assert returned.match_kind == "reidentified"
    assert returned.match_similarity > .94


def test_simultaneously_visible_people_never_share_id():
    gallery = IdentityGallery(similarity=.80)
    people = gallery.update([person(1, [1, 0]), person(2, [1, 0])], 1)
    assert {p.stable_id for p in people} == {"P0001", "P0002"}
    again = gallery.update([person(1, [1, 0]), person(2, [1, 0])], 2)
    assert {p.stable_id for p in again} == {"P0001", "P0002"}


def test_ambiguous_gallery_match_creates_new_id():
    gallery = IdentityGallery(similarity=.8, ambiguity_margin=.10)
    gallery.update([person(1, [1, 0]), person(2, [.98, .2])], 1)
    gallery.update([], 2)
    uncertain = gallery.update([person(3, [1, .1])], 3)[0]
    assert uncertain.stable_id == "P0003"
    assert uncertain.match_kind == "ambiguous"


def test_gallery_expiry_does_not_guess_identity():
    gallery = IdentityGallery(similarity=.95, max_gap_frames=2)
    gallery.update([person(1, [1, 0])], 1)
    gallery.update([], 2)
    return_value = gallery.update([person(2, [1, 0])], 5)[0]
    assert return_value.stable_id == "P0002"
    assert return_value.match_kind == "new"


def test_two_new_ids_cannot_both_reacquire_one_inactive_id():
    gallery = IdentityGallery(similarity=.9)
    gallery.update([person(1, [1, 0])], 1)
    gallery.update([], 2)
    reassociated = gallery.update([person(2, [1, .02]), person(3, [1, .02])], 3)
    assert len({p.stable_id for p in reassociated}) == 2
    assert sum(p.match_kind == "reidentified" for p in reassociated) == 1


def test_duplicate_local_id_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        IdentityGallery().update([person(1, [1, 0]), person(1, [0, 1])], 1)


def test_zero_feature_rejected():
    with pytest.raises(ValueError, match="finite nonzero"):
        IdentityGallery().update([person(1, [0, 0])], 1)

