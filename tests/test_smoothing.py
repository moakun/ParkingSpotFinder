from parking.temporal import SmoothingBank, SpotState
from parking.types import Occupancy, Prediction


def test_single_frame_transient_does_not_flip():
    st = SpotState("A1", k=5, initial=Occupancy.EMPTY)
    # one stray "occupied" frame amid empties must not change state
    assert st.update(Occupancy.OCCUPIED) is Occupancy.EMPTY
    assert st.update(Occupancy.EMPTY) is Occupancy.EMPTY


def test_k_consecutive_frames_flip_state():
    st = SpotState("A1", k=3, initial=Occupancy.EMPTY)
    assert st.update(Occupancy.OCCUPIED) is Occupancy.EMPTY  # streak 1
    assert st.update(Occupancy.OCCUPIED) is Occupancy.EMPTY  # streak 2
    assert st.update(Occupancy.OCCUPIED) is Occupancy.OCCUPIED  # streak 3 -> flip


def test_interrupted_streak_resets():
    st = SpotState("A1", k=3, initial=Occupancy.EMPTY)
    st.update(Occupancy.OCCUPIED)
    st.update(Occupancy.EMPTY)  # resets pending
    st.update(Occupancy.OCCUPIED)
    st.update(Occupancy.OCCUPIED)
    assert st.state is Occupancy.EMPTY  # only 2 in a row, no flip yet


def test_bank_preserves_spot_order():
    bank = SmoothingBank(["A1", "A2"], k=1)
    preds = [Prediction(Occupancy.OCCUPIED, 0.9), Prediction(Occupancy.EMPTY, 0.8)]
    results = bank.update(["A1", "A2"], preds)
    assert [r.spot_id for r in results] == ["A1", "A2"]
    assert results[0].state is Occupancy.OCCUPIED
    assert results[1].state is Occupancy.EMPTY
