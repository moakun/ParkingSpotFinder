import numpy as np

from parking.eval import compute_metrics, parse_weather


def test_parse_weather_from_prepare_pklot_names():
    assert parse_weather("PUC__Sunny__2012-09-17__Occupied__x.jpg") == "Sunny"
    assert parse_weather("UFPR04__Cloudy__2012-12-15__Empty__y.jpg") == "Cloudy"
    assert parse_weather("PUC__Rainy__2012-10-25__Occupied__z.jpg") == "Rainy"


def test_parse_weather_absent_in_coco_names():
    # coco_to_rois names carry no weather token
    assert parse_weather("train_12_3.jpg") is None
    assert parse_weather("val_0_44.jpg") is None


def test_metrics_perfect():
    preds = np.array([0, 0, 1, 1])
    labels = np.array([0, 0, 1, 1])
    m = compute_metrics(preds, labels)
    assert m["acc"] == 1.0
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (2, 2, 0, 0)
    assert m["false_available_rate"] == 0.0
    assert m["false_occupied_rate"] == 0.0


def test_metrics_error_rates():
    # true:  empty, occ,  occ,  empty
    # pred:  empty, empty, occ, occ
    labels = np.array([0, 1, 1, 0])
    preds = np.array([0, 0, 1, 1])
    m = compute_metrics(preds, labels)
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (1, 1, 1, 1)
    assert m["acc"] == 0.5
    assert m["false_available_rate"] == 0.5  # 1 missed occupied / 2 truly occupied
    assert m["false_occupied_rate"] == 0.5   # 1 false full / 2 truly empty
    assert m["recall_occupied"] == 0.5
