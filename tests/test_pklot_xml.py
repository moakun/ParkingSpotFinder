from parking.data.pklot_xml_to_geometry import parse_pklot_xml
from parking.types import Occupancy

_XML = """<parking id="TEST">
  <space id="1" occupied="1">
    <contour><point x="10" y="10"/><point x="30" y="10"/><point x="30" y="40"/><point x="10" y="40"/></contour>
  </space>
  <space id="2" occupied="0">
    <contour><point x="40" y="10"/><point x="60" y="10"/><point x="60" y="40"/><point x="40" y="40"/></contour>
  </space>
  <space id="3">
    <contour><point x="70" y="10"/><point x="90" y="10"/><point x="90" y="40"/></contour>
  </space>
</parking>"""


def test_parse_pklot_xml(tmp_path):
    xml = tmp_path / "f.xml"
    xml.write_text(_XML, encoding="utf-8")
    spots, gt = parse_pklot_xml(xml)

    # spaces 1 and 2 have valid 4-point contours; space 3 has only 3 points -> skipped
    assert [s.id for s in spots] == ["1", "2"]
    assert spots[0].polygon.shape == (4, 2)

    # occupancy ground truth parsed from the `occupied` attr; unlabeled space absent
    assert gt["1"] is Occupancy.OCCUPIED
    assert gt["2"] is Occupancy.EMPTY
    assert "3" not in gt
