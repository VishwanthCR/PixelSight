import pytest
from backend.app.services.aoi import (
    AOIValidationError,
    bbox_to_geojson_polygon,
    parse_bbox_from_geometry,
    validate_and_estimate_aoi,
)


def test_parse_bbox_from_geometry():
    bbox = [12.4, 41.8, 12.6, 42.0]
    assert parse_bbox_from_geometry(bbox) == bbox

    poly = bbox_to_geojson_polygon(bbox)
    assert parse_bbox_from_geometry(poly) == bbox


def test_validate_and_estimate_aoi_dimensions():
    # Indian AOI (Chennai: ~1.28 km x 1.28 km)
    bbox = [80.264798, 13.076941, 80.276602, 13.088459]
    res = validate_and_estimate_aoi(bbox)

    assert res["valid"] is True
    assert res["inside_india"] is True
    assert res["supported_country"] == "INDIA"
    assert res["area_sqkm"] > 0.5
    assert res["category"] == "interactive"
    assert res["native_10m"]["width"] > 50
    assert res["native_10m"]["height"] > 50
    assert res["super_resolution_2_5m"]["width"] == res["native_10m"]["width"] * 4
    assert res["super_resolution_2_5m"]["height"] == res["native_10m"]["height"] * 4
    assert "tiles" in res
    assert res["tiles"]["total_tiles"] >= 1


def test_outside_india_raises_validation_error():
    # Rome (outside India)
    with pytest.raises(AOIValidationError) as exc:
        validate_and_estimate_aoi([12.48, 41.89, 12.50, 41.91])
    assert "outside India" in str(exc.value)

    # New York (outside India)
    with pytest.raises(AOIValidationError) as exc:
        validate_and_estimate_aoi([-74.01, 40.71, -73.99, 40.73])
    assert "outside India" in str(exc.value)


def test_invalid_aoi_raises_validation_error():
    # Invalid longitude
    with pytest.raises(AOIValidationError):
        validate_and_estimate_aoi([190.0, 10.0, 195.0, 12.0])

    # min >= max
    with pytest.raises(AOIValidationError):
        validate_and_estimate_aoi([80.28, 13.09, 80.26, 13.07])

    # Exceeds max area limit
    with pytest.raises(AOIValidationError):
        validate_and_estimate_aoi([75.0, 12.0, 80.0, 17.0], max_area_sqkm=10.0)
