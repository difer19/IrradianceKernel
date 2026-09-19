#!/usr/bin/env python3
"""Convierte el shapefile EPSG:3857 de Nariño a GeoJSON WGS84."""

from __future__ import annotations

import json
import math
from pathlib import Path

import shapefile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "geography" / "narino_3857.shp"
OUTPUT = ROOT / "data" / "geography" / "narino_boundary.geojson"


def point_to_wgs84(point):
    x, y = point
    longitude = math.degrees(x / 6378137.0)
    latitude = math.degrees(2.0 * math.atan(math.exp(y / 6378137.0)) - math.pi / 2.0)
    return [round(longitude, 7), round(latitude, 7)]


def transform_coordinates(value):
    if value and isinstance(value[0], (int, float)):
        return point_to_wgs84(value)
    return [transform_coordinates(item) for item in value]


def main():
    reader = shapefile.Reader(str(SOURCE), encoding="latin1")
    field_names = [field[0] for field in reader.fields[1:]]
    features = []
    for shape_record in reader.iterShapeRecords():
        geometry = shape_record.shape.__geo_interface__
        geometry["coordinates"] = transform_coordinates(geometry["coordinates"])
        properties = dict(zip(field_names, shape_record.record))
        if str(properties.get("NAME_1", "")).startswith("Nari"):
            properties["NAME_1"] = "Nariño"
        features.append(
            {"type": "Feature", "properties": properties, "geometry": geometry}
        )
    payload = {
        "type": "FeatureCollection",
        "name": "narino_boundary",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": features,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
