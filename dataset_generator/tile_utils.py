import math


def latlon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)

    n = 2 ** zoom

    xtile = int((lon + 180.0) / 360.0 * n)

    ytile = int(
        (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi)
        / 2.0
        * n
    )

    return xtile, ytile


def tile_to_latlon(xtile, ytile, zoom):
    n = 2 ** zoom

    lon = xtile / n * 360.0 - 180.0

    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))

    lat = math.degrees(lat_rad)

    return lat, lon