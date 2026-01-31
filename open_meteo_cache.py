"""
Siatka punktów nad Europą oraz cache Open-Meteo na dysku.
Ścieżka cache: cache/open_meteo/{year}/{lat}_{lon}.json
Zakres: lat 35–55, lon -10–25 (co 1).
"""

import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import Point

LAT_MIN = 35
LAT_MAX = 55
LON_MIN = -10
LON_MAX = 25
STEP = 1

CACHE_ROOT = Path("cache/open_meteo")
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


def build_grid():
    """Zwraca listę par (lat, lon)"""
    lats = list(range(LAT_MIN, LAT_MAX + 1, STEP))  # 35, 40, 45, 50, 55
    lons = list(range(LON_MIN, LON_MAX + 1, STEP))  # -10, -5, 0, 5, 10, 15, 20, 25
    return [(lat, lon) for lat in lats for lon in lons]


def _load_europe_map():
    """Ładuje mapę Europy do filtrowania punktów na lądzie."""
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)
    name_column = "NAME" if "NAME" in world.columns else "ADMIN"
    europe_countries = [
        "Portugal", "Spain", "France", "Italy", "Germany", "Switzerland",
        "Austria", "Poland", "Netherlands", "Belgium", "Luxembourg",
        "Czechia", "Slovakia", "Greece", "Slovenia", "Croatia",
        "Bosnia and Herz.", "Serbia", "Montenegro", "Albania",
        "North Macedonia", "Hungary", "Kosovo", "Romania", "Bulgaria",
    ]
    europe = world[world[name_column].isin(europe_countries)].copy()
    europe = europe.cx[LON_MIN : LON_MAX + 1, LAT_MIN : LAT_MAX + 1]
    return europe


def filter_grid_to_land(grid: list[tuple[float, float]] | None = None) -> list[tuple[float, float]]:
    """
    Filtruje siatkę, zwracając tylko punkty na lądzie (w obrębie krajów Europy).
    Jeśli grid=None, używa build_grid().
    """
    if grid is None:
        grid = build_grid()

    europe = _load_europe_map()
    land_points = []

    for lat, lon in grid:
        point = Point(lon, lat)
        if europe.geometry.contains(point).any():
            land_points.append((lat, lon))

    return land_points


def get_cache_path(year: int, lat: float, lon: float) -> Path:
    """Ścieżka pliku cache: cache/open_meteo/{year}/{lat}_{lon}.json"""
    folder = CACHE_ROOT / str(year)
    # np. 49.5_6.5  lub  43.0_-1.5
    name = f"{lat}_{lon}.json"
    return folder / name


def load_cached(year: int, lat: float, lon: float) -> dict | None:
    """Wczytuje dane z dysku. Zwraca None, jeśli plik nie istnieje."""
    path = get_cache_path(year, lat, lon)
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_to_cache(year: int, lat: float, lon: float, data: dict) -> None:
    """Zapisuje odpowiedź API do cache/open_meteo/{year}/{lat}_{lon}.json"""
    path = get_cache_path(year, lat, lon)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=None)


def fetch_daily(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    daily: list[str] | None = None,
    max_retries: int = 3,
) -> dict:
    """Pobiera dane dzienne z Open-Meteo Archive. Zwraca surową odpowiedź JSON."""
    if daily is None:
        daily = [
            "temperature_2m_mean",
            "temperature_2m_min",
            "temperature_2m_max",
            "precipitation_sum",
        ]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(daily),
        "timezone": "auto",
    }
    for attempt in range(max_retries):
        try:
            # timeout, żeby nie wisieć wiecznie na jednym punkcie
            resp = requests.get(BASE_URL, params=params, timeout=5)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            wait = (attempt) * 5
            if attempt < max_retries - 1:
                print(f"    {e.__class__.__name__} – czekam {wait}s i ponaw")
                time.sleep(wait)
                continue
            raise

        if resp.status_code == 429:
            wait = (attempt + 1) * 5
            if attempt < max_retries - 1:
                print(f"    429 – czekam {wait}s...")
                time.sleep(wait)
                continue

        resp.raise_for_status()
        return resp.json()

    raise RuntimeError("fetch_daily: wyczerpano próby")


def get_data(
    year: int,
    lat: float,
    lon: float,
    start_date: str | None = None,
    end_date: str | None = None,
    daily: list[str] | None = None,
) -> dict:
    """
    Zwraca dane dzienne dla (year, lat, lon):
    - z cache, jeśli plik istnieje,
    - w przeciwnym razie pobiera z API, zapisuje do cache i zwraca.
    Domyślny okres: 1 IV – 31 X danego roku.
    """
    start_date = start_date or f"{year}-04-01"
    end_date = end_date or f"{year}-10-31"

    cached = load_cached(year, lat, lon)
    if cached is not None:
        return cached

    data = fetch_daily(lat, lon, start_date, end_date, daily)
    save_to_cache(year, lat, lon, data)
    return data


def fill_cache_for_year(
    year: int,
    *,
    delay_seconds: float = 1.0,
    land_only: bool = True,
) -> None:
    """
    Pobiera i zapisuje w cache dane za pełny rok (01.01–31.12) dla każdego
    punktu siatki. Punkty już w cache są pomijane (nie wywołuje API).

    Args:
        year: Rok danych
        delay_seconds: Opóźnienie między requestami (sekundy)
        land_only: Jeśli True, pobiera tylko punkty na lądzie (oszczędza requesty)
    """
    if land_only:
        print("Filtrowanie siatki do punktów na lądzie...")
        grid = filter_grid_to_land()
        print(f"Znaleziono {len(grid)} punktów na lądzie (z {len(build_grid())} wszystkich).")
    else:
        grid = build_grid()

    start = f"{year}-01-01"
    end = f"{year}-12-31"
    total = len(grid)
    fetched = 0
    for i, (lat, lon) in enumerate(grid):
        if load_cached(year, lat, lon) is None:
            try:
                get_data(year, lat, lon, start_date=start, end_date=end)
                fetched += 1
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
            except Exception as e:  # noqa: BLE001
                # Nie zatrzymujemy całej pętli – logujemy i idziemy dalej.
                print(f"    BŁĄD dla punktu (lat={lat}, lon={lon}): {e!r} – pomijam.")
        if (i + 1) % 50 == 0 or i == 0 or i == total - 1:
            print(f"  {i + 1}/{total} punktów (pobrano w tej sesji: {fetched})")
    print(f"Gotowe. Rok {year}: {total} punktów, w tej sesji pobrano {fetched}.")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        arg = sys.argv[1].lower()
        if arg == "fill":
            y = int(sys.argv[2]) if len(sys.argv) >= 3 else 2024
        elif arg.isdigit():
            y = int(arg)
        else:
            y = None
        if y is not None:
            print(f"Uzupełniam cache dla roku {y} (pełny rok, wszystkie punkty siatki)")
            fill_cache_for_year(y)
            sys.exit(0)

    grid = build_grid()
    print(f"Siatka: {len(grid)} punktów (krok {STEP}°, lon {LON_MIN}–{LON_MAX}, lat {LAT_MIN}–{LAT_MAX})")
    print("Przykład punktów:", grid[:5], "...", grid[-3:])

    # Test cache dla jednego punktu
    year = 2024
    lat, lon = 49.98, 6.64
    data = get_data(year, lat, lon)
    key = "daily"
    print(f"\nPrzykład get_data({year}, {lat}, {lon}): klucze = {list(data.keys())}")
    if key in data and data[key]:
        print(f"  daily: {list(data[key].keys())}, liczba dni = {len(data[key]['time'])}")

    p = get_cache_path(year, lat, lon)
    print(f"Cache dla tego punktu: {p} (exists: {p.is_file()})")
    print("\nAby pobrać cały 2024 dla wszystkich regionów:  python open_meteo_cache.py 2024")