"""
Mapa przydatności regionów Europy dla szczepów winogron – model Tonietto (HI, CI, DI).
"""

import warnings
from datetime import datetime
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from scipy.interpolate import griddata
from shapely.geometry import Point
from open_meteo_cache import build_grid, get_cache_path, get_data

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Parametry modelu Tonietto & Carbonneau
# -----------------------------------------------------------------------------

# Okresy obliczeń (dla półkuli północnej, rok wegetacyjny)
DATE_HI_START = "04-01"   # Huglin Index: 1 kwietnia – 30 września
DATE_HI_END = "09-30"
DATE_CI_START = "09-01"   # Cool Night Index: wrzesień
DATE_CI_END = "09-30"
DATE_DI_START = "04-01"   # Dryness Index: 1 kwietnia – 30 września
DATE_DI_END = "09-30"

#YEARS = [2022, 2023, 2024, 2025] # uncomment for years slider
YEARS = [2025]

YEAR_DATA_MAPPING = {
    2022: 2022,
    2023: 2023,
    2024: 2024,
    2025: 2025,
}

def huglin_k(lat: float) -> float:
    if lat <= 40:
        return 1.00
    if lat <= 42:
        return 1.02
    if lat <= 44:
        return 1.03
    if lat <= 46:
        return 1.04
    if lat <= 48:
        return 1.05
    if lat <= 50:
        return 1.06
    return 1.06

def compute_huglin_index(tmean: np.ndarray, tmax: np.ndarray, lat: float) -> float:
    """
    Huglin Index (HI): HI = K · Σ [(Tmean + Tmax)/2 - 10]+
    Okres: 1 IV – 30 IX. Sumowane tylko dni, gdy (Tmean+Tmax)/2 > 10°C.
    """
    term = (tmean + tmax) / 2.0 - 10.0
    positive = np.maximum(term, 0.0)
    return huglin_k(lat) * float(np.sum(positive))

def compute_cool_night_index(tmin: np.ndarray) -> float:
    """Cool Night Index (CI): średnia Tmin w okresie dojrzałości"""
    return float(np.mean(tmin))

def compute_dryness_index(precip_mm: np.ndarray) -> float:
    """
    Dryness Index (DI): uproszczenie – suma opadów w okresie 1 IV – 30 IX.
    Pełny model Riou: W = Wo + P − Tv
    """
    return float(np.sum(precip_mm))

GRAPE_VARIETIES = {
    "Riesling": {
        "HI": (1700, 1800),  # Huglin (1986)
        "CI": (8, 10),  # Cool Night Index
        "DI": (500, 700),  # Dryness Index
        "WEIGHT_HI": 0.6,  # Waga dla Huglin Index
        "WEIGHT_CI": 0.3,  # Waga dla Cool Night Index
        "WEIGHT_DI": 0.1,  # Waga dla Dryness Index
    },
    "Cabernet Sauvignon": {
        "HI": (1900, 2000),  # Huglin (1986)
        "CI": (14, 18),  # Cool Night Index
        "DI": (0, 200),  # Dryness Index
        "WEIGHT_HI": 0.6,
        "WEIGHT_CI": 0.3,
        "WEIGHT_DI": 0.1,
    },
}

def _mask_dates(times: list[str], start_mmdd: str, end_mmdd: str, year: int) -> np.ndarray:
    """Zwraca maskę bool: True dla dni w [start_mmdd, end_mmdd] w podanym roku."""
    start = datetime(year, int(start_mmdd[:2]), int(start_mmdd[3:5]))
    end = datetime(year, int(end_mmdd[:2]), int(end_mmdd[3:5]))
    mask = np.zeros(len(times), dtype=bool)
    for i, t in enumerate(times):
        d = datetime.strptime(t[:10], "%Y-%m-%d")
        mask[i] = start <= d <= end
    return mask

def calculate_suitability(climate_data: dict, variety: str = "Riesling") -> float:
    """
    Oblicza przydatność lokalizacji dla danego szczepu winogron.
    Używa funkcji oceny ze skalowaniem kary względem zakresu indeksu.
    Kara jest proporcjonalna do % odchylenia od optymalnego zakresu.
    """
    grape_params = GRAPE_VARIETIES[variety]
    hi_min, hi_max = grape_params["HI"]
    ci_min, ci_max = grape_params["CI"]
    di_min, di_max = grape_params["DI"]

    # Pobierz wagi specyficzne dla danego szczepu
    weight_hi = grape_params["WEIGHT_HI"]
    weight_ci = grape_params["WEIGHT_CI"]
    weight_di = grape_params["WEIGHT_DI"]

    def score_index(value: float, min_val: float, max_val: float) -> float:
        range_size = max_val - min_val
        if range_size == 0:
            # Degenerate case: jeśli min == max, tylko dokładna wartość = 100
            return 100.0 if value == min_val else 0.0

        tolerance = range_size * 0.2  # 20% zakresu jako strefa tolerancji
        if min_val <= value <= max_val:
            return 100.0

        if value < min_val:
            distance = min_val - value

            # Oblicz karę jako % odchylenia od zakresu
            if distance <= tolerance:
                # Łagodniejsza kara: 25 punktów na 100% odchylenia
                penalty = (distance / range_size) * 100 * 0.25
                return max(0.0, 100.0 - penalty)

            else:
                # Kara za strefę tolerancji
                penalty_in_tolerance = (tolerance / range_size) * 100 * 0.25
                # Ostrzejsza kara poza tolerancją: 50 punktów na 100% odchylenia
                penalty_beyond = ((distance - tolerance) / range_size) * 100 * 0.50
                return max(0.0, 100.0 - penalty_in_tolerance - penalty_beyond)

        # value > max_val
        distance = value - max_val
        if distance <= tolerance:
            penalty = (distance / range_size) * 100 * 0.25
            return max(0.0, 100.0 - penalty)

        else:
            penalty_in_tolerance = (tolerance / range_size) * 100 * 0.25
            penalty_beyond = ((distance - tolerance) / range_size) * 100 * 0.50
            return max(0.0, 100.0 - penalty_in_tolerance - penalty_beyond)

    hi_score = score_index(climate_data["HI"], hi_min, hi_max)
    ci_score = score_index(climate_data["CI"], ci_min, ci_max)
    di_score = score_index(climate_data["DI"], di_min, di_max)
    total_score = hi_score * weight_hi + ci_score * weight_ci + di_score * weight_di
    return round(total_score, 1)

def load_europe_map():
    """Ładuje mapę Europy"""
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)
    name_column = "NAME" if "NAME" in world.columns else "ADMIN"
    europe_countries = [
        "Portugal",
        "Spain",
        "France",
        "Italy",
        "Germany",
        "Switzerland",
        "Austria",
        "Poland",
        "Netherlands",
        "Belgium",
        "Luxembourg",
        "Czechia",
        "Slovakia",
        "Greece",
        "Slovenia",
        "Croatia",
        "Bosnia and Herz.",
        "Serbia",
        "Montenegro",
        "Albania",
        "North Macedonia",
        "Hungary",
        "Kosovo",
        "Romania",
        "Bulgaria",
    ]
    europe = world[world[name_column].isin(europe_countries)].copy()
    europe = europe.cx[-10:25, 35:55]
    return europe

def compute_indices_for_point(year_display: int, lat: float, lon: float) -> dict | None:
    """
    Dla danego punktu (lat, lon) i roku WYŚWIETLANEGO:
    - wczytuje dzienne dane z cache (lub pobiera, jeśli brak),
    - liczy HI, CI, DI wg Tonietto,
    - zwraca słownik z indeksami.
    """
    # Mapowanie roku wyświetlanego na rok danych
    year_data = YEAR_DATA_MAPPING.get(year_display, year_display)

    data = get_data(year_data, lat, lon, start_date=f"{year_data}-01-01", end_date=f"{year_data}-12-31")
    if "daily" not in data or "time" not in data["daily"]:
        return None

    times = data["daily"]["time"]
    if not times:
        return None

    tmin = np.array(data["daily"]["temperature_2m_min"], dtype=float)
    tmax = np.array(data["daily"]["temperature_2m_max"], dtype=float)
    precip = np.array(data["daily"]["precipitation_sum"], dtype=float)

    tmean = (tmin + tmax) / 2.0
    mask_hi = _mask_dates(times, DATE_HI_START, DATE_HI_END, year_data)
    mask_ci = _mask_dates(times, DATE_CI_START, DATE_CI_END, year_data)
    mask_di = _mask_dates(times, DATE_DI_START, DATE_DI_END, year_data)

    # Sprawdź, czy mamy dane dla wymaganych okresów
    if not np.any(mask_hi) or not np.any(mask_ci) or not np.any(mask_di):
        # Brak danych dla wymaganych okresów (np. niepełne dane dla przyszłego roku)
        return None

    HI = compute_huglin_index(tmean[mask_hi], tmax[mask_hi], lat)
    CI = compute_cool_night_index(tmin[mask_ci])
    DI = compute_dryness_index(precip[mask_di])
    return {
        "lat": lat,
        "lon": lon,
        "HI": round(HI, 1),
        "CI": round(CI, 1),
        "DI": round(DI, 1),
    }

def is_point_on_land(lat: float, lon: float, europe_gdf: gpd.GeoDataFrame) -> bool:
    """
    Sprawdza, czy punkt (lat, lon) znajduje się na lądzie
    Zwraca True, jeśli punkt jest w obrębie krajów z mapy Europy.
    """
    point = Point(lon, lat)
    # Sprawdź, czy punkt jest w obrębie któregoś z krajów
    return europe_gdf.geometry.contains(point).any()

def create_heatmap(variety: str = "Riesling", years: list[int] | None = None) -> None:
    """
    Tworzy interaktywną heatmapę przydatności dla danego szczepu z suwakiem roku
    """
    europe = load_europe_map()
    grid = build_grid()
    # Filtruje siatkę do punktów na lądzie – tylko raz
    land_points: list[tuple[float, float]] = []
    for lat, lon in grid:
        if is_point_on_land(lat, lon, europe):
            land_points.append((lat, lon))
    skipped_sea = len(grid) - len(land_points)
    if skipped_sea > 0:
        print(f"Pominięto {skipped_sea} punktów na morzu")
    if years is None:
        years = YEARS
    years = sorted(years)
    initial_year = years[0]

    def compute_results_for_year(year_display: int) -> tuple[list[dict], int]:
        """Zwraca (lista wyników, liczba punktów z brakującymi danymi)."""
        results: list[dict] = []
        skipped_data = 0
        # Pobierz rzeczywisty rok danych
        year_data = YEAR_DATA_MAPPING.get(year_display, year_display)

        for lat, lon in land_points:
            indices = compute_indices_for_point(year_display, lat, lon)
            if indices is None:
                skipped_data += 1
                continue
            suitability = calculate_suitability(indices, variety)
            results.append(
                {
                    "lat": indices["lat"],
                    "lon": indices["lon"],
                    "suitability": suitability,
                    "HI": indices["HI"],
                    "CI": indices["CI"],
                    "DI": indices["DI"],
                }
            )
        return results, skipped_data

    fig, ax = plt.subplots(figsize=(12, 8))

    # miejsce na suwak na dole i colorbar po prawej
    plt.subplots_adjust(bottom=0.22, right=0.88)

    # Utworzenie stałego colorbar z zakresem 0-100
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=0, vmax=100)
    sm = ScalarMappable(cmap="RdYlGn", norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.90, 0.22, 0.02, 0.66])  # [left, bottom, width, height]
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"Przydatność dla {variety} (%)", fontsize=11, fontweight="bold")

    def plot_for_year(year_display: int) -> None:
        # Pobierz rzeczywisty rok danych
        year_data = YEAR_DATA_MAPPING.get(year_display, year_display)

        results, skipped_data = compute_results_for_year(year_display)
        if skipped_data > 0:
            print(f"Rok {year_display}: pominięto {skipped_data} punktów z powodu niepełnych danych.")

        # Debug
        if results:
            ci_vals = [r["CI"] for r in results]
            suit_vals = [r["suitability"] for r in results]
            ci_min, ci_max = GRAPE_VARIETIES[variety]["CI"]
            print(
                f"Rok {year_display} (dane z {year_data}): CI min={min(ci_vals):.1f} max={max(ci_vals):.1f} | "
                f"przydatność min={min(suit_vals):.1f} max={max(suit_vals):.1f} (punktów: {len(results)})"
            )
            print(f"   Zakres CI dla {variety}: [{ci_min}, {ci_max}] C")
            if len(results) == 1:
                print("Tylko 1 punkt ma dane – mapa będzie jednolita. Uzupełnij cache")
            elif len(set(ci_vals)) <= 1:
                print("Wszystkie punkty mają ten sam CI – ładowany jest ten sam plik cache.")
                p0 = get_cache_path(year_data, results[0]["lat"], results[0]["lon"])
                print(f"Ścieżka cache - pierwszy punkt: {p0}")
                for r in results[:5]:
                    print(f"      ({r['lat']}, {r['lon']}) → CI={r['CI']}, suit={r['suitability']}")
            elif all(s == 100.0 for s in suit_vals):
                print("Wszystkie punkty mają przydatność 100 – CI mieści się w zakresie")
                for r in results[:5]:
                    print(f"      ({r['lat']}, {r['lon']}) → CI={r['CI']}, suit={r['suitability']}")
            else:
                # Przykłady: najniższa i najwyższa przydatność
                lo = min(results, key=lambda x: x["suitability"])
                hi = max(results, key=lambda x: x["suitability"])
                print(f"   Przykład niski:  ({lo['lat']}, {lo['lon']}) CI={lo['CI']} suit={lo['suitability']}")
                print(f"   Przykład wysoki: ({hi['lat']}, {hi['lon']}) CI={hi['CI']} suit={hi['suitability']}")
        ax.clear()

        # limity osi przed rysowaniem, aby zakres zawsze był taki sam
        ax.set_xlim(-10, 25)
        ax.set_ylim(35, 55)
        europe.boundary.plot(ax=ax, edgecolor="black", linewidth=0.8)
        if not results:
            ax.set_title(
                f"Brak danych klimatycznych dla roku {year_display} (dane z {year_data})\n"
                f"sprawdzic cache",
                fontsize=13,
                fontweight="bold",
            )
            fig.canvas.draw_idle()
            return
        lons_grid = np.array([r["lon"] for r in results])
        lats_grid = np.array([r["lat"] for r in results])
        suitability_grid = np.array([r["suitability"] for r in results])

        # Tworzenie gęstej siatki do interpolacji
        lon_min, lon_max = lons_grid.min(), lons_grid.max()
        lat_min, lat_max = lats_grid.min(), lats_grid.max()

        # Rozdzielczość interpolacji
        grid_resolution = 0.1  # stopnie
        lon_interp = np.arange(lon_min, lon_max + grid_resolution, grid_resolution)
        lat_interp = np.arange(lat_min, lat_max + grid_resolution, grid_resolution)
        lon_mesh, lat_mesh = np.meshgrid(lon_interp, lat_interp)

        # Interpolacja danych na gęstą siatkę
        points = np.column_stack((lons_grid, lats_grid))
        values = suitability_grid
        grid_points = np.column_stack((lon_mesh.ravel(), lat_mesh.ravel()))

        # Interpolacja metodą 'cubic' dla płynnych przejść
        try:
            suitability_interp = griddata(
                points,
                values,
                grid_points,
                method="cubic",
                fill_value=np.nan,
            )
        except ValueError:
            suitability_interp = griddata(
                points,
                values,
                grid_points,
                method="linear",
                fill_value=np.nan,
            )
        suitability_interp = suitability_interp.reshape(lon_mesh.shape)

        # Maskowanie obszarów poza lądem
        grid_points_list = [Point(lon, lat) for lon, lat in zip(lon_mesh.ravel(), lat_mesh.ravel())]
        grid_points_gdf = gpd.GeoDataFrame(geometry=grid_points_list, crs=europe.crs)
        mask_land_flat = grid_points_gdf.geometry.within(europe.geometry.unary_union)
        mask_land = mask_land_flat.values.reshape(lon_mesh.shape)
        suitability_interp[~mask_land] = np.nan

        # Tworzenie interpolowanej heatmapy ze stałym zakresem 0-100
        contour = ax.contourf(
            lon_mesh,
            lat_mesh,
            suitability_interp,
            levels=20,
            cmap="RdYlGn",
            vmin=0,
            vmax=100,
            alpha=0.8,
            zorder=1,
            extend="both",
        )
        # Kontury dla lepszej czytelności
        ax.contour(
            lon_mesh,
            lat_mesh,
            suitability_interp,
            levels=10,
            colors="black",
            alpha=0.2,
            linewidths=0.5,
            zorder=2,
        )
        # Limity osi już ustawione na początku funkcji
        ax.set_xlabel("Długość geograficzna (°E)", fontsize=11)
        ax.set_ylabel("Szerokość geograficzna (°N)", fontsize=11)

        title_year_info = f"rok {year_display}"
        if year_data != year_display:
            title_year_info += f" (dane z {year_data})"

        ax.set_title(
            f"Heatmapa przydatności dla szczepu: {variety}\n"
            f"(HI: {GRAPE_VARIETIES[variety]['HI']}, "
            f"CI: {GRAPE_VARIETIES[variety]['CI']}°C, "
            f"DI: {GRAPE_VARIETIES[variety]['DI']} mm)\n"
            f"Dane dzienne: {title_year_info}, Open-Meteo (cache)",
            fontsize=13,
            fontweight="bold",
            pad=18,
        )
        ax.grid(True, alpha=0.3, linestyle="--")
        legend_text = (
            "HI: Huglin Index\n"
            "CI: Cool Night Index\n"
            "DI: Dryness Index\n"
            f"Punktów danych: {len(results)}"
        )
        ax.text(
            0.02,
            0.98,
            legend_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    # Suwak roku
    ax_year = plt.axes([0.15, 0.06, 0.7, 0.03])
    year_slider = Slider(
        ax=ax_year,
        label="Rok",
        valmin=float(years[0]),
        valmax=float(years[-1]),
        valinit=float(initial_year),
        valstep=1.0,
    )

    def on_year_change(val: float) -> None:
        year = int(round(val))
        plot_for_year(year)
        fig.canvas.draw_idle()
    year_slider.on_changed(on_year_change)
    # Pierwsze rysowanie
    plot_for_year(initial_year)
    plt.show()

if __name__ == "__main__":
    create_heatmap("Riesling")