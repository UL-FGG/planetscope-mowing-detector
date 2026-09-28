# Detektor košenj na posnetkih PlanetScope

Orodje za pripravo podatkovnih časovnih kock PlanetScope, zaznavanje možnih košenj na travnikih in ročni pregled zaznanih dogodkov.

Avtorja: **Ana Potočnik Buhvald, Krištof Oštir**  
Univerza v Ljubljani, Fakulteta za gradbeništvo in geodezijo, 2026  
Copyright (c) 2026 Ana Potočnik Buhvald, Krištof Oštir

## Namen

Paket je pripravljen kot ponovljiv delovni postopek za uporabnike, ki imajo dostop do PlanetScope podatkov prek obstoječe Planet/Sentinel Hub BYOC kolekcije.

Paket za izbrano območje in leto:

1. izdela kvadratno območje obdelave iz središčne koordinate,
2. obreže državno masko travnikov na območje obdelave,
3. iz obstoječe BYOC kolekcije zgradi `PlanetCube.nc`,
4. izračuna vegetacijske indekse `NDVI`, `NDRE` in `GNDVI`,
5. s časovnimi vrstami in dodatnimi atributi zazna možne košnje samo znotraj maske travnikov,
6. pripravi rastrske produkte in katalog zaznanih segmentov,
7. omogoči interaktivni pregled dogodkov in ročno potrditev ali zavrnitev zaznave.

Paket ni namenjen samodejni uradni klasifikaciji brez pregleda. Namenjen je pripravi produkta, interpretaciji in validaciji zaznanih košenj.

## Struktura Paketa

```text
detektor_kosenj/
    settings.py
    environment.yml
    requirements.txt
    README.md

    notebooks/
        01_build_product.ipynb
        02_review_events.ipynb

    scripts/
        01_build_product.py
        02_review_events.py

    src/
        aoi.py
        collection.py
        config.py
        cube.py
        grassland.py
        indices.py
        product.py
        timeseries_detection.py
        ...

    data/
        grassland/
            travniki_RABA_GERK_20260831.gpkg
```

## Namestitev

Priporočena je uporaba conda okolja, ker paket uporablja več geoprostorskih knjižnic.

V terminalu odpri mapo paketa in zaženi:

```powershell
conda env create -f environment.yml
conda activate mowing
python -m ipykernel install --user --name mowing-detector --display-name "mowing-detector"
```

Če okolje `mowing` že obstaja:

```powershell
conda activate mowing
conda env update -f environment.yml --prune
python -m ipykernel install --user --name mowing-detector --display-name "mowing-detector"
```

## Nastavitve

Vse glavne nastavitve so v datoteki:

```text
settings.py
```

Primer:

```python
AREA_ID = "area_01_2025"
CENTER_LAT = 46.28735
CENTER_LON = 14.00218
AOI_SIZE = 1000

START_DATE = "2025-03-01"
END_DATE = "2025-10-31"

COLLECTION_ID = " " <- dodajte številko collection_id

USE_GRASSLAND_MASK = True
GRASSLAND_SOURCE = Path("data/grassland/travniki_RABA_GERK_20260831.gpkg")

MIN_COMPONENT_PIXELS = 10
```

Pomen glavnih nastavitev:

- `AREA_ID`: ime območja in leta; uporabi se tudi kot ime izhodne mape,
- `CENTER_LAT`, `CENTER_LON`: središčna koordinata območja v WGS84,
- `AOI_SIZE`: velikost kvadratnega območja v metrih; privzeto `1000`, torej 1000 m x 1000 m,
- `START_DATE`, `END_DATE`: obdobje pridobivanja posnetkov,
- `COLLECTION_ID`: ID obstoječe Planet/Sentinel Hub BYOC kolekcije,
- `USE_GRASSLAND_MASK`: ali se detekcija omeji samo na travnike,
- `MIN_COMPONENT_PIXELS`: najmanjša dovoljena velikost zaznanega segmenta v povezanih pikslih.

Za ponovno izdelavo rezultatov lahko nastaviš:

```python
REBUILD_PLANET_CUBE = True
REBUILD_INDICES_CUBE = True
REBUILD_DETECTION = True
```

Za ponoven zagon brez ponovne obdelave jih nastavi na `False`.

## Dostop do podatkov

Paket uporablja Sentinel Hub credentials:

- `SH_CLIENT_ID`
- `SH_CLIENT_SECRET`

Ob zagonu te skripta vpraša za obe vrednosti. Lahko ju nastaviš tudi kot okoljski spremenljivki:

```powershell
$env:SH_CLIENT_ID="..."
$env:SH_CLIENT_SECRET="..."
```

Credentials se ne zapisujejo v notebook ali v kodo.

## Zagon

### 1. Izdelava Produkta

Odpri:

```text
notebooks/01_build_product.ipynb
```

Izberi kernel `mowing-detector` in poženi edino celico.

Ta korak:

1. izdela AOI,
2. obreže in združi masko travnikov,
3. zgradi `PlanetCube.nc`,
4. izračuna `IndicesCube.nc`,
5. izvede detekcijo košenj,
6. pripravi katalog dogodkov za pregled.

### 2. Pregled Dogodkov

Odpri:

```text
notebooks/02_review_events.ipynb
```

Izberi kernel `mowing-detector` in poženi celico.

Pregledovalnik prikaže:

- prejšnji jasen posnetek,
- posnetek na datum zaznane košnje,
- naslednji jasen posnetek,
- razliko `delta NDVI`,
- časovno vrsto `NDVI`, `NDRE` in `GNDVI`.

Na sliki dogodka so:

- oranžno: vsi zaznani segmenti na izbrani datum,
- rumeno: trenutno izbrani segment.

Za vsak dogodek uporabnik izbere:

- `Mowing`: dogodek je prava košnja,
- `Not mowing`: dogodek ni košnja,
- `Uncertain`: dogodek ni zanesljivo interpretabilen.

Ocene se shranijo v:

```text
data/<AREA_ID>/outputs/events/event_reviews.csv
```

## Izhodni Podatki

Za vsako območje se rezultati shranijo v:

```text
data/<AREA_ID>/
```

Glavne datoteke:

```text
data/<AREA_ID>/aoi/
data/<AREA_ID>/cube/PlanetCube.nc
data/<AREA_ID>/indices/IndicesCube.nc
data/<AREA_ID>/masks/grasslands_clipped.gpkg
data/<AREA_ID>/outputs/product_detection/
data/<AREA_ID>/outputs/events/event_catalog.csv
data/<AREA_ID>/outputs/events/event_reviews.csv
```

Glavni rastrski produkti:

- `MOWING_COUNT.tif`: število zaznanih košenj,
- `MOWING_DOY_1.tif` do `MOWING_DOY_6.tif`: dan v letu za posamezne košnje,
- `LAST_MOWING_DOY.tif`: dan zadnje zaznane košnje,
- `MOWING_CONFIDENCE.tif`: stopnja zaupanja,
- `VALID_OBSERVATIONS.tif`: število veljavnih opazovanj,
- `SCL_MASKED_PERCENT.tif`: delež izločenih opazovanj,
- `QUALITY_FLAG.tif`: ocena kakovosti časovne vrste,
- `MOWING_EVENT_STACK.nc`: časovna kocka binarnih dogodkov košnje.

## Maska Travnikov

V paketu je vključena maska travnikov:

```text
data/grassland/travniki_RABA_GERK_20260831.gpkg
```

Skripta jo za vsako območje samodejno:

1. obreže na AOI,
2. združi v en sloj z `dissolve`,
3. shrani kot:

```text
data/<AREA_ID>/masks/grasslands_clipped.gpkg
```

Detekcija uporablja to lokalno masko. Piksli zunaj travnikov so v končnih rasterjih zapisani kot `NoData`.

## Omejitve

Detekcija je algoritemska ocena. Rezultati so odvisni od:

- gostote časovne vrste,
- oblakov, senc in kakovosti maskiranja,
- pravilnosti maske travnikov,
- fenološkega stanja vegetacije,
- izbranih pragov detektorja.

Zato je ročni pregled dogodkov pomemben del postopka.

## Citiranje

Če uporabljaš ali nadgrajuješ paket, navedi:

Ana Potočnik Buhvald, Krištof Oštir. 2026. Detektor košenj PlanetScope. Univerza v Ljubljani, Fakulteta za gradbeništvo in geodezijo.

## Avtorstvo In Pravice

Avtorja: Ana Potočnik Buhvald, Krištof Oštir  
Institucija: Univerza v Ljubljani, Fakulteta za gradbeništvo in geodezijo  
Leto: 2026  
Copyright (c) 2026 Ana Potočnik Buhvald, Krištof Oštir
