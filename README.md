# Pişirme Laboratuvarı — Modular V2

This branch/repository is a mechanical modularization of the tested V1.1 app.
Analysis algorithms, thresholds, widget keys, and page behavior were preserved.

## Structure

```text
app.py
requirements.txt
core/
  data_merger_core.py
sections/
  potato.py
  pizza.py
  borek.py
  smallcake.py
  pyrocam.py
  bread.py
  data_merger.py
  teflon_block.py
  cookie.py
  flour_disk.py
ui/
  home.py
  layout.py
  navigation.py
scripts/
  verify_structure.py
```

## Assets

Copy the existing logo/image assets into the repository root next to `app.py`:

- `Lab_Logo.png`
- `Patates_Logo.png`
- `Pizza_Logo.png`
- `Borek_Logo.png`
- `Smallcake_Logo.png`
- `PyroCam_Logo.png`
- `Bread_Logo.png`
- `DataMerger_Logo.png`
- `TeflonBlock_Logo.png`
- `Cookie_Logo.png`
- `FlourDisk_Logo.png`

Missing product logos fall back to emoji cards. The main logo still uses the original fallback behavior.

## Run

```bash
python -m pip install -r requirements.txt
python scripts/verify_structure.py
streamlit run app.py
```

## Recommended Git workflow

From the stable V1.1 repository:

```bash
git checkout -b refactor/modular-v2
```

Copy this repository content into that branch, keep the existing asset files, then run the smoke checklist.

## Important

Do not delete the stable V1.1 branch/tag until every page passes the regression checklist.
