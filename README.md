# Verbrauchsinfo

This is a chart generator for the consumption data, provided in a CSV file.

## CSV schema

Expected CSV columns:

- One month column (default: `Monat`)
- One or more numeric tenant columns (kWh values)

Example:

```csv
Monat,Haus A,Haus B
2025-01,1200,980
2025-02,1100,1030
2025-03,990,1015
```

## Usage

Grouped bars (default order from CSV files):

```bash
python main.py --folder ./data --output png --title "Heizungsverbrauch 2025" --ylabel "kWh" --outdir charts
```

Stacked bars with chronological month sorting:

```bash
python main.py --folder ./data --stacked --month-sort chronological --month-column Monat --output pdf
```

## Notes

- The tool validates input before plotting (required month column, numeric tenant columns, duplicate months).
- Month input must be date-like (for example `YYYY-MM`, `YYYY/MM`, `MM/YYYY`, `YYYY-MM-DD`, `DD.MM.YYYY`).
- Month labels are rendered in German on the chart.
