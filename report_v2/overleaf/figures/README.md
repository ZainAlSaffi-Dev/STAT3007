# figures

This folder will hold the PDF figures of the report. Each one is made by a script in `report_v2/code/`, which does not exist yet. `report_v2/EXPERIMENTS_TODO.md` lists the items E1 to E18 and the file name of each.

Until a figure exists, `main.tex` draws a red placeholder box in its place, so the report builds without any file here. When a figure is ready, replace its `\placeholderfigure{...}` with `\includegraphics[width=\linewidth]{figures/<name>.pdf}`.
