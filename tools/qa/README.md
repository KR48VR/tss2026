# Browser and accuracy checks for the Work and AI Map

Headless-browser checks (Playwright, chromium at /opt/pw-browsers/chromium-1243/chrome-linux64/chrome in the cloud environment) and an accuracy audit of the page against the processed tables.

Setup once per session: `cd tools/qa && npm install axe-core` (the scripts read `tools/qa/node_modules/axe-core/axe.min.js`), and serve the site: `cd site && python3 -m http.server 8765 --bind 127.0.0.1 &`.

- `quick_site.py` rebuilds index.html and the task files without the search assets (fast loop); `python -m onetx site` is the full build.
- `run_all.sh` rebuilds the page, runs every check and the full site build; read its log for the results.
- `many_country_qa.py CODE ...` walks each country's views; `seven_country_accuracy_qa.py` audits every record in `countries.SPECS` and the Europe table; `leads_qa.py` checks the lead wording and the adoption chart; `bar_qa.py` the pinned bar; `readability_qa.py` the method view and the Europe chart; `europe_tab_qa.py` the Europe tab; `companion_qa.py` the companion page.

Screenshots and logs go to `tools/qa/out` (or `$QA_OUT`), which is ignored by git.
