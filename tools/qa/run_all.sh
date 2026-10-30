#!/bin/bash
# After the full forecast: rebuild the page, run every QA script with Korea included, then the full site build with search assets.
set -u
cd /home/user/tss2026
Q=/home/user/tss2026/tools/qa
mkdir -p $Q/r21 $Q/out
python3 $Q/quick_site.py 2>&1 | tail -2
curl -s -o /dev/null -w "server %{http_code}\n" http://127.0.0.1:8765/index.html || (cd site && setsid nohup python3 -m http.server 8765 --bind 127.0.0.1 > server.log 2>&1 &)
sleep 1
echo "== bar_qa"; python3 $Q/bar_qa.py 2>&1 | grep "header\|axe\|overflow"
echo "== leads_qa"; python3 $Q/leads_qa.py > $Q/r21/leads_qa.log 2>&1; echo "clean axe: $(grep -c '0 violations' $Q/r21/leads_qa.log)"; grep "errors:\|KOR breath\|KOR tile" $Q/r21/leads_qa.log | cut -c1-400
echo "== readability_qa"; python3 $Q/readability_qa.py > $Q/r21/readability_qa.log 2>&1; echo "clean axe: $(grep -c '0 violations' $Q/r21/readability_qa.log)"; grep "errors:" $Q/r21/readability_qa.log
echo "== europe_tab_qa"; python3 $Q/europe_tab_qa.py > $Q/r21/europe.log 2>&1; echo "clean axe: $(grep -c '0 violations' $Q/r21/europe.log)"; grep "errors:" $Q/r21/europe.log
echo "== many_country_qa"; python3 $Q/many_country_qa.py SWE FIN DNK NLD IRL JPN KOR DEU FRA NZL > $Q/r21/many_qa.log 2>&1; echo "clean axe runs: $(grep -c '0 violations' $Q/r21/many_qa.log); empty error lists: $(grep -c 'errors: \[\]' $Q/r21/many_qa.log); non-empty: $(grep -c 'errors: \[[^]]' $Q/r21/many_qa.log)"; grep "KOR\|DEU\|FRA\|NZL" $Q/r21/many_qa.log | head -12 | cut -c1-300
echo "== accuracy"; python3 $Q/seven_country_accuracy_qa.py > $Q/r21/accuracy.log 2>&1; grep "PROBLEMS\|KOR\|DEU\|FRA\|NZL" $Q/r21/accuracy.log | head -12
echo "== companion"; python3 $Q/companion_qa.py 2>&1 | tail -4
echo "== tests"; python -m pytest -q tests 2>&1 | tail -1
echo "== full site build"; python -m onetx site > $Q/r21/site_build.log 2>&1; tail -2 $Q/r21/site_build.log
ls -la site/index.html site/search.json site/emb.bin.txt site/kor_tasks.json site/deu_tasks.json site/fra_tasks.json 2>/dev/null | awk '{print $5, $9}'
