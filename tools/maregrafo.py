"""Primeira versão: guarda a página e lista scripts e URLs que parecem trazer os dados do marégrafo."""
import json
import re
import sys

html = open(sys.argv[1], encoding="utf-8", errors="replace").read()
info = {
    "bytes": len(html),
    "scripts": re.findall(r'<script[^>]+src="([^"]+)"', html)[:80],
    "iframes": re.findall(r'<iframe[^>]+src="([^"]+)"', html),
    "urls_dados": sorted(set(re.findall(r'["\']((?:https?:)?/[^"\']*(?:json|csv|php\?|api|mare|dados)[^"\']*)["\']', html, re.I)))[:80],
    "trechos": [html[max(0, m.start() - 400):m.start() + 400] for m in re.finditer(r"Florian", html)][:6],
}
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=1)
print(json.dumps({k: (v if k != "trechos" else len(v)) for k, v in info.items()}, ensure_ascii=False, indent=1))
