"""Converte o JSON (formato Google Charts DataTable) dos marégrafos EPAGRI num arquivo simples para o app."""
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

pasta, saida = sys.argv[1], sys.argv[2]
BRT = timezone(timedelta(hours=-3))


def quando(v):
    if v is None:
        return None
    s = str(v)
    m = re.match(r"Date\((\d+),(\d+),(\d+)(?:,(\d+))?(?:,(\d+))?", s)  # mês começa em 0
    if m:
        y, mo, d, h, mi = (int(x or 0) for x in m.groups())
        return datetime(y, mo + 1, d, h, mi, tzinfo=BRT)
    m = re.match(r"(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\s+(\d{1,2}):(\d{2})", s)
    if m:
        d, mo, y, h, mi = m.groups()
        y = int(y) if y else datetime.now(BRT).year
        y = y + 2000 if y < 100 else y
        t = datetime(y, int(mo), int(d), int(h), int(mi), tzinfo=BRT)
        if t - datetime.now(BRT) > timedelta(days=60):  # virada de ano
            t = t.replace(year=y - 1)
        return t
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})", s)
    if m:
        return datetime(*map(int, m.groups()), tzinfo=BRT)
    return None


out = {"gerado": datetime.now(timezone.utc).isoformat(), "estacoes": {}}
for nome in ("floripa", "imbituba"):
    arq = os.path.join(pasta, f"{nome}_raw.json")
    try:
        txt = open(arq, encoding="utf-8-sig", errors="replace").read()
        dt = json.loads(txt)
    except Exception as e:  # noqa: BLE001
        out["estacoes"][nome] = {"erro": str(e)}
        continue
    cols = [c.get("label") or c.get("id") or "" for c in dt.get("cols", [])]
    linhas = []
    for r in dt.get("rows", []):
        c = [(x or {}).get("v") if isinstance(x, dict) else x for x in r.get("c", [])]
        t = quando(c[0]) if c else None
        if not t:
            continue
        vals = [None if v in ("", None) else float(v) for v in c[1:]]
        linhas.append([int(t.timestamp() * 1000)] + vals)
    out["estacoes"][nome] = {"colunas": cols, "linhas": linhas, "exemplo_bruto": dt.get("rows", [])[:2]}
with open(saida, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
for k, v in out["estacoes"].items():
    print(k, v.get("erro") or (v["colunas"], len(v["linhas"])))
