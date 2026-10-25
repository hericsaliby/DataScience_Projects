#!/usr/bin/env python3
"""
Calibra o vento de um modelo (GFS, ECMWF ou ICON) com uma estação EPAGRI/CIRAM.

Técnica: MOS (Model Output Statistics). Para cada hora em que existe
observação da estação, buscamos o que o modelo previu para aquela hora
(API de previsões históricas da Open-Meteo) e ajustamos:

    vento_real = a + b * vento_modelo        (um ajuste global + um por setor de 45°)

O ajuste por setor captura efeitos locais que o GFS (~25 km) não enxerga:
morro que segura o sul, canalização na baía, etc. Também treinamos um
GradientBoosting só para comparar, com validação temporal (treina no início,
testa no fim) para não enganar o erro.

Uso:
    python calibrar_gfs_epagri.py --csv estacao.csv --lat -27.60 --lon -48.55 \
        --unidade ms --modelo gfs_seamless --ponto ribeirao --saida calib.json

O JSON gerado é importado no app em "Dados > JSON de calibração".
"""
import argparse
import json
import math
import re
import sys

import numpy as np
import pandas as pd
import requests

KMH = {"ms": 3.6, "kmh": 1.0, "kn": 1.852}


def ler_estacao(caminho, unidade):
    """Lê o CSV exportado do CIRAM tentando adivinhar separador e colunas."""
    with open(caminho, encoding="utf-8", errors="replace") as f:
        txt = f.read()
    linhas = [l for l in txt.splitlines() if l.strip()]
    hdr = next((i for i, l in enumerate(linhas) if re.search(r"vel|vento|wind", l, re.I)), 0)
    sep = ";" if linhas[hdr].count(";") >= linhas[hdr].count(",") else ","
    df = pd.read_csv(caminho, sep=sep, skiprows=hdr, decimal="," if sep == ";" else ".",
                     encoding="utf-8", encoding_errors="replace")
    cols = {c: c.strip().lower() for c in df.columns}

    def acha(rx):
        for c, l in cols.items():
            if re.search(rx, l):
                return c
        return None

    c_data = acha(r"^data|date|timestamp")
    c_hora = acha(r"^hora|hour")
    c_vel = acha(r"vel.*(vento|med)|vento.*vel|wind.?speed|^vel")
    c_dir = acha(r"dir")
    if not c_data or not c_vel:
        sys.exit(f"Não achei colunas de data/velocidade. Colunas: {list(df.columns)}")
    texto = df[c_data].astype(str)
    if c_hora:
        texto = texto + " " + df[c_hora].astype(str)
    t = pd.to_datetime(texto, dayfirst=True, errors="coerce")
    out = pd.DataFrame({
        "t": t.dt.floor("h"),
        "obs": pd.to_numeric(df[c_vel], errors="coerce") * KMH[unidade],
        "dir_obs": pd.to_numeric(df[c_dir], errors="coerce") if c_dir else np.nan,
    }).dropna(subset=["t", "obs"])
    return out.groupby("t", as_index=False).mean()


def buscar_modelo(lat, lon, ini, fim, modelo):
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    p = dict(latitude=lat, longitude=lon, start_date=ini, end_date=fim,
             hourly="wind_speed_10m,wind_direction_10m", models=modelo,
             wind_speed_unit="kmh", timezone="America/Sao_Paulo")
    r = requests.get(url, params=p, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    return pd.DataFrame({"t": pd.to_datetime(h["time"]),
                         "mod": h["wind_speed_10m"], "dir": h["wind_direction_10m"]}).dropna()


def ajuste(x, y):
    if len(x) < 12:
        return None
    b, a = np.polyfit(x, y, 1)
    b = float(np.clip(b, 0.3, 2.0))
    a = float(np.mean(y) - b * np.mean(x))
    return {"a": round(a, 3), "b": round(b, 3), "n": int(len(x))}


def aplicar(cal, x, d):
    p = cal["global"]
    s = cal["sectors"][int(round((d % 360) / 45)) % 8]
    if s:
        p = s
    return max(0.0, p["a"] + p["b"] * x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--unidade", choices=KMH, default="ms")
    ap.add_argument("--modelo", default="gfs_seamless", help="gfs_seamless, ecmwf_ifs025, icon_seamless")
    ap.add_argument("--ponto", default="all", help="id do ponto no app (ex.: ribeirao) ou all")
    ap.add_argument("--saida", default="calibracao.json")
    a = ap.parse_args()

    obs = ler_estacao(a.csv, a.unidade)
    ini, fim = obs.t.min().date().isoformat(), obs.t.max().date().isoformat()
    print(f"Estação: {len(obs)} horas de {ini} a {fim}")
    mod = buscar_modelo(a.lat, a.lon, ini, fim, a.modelo)
    df = obs.merge(mod, on="t").dropna(subset=["obs", "mod", "dir"])
    print(f"Horas pareadas com {a.modelo}: {len(df)}")
    if len(df) < 48:
        sys.exit("Poucos dados pareados (mínimo 48 horas).")

    # validação temporal: 70% iniciais treinam, 30% finais testam
    df = df.sort_values("t").reset_index(drop=True)
    corte = int(len(df) * 0.7)
    tr, te = df.iloc[:corte], df.iloc[corte:]

    def calibra(base):
        setor = (np.round((base["dir"] % 360) / 45) % 8).astype(int)
        return {"global": ajuste(base["mod"].values, base["obs"].values),
                "sectors": [ajuste(base["mod"][setor == k].values, base["obs"][setor == k].values) for k in range(8)]}

    cal_tr = calibra(tr)
    rmse = lambda p, y: float(math.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))
    e0 = rmse(te["mod"], te["obs"])
    e1 = rmse([aplicar(cal_tr, x, d) for x, d in zip(te["mod"], te["dir"])], te["obs"])
    print(f"Teste (últimos 30%): erro bruto {e0:.2f} km/h -> MOS por setor {e1:.2f} km/h")

    try:  # comparação com um modelo não linear
        from sklearn.ensemble import GradientBoostingRegressor
        X = lambda d: np.c_[d["mod"], np.sin(np.radians(d["dir"])), np.cos(np.radians(d["dir"])),
                            np.sin(2 * np.pi * d["t"].dt.hour / 24), np.cos(2 * np.pi * d["t"].dt.hour / 24)]
        gb = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8)
        gb.fit(X(tr), tr["obs"])
        print(f"Comparação GradientBoosting (vento+direção+hora): {rmse(gb.predict(X(te)), te['obs']):.2f} km/h")
    except ImportError:
        pass

    # calibração final com todos os dados
    cal = calibra(df)
    e_all0 = rmse(df["mod"], df["obs"])
    e_all1 = rmse([aplicar(cal, x, d) for x, d in zip(df["mod"], df["dir"])], df["obs"])
    nomes = ["N", "NE", "L", "SE", "S", "SO", "O", "NO"]
    for k, s in enumerate(cal["sectors"]):
        if s:
            print(f"  vento de {nomes[k]:>2}: real = {s['a']:+.1f} + {s['b']:.2f} x modelo  ({s['n']} h)")
    out = dict(unit="kmh", spot=a.ponto, model=a.modelo, n=int(len(df)), periodo=[ini, fim], estacao=[a.lat, a.lon],
               rmse_before=round(e_all0, 2), rmse_after=round(e_all1, 2), rmse_teste=[round(e0, 2), round(e1, 2)], **cal)
    with open(a.saida, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"Salvo em {a.saida}. Importe no app: Dados > JSON de calibração.")


if __name__ == "__main__":
    main()
