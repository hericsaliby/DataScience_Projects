#!/usr/bin/env python3
"""
Analisa o diário exportado do app (diario-pesca.json) com validação cruzada.

Mostra o que mais pesa nas suas saídas boas (resultado Bom ou Muito bom)
usando regressão logística e GradientBoosting, e compara com a nota de regras.

Uso:
    python analisar_diario.py diario-pesca.json
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NOMES = {"vento": "Vento no ponto (fraco)", "mar": "Mar adequado", "mare": "Movimento de maré",
         "enchendo": "Maré enchendo", "nivel": "Maré alta", "sol": "Sol/lua", "pres": "Pressão favorável",
         "sizigia": "Sizígia", "noite": "Noite", "agua": "Água quente"}


def main(caminho):
    with open(caminho, encoding="utf-8") as f:
        j = json.load(f)
    log = [x for x in j.get("log", j) if x.get("f")]
    if len(log) < 15:
        sys.exit(f"Só {len(log)} registros com condições. Junte pelo menos 15 para uma análise útil.")
    X = pd.DataFrame([x["f"] for x in log])[list(NOMES)]
    y = np.array([1 if x["res"] >= 2 else 0 for x in log])
    if y.min() == y.max():
        sys.exit("Todos os registros têm o mesmo resultado; registre também as saídas fracas.")
    k = min(5, int(np.bincount(y).min()))
    cv = StratifiedKFold(n_splits=max(2, k), shuffle=True, random_state=0)

    lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8)
    print(f"{len(log)} saídas, {y.sum()} boas ({y.mean():.0%})")
    for nome, m in [("Regressão logística", lr), ("GradientBoosting", gb)]:
        s = cross_val_score(m, X, y, cv=cv, scoring="roc_auc")
        print(f"{nome:22s} AUC {s.mean():.2f} ± {s.std():.2f}  (0.5 = chute)")

    lr.fit(X, y)
    coef = lr[-1].coef_[0]
    print("\nPesos da regressão logística (+ ajuda, - atrapalha):")
    for i in np.argsort(-np.abs(coef)):
        print(f"  {NOMES[X.columns[i]]:24s} {coef[i]:+.2f}")

    gb.fit(X, y)
    imp = permutation_importance(gb, X, y, n_repeats=20, random_state=0)
    print("\nImportância (GradientBoosting, permutação):")
    for i in np.argsort(-imp.importances_mean):
        print(f"  {NOMES[X.columns[i]]:24s} {imp.importances_mean[i]:.3f}")

    por_ponto = pd.DataFrame({"ponto": [x["spot"] for x in log], "boa": y}).groupby("ponto")["boa"].agg(["count", "mean"])
    print("\nTaxa de saídas boas por ponto:\n", por_ponto.sort_values("mean", ascending=False).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "diario-pesca.json")
