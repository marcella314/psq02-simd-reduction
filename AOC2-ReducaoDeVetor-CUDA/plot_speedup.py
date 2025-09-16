# plot_speedup.py
import csv
from pathlib import Path
import math

import matplotlib.pyplot as plt


CSV_PATH = Path("resultados.csv")
OUT_PATH = Path("speedup.png")

if not CSV_PATH.exists():
    raise SystemExit(f"Arquivo não encontrado: {CSV_PATH.resolve()}")

# Lê o CSV em dicionários (sem depender de pandas)
with CSV_PATH.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

def fnum(v):
    # Converte strings tipo "0.0946" ou "0,0946" para float
    if isinstance(v, (int, float)): return float(v)
    s = str(v).strip()
    s = s.replace(" ", "")
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    return float(s)

# Extrai colunas com fallback
N = []
speedup_sum = []
speedup_max = []

for r in rows:
    n = int(fnum(r["N"]))
    N.append(n)

    # tenta usar speedups prontos
    ss = r.get("speedup_sum")
    sm = r.get("speedup_max")

    if ss is None or sm is None or str(ss).strip() == "" or str(sm).strip() == "":
        # recalcula se não existir
        sum_scalar = fnum(r["sum_scalar(ms)"])
        sum_avx2   = fnum(r["sum_avx2(ms)"])
        max_scalar = fnum(r["max_scalar(ms)"])
        max_avx2   = fnum(r["max_avx2(ms)"])
        speedup_sum.append(sum_scalar / sum_avx2)
        speedup_max.append(max_scalar / max_avx2)
    else:
        speedup_sum.append(fnum(ss))
        speedup_max.append(fnum(sm))

# Ordena por N (caso o CSV não esteja ordenado)
z = sorted(zip(N, speedup_sum, speedup_max), key=lambda t: t[0])
N, speedup_sum, speedup_max = map(list, zip(*z))

# --- Plot (matplotlib, sem estilos/cores específicas) ---
plt.figure()
plt.plot(N, speedup_sum, marker="o", label="speedup_sum")
plt.plot(N, speedup_max, marker="o", label="speedup_max")
plt.title("Speedup (AVX2 vs Escalar) × Tamanho N")
plt.xlabel("N (nº de elementos)")
plt.ylabel("Speedup (×)")
plt.xscale("log")  # opcional: comente esta linha se não quiser escala log
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=180)

print(f"OK! Gráfico salvo em: {OUT_PATH.resolve()}")
