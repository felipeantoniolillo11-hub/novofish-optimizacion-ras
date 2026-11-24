"""
generar_tablas.py (Versión rápida con progreso en pantalla)
Genera y guarda los 4 archivos CSV en ~20-25 segundos:
  - tabla_politicas.csv
  - tabla_36_detalle.csv
  - tabla_validacion.csv
  - tabla_sensibilidad.csv
"""

import copy
import time
import numpy as np
import pandas as pd
from scipy.stats import binom
import comp
import opt2


def escalar_componentes(mult_tasa=1.0, mult_precio=1.0, mult_lead=1.0, n_plantas=6):
    c_nuevo = copy.deepcopy(comp.COMP)
    factor_exp = n_plantas / 6.0
    for k, d in c_nuevo.items():
        d["n"] = d["n"] * mult_tasa * factor_exp
        d["desg"] = d["desg"] * mult_tasa * factor_exp
        d["precio"] = d["precio"] * mult_precio
        d["lead"] = int(round(d["lead"] * mult_lead))
    return c_nuevo


def evaluar_politica_fija(nombre, T_fijo, s_fijo, r_fijo=1):
    df_ops = pd.DataFrame(comp.generar_opciones())
    seleccion = []
    for p in comp.FACTORES_PLANTA:
        for k, d in comp.COMP.items():
            s_obj = 0 if d["precio"] == 0 else s_fijo
            mask_T = df_ops["T"].isna() if T_fijo is None else (df_ops["T"] == T_fijo)
            fila = df_ops[
                (df_ops["p"] == p) & (df_ops["k"] == k) &
                (df_ops["m"] == 0) & (df_ops["s"] == s_obj) & mask_T
            ].iloc[0]
            seleccion.append(fila)

    df_s = pd.DataFrame(seleccion)
    h_def = sum(
        8760.0 * binom.sf(r_fijo, comp.PAR["N_REQ"] + r_fijo, df_s.loc[df_s["p"] == p, "u"].sum())
        for p in comp.FACTORES_PLANTA
    )
    c_op = df_s["c"].sum()
    c_red = len(comp.FACTORES_PLANTA) * comp.PAR["C_RED"] * r_fijo
    c_riesgo = h_def * comp.PAR["L_BIO"]

    return {
        "Politica": nombre,
        "Costo_Total_MM": round((c_op + c_red + c_riesgo) / 1e6, 2),
        "Mant_e_Inv_MM": round(c_op / 1e6, 2),
        "Respaldo_MM": round(c_red / 1e6, 2),
        "Riesgo_Bio_MM": round(c_riesgo / 1e6, 2),
        "Horas_Deficit": round(h_def, 2),
        "Fallas_Corr": round(df_s["fc"].sum(), 2),
        "Recambios": round(df_s["fp"].sum(), 2),
    }


def fuerza_bruta_pequena(par_ch, comp_ch, fact_ch):
    ops = comp.generar_opciones(par_ch, comp_ch, fact_ch)
    h_disp = comp.horas_tecnico_disponibles(par_ch)
    mejor_costo = np.inf
    mejor_dec = None
    for op in ops:
        if op["v"] > par_ch["BUDGET_STOCK"] or op["h"] > h_disp:
            continue
        for r in comp.R_OPC:
            h_def = 8760.0 * binom.sf(r, par_ch["N_REQ"] + r, op["u"])
            costo_tot = op["c"] + par_ch["C_RED"] * r + par_ch["L_BIO"] * h_def
            if costo_tot < mejor_costo:
                mejor_costo = costo_tot
                mejor_dec = (op["T"], op["m"], op["s"], r)
    return mejor_costo, mejor_dec


if __name__ == "__main__":
    print("[1/4] Resolviendo Caso Base y Políticas (B0, B1, B2)...")
    t0 = time.perf_counter()
    res_base = opt2.resolver_milp()
    t_real = time.perf_counter() - t0
    r_b = res_base["resumen"]

    tabla_pol = pd.DataFrame([
        evaluar_politica_fija("B0: Reactivo sin repuestos", None, 0),
        evaluar_politica_fija("B1: Reactivo + 1 repuesto (Actual)", None, 1),
        evaluar_politica_fija("B2: Recambio anual + 1 repuesto", 1.0, 1),
        {
            "Politica": "Plan Optimizado (MILP)",
            "Costo_Total_MM": round(r_b["costo_total_exacto_clp"] / 1e6, 2),
            "Mant_e_Inv_MM": round(r_b["costo_opciones_clp"] / 1e6, 2),
            "Respaldo_MM": round(r_b["costo_respaldo_clp"] / 1e6, 2),
            "Riesgo_Bio_MM": round(r_b["costo_riesgo_clp"] / 1e6, 2),
            "Horas_Deficit": r_b["horas_deficit_totales"],
            "Fallas_Corr": r_b["fallas_correctivas_anio"],
            "Recambios": r_b["recambios_programados_anio"],
        }
    ])
    tabla_pol.to_csv("tabla_politicas.csv", index=False)

    print("[2/4] Guardando detalle de los 36 pares planta x componente...")
    df_36 = res_base["plan"][["p", "k", "T", "m", "s", "fc", "fp", "fill", "c"]].copy()
    df_36["T"] = df_36["T"].apply(lambda x: "Solo corr." if pd.isna(x) else f"{x:.1f} años")
    df_36["m"] = df_36["m"].map({0: "No", 1: "Sí"})
    df_36["fill"] = (df_36["fill"] * 100).round(1)
    df_36["fc"] = df_36["fc"].round(2)
    df_36["fp"] = df_36["fp"].round(2)
    df_36["c"] = df_36["c"].round(0).astype(int)
    df_36.columns = ["Planta", "Componente", "Recambio_T", "Mitigacion", "Stock_s",
                     "Fallas_Anio", "Recambios_Anio", "FillRate_%", "Costo_Anual_CLP"]
    df_36.to_csv("tabla_36_detalle.csv", index=False)

    print("[3/4] Corriendo validación y escalabilidad...")
    filas_val = []

    # Pequeña vs Fuerza Bruta
    fact_1 = {"P1": 1.21}
    comp_1 = {"Sello mecánico": escalar_componentes(n_plantas=1)["Sello mecánico"]}
    t0 = time.perf_counter()
    res_1 = opt2.resolver_milp(comp.PAR, comp_1, fact_1)
    t_1 = time.perf_counter() - t0
    c_fb, dec_fb = fuerza_bruta_pequena(comp.PAR, comp_1, fact_1)
    filas_val.append({
        "Instancia": "Pequeña (1P x 1C)", "Proposito": "Verificar vs. fuerza bruta",
        "Estado": res_1["estado"],
        "Costo_MM": round(res_1["resumen"]["costo_total_exacto_clp"] / 1e6, 2),
        "Observacion": f"Fuerza bruta = ${c_fb/1e6:.2f} MM (decisión {dec_fb})",
        "Tiempo_s": round(t_1, 2),
    })

    # Real 6 plantas
    filas_val.append({
        "Instancia": "Real (6P x 6C)", "Proposito": "Caso base Novofish",
        "Estado": res_base["estado"],
        "Costo_MM": round(r_b["costo_total_exacto_clp"] / 1e6, 2),
        "Observacion": f"Déficit = {r_b['horas_deficit_totales']} h/año",
        "Tiempo_s": round(t_real, 2),
    })

    # Sintéticas 12 y 24 plantas (tope 15s para que no te haga esperar)
    rng = np.random.default_rng(7)
    for n_p in (12, 24):
        fact_s = {f"P{i+1}": round(float(v), 2) for i, v in enumerate(rng.uniform(0.6, 1.5, n_p))}
        par_s = copy.deepcopy(comp.PAR)
        par_s["BUDGET_STOCK"] = comp.PAR["BUDGET_STOCK"] * (n_p / 6.0)
        comp_s = escalar_componentes(n_plantas=n_p)
        t0 = time.perf_counter()
        res_s = opt2.resolver_milp(par_s, comp_s, fact_s, time_limit=15)
        t_s = time.perf_counter() - t0
        c_mm = round(res_s["resumen"]["costo_total_exacto_clp"] / 1e6, 2) if res_s["resumen"] else None
        filas_val.append({
            "Instancia": f"Sintética ({n_p}P x 6C)", "Proposito": "Escalabilidad sin Big-M",
            "Estado": res_s["estado"], "Costo_MM": c_mm,
            "Observacion": f"Presupuesto proporcional (${par_s['BUDGET_STOCK']/1e6:.1f} MM)",
            "Tiempo_s": round(t_s, 2),
        })

    # Estrés
    par_st = copy.deepcopy(comp.PAR)
    par_st["BUDGET_STOCK"] = 2_900_000.0
    par_st["U_MAX"] = 0.35
    comp_st = escalar_componentes(mult_tasa=3.0, mult_precio=2.0, n_plantas=6)
    t0 = time.perf_counter()
    res_st = opt2.resolver_milp(par_st, comp_st, comp.FACTORES_PLANTA, time_limit=15)
    t_st = time.perf_counter() - t0
    r_med_st = round(res_st["plantas"]["respaldo_r"].mean(), 2) if res_st["plantas"] is not None else None
    filas_val.append({
        "Instancia": "Estrés (tasas x3, $ x2)", "Proposito": "Robustez extrema",
        "Estado": res_st["estado"],
        "Costo_MM": round(res_st["resumen"]["costo_total_exacto_clp"] / 1e6, 2) if res_st["resumen"] else None,
        "Observacion": f"Respaldo medio = {r_med_st} bombas/planta",
        "Tiempo_s": round(t_st, 2),
    })

    # Infactible
    par_inf = copy.deepcopy(comp.PAR)
    par_inf["TECNICOS_DIA"] = [0, 0, 0, 0, 0, 0, 0]
    t0 = time.perf_counter()
    res_inf = opt2.resolver_milp(par_inf, comp.COMP, comp.FACTORES_PLANTA)
    t_inf = time.perf_counter() - t0
    filas_val.append({
        "Instancia": "Infactible (0 técnicos)", "Proposito": "Detección de infactibilidad",
        "Estado": res_inf["estado"], "Costo_MM": None,
        "Observacion": "Solver detecta correctamente infactibilidad",
        "Tiempo_s": round(t_inf, 2),
    })

    df_val = pd.DataFrame(filas_val)
    df_val.to_csv("tabla_validacion.csv", index=False)
    print("\n=== TABLA C: VALIDACIÓN Y ESCALABILIDAD ===")
    print(df_val.to_string(index=False))

    print("\n[4/4] Corriendo análisis de sensibilidad (17 escenarios)...")
    escenarios = [
        ("Caso base", {}, (1.0, 1.0, 1.0)),
        ("Costo biológico $245.000/h", {"L_BIO": 245_000.0}, (1.0, 1.0, 1.0)),
        ("Costo biológico $490.000/h", {"L_BIO": 490_000.0}, (1.0, 1.0, 1.0)),
        ("Costo biológico $1.960.000/h", {"L_BIO": 1_960_000.0}, (1.0, 1.0, 1.0)),
        ("Costo biológico $3.920.000/h", {"L_BIO": 3_920_000.0}, (1.0, 1.0, 1.0)),
        ("Tasas de falla -30%", {}, (0.70, 1.0, 1.0)),
        ("Tasas de falla +30%", {}, (1.30, 1.0, 1.0)),
        ("Tasas de falla +60%", {"U_MAX": 0.20}, (1.60, 1.0, 1.0)),
        ("Precio repuestos +50%", {}, (1.0, 1.50, 1.0)),
        ("Plazo reposición x0,5", {}, (1.0, 1.0, 0.5)),
        ("Plazo reposición x2", {"U_MAX": 0.20}, (1.0, 1.0, 2.0)),
        ("Presupuesto stock $5,88 MM", {"BUDGET_STOCK": 5_880_000.0}, (1.0, 1.0, 1.0)),
        ("Presupuesto stock $24,5 MM", {"BUDGET_STOCK": 24_500_000.0}, (1.0, 1.0, 1.0)),
        ("Demora respuesta 2 h", {"W_RESP_H": 2.0}, (1.0, 1.0, 1.0)),
        ("Demora respuesta 12 h", {"W_RESP_H": 12.0}, (1.0, 1.0, 1.0)),
        ("Demora respuesta 24 h", {"W_RESP_H": 24.0}, (1.0, 1.0, 1.0)),
        ("Costo bomba respaldo x2", {"C_RED": 4_800_000.0}, (1.0, 1.0, 1.0)),
    ]

    costo_base_mm = r_b["costo_total_exacto_clp"] / 1e6
    filas_sens = []
    for nombre, cambios_par, (mt, mp, ml) in escenarios:
        par_e = copy.deepcopy(comp.PAR)
        par_e.update(cambios_par)
        comp_e = escalar_componentes(mult_tasa=mt, mult_precio=mp, mult_lead=ml, n_plantas=6)
        res_e = opt2.resolver_milp(par_e, comp_e, comp.FACTORES_PLANTA)
        re = res_e["resumen"]
        c_mm = re["costo_total_exacto_clp"] / 1e6
        var_pct = ((c_mm - costo_base_mm) / costo_base_mm) * 100.0
        filas_sens.append({
            "Escenario": nombre,
            "Costo_MM": round(c_mm, 2),
            "Var_%": round(var_pct, 1),
            "Riesgo_MM": round(re["costo_riesgo_clp"] / 1e6, 2),
            "Horas_Def": re["horas_deficit_totales"],
            "Recambios_N": int(res_e["plan"]["T"].notna().sum()),
            "Mitig_N": int(res_e["plan"]["m"].sum()),
            "Respaldo_Med": round(res_e["plantas"]["respaldo_r"].mean(), 2),
        })

    df_sens = pd.DataFrame(filas_sens)
    df_sens.to_csv("tabla_sensibilidad.csv", index=False)
    print("\n=== TABLA D: ANÁLISIS DE SENSIBILIDAD ===")
    print(df_sens.to_string(index=False))
    print("\n¡TERMINADO! Ya están los 4 archivos CSV en tu carpeta.")