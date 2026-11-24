"""
opt2.py
Construcción y resolución del modelo MILP (Programación Lineal Entera Mixta)
con PuLP + CBC. Utiliza desagregación de indisponibilidad (U_pr) para eliminar
la formulación 'Big-M' y escala costos a millones de CLP para estabilidad numérica.

Empresa: Novofish S.A. - Capstone Analytics
"""

import numpy as np
import pandas as pd
import pulp
from scipy.stats import binom
import comp

ESCALA_MM = 1e6  # Trabajamos internamente en Millones de CLP (MM$) en el solver


def generar_tangentes_riesgo(par=comp.PAR, r_opc=comp.R_OPC, n_pts=11):
    """
    Genera rectas tangentes (a_rj, b_rj) que aproximan por debajo la función
    convexa de riesgo biológico: g_r(U) = L_BIO * 8760 * P[Bin(N_REQ + r, U) > r].
    Retorna coeficientes escalados a Millones de CLP (MM$).
    """
    puntos_U = np.geomspace(0.0005, 0.07, n_pts)
    tangentes = {}
    eps = 1e-6  # Paso para derivada numérica centrada

    for r in r_opc:
        n_bombas = par["N_REQ"] + r
        tangentes[r] = []
        for u0 in puntos_U:
            # Costo exacto del riesgo biológico en el punto u0 (en MM$)
            g_u0 = (par["L_BIO"] * 8760.0 * binom.sf(r, n_bombas, u0)) / ESCALA_MM
            # Pendiente a_rj = dg_r / dU evaluada en u0
            g_mas = (par["L_BIO"] * 8760.0 * binom.sf(r, n_bombas, u0 + eps)) / ESCALA_MM
            g_men = (par["L_BIO"] * 8760.0 * binom.sf(r, n_bombas, max(0.0, u0 - eps))) / ESCALA_MM
            a_rj = (g_mas - g_men) / (2.0 * eps)
            # Intercepto b_rj = g(u0) - a_rj * u0
            b_rj = g_u0 - a_rj * u0
            tangentes[r].append((a_rj, b_rj))

    return tangentes


def resolver_milp(par=comp.PAR, comp_dict=comp.COMP, factores=comp.FACTORES_PLANTA,
                  time_limit=120, msg_solver=False):
    """
    Arma y resuelve el modelo MILP desagregado sin Big-M.
    Luego reevalúa el costo biológico con la distribución Binomial exacta.
    """
    plantas = list(factores.keys())
    componentes = list(comp_dict.keys())
    opciones = comp.generar_opciones(par, comp_dict, factores)
    tangentes = generar_tangentes_riesgo(par)
    h_disponibles = comp.horas_tecnico_disponibles(par)

    # Inicialización del problema de minimización en PuLP
    modelo = pulp.LpProblem("Optimizacion_Recambio_Novofish", pulp.LpMinimize)

    # --------------------------------------------------------------------------
    # 1. VARIABLES DE DECISIÓN
    # --------------------------------------------------------------------------
    # x[i] = 1 si se elige la opción i = (p, k, T, m, s)
    x = pulp.LpVariable.dicts("x", range(len(opciones)), cat=pulp.LpBinary)

    # y[p, r] = 1 si la planta p opera con r bombas de respaldo
    y = pulp.LpVariable.dicts("y", [(p, r) for p in plantas for r in comp.R_OPC],
                              cat=pulp.LpBinary)

    # U_pr[p, r] = indisponibilidad de una bomba en planta p si elige respaldo r (sin Big-M)
    U_pr = pulp.LpVariable.dicts("U", [(p, r) for p in plantas for r in comp.R_OPC],
                                 lowBound=0.0, upBound=par["U_MAX"], cat=pulp.LpContinuous)

    # rho[p] = costo anual esperado de riesgo biológico en planta p (en MM$)
    rho = pulp.LpVariable.dicts("rho", plantas, lowBound=0.0, cat=pulp.LpContinuous)

    # --------------------------------------------------------------------------
    # 2. FUNCIÓN OBJETIVO (Minimizar costo anual esperado total en MM$)
    # --------------------------------------------------------------------------
    costo_opciones = pulp.lpSum((op["c"] / ESCALA_MM) * x[i] for i, op in enumerate(opciones))
    costo_respaldo = pulp.lpSum((par["C_RED"] * r / ESCALA_MM) * y[(p, r)]
                                for p in plantas for r in comp.R_OPC)
    costo_riesgo = pulp.lpSum(rho[p] for p in plantas)

    modelo += costo_opciones + costo_respaldo + costo_riesgo, "Costo_Total_Anual_MM"

    # --------------------------------------------------------------------------
    # 3. RESTRICCIONES DEL MODELO
    # --------------------------------------------------------------------------
    # Agrupamos índices de opciones por (planta, componente) para agilizar el armado
    idx_pk = {(p, k): [] for p in plantas for k in componentes}
    idx_p = {p: [] for p in plantas}
    for i, op in enumerate(opciones):
        idx_pk[(op["p"], op["k"])].append(i)
        idx_p[op["p"]].append(i)

    # R1: Exactamente una política operativa por cada planta y componente
    for p in plantas:
        for k in componentes:
            modelo += pulp.lpSum(x[i] for i in idx_pk[(p, k)]) == 1, f"Una_Politica_{p}_{k}"

    # R2: Exactamente un nivel entero de respaldo por cada planta
    for p in plantas:
        modelo += pulp.lpSum(y[(p, r)] for r in comp.R_OPC) == 1, f"Un_Respaldo_{p}"

    # R3: Presupuesto máximo de inventario en bodega
    modelo += (
        pulp.lpSum(opciones[i]["v"] * x[i] for i in range(len(opciones))) <= par["BUDGET_STOCK"],
        "Presupuesto_Inventario"
    )

    # R4: Capacidad anual de horas-técnico disponibles
    modelo += (
        pulp.lpSum(opciones[i]["h"] * x[i] for i in range(len(opciones))) <= h_disponibles,
        "Capacidad_Horas_Tecnico"
    )

    # R5: Desagregación exacta de indisponibilidad U_pr (reemplaza la M grande)
    for p in plantas:
        indisp_planta = pulp.lpSum(opciones[i]["u"] * x[i] for i in idx_p[p])
        modelo += pulp.lpSum(U_pr[(p, r)] for r in comp.R_OPC) == indisp_planta, f"Suma_U_{p}"
        for r in comp.R_OPC:
            modelo += U_pr[(p, r)] <= par["U_MAX"] * y[(p, r)], f"Activar_U_{p}_{r}"

    # R6: Linealización convexa del riesgo biológico mediante rectas tangentes
    n_tangentes = len(tangentes[comp.R_OPC[0]])
    for p in plantas:
        for j in range(n_tangentes):
            cota_riesgo = pulp.lpSum(
                tangentes[r][j][0] * U_pr[(p, r)] + tangentes[r][j][1] * y[(p, r)]
                for r in comp.R_OPC
            )
            modelo += rho[p] >= cota_riesgo, f"Tangente_{p}_{j}"

    # --------------------------------------------------------------------------
    # 4. RESOLUCIÓN CON SOLVER CBC
    # --------------------------------------------------------------------------
    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=msg_solver)
    modelo.solve(solver)
    estado = pulp.LpStatus[modelo.status]

    if estado not in ("Optimal", "Not Solved") or pulp.value(modelo.objective) is None:
        return {"estado": estado, "plan": None, "plantas": None, "resumen": None}

    # --------------------------------------------------------------------------
    # 5. EXTRACCIÓN Y REEVALUACIÓN EXACTA CON BINOMIAL
    # --------------------------------------------------------------------------
    filas_plan = [opciones[i] for i in range(len(opciones)) if pulp.value(x[i]) > 0.5]
    df_plan = pd.DataFrame(filas_plan)

    filas_plantas = []
    for p in plantas:
        r_opt = [r for r in comp.R_OPC if pulp.value(y[(p, r)]) > 0.5][0]
        u_opt = df_plan.loc[df_plan["p"] == p, "u"].sum()
        # Recálculo exacto (sin aproximación lineal) de horas de déficit y costo biológico
        prob_deficit = binom.sf(r_opt, par["N_REQ"] + r_opt, u_opt)
        horas_deficit = 8760.0 * prob_deficit
        riesgo_exacto = par["L_BIO"] * horas_deficit
        costo_red = par["C_RED"] * r_opt

        filas_plantas.append({
            "planta": p,
            "respaldo_r": r_opt,
            "indisp_bomba_pct": round(u_opt * 100.0, 3),
            "horas_deficit_anio": round(horas_deficit, 2),
            "riesgo_biologico_clp": round(riesgo_exacto, 0),
            "costo_respaldo_clp": round(costo_red, 0),
        })
    df_plantas = pd.DataFrame(filas_plantas)

    costo_op_total = df_plan["c"].sum()
    costo_red_total = df_plantas["costo_respaldo_clp"].sum()
    costo_riesgo_total = df_plantas["riesgo_biologico_clp"].sum()

    resumen = {
        "estado": estado,
        "costo_total_exacto_clp": round(costo_op_total + costo_red_total + costo_riesgo_total, 0),
        "costo_opciones_clp": round(costo_op_total, 0),
        "costo_respaldo_clp": round(costo_red_total, 0),
        "costo_riesgo_clp": round(costo_riesgo_total, 0),
        "horas_deficit_totales": round(df_plantas["horas_deficit_anio"].sum(), 2),
        "fallas_correctivas_anio": round(df_plan["fc"].sum(), 2),
        "recambios_programados_anio": round(df_plan["fp"].sum(), 2),
        "horas_tecnico_usadas": round(df_plan["h"].sum(), 1),
        "horas_tecnico_disponibles": round(h_disponibles, 1),
        "inventario_usado_clp": round(df_plan["v"].sum(), 0),
    }

    return {"estado": estado, "plan": df_plan, "plantas": df_plantas, "resumen": resumen}


# Ejecución rápida de prueba al correr directamente este archivo
if __name__ == "__main__":
    res = resolver_milp()
    print("Estado del solver:", res["estado"])
    if res["resumen"]:
        print("\n--- RESUMEN DEL PLAN ÓPTIMO ---")
        for k, v in res["resumen"].items():
            print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v}")
        print("\n--- RESUMEN POR PLANTA ---")
        print(res["plantas"].to_string(index=False))