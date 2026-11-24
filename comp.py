"""
comp.py
Módulo de parámetros operacionales y etapa estocástica (teoría de confiabilidad).
Calcula tasas de falla por riesgos competitivos (Weibull + Exponencial),
nivel de servicio de bodega (fill rate) y coeficientes técnicos para el MILP.

Empresa: Novofish S.A. - Capstone Analytics
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gamma
from scipy.stats import poisson

# ==============================================================================
# 1. PARÁMETROS GLOBALES DEL SISTEMA (En Pesos Chilenos - CLP)
# ==============================================================================
PAR = {
    "N_REQ": 4,                  # Bombas mínimas operando por planta para evitar déficit
    "N_BASE": 5,                 # Bombas instaladas actualmente por planta (4 op + 1 base)
    "DIAS_REGISTRO": 212,        # Ventana del registro histórico (23-feb a 22-sep 2026)
    "L_BIO": 980_000.0,          # Costo supuesto por hora de déficit de circulación (CLP/h)
    "C_RED": 2_400_000.0,        # Costo Anual Equivalente (CAE) de 1 bomba de respaldo (CLP/año)
    "BUDGET_STOCK": 11_800_000.0,# Presupuesto máximo de capital inmovilizado en bodega (CLP)
    "HOLD": 0.20,                # Tasa anual de mantenimiento de inventario (20% del valor)
    "W_RESP_H": 5.0,             # Demora promedio entre detección e inicio de reparación (h)
    "MO_H": 15_000.0,            # Costo de mano de obra técnica (CLP/hora-hombre)
    "KAPPA": 0.60,               # Duración de un recambio preventivo respecto a uno correctivo
    "TECNICOS_DIA": [4, 4, 4, 4, 5, 5, 5],  # Dotación diaria (Lun-Jue: 4, Vie-Dom: 5)
    "HORAS_TURNO": 8.0,          # Horas efectivas por turno técnico
    "PHI": 0.50,                 # Fracción de horas-técnico dedicables a bombas (6.448 h/año)
    "U_MAX": 0.10,               # Cota física superior de indisponibilidad por bomba (10%)
}

# Exposición total observada en el registro: 6 plantas * 5 bombas * (212 / 365) años
FACTORES_PLANTA = {
    "P1": 1.21,  # Factor suavizado f_p = (n_planta + n_prom) / (2 * n_prom)
    "P2": 0.82,
    "P3": 0.66,
    "P4": 1.21,
    "P5": 0.74,
    "P8": 1.37,
}

EXPOSICION_BOMBA_ANIOS = len(FACTORES_PLANTA) * PAR["N_BASE"] * (PAR["DIAS_REGISTRO"] / 365.0)

# ==============================================================================
# 2. PARÁMETROS POR COMPONENTE CRÍTICO DE BOMBA
# ==============================================================================
# n: fallas observadas | desg: fallas atribuidas a desgaste | beta: forma Weibull (>1 desgaste)
# precio: costo del repuesto (CLP) | dano: daño colateral por falla correctiva (CLP)
# lead: plazo de reposición (días) | rep_h: tiempo medio de reparación (horas)
# mit_eta: multiplicador de vida útil con mitigación | mit_lam0: reducción de fallas externas
# mit_costo: costo anualizado de aplicar la medida de mitigación por bomba (CLP/año)
COMP = {
    "Sello mecánico": {
        "n": 11, "desg": 11, "beta": 2.0, "precio": 343_000, "dano": 490_000,
        "lead": 14, "rep_h": 3.5, "tec": 2,
        "mit_eta": 1.40, "mit_lam0": 1.00, "mit_costo": 120_000,
        "mit_txt": "Sello de carburo de silicio / mejor estándar"
    },
    "Rodamientos": {
        "n": 2, "desg": 1, "beta": 2.5, "precio": 441_000, "dano": 1_500_000,
        "lead": 10, "rep_h": 4.7, "tec": 2,
        "mit_eta": 1.40, "mit_lam0": 0.70, "mit_costo": 100_000,
        "mit_txt": "Alineación láser y lubricación programada"
    },
    "Válvulas": {
        "n": 7, "desg": 2, "beta": 1.5, "precio": 118_000, "dano": 98_000,
        "lead": 7, "rep_h": 2.7, "tec": 1,
        "mit_eta": 1.30, "mit_lam0": 0.70, "mit_costo": 80_000,
        "mit_txt": "Upgrade a válvulas de acero inoxidable"
    },
    "Accionamiento eléctrico": {
        "n": 8, "desg": 0, "beta": 1.0, "precio": 882_000, "dano": 294_000,
        "lead": 21, "rep_h": 4.2, "tec": 2,
        "mit_eta": 1.00, "mit_lam0": 0.60, "mit_costo": 150_000,
        "mit_txt": "Protecciones de voltaje y ventilación de tablero VDF"
    },
    "Sólidos en boluta": {
        "n": 5, "desg": 0, "beta": 1.0, "precio": 0, "dano": 392_000,
        "lead": 0, "rep_h": 3.2, "tec": 2,
        "mit_eta": 1.00, "mit_lam0": 0.50, "mit_costo": 120_000,
        "mit_txt": "Rejillas de retención de sólidos en succión"
    },
    "Hidráulica": {
        "n": 4, "desg": 2, "beta": 1.5, "precio": 147_000, "dano": 147_000,
        "lead": 3, "rep_h": 2.7, "tec": 1,
        "mit_eta": 1.30, "mit_lam0": 0.70, "mit_costo": 60_000,
        "mit_txt": "Acoples reforzados y revisión de torques"
    },
}

# Conjuntos discretos de decisiones candidatas (horizonte T ampliado hasta 5 años)
T_OPC = [None, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]  # None = solo correctivo (T -> infinito)
M_OPC = [0, 1]                                     # 0 = sin mitigación, 1 = con mitigación
S_OPC = [0, 1, 2, 3]                               # Stock objetivo en bodega (política S-1, S)
R_OPC = [0, 1, 2]                                  # Bombas de respaldo por sobre las 4 requeridas


# ==============================================================================
# 3. FUNCIONES DE CONFIABILIDAD E INVENTARIO
# ==============================================================================
def horas_tecnico_disponibles(par=PAR):
    """Calcula el presupuesto anual de horas-técnico disponibles para bombeo."""
    horas_semana = sum(par["TECNICOS_DIA"]) * par["HORAS_TURNO"]
    return horas_semana * 52.0 * par["PHI"]


def calibrar_riesgos_competitivos(lam_total, w_desg, beta):
    """
    Calibra la vida característica Weibull (eta) y la tasa aleatoria (lam0)
    acotando el límite superior de integración para evitar subdesbordamiento numérico.
    """
    if w_desg <= 0.0 or beta <= 1.0:
        return np.inf, lam_total

    lam_desg = lam_total * w_desg
    lam0 = lam_total * (1.0 - w_desg)

    # Caso 100% desgaste: solución analítica exacta con función Gamma
    if lam0 == 0.0:
        eta = 1.0 / (lam_desg * gamma(1.0 + 1.0 / beta))
        return eta, 0.0

    # Caso mixto: acotamos t_max donde exp(-lam0*t) ya es prácticamente cero (< 1e-15)
    def objetivo(eta_val):
        t_max = min(50.0 * eta_val, 35.0 / lam0)
        mttf, _ = quad(lambda t: np.exp(-(t / eta_val) ** beta - lam0 * t), 0.0, t_max)
        return (1.0 / max(mttf, 1e-12)) - lam_total

    eta = brentq(objetivo, 1e-3, 1e3)
    return eta, lam0


def calcular_tasas_recambio(eta, beta, lam0, T):
    """
    Aplica la política de recambio por edad (Barlow & Proschan, 1965).
    Retorna (lambda_c, lambda_p): fallas correctivas y recambios programados por bomba-año.
    """
    if T is None:
        if np.isinf(eta):
            return lam0, 0.0
        t_max = min(50.0 * eta, 35.0 / lam0) if lam0 > 0 else 50.0 * eta
        mttf, _ = quad(lambda t: np.exp(-(t / eta) ** beta - lam0 * t), 0.0, t_max)
        return 1.0 / mttf, 0.0

    def S(t):
        desgaste = 0.0 if np.isinf(eta) else (t / eta) ** beta
        return np.exp(-desgaste - lam0 * t)

    esperanza_ciclo, _ = quad(S, 0.0, T)
    prob_sobrevivir_T = S(T)

    lam_c = (1.0 - prob_sobrevivir_T) / esperanza_ciclo
    lam_p = prob_sobrevivir_T / esperanza_ciclo
    return lam_c, lam_p


def calcular_fill_rate(demanda_anual, lead_dias, stock_s):
    """
    Calcula la probabilidad de tener repuesto disponible al ocurrir una falla
    bajo política uno-a-uno (S-1, S) con llegadas Poisson durante el lead time.
    """
    if lead_dias <= 0 or demanda_anual <= 0:
        return 1.0
    if stock_s <= 0:
        return 0.0
    demanda_en_transito = demanda_anual * (lead_dias / 365.0)
    return float(poisson.cdf(stock_s - 1, demanda_en_transito))


def generar_opciones(par=PAR, comp=COMP, factores=FACTORES_PLANTA):
    """
    Precalcula los coeficientes técnicos (costo, indisponibilidad, horas, stock)
    para todas las combinaciones (planta, componente, T, m, s).
    """
    opciones = []
    exposicion = len(factores) * par["N_BASE"] * (par["DIAS_REGISTRO"] / 365.0)

    for p, f_p in factores.items():
        for k, d in comp.items():
            # 1. Tasa base ajustada por planta y calibración Weibull + Exponencial
            lam_obs_planta = (d["n"] / exposicion) * f_p
            w_desg = d["desg"] / d["n"] if d["n"] > 0 else 0.0
            eta_base, lam0_base = calibrar_riesgos_competitivos(lam_obs_planta, w_desg, d["beta"])

            # Si el componente no requiere repuesto físico (ej. limpieza de boluta), s = 0
            stocks_validos = [0] if d["precio"] == 0 else S_OPC

            for m in M_OPC:
                # 2. Aplicar efecto de la medida de mitigación (si m == 1)
                eta_m = eta_base * d["mit_eta"] if (m == 1 and not np.isinf(eta_base)) else eta_base
                lam0_m = lam0_base * d["mit_lam0"] if m == 1 else lam0_base
                costo_mit = d["mit_costo"] if m == 1 else 0.0

                for T in T_OPC:
                    # 3. Tasas de intervención por bomba-año
                    lam_c, lam_p = calcular_tasas_recambio(eta_m, d["beta"], lam0_m, T)
                    demanda_total = par["N_BASE"] * (lam_c + lam_p)
                    mo_correctiva = d["rep_h"] * d["tec"] * par["MO_H"]

                    for s in stocks_validos:
                        # 4. Nivel de servicio (fill rate) y riesgo de quiebre de stock
                        fill = calcular_fill_rate(demanda_total, d["lead"], s)
                        espera_stock_h = (1.0 - fill) * d["lead"] * 24.0

                        # 5. Indisponibilidad anual por bomba (fracción de las 8.760 horas)
                        horas_fuera_corr = lam_c * (par["W_RESP_H"] + d["rep_h"] + espera_stock_h)
                        horas_fuera_prev = lam_p * (par["KAPPA"] * d["rep_h"] + 2.0)
                        u_o = (horas_fuera_corr + horas_fuera_prev) / 8760.0

                        # 6. Costo anual esperado de la opción (CLP/año)
                        costo_interv = par["N_BASE"] * (
                            lam_c * (d["precio"] + d["dano"] + mo_correctiva)
                            + lam_p * (d["precio"] + par["KAPPA"] * mo_correctiva)
                            + costo_mit
                        )
                        costo_inv = par["HOLD"] * d["precio"] * s
                        c_o = costo_interv + costo_inv

                        # 7. Horas-técnico anuales y capital inmovilizado en bodega
                        h_o = par["N_BASE"] * (
                            lam_c * d["rep_h"] * d["tec"]
                            + lam_p * par["KAPPA"] * d["rep_h"] * d["tec"]
                        )
                        v_o = d["precio"] * s

                        opciones.append({
                            "p": p, "k": k, "T": T, "m": m, "s": s,
                            "c": c_o, "u": u_o, "h": h_o, "v": v_o,
                            "fill": fill, "prob_quiebre": 1.0 - fill,
                            "fc": par["N_BASE"] * lam_c,
                            "fp": par["N_BASE"] * lam_p,
                        })
    return opciones