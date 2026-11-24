"""
================================================================================
DASHBOARD INTERACTIVO DE OPTIMIZACIÓN - NOVOFISH S.A. (Capstone Analytics)
================================================================================
Interfaz web local construida en Streamlit + Plotly para ejecutar, sensibilizar
y visualizar el modelo MILP sin Big-M (opt2.py + comp.py).

Ejecución en la terminal de VS Code:
    streamlit run dashboard_novofish.py
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import binom

# Importamos los módulos del proyecto (deben estar en la misma carpeta)
try:
    import comp
    import opt2
    MODULOS_CARGADOS = True
except ImportError as e:
    MODULOS_CARGADOS = False
    ERROR_IMPORT = str(e)

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS VISUALES
# ==============================================================================
st.set_page_config(
    page_title="Novofish Analytics | Optimización de Bombas RAS",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1B365D;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        color: #1B365D;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-header">🐟 Novofish S.A. — Dashboard de Optimización de Mantenimiento e Inventario</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">Modelo Híbrido Estocástico-Determinístico (MILP sin Big-M | Evaluación Binomial Exacta | CLP)</div>',
    unsafe_allow_html=True
)

if not MODULOS_CARGADOS:
    st.error(
        f"⚠️ No se pudieron importar `comp.py` u `opt2.py`: {ERROR_IMPORT}. "
        "Asegúrate de ejecutar este dashboard dentro de la misma carpeta donde están tus scripts."
    )
    st.stop()

# ==============================================================================
# BARRA LATERAL: FORMULARIO DE PARÁMETROS Y SENSIBILIDAD EN VIVO
# ==============================================================================
st.sidebar.header("⚙️ Parámetros del Escenario")
st.sidebar.caption("Ajusta los parámetros y presiona 'Ejecutar Optimización' para recalcular el plan óptimo.")

par_sim = dict(comp.PAR)

# Usamos st.sidebar.form para que el modelo recalcule solo al presionar el botón
with st.sidebar.form("form_escenario"):
    l_bio_mm = st.slider(
        "Costo hora déficit biológico (MM$ CLP/h)",
        min_value=0.20, max_value=4.00,
        value=float(par_sim["L_BIO"] / 1e6), step=0.05, format="$%.2f MM"
    )

    c_red_mm = st.slider(
        "Costo anual bomba respaldo - CAE (MM$ CLP/año)",
        min_value=1.00, max_value=6.00,
        value=float(par_sim["C_RED"] / 1e6), step=0.10, format="$%.2f MM"
    )

    budget_mm = st.slider(
        "Presupuesto máximo inventario (MM$ CLP)",
        min_value=3.00, max_value=30.00,
        value=float(par_sim["BUDGET_STOCK"] / 1e6), step=0.50, format="$%.1f MM"
    )

    w_resp = st.slider(
        "Demora media respuesta técnica (horas)",
        min_value=1.0, max_value=24.0,
        value=float(par_sim.get("W_RESP_H", 5.0)), step=0.5
    )

    mult_tasa = st.slider(
        "Multiplicador tasas de falla (estrés)",
        min_value=0.50, max_value=2.00,
        value=1.00, step=0.10, format="%.2fx"
    )

    time_limit = st.number_input(
        "Tiempo límite Solver CBC (seg)",
        min_value=10, max_value=300, value=120
    )

    st.form_submit_button("🚀 Ejecutar Optimización", type="primary", use_container_width=True)

# Aplicamos los valores elegidos en el formulario
par_sim["L_BIO"] = l_bio_mm * 1e6
par_sim["C_RED"] = c_red_mm * 1e6
par_sim["BUDGET_STOCK"] = budget_mm * 1e6
par_sim["W_RESP_H"] = w_resp
# Ajuste dinámico de U_MAX para evitar falsos infactibles en escenarios de estrés alto
par_sim["U_MAX"] = max(0.10, 0.20 * mult_tasa + 0.005 * w_resp)

factores_sim = {p: f * mult_tasa for p, f in comp.FACTORES_PLANTA.items()}

# ==============================================================================
# EJECUCIÓN DEL MODELO (CON CACHE PARA VELOCIDAD)
# ==============================================================================
@st.cache_data(show_spinner="Resolviendo MILP sin Big-M con CBC...")
def correr_optimizacion_cache(p_dict, f_dict, t_lim):
    return opt2.resolver_milp(
        par=p_dict,
        comp_dict=comp.COMP,
        factores=f_dict,
        time_limit=t_lim,
        msg_solver=False
    )

res = correr_optimizacion_cache(par_sim, factores_sim, time_limit)

if res["estado"] not in ("Optimal", "Not Solved") or res["resumen"] is None:
    st.error(f"❌ El modelo no encontró solución factible. Estado del solver: **{res['estado']}**")
    st.stop()

resumen = res["resumen"]
df_plan = res["plan"].copy()
df_plantas = res["plantas"].copy()

# ==============================================================================
# FILA 1: TARJETAS KPI PRINCIPALES
# ==============================================================================
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric(
    "Costo Total Anual Óptimo",
    f"${resumen['costo_total_exacto_clp']/1e6:,.2f} MM",
    help="Suma exacta de mantenimiento, inventario, respaldo (CAE) y riesgo biológico binomial."
)
kpi2.metric(
    "Horas Déficit Caudal / Año",
    f"{resumen['horas_deficit_totales']:.2f} h",
    help="Horas esperadas al año con menos de 4 bombas operativas en el sistema."
)
kpi3.metric(
    "Riesgo Biológico Anual",
    f"${resumen['costo_riesgo_clp']/1e6:,.2f} MM",
    help="Valorización exacta (Binomial) de las horas de déficit."
)
kpi4.metric(
    "Capital en Bodega",
    f"${resumen['inventario_usado_clp']/1e6:,.2f} MM",
    delta=f"Tope: ${par_sim['BUDGET_STOCK']/1e6:,.1f} MM",
    delta_color="off"
)
kpi5.metric(
    "Uso Dotación Técnica",
    f"{resumen['horas_tecnico_usadas']:,.0f} HH",
    delta=f"Disp: {resumen['horas_tecnico_disponibles']:,.0f} HH",
    delta_color="off"
)

st.divider()

# ==============================================================================
# PESTAÑAS DE VISUALIZACIÓN
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Resumen Ejecutivo y Costos",
    "🏭 Redundancia y Riesgo por Planta",
    "🔧 Plan Óptimo por Componente",
    "📈 Curva Convexa y Linealización"
])

# ------------------------------------------------------------------------------
# TAB 1: DESGLOSE DE COSTOS Y COMPARACIÓN
# ------------------------------------------------------------------------------
with tab1:
    col_a, col_b = st.columns([1.1, 1])

    with col_a:
        st.subheader("Composición del Costo Anual Esperado (CLP)")
        df_costos = pd.DataFrame({
            "Categoría": [
                "Mantenimiento e Inventario",
                "Bombas de Respaldo (CAE)",
                "Riesgo Biológico (Binomial)"
            ],
            "Monto_CLP": [
                resumen["costo_opciones_clp"],
                resumen["costo_respaldo_clp"],
                resumen["costo_riesgo_clp"]
            ]
        })
        df_costos["Monto_MM"] = df_costos["Monto_CLP"] / 1e6
        fig_pie = px.pie(
            df_costos, values="Monto_MM", names="Categoría", hole=0.45,
            color_discrete_sequence=["#1B365D", "#3182CE", "#E53E3E"]
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=360)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.subheader("Indicadores Operacionales del Plan")
        df_ind = pd.DataFrame({
            "Indicador": [
                "Estado del Solver CBC",
                "Costo Total Anual Exacto",
                "Fallas Correctivas Esperadas",
                "Recambios Preventivos Programados",
                "Utilización Presupuesto Stock",
                "Utilización Horas-Técnico"
            ],
            "Valor": [
                resumen["estado"],
                f"${resumen['costo_total_exacto_clp']:,.0f} CLP",
                f"{resumen['fallas_correctivas_anio']:.1f} eventos/año",
                f"{resumen['recambios_programados_anio']:.1f} intervenciones/año",
                f"{(resumen['inventario_usado_clp'] / par_sim['BUDGET_STOCK'])*100:.1f}%",
                f"{(resumen['horas_tecnico_usadas'] / resumen['horas_tecnico_disponibles'])*100:.1f}%"
            ]
        })
        st.dataframe(df_ind, use_container_width=True, hide_index=True, height=340)

# ------------------------------------------------------------------------------
# TAB 2: RESUMEN POR PLANTA PRODUCTIVA
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Asignación de Bombas de Respaldo y Riesgo Biológico por Planta")
    c1, c2 = st.columns(2)

    # Categoría discreta para que la leyenda muestre "0", "1" o "2" bombas limpiamente
    df_plantas["Respaldo_Cat"] = df_plantas["respaldo_r"].apply(
        lambda x: f"{x} bomba" if x == 1 else f"{x} bombas"
    )

    with c1:
        fig_horas = px.bar(
            df_plantas, x="planta", y="horas_deficit_anio",
            color="Respaldo_Cat",
            labels={
                "planta": "Unidad Productiva",
                "horas_deficit_anio": "Horas de Déficit / Año",
                "Respaldo_Cat": "Respaldo Asignado"
            },
            title="Horas Esperadas de Déficit de Caudal por Planta",
            text_auto=".2f",
            color_discrete_map={
                "0 bombas": "#E53E3E",
                "1 bomba": "#3182CE",
                "2 bombas": "#1B365D"
            }
        )
        fig_horas.update_layout(height=360)
        st.plotly_chart(fig_horas, use_container_width=True)

    with c2:
        df_plantas_melt = df_plantas.melt(
            id_vars=["planta"],
            value_vars=["riesgo_biologico_clp", "costo_respaldo_clp"],
            var_name="Tipo_Costo", value_name="CLP"
        )
        df_plantas_melt["MM_CLP"] = df_plantas_melt["CLP"] / 1e6
        df_plantas_melt["Tipo_Costo"] = df_plantas_melt["Tipo_Costo"].replace({
            "riesgo_biologico_clp": "Riesgo Biológico",
            "costo_respaldo_clp": "Bombas Respaldo (CAE)"
        })
        fig_cost_p = px.bar(
            df_plantas_melt, x="planta", y="MM_CLP", color="Tipo_Costo",
            barmode="group",
            labels={
                "planta": "Unidad Productiva",
                "MM_CLP": "Costo Anual (MM$ CLP)",
                "Tipo_Costo": "Concepto"
            },
            title="Trade-off: Inversión en Respaldo vs. Riesgo Biológico",
            color_discrete_map={"Riesgo Biológico": "#E53E3E", "Bombas Respaldo (CAE)": "#2B6CB0"}
        )
        fig_cost_p.update_layout(height=360)
        st.plotly_chart(fig_cost_p, use_container_width=True)

    # Tabla con nombres claros en español para el usuario
    df_plantas_vista = df_plantas[[
        "planta", "respaldo_r", "indisp_bomba_pct",
        "horas_deficit_anio", "riesgo_biologico_clp", "costo_respaldo_clp"
    ]].copy()
    df_plantas_vista["riesgo_biologico_clp"] = df_plantas_vista["riesgo_biologico_clp"].apply(lambda x: f"${x:,.0f}")
    df_plantas_vista["costo_respaldo_clp"] = df_plantas_vista["costo_respaldo_clp"].apply(lambda x: f"${x:,.0f}")
    df_plantas_vista.columns = [
        "Unidad Productiva", "Bombas Respaldo (r)", "Indisponibilidad Bomba (%)",
        "Horas Déficit / Año", "Riesgo Biológico (CLP)", "Costo Respaldo (CLP)"
    ]
    st.dataframe(df_plantas_vista, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# TAB 3: PLAN ÓPTIMO DETALLADO POR COMPONENTE
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Políticas Óptimas Seleccionadas (Recambio T, Mitigación m, Inventario s)")

    filtro_planta = st.multiselect(
        "Filtrar por Unidad Productiva:",
        options=sorted(df_plan["p"].unique()),
        default=sorted(df_plan["p"].unique())
    )

    df_show = df_plan[df_plan["p"].isin(filtro_planta)].copy()
    df_show["Intervalo Recambio"] = df_show["T"].apply(lambda v: f"{v:.1f} años" if pd.notnull(v) else "Solo correctivo")
    df_show["¿Mitigación?"] = df_show["m"].apply(lambda v: "Sí" if v == 1 else "No")
    df_show["Nivel Servicio (%)"] = (df_show["fill"] * 100).round(1)
    df_show["Fallas / Año"] = df_show["fc"].round(2)
    df_show["Recambios / Año"] = df_show["fp"].round(2)
    df_show["Costo Anual (CLP)"] = df_show["c"].apply(lambda x: f"${x:,.0f}")

    col_left, col_right = st.columns([1, 1.5])
    with col_left:
        st.markdown("**Inventario Consolidado en Bodega por Componente**")
        df_stock_comp = df_plan.groupby("k", as_index=False).agg(
            Repuestos_Totales=("s", "sum"),
            Plantas_Mitigadas=("m", "sum"),
            Costo_Anual_MM=("c", lambda x: round(x.sum() / 1e6, 2))
        )
        df_stock_comp.columns = [
            "Componente Crítico", "Repuestos en Bodega", "Plantas Mitigadas", "Costo Anual (MM$)"
        ]
        st.dataframe(df_stock_comp, use_container_width=True, hide_index=True)

    with col_right:
        st.markdown("**Detalle Operativo por Planta y Subcomponente**")
        df_detalle_vista = df_show[[
            "p", "k", "Intervalo Recambio", "¿Mitigación?", "s",
            "Nivel Servicio (%)", "Fallas / Año", "Recambios / Año", "Costo Anual (CLP)"
        ]].rename(columns={
            "p": "Planta",
            "k": "Componente",
            "s": "Stock (Unid.)"
        })
        st.dataframe(df_detalle_vista, use_container_width=True, hide_index=True, height=380)

# ------------------------------------------------------------------------------
# TAB 4: CURVA CONVEXA DE RIESGO Y RECTAS TANGENTES (SOPORTE ACADÉMICO)
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Validación Matemática: Linealización por Envolvente Convexa (11 Tangentes)")
    # Usamos raw string (r"...") para que Python no rompa el código LaTeX (\text, \cdot)
    st.markdown(
        r"Visualización de la función exacta de riesgo biológico "
        r"$g_r(U) = L_{\text{bio}} \cdot 8760 \cdot P[\text{Binomial}(4+r, U) > r]$ "
        r"y las 11 rectas tangentes utilizadas en el MILP sin *Big-M*."
    )

    r_sel = st.radio("Selecciona nivel de respaldo (r):", options=comp.R_OPC, horizontal=True, index=1)
    u_grid = np.linspace(0.0001, 0.04, 200)
    g_exact = [par_sim["L_BIO"] * 8760.0 * binom.sf(r_sel, par_sim["N_REQ"] + r_sel, u) / 1e6 for u in u_grid]

    tangentes_r = opt2.generar_tangentes_riesgo(par_sim, comp.R_OPC)[r_sel]
    g_approx = [max(a * u + b for (a, b) in tangentes_r) for u in u_grid]

    fig_tan = go.Figure()
    fig_tan.add_trace(go.Scatter(
        x=u_grid * 100, y=g_exact, mode="lines",
        name="Curva Exacta Binomial", line=dict(color="#1B365D", width=3)
    ))
    fig_tan.add_trace(go.Scatter(
        x=u_grid * 100, y=g_approx, mode="lines",
        name="Aproximación MILP (11 Tangentes)", line=dict(color="#E53E3E", dash="dash", width=2)
    ))
    fig_tan.update_layout(
        xaxis_title="Indisponibilidad por bomba U_p (%)",
        yaxis_title="Costo Anual de Riesgo Biológico (MM$ CLP)",
        height=400
    )
    st.plotly_chart(fig_tan, use_container_width=True)