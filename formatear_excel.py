"""
formatear_excel.py
Toma los 4 archivos CSV generados por generar_tablas.py y crea un único
libro de Excel profesional ('Resultados_Optimizacion_Novofish.xlsx') con
4 pestañas ordenadas, nombres claros en español, formato moneda y bordes.
"""

import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Diccionario que define el nombre de cada pestaña, su CSV origen y los títulos humanos
HOJAS = {
    "1. Comparación Políticas": (
        "tabla_politicas.csv",
        {
            "Politica": "Política Evaluada",
            "Costo_Total_MM": "Costo Total Anual (MM$)",
            "Mant_e_Inv_MM": "Mantención e Inventario (MM$)",
            "Respaldo_MM": "Bombas de Respaldo (MM$)",
            "Riesgo_Bio_MM": "Riesgo Biológico (MM$)",
            "Horas_Deficit": "Horas de Déficit / Año",
            "Fallas_Corr": "Fallas Correctivas / Año",
            "Recambios": "Recambios Programados / Año",
        },
    ),
    "2. Detalle 36 Componentes": (
        "tabla_36_detalle.csv",
        {
            "Planta": "Unidad Productiva",
            "Componente": "Componente Crítico",
            "Recambio_T": "Intervalo Recambio Preventivo",
            "Mitigacion": "¿Aplica Mitigación?",
            "Stock_s": "Repuestos en Bodega (Unid.)",
            "Fallas_Anio": "Fallas Esperadas / Año",
            "Recambios_Anio": "Recambios Programados / Año",
            "FillRate_%": "Nivel de Servicio Bodega (%)",
            "Costo_Anual_CLP": "Costo Anual Opción (CLP)",
        },
    ),
    "3. Validación Modelo": (
        "tabla_validacion.csv",
        {
            "Instancia": "Escenario de Prueba",
            "Proposito": "Objetivo de Validación",
            "Estado": "Estado del Solver",
            "Costo_MM": "Costo Óptimo (MM$)",
            "Observacion": "Resultado y Verificación",
            "Tiempo_s": "Tiempo Resolución (seg)",
        },
    ),
    "4. Análisis Sensibilidad": (
        "tabla_sensibilidad.csv",
        {
            "Escenario": "Parámetro Evaluado",
            "Costo_MM": "Costo Total (MM$)",
            "Var_%": "Variación vs. Base (%)",
            "Riesgo_MM": "Costo Riesgo Biológico (MM$)",
            "Horas_Def": "Horas de Déficit / Año",
            "Recambios_N": "Componentes con Recambio (N°)",
            "Mitig_N": "Componentes Mitigados (N°)",
            "Respaldo_Med": "Promedio Bombas Respaldo",
        },
    ),
}

ARCHIVO_SALIDA = "Resultados_Optimizacion_Novofish.xlsx"

# Estilos visuales de tabla ejecutiva
estilo_header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
estilo_header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
estilo_zebra_fill = PatternFill(start_color="F2F6FA", end_color="F2F6FA", fill_type="solid")
estilo_celda_font = Font(name="Calibri", size=10, color="000000")
borde_fino = Side(border_style="thin", color="D9D9D9")
borde_celda = Border(left=borde_fino, right=borde_fino, top=borde_fino, bottom=borde_fino)

with pd.ExcelWriter(ARCHIVO_SALIDA, engine="openpyxl") as writer:
    for nombre_hoja, (archivo_csv, renombres) in HOJAS.items():
        # Leer CSV, traducir columnas a español claro y escribir en su pestaña
        df = pd.read_csv(archivo_csv).rename(columns=renombres)
        df.to_excel(writer, sheet_name=nombre_hoja, index=False)

        ws = writer.sheets[nombre_hoja]
        ws.views.sheetView[0].showGridLines = True

        # Dar formato al encabezado (fila 1)
        ws.row_dimensions[1].height = 26
        for col_idx in range(1, df.shape[1] + 1):
            celda = ws.cell(row=1, column=col_idx)
            celda.fill = estilo_header_fill
            celda.font = estilo_header_font
            celda.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            celda.border = borde_celda

        # Dar formato a las filas de datos y formatos numéricos
        for row_idx in range(2, df.shape[0] + 2):
            ws.row_dimensions[row_idx].height = 20
            for col_idx in range(1, df.shape[1] + 1):
                celda = ws.cell(row=row_idx, column=col_idx)
                col_nombre = df.columns[col_idx - 1]
                celda.font = estilo_celda_font
                celda.border = borde_celda

                # Color intercalado suave en filas pares
                if row_idx % 2 == 0:
                    celda.fill = estilo_zebra_fill

                # Alineación y formato numérico según el tipo de columna
                if col_idx == 1 or "Objetivo" in col_nombre or "Verificación" in col_nombre or "Componente" in col_nombre:
                    celda.alignment = Alignment(horizontal="left", vertical="center")
                else:
                    celda.alignment = Alignment(horizontal="center", vertical="center")

                if "(CLP)" in col_nombre:
                    celda.number_format = "$#,##0"
                elif "(MM$)" in col_nombre:
                    celda.number_format = "$#,##0.00"
                elif "(%)" in col_nombre:
                    celda.number_format = "0.0"
                elif isinstance(celda.value, float):
                    celda.number_format = "0.00"

        # Ajustar automáticamente el ancho de cada columna para que no se corte ningún texto
        for col_idx, col_nombre in enumerate(df.columns, start=1):
            largo_max = max(
                len(str(col_nombre)),
                df.iloc[:, col_idx - 1].astype(str).map(len).max(),
            )
            ws.column_dimensions[get_column_letter(col_idx)].width = min(largo_max + 5, 42)

print(f"¡Listo, mano! Se creó '{ARCHIVO_SALIDA}' con las 4 tablas formateadas.")