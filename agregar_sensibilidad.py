"""
agregar_sensibilidad.py
Abre 'Resultados_Optimizacion_Novofish.xlsx' y le agrega la 4ta pestaña
('4. Análisis Sensibilidad') con el mismo formato ejecutivo.
"""

import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ARCHIVO_EXCEL = "Resultados_Optimizacion_Novofish.xlsx"
NOMBRE_HOJA = "4. Análisis Sensibilidad"

# 1. Leer el CSV de sensibilidad y traducir encabezados
df = pd.read_csv("tabla_sensibilidad.csv").rename(columns={
    "Escenario": "Parámetro Evaluado",
    "Costo_MM": "Costo Total (MM$)",
    "Var_%": "Variación vs. Base (%)",
    "Riesgo_MM": "Costo Riesgo Biológico (MM$)",
    "Horas_Def": "Horas de Déficit / Año",
    "Recambios_N": "Componentes con Recambio (N°)",
    "Mitig_N": "Componentes Mitigados (N°)",
    "Respaldo_Med": "Promedio Bombas Respaldo",
})

# 2. Agregar (o reemplazar) la 4ta pestaña en el Excel existente
with pd.ExcelWriter(ARCHIVO_EXCEL, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df.to_excel(writer, sheet_name=NOMBRE_HOJA, index=False)
    ws = writer.sheets[NOMBRE_HOJA]

    fill_head = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    font_head = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    fill_zebra = PatternFill(start_color="F2F6FA", end_color="F2F6FA", fill_type="solid")
    font_celda = Font(name="Calibri", size=10, color="000000")
    borde = Border(
        left=Side(border_style="thin", color="D9D9D9"),
        right=Side(border_style="thin", color="D9D9D9"),
        top=Side(border_style="thin", color="D9D9D9"),
        bottom=Side(border_style="thin", color="D9D9D9"),
    )

    ws.row_dimensions[1].height = 26
    for col_idx in range(1, df.shape[1] + 1):
        c = ws.cell(row=1, column=col_idx)
        c.fill, c.font, c.border = fill_head, font_head, borde
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for row_idx in range(2, df.shape[0] + 2):
        ws.row_dimensions[row_idx].height = 20
        for col_idx in range(1, df.shape[1] + 1):
            c = ws.cell(row=row_idx, column=col_idx)
            col_nombre = df.columns[col_idx - 1]
            c.font, c.border = font_celda, borde
            if row_idx % 2 == 0:
                c.fill = fill_zebra
            c.alignment = Alignment(horizontal="left" if col_idx == 1 else "center", vertical="center")
            if "(MM$)" in col_nombre:
                c.number_format = "$#,##0.00"
            elif "(%)" in col_nombre:
                c.number_format = "0.0"
            elif isinstance(c.value, float):
                c.number_format = "0.00"

    for col_idx, col_nombre in enumerate(df.columns, start=1):
        ancho = max(len(str(col_nombre)), df.iloc[:, col_idx - 1].astype(str).map(len).max())
        ws.column_dimensions[get_column_letter(col_idx)].width = min(ancho + 5, 42)

# 3. Verificar en consola que quedaron las 4 pestañas guardadas
wb = openpyxl.load_workbook(ARCHIVO_EXCEL)
print("Pestañas actuales en tu Excel:", wb.sheetnames)