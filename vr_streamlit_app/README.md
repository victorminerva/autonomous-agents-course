# VR Agent — Streamlit

App simples para automatizar o processo mensal de compra de VR (Vale Refeição),.

## Rodar local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Abra: http://localhost:8501

## Entradas (na barra lateral)
- ATIVOS.csv (obrigatório)
- APRENDIZ.csv, ESTÁGIO.csv (opcionais) — excluem do cálculo
- AFASTAMENTOS.csv (opcional) — coluna `na compra?` (sim/não)
- FÉRIAS.csv (opcional) — coluna `DIAS DE FÉRIAS`
- ADMISSÕES.csv (opcional) — coluna de data: `DATA ADMISSÃO`/`ADMISSÃO`/`DATA`…
- DESLIGADOS.csv (opcional) — `DATA DEMISSÃO`/`DEMISSAO`…
- Base dias uteis.csv — colunas `SINDICATO, DIAS UTEIS`
- valor_por_sindicato.csv (opcional) — `SINDICATO, VALOR`
- EXTERIOR.csv (opcional) — `MATRICULA, Valor` (sobrepõe valor por sindicato)

## Saídas
- Planilha de **Cálculo** (CSV)
- Planilha de **Auditoria** (CSV)
