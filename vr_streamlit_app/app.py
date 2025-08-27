import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from dateutil import parser as dateparser
import io
import requests
import os
from dotenv import load_dotenv

st.set_page_config(page_title="VR Agent — Streamlit", layout="wide")

st.title("🧮 VR Agent — Cálculo de Vale Refeição (Streamlit)")

st.markdown("""
Carregue suas planilhas, escolha a competência e gere o **cálculo** e a **auditoria**.
- Exclui **Aprendizes** e **Estagiários**
- Aplica **pró-rata** (admissão/desligamento) — dias úteis (seg–sex)
- Subtrai **férias**
- Respeita **afastamentos** (se `na compra?` = não → zera)
- Valor por **Sindicato** (opcional) e **EXTERIOR** por matrícula (opcional)
""")

# ---------- Helpers ----------
def read_csv_robust(file):
    # file can be UploadedFile or bytes/str path
    encodings = ["utf-8-sig", "latin-1", "cp1252", "utf-8"]
    if hasattr(file, "read"):
        content = file.read()
        for enc in encodings:
            try:
                return pd.read_csv(io.BytesIO(content), sep=None, engine="python", encoding=enc, dtype=str)
            except Exception:
                try:
                    return pd.read_csv(io.BytesIO(content), sep=";", encoding=enc, dtype=str)
                except Exception:
                    continue
    else:
        for enc in encodings:
            try:
                return pd.read_csv(file, sep=None, engine="python", encoding=enc, dtype=str)
            except Exception:
                try:
                    return pd.read_csv(file, sep=";", encoding=enc, dtype=str)
                except Exception:
                    continue
    raise ValueError("Falha ao ler CSV")

def clean_id(df, col="MATRICULA"):
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
                 .str.replace(r"[^0-9]", "", regex=True)
                 .str.lstrip("0")
        )
    return df

def parse_br_date(x):
    x = str(x).strip()
    if x == "" or x.lower() in ("nan","none"):
        return None
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(x, fmt).date()
        except Exception:
            pass
    try:
        return dateparser.parse(x, dayfirst=True).date()
    except Exception:
        return None

def month_bounds(competencia):
    y, m = map(int, competencia.split("-"))
    first = date(y, m, 1)
    if m == 12:
        last = date(y+1, 1, 1) - timedelta(days=1)
    else:
        last = date(y, m+1, 1) - timedelta(days=1)
    return first, last

def working_days_between(d1, d2, first_day, last_day):
    if d1 is None or d2 is None:
        return None
    d1 = max(d1, first_day)
    d2 = min(d2, last_day)
    if d2 < d1:
        return 0
    # Feriados
    feriados = set()
    try:
        # file_feriados é global, mas pode não existir
        global file_feriados
        if 'file_feriados' in globals() and file_feriados:
            df_fer = read_csv_robust(file_feriados)
            col = [c for c in df_fer.columns if 'data' in c.lower()][0]
            feriados = set(df_fer[col].apply(parse_br_date).dropna())
    except Exception:
        pass
    rng = [d for d in pd.date_range(d1, d2, freq="D").date if d.weekday() < 5 and d not in feriados]
    return len(rng)

# ---------- Sidebar: inputs ----------

st.sidebar.header("⚙️ Parâmetros")
competencia = st.sidebar.text_input("Competência (YYYY-MM)", value=date.today().strftime("%Y-%m"))
cap_dias_mes = st.sidebar.number_input("Cap de dias por mês", min_value=1, max_value=31, value=22, step=1)

st.sidebar.header("🤖 LLM Agent (OpenRouter)")
# Carrega chave do .env se existir
load_dotenv()
openrouter_api_key = os.getenv("OPENAI_API_KEY", "")
llm_enabled = True  # Sempre usa LLM para consolidar

st.sidebar.header("📥 Planilhas")

# Bases principais
file_ativos = st.sidebar.file_uploader("ATIVOS.csv", type=["csv"], accept_multiple_files=False)
file_ferias = st.sidebar.file_uploader("FÉRIAS.csv", type=["csv"], accept_multiple_files=False)
file_desligados = st.sidebar.file_uploader("DESLIGADOS.csv", type=["csv"], accept_multiple_files=False)
file_admissoes = st.sidebar.file_uploader("ADMISSÕES.csv (ex.: ADMISSÃO ABRIL.csv)", type=["csv"], accept_multiple_files=False)
file_sindicato_valor = st.sidebar.file_uploader("valor_por_sindicato.csv (SINDICATO, VALOR)", type=["csv"], accept_multiple_files=False)
file_dias_uteis = st.sidebar.file_uploader("Base dias uteis.csv (SINDICATO, DIAS UTEIS)", type=["csv"], accept_multiple_files=False)
# Bases opcionais (mantidas para compatibilidade com o restante do código)
file_aprendiz = st.sidebar.file_uploader("APRENDIZ.csv (opcional)", type=["csv"], accept_multiple_files=False)
file_estagio = st.sidebar.file_uploader("ESTÁGIO.csv (opcional)", type=["csv"], accept_multiple_files=False)
file_afastamentos = st.sidebar.file_uploader("AFASTAMENTOS.csv (opcional)", type=["csv"], accept_multiple_files=False)
file_valor_sind = file_sindicato_valor  # compatibilidade
file_exterior = st.sidebar.file_uploader("EXTERIOR.csv (MATRICULA, Valor) (opcional)", type=["csv"], accept_multiple_files=False)

# ...existing code...

run = st.sidebar.button("▶️ Calcular")

# ---------- Main logic ----------

def call_openrouter_agent(prompt, api_key, model="openai/gpt-3.5-turbo"):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Você é um agente Python especialista em consolidação de bases de dados."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1200,
        "temperature": 0.2
    }
    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    else:
        return f"Erro na chamada OpenRouter: {resp.text}"

if run:
    errors = []
    if not file_ativos:
        st.error("Envie o arquivo ATIVOS.csv.")
        st.stop()


    # Leitura das 5 bases principais
    try:
        ativos = read_csv_robust(file_ativos); ativos.columns = [c.strip() for c in ativos.columns]
        ativos = clean_id(ativos, "MATRICULA")
    except Exception as e:
        st.error(f"Falha ao ler ATIVOS.csv: {e}")
        st.stop()

    try:
        ferias = read_csv_robust(file_ferias) if file_ferias else pd.DataFrame()
        if not ferias.empty: ferias.columns = [c.strip() for c in ferias.columns]; ferias = clean_id(ferias, "MATRICULA")
    except Exception as e:
        errors.append(f"Falha ao ler FÉRIAS.csv: {e}")
        ferias = pd.DataFrame()

    try:
        desligados = read_csv_robust(file_desligados) if file_desligados else pd.DataFrame()
        if not desligados.empty: desligados.columns = [c.strip() for c in desligados.columns]; desligados = clean_id(desligados, "MATRICULA")
    except Exception as e:
        errors.append(f"Falha ao ler DESLIGADOS.csv: {e}")
        desligados = pd.DataFrame()

    try:
        admitidos = read_csv_robust(file_admissoes) if file_admissoes else pd.DataFrame()
        if not admitidos.empty: admitidos.columns = [c.strip() for c in admitidos.columns]; admitidos = clean_id(admitidos, "MATRICULA")
    except Exception as e:
        errors.append(f"Falha ao ler ADMISSÕES.csv: {e}")
        admitidos = pd.DataFrame()

    try:
        sindicato_valor = read_csv_robust(file_sindicato_valor) if file_sindicato_valor else pd.DataFrame()
        if not sindicato_valor.empty: sindicato_valor.columns = [c.strip() for c in sindicato_valor.columns]
    except Exception as e:
        errors.append(f"Falha ao ler valor_por_sindicato.csv: {e}")
        sindicato_valor = pd.DataFrame()

    try:
        dias_uteis = read_csv_robust(file_dias_uteis) if file_dias_uteis else pd.DataFrame()
        if not dias_uteis.empty: dias_uteis.columns = [c.strip() for c in dias_uteis.columns]
    except Exception as e:
        errors.append(f"Falha ao ler Base dias uteis.csv: {e}")
        dias_uteis = pd.DataFrame()



    # Consolidação direta das bases
    st.info("Consolidando bases localmente...")
    base_final = ativos.copy()
    # Merge com as demais bases
    if not ferias.empty:
        base_final = base_final.merge(ferias, on="MATRICULA", how="left")
    if not desligados.empty:
        base_final = base_final.merge(desligados, on="MATRICULA", how="left")
    if not admitidos.empty:
        base_final = base_final.merge(admitidos, on="MATRICULA", how="left")
    if not sindicato_valor.empty:
        sindicato_col = "SINDICATO" if "SINDICATO" in sindicato_valor.columns else None
        if sindicato_col and "Sindicato" in base_final.columns:
            base_final = base_final.merge(sindicato_valor, left_on="Sindicato", right_on=sindicato_col, how="left")
    if not dias_uteis.empty:
        dias_col = "SINDICATO" if "SINDICATO" in dias_uteis.columns else None
        if dias_col and "Sindicato" in base_final.columns:
            base_final = base_final.merge(dias_uteis, left_on="Sindicato", right_on=dias_col, how="left")

    # Preencher valores ausentes
    for col in ["DIAS DE FÉRIAS", "DIAS UTEIS"]:
        if col in base_final.columns:
            base_final[col] = base_final[col].fillna(0)

    # Exclusão por cargo, país, afastamento, estagiário/aprendiz
    cargos_excluir = ["diretor", "estagiário", "estagiario", "aprendiz"]
    if "TITULO DO CARGO" in base_final.columns:
        base_final = base_final[~base_final["TITULO DO CARGO"].str.lower().str.contains('|'.join(cargos_excluir), na=False)]
    if "DESC. SITUACAO" in base_final.columns:
        base_final = base_final[~base_final["DESC. SITUACAO"].str.lower().str.contains("afast", na=False)]
    if "PAIS" in base_final.columns:
        base_final = base_final[~base_final["PAIS"].str.lower().str.contains("exterior|international|estrangeiro", na=False)]
    # Excluir por matrícula nas planilhas de estagiário/aprendiz/exterior
    exclu_set = set()
    for f in [file_aprendiz, file_estagio, file_exterior]:
        if f:
            try:
                df = read_csv_robust(f); df = clean_id(df, "MATRICULA")
                exclu_set.update(df["MATRICULA"].dropna().astype(str))
            except Exception:
                pass
    if exclu_set:
        base_final = base_final[~base_final["MATRICULA"].astype(str).isin(exclu_set)]

    # Validação/correção de datas e férias
    date_cols = [c for c in base_final.columns if "data" in c.lower() or "admiss" in c.lower() or "demiss" in c.lower()]
    for col in date_cols:
        base_final[col] = base_final[col].apply(parse_br_date)
    if "DIAS DE FÉRIAS" in base_final.columns:
        base_final["DIAS DE FÉRIAS"] = pd.to_numeric(base_final["DIAS DE FÉRIAS"], errors="coerce").fillna(0).astype(int)
    # Corrige férias mal preenchidas (negativos ou > dias mês)
    if "DIAS DE FÉRIAS" in base_final.columns:
        base_final["DIAS DE FÉRIAS"] = base_final["DIAS DE FÉRIAS"].clip(lower=0, upper=cap_dias_mes)

    # --- LLM para exclusão/correção inteligente ---
    if llm_enabled and openrouter_api_key:
        st.info("Analisando exclusões e correções com LLM...")
        # Amostra da base consolidada para prompt
        sample_csv = base_final.head(30).to_csv(index=False)
        prompt = f"""
Você é um agente Python especialista em tratamento de dados de RH. Analise a base abaixo e:
1. Liste as matrículas que devem ser excluídas, considerando: diretores, estagiários, aprendizes, afastados (ex: licença maternidade), profissionais no exterior. Use as colunas de cargo, situação, afastamento, país, etc.
2. Liste correções necessárias: datas quebradas, campos faltantes, férias mal preenchidas, feriados estaduais/municipais não aplicados.
Base (CSV):\n{sample_csv}
Responda em JSON com duas chaves: 'excluir' (lista de matrículas) e 'correcoes' (lista de instruções).
"""
        import json
        llm_response = call_openrouter_agent(prompt, openrouter_api_key, model="openai/gpt-3.5-turbo")
        # Tenta limpar espaços/quebras de linha antes do parsing
        llm_response_clean = llm_response.strip()
        try:
            result = json.loads(llm_response_clean)
            excluir = result.get("excluir", [])
            correcoes = result.get("correcoes", [])
        except Exception:
            # Tenta encontrar o JSON dentro da resposta
            import re
            match = re.search(r'\{[\s\S]*\}', llm_response_clean)
            if match:
                try:
                    result = json.loads(match.group(0))
                    excluir = result.get("excluir", [])
                    correcoes = result.get("correcoes", [])
                except Exception:
                    st.warning("Não foi possível interpretar a resposta da LLM. Veja abaixo:")
                    st.code(llm_response)
                    excluir, correcoes = [], []
            else:
                st.warning("Não foi possível interpretar a resposta da LLM. Veja abaixo:")
                st.code(llm_response)
                excluir, correcoes = [], []
        # Aplicar exclusões
        if excluir:
            base_final = base_final[~base_final["MATRICULA"].isin(excluir)]
            st.success(f"{len(excluir)} registros excluídos conforme instrução da LLM.")

        # Se houver correções, pedir para a LLM aplicar e devolver a base corrigida em lotes
        if correcoes:
            st.info("Correções sugeridas pela LLM:")
            for corr in correcoes:
                st.write(f"- {corr}")
            st.info("Solicitando à LLM a base corrigida em lotes...")
            import io
            import pandas as pd
            batch_size = 20
            batches = [base_final.iloc[i:i+batch_size] for i in range(0, len(base_final), batch_size)]
            base_corrigida_list = []
            for idx, batch in enumerate(batches):
                st.write(f"Processando lote {idx+1}/{len(batches)}...")
                corr_prompt = f"""
Você é um agente Python especialista em tratamento de dados de RH. Aplique as seguintes correções na base abaixo e devolva a base corrigida em formato CSV. Correções:
{json.dumps(correcoes, ensure_ascii=False)}
Base (CSV):\n{batch.to_csv(index=False)}
"""
                corr_response = call_openrouter_agent(corr_prompt, openrouter_api_key, model="openai/gpt-3.5-turbo")
                csv_start = corr_response.find("MATRICULA")
                if csv_start != -1:
                    csv_data = corr_response[csv_start:]
                else:
                    csv_data = corr_response
                try:
                    batch_corrigido = pd.read_csv(io.StringIO(csv_data))
                    base_corrigida_list.append(batch_corrigido)
                except Exception:
                    st.warning(f"Não foi possível interpretar o lote {idx+1} corrigido pela LLM. Veja abaixo:")
                    st.code(corr_response)
            if base_corrigida_list:
                base_final = pd.concat(base_corrigida_list, ignore_index=True)
                st.success("Base corrigida pela LLM aplicada em todos os lotes!")

    # Exibir resultado consolidado
    st.subheader("📊 Base Consolidada")
    st.dataframe(base_final.head(1000), use_container_width=True)
    # Download
    csv_final = base_final.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("⬇️ Baixar Base Consolidada (CSV)", data=csv_final, file_name=f"base_consolidada_{competencia.replace('-','')}.csv", mime="text/csv")

# ...existing code...

    # Excluir aprendiz/estagiário
    def safe_ids(file):
        if not file: return set()
        try:
            df = read_csv_robust(file); df.columns = [c.strip() for c in df.columns]
            df = clean_id(df, "MATRICULA")
            return set(df.get("MATRICULA", pd.Series(dtype=str)).dropna().tolist())
        except Exception as e:
            errors.append(f"Falha ao ler lista: {e}")
            return set()

    apr_set = safe_ids(file_aprendiz)
    est_set = safe_ids(file_estagio)

    base = ativos[~(ativos["MATRICULA"].isin(apr_set) | ativos["MATRICULA"].isin(est_set))].copy()

    # Dias úteis por sindicato
    if file_dias_uteis:
        try:
            du = read_csv_robust(file_dias_uteis); du.columns = [c.strip() for c in du.columns]
            if "SINDICATO" in du.columns and "DIAS UTEIS" in du.columns:
                du["SINDICATO"] = du["SINDICATO"].str.strip()
                du["DIAS_UTEIS_INT"] = pd.to_numeric(du["DIAS UTEIS"].str.replace(",",".", regex=False), errors="coerce").fillna(0).astype(int)
                base = base.merge(du[["SINDICATO","DIAS_UTEIS_INT"]], left_on="Sindicato", right_on="SINDICATO", how="left")
                base.rename(columns={"DIAS_UTEIS_INT": "DIAS_BASE"}, inplace=True)
                base.drop(columns=["SINDICATO"], inplace=True, errors="ignore")
            else:
                base["DIAS_BASE"] = cap_dias_mes
        except Exception as e:
            errors.append(f"Base dias úteis: {e}")
            base["DIAS_BASE"] = cap_dias_mes
    else:
        base["DIAS_BASE"] = cap_dias_mes

    # Férias
    if file_ferias:
        try:
            fer = read_csv_robust(file_ferias); fer.columns = [c.strip() for c in fer.columns]
            fer = clean_id(fer, "MATRICULA")
            col = "DIAS DE FÉRIAS" if "DIAS DE FÉRIAS" in fer.columns else None
            if col:
                fer["DIAS_FERIAS_INT"] = pd.to_numeric(fer[col].str.replace(",",".", regex=False), errors="coerce").fillna(0).astype(int)
            else:
                fer["DIAS_FERIAS_INT"] = 0
            base = base.merge(fer[["MATRICULA","DIAS_FERIAS_INT"]], on="MATRICULA", how="left")
        except Exception as e:
            errors.append(f"Férias: {e}")
            base["DIAS_FERIAS_INT"] = 0
    else:
        base["DIAS_FERIAS_INT"] = 0
    base["DIAS_FERIAS_INT"] = base["DIAS_FERIAS_INT"].fillna(0).astype(int)

    # Afastamentos
    if file_afastamentos:
        try:
            af = read_csv_robust(file_afastamentos); af.columns = [c.strip() for c in af.columns]
            af = clean_id(af, "MATRICULA")
            fmap = {"sim": True, "s": True, "y": True, "yes": True, "1": True, "true": True, "verdadeiro": True,
                    "não": False, "nao": False, "n": False, "no": False, "0": False, "false": False}
            if "na compra?" in af.columns:
                af["na_compra_flag"] = af["na compra?"].str.lower().map(fmap).fillna(True)
            else:
                af["na_compra_flag"] = True
            base = base.merge(af[["MATRICULA","na_compra_flag"]], on="MATRICULA", how="left")
        except Exception as e:
            errors.append(f"Afastamentos: {e}")
            base["na_compra_flag"] = True
    else:
        base["na_compra_flag"] = True
    base["na_compra_flag"] = base["na_compra_flag"].fillna(True)

    # Admissões
    adm_date = None
    if file_admissoes:
        try:
            adm = read_csv_robust(file_admissoes); adm.columns = [c.strip() for c in adm.columns]
            adm = clean_id(adm, "MATRICULA")
            adm_col = None
            for cand in ["DATA ADMISSAO","DATA ADMISSÃO","ADMISSÃO","DATA","ADMISSAO"]:
                if cand in adm.columns:
                    adm_col = cand; break
            if adm_col:
                adm["ADMISSAO_DATE"] = adm[adm_col].apply(parse_br_date)
                base = base.merge(adm[["MATRICULA","ADMISSAO_DATE"]], on="MATRICULA", how="left")
        except Exception as e:
            errors.append(f"Admissões: {e}")
    if "ADMISSAO_DATE" not in base.columns:
        base["ADMISSAO_DATE"] = None

    # Desligados
    if file_desligados:
        try:
            des = read_csv_robust(file_desligados); des.columns = [c.strip() for c in des.columns]
            des = clean_id(des, "MATRICULA")
            dcol = None
            for cand in ["DATA DEMISSÃO","DATA DEMISSAO","DEMISSAO","DATA DESLIGAMENTO"]:
                if cand in des.columns:
                    dcol = cand; break
            if dcol:
                des["DESLIG_DATE"] = des[dcol].apply(parse_br_date)
                base = base.merge(des[["MATRICULA","DESLIG_DATE"]], on="MATRICULA", how="left")
        except Exception as e:
            errors.append(f"Desligados: {e}")
    if "DESLIG_DATE" not in base.columns:
        base["DESLIG_DATE"] = None

    # Pró-rata
    first_day, last_day = month_bounds(competencia)
    total_weekdays_in_month = len([d for d in pd.date_range(first_day, last_day, freq="D").date if d.weekday() < 5])

    def wd_between(d1, d2):
        if d1 is None or d2 is None: return None
        return working_days_between(d1, d2, first_day, last_day)

    base["DIAS_FROM_ADM"] = base["ADMISSAO_DATE"].apply(lambda d: wd_between(d, last_day) if pd.notna(d) else None)
    base["DIAS_UNTIL_DESL"] = base["DESLIG_DATE"].apply(lambda d: wd_between(first_day, d) if pd.notna(d) else None)

    def apply_prorata(row):
        dias = int(row.get("DIAS_BASE", cap_dias_mes)) if pd.notna(row.get("DIAS_BASE", np.nan)) else cap_dias_mes
        dfa = row.get("DIAS_FROM_ADM", None)
        if dfa is not None and pd.notna(dfa):
            dias = min(dias, max(0, int(round(dias * (float(dfa) / max(1, total_weekdays_in_month))))))
        dtd = row.get("DIAS_UNTIL_DESL", None)
        if dtd is not None and pd.notna(dtd):
            dias = min(dias, max(0, int(round(dias * (float(dtd) / max(1, total_weekdays_in_month))))))
        return dias

    base["DIAS_BASE"] = base["DIAS_BASE"].fillna(cap_dias_mes).astype(int)
    base["DIAS_APOS_PRORATA"] = base.apply(apply_prorata, axis=1)
    base["DIAS_APOS_FERIAS"] = (base["DIAS_APOS_PRORATA"] - base["DIAS_FERIAS_INT"]).clip(lower=0)
    base["DIAS_CALCULADOS"] = np.where(base["na_compra_flag"] == False, 0, base["DIAS_APOS_FERIAS"])

    # Valor: sindicato e/ou exterior
    base["VALOR_DIARIO"] = np.nan
    if file_valor_sind:
        try:
            vs = read_csv_robust(file_valor_sind); vs.columns = [c.strip() for c in vs.columns]
            if "SINDICATO" in vs.columns and "VALOR" in vs.columns:
                vs["VALOR"] = vs["VALOR"].apply(lambda x: float(str(x).replace(",",".").strip()))
                base = base.merge(vs[["SINDICATO","VALOR"]], left_on="Sindicato", right_on="SINDICATO", how="left")
                base["VALOR_DIARIO"] = base["VALOR"].astype(float)
                base.drop(columns=["SINDICATO","VALOR"], inplace=True, errors="ignore")
        except Exception as e:
            errors.append(f"Valor por sindicato: {e}")

    if file_exterior:
        try:
            ext = read_csv_robust(file_exterior); ext.columns = [c.strip() for c in ext.columns]
            if "MATRICULA" in ext.columns and "Valor" in ext.columns:
                ext["MATRICULA"] = ext["MATRICULA"].astype(str).str.replace(r"[^0-9]","", regex=True).str.lstrip("0")
                ext["VALOR_DIARIO_EXT"] = ext["Valor"].apply(lambda x: float(str(x).replace(",",".").strip()))
                base = base.merge(ext[["MATRICULA","VALOR_DIARIO_EXT"]], on="MATRICULA", how="left")
                base["VALOR_DIARIO"] = base["VALOR_DIARIO_EXT"].combine_first(base["VALOR_DIARIO"])
                base.drop(columns=["VALOR_DIARIO_EXT"], inplace=True, errors="ignore")
        except Exception as e:
            errors.append(f"Exterior: {e}")

    base["VALOR_DIARIO"] = base["VALOR_DIARIO"].fillna(0.0)
    base["TOTAL"] = (base["DIAS_CALCULADOS"] * base["VALOR_DIARIO"]).round(2)

    # ---- Outputs ----
    calc_cols = ["MATRICULA","EMPRESA","TITULO DO CARGO","DESC. SITUACAO","Sindicato",
                 "DIAS_BASE","DIAS_FERIAS_INT","DIAS_CALCULADOS","VALOR_DIARIO","TOTAL"]
    calc = base[calc_cols].copy()

    audit_cols = ["MATRICULA","EMPRESA","TITULO DO CARGO","DESC. SITUACAO","Sindicato",
                  "DIAS_BASE","ADMISSAO_DATE","DIAS_FROM_ADM","DESLIG_DATE","DIAS_UNTIL_DESL",
                  "DIAS_APOS_PRORATA","DIAS_FERIAS_INT","na_compra_flag","DIAS_CALCULADOS",
                  "VALOR_DIARIO","TOTAL"]
    audit = base[audit_cols].copy()

    # KPIs
    kpis = {
        "Competência": competencia,
        "Colaboradores (ativos base)": int(ativos.shape[0]),
        "Excluídos (aprendizes)": int(len(apr_set)),
        "Excluídos (estagiários)": int(len(est_set)),
        "Com férias (>0 dias)": int((base["DIAS_FERIAS_INT"] > 0).sum()),
        "Afastamento (fora da compra)": int((base["na_compra_flag"] == False).sum()),
        "Dias VR — Soma": int(base["DIAS_CALCULADOS"].sum()),
        "Valor total estimado": float(base["TOTAL"].sum()),
    }

    st.success("Cálculo concluído.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dias (soma)", kpis["Dias VR — Soma"])
    c2.metric("Valor total estimado", f"R$ {kpis['Valor total estimado']:.2f}")
    c3.metric("Com férias", kpis["Com férias (>0 dias)"])
    c4.metric("Afast. fora da compra", kpis["Afastamento (fora da compra)"])

    st.subheader("📄 Planilha de Cálculo")
    st.dataframe(calc.head(1000), use_container_width=True)

    st.subheader("🧾 Auditoria (explicativa)")
    st.dataframe(audit.head(1000), use_container_width=True)

    # Downloads
    calc_csv = calc.to_csv(index=False, encoding="utf-8-sig")
    audit_csv = audit.to_csv(index=False, encoding="utf-8-sig")

    st.download_button("⬇️ Baixar Cálculo (CSV)", data=calc_csv, file_name=f"VR_calculo_{competencia.replace('-','')}.csv", mime="text/csv")
    st.download_button("⬇️ Baixar Auditoria (CSV)", data=audit_csv, file_name=f"VR_auditoria_{competencia.replace('-','')}.csv", mime="text/csv")


    # Planilha final para operadora (80/20)
    final_df = calc.copy()
    final_df["VALOR_CONCEDIDO"] = (final_df["DIAS_CALCULADOS"] * final_df["VALOR_DIARIO"]).round(2)
    final_df["CUSTO_EMPRESA_80"] = (final_df["VALOR_CONCEDIDO"] * 0.80).round(2)
    final_df["DESCONTO_PROF_20"] = (final_df["VALOR_CONCEDIDO"] * 0.20).round(2)

    st.subheader("📦 Planilha Final (Operadora) — 80% Empresa / 20% Profissional")
    show_cols = ["MATRICULA","EMPRESA","TITULO DO CARGO","Sindicato","DIAS_CALCULADOS","VALOR_DIARIO","VALOR_CONCEDIDO","CUSTO_EMPRESA_80","DESCONTO_PROF_20"]
    show_cols = [c for c in show_cols if c in final_df.columns]
    st.dataframe(final_df[show_cols].head(1000), use_container_width=True)
    final_csv = final_df[show_cols].to_csv(index=False, encoding="utf-8-sig")
    st.download_button("⬇️ Baixar Planilha Final (CSV)", data=final_csv, file_name=f"VR_operadora_{competencia.replace('-','')}.csv", mime="text/csv")

    if errors:
        with st.expander("Avisos/Logs"):
            for e in errors:
                st.warning(e)
else:
    st.info("👈 Envie os arquivos na barra lateral e clique em **Calcular**.")
