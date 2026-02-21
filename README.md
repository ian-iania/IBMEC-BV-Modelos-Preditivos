# IBMEC-BV-Modelos-Preditivos
MODELOS PREDITIVOS PARA SUPORTE À TOMADA DE DECISÃO

## Estrutura

- `data/bv_originacao_auto_sintetico.csv`
- `data/bv_originacao_auto_sintetico.xlsx`
- `notebooks/NB00_WarmUp_Colab_Python_BV.ipynb`
- `scripts/generate_bv_synth_data.py`

## NB00 Warm-up (Colab)

Objetivo do NB00: primeiro contato com notebook e DataFrame, sem graficos.

Fluxo recomendado em aula:

1. Abrir `notebooks/NB00_WarmUp_Colab_Python_BV.ipynb` no Colab.
2. Fazer upload do arquivo `data/bv_originacao_auto_sintetico.xlsx` no painel Files.
3. Rodar as celulas `RUN_ME` na ordem.
4. Se o upload do Excel falhar, usar a celula de Plano B com CSV raw do GitHub.

CSV raw:

`https://raw.githubusercontent.com/ian-iania/IBMEC-BV-Modelos-Preditivos/main/data/bv_originacao_auto_sintetico.csv`

## Regerar datasets

O script abaixo recria os arquivos `CSV` e `XLSX` sintéticos:

```bash
python3 scripts/generate_bv_synth_data.py
```

Se quiser usar pandas/openpyxl localmente em outros scripts:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```
