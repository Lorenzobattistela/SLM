# Exemplos de Execução

Este guia centraliza os comandos que antes ficavam no README. Execute os
exemplos a partir da raiz do repositório e, quando estiver dentro do ambiente
virtual, use `python` normalmente. Fora dele, `python3` pode ser necessário.

## Ambiente

Crie e ative o ambiente local:

```bash
uv venv --python /usr/bin/python3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Instale o backend SuperBPE antes de treinar ou carregar o tokenizador local:

```bash
git clone --recurse-submodules https://github.com/PythonNut/superbpe.git /tmp/superbpe
pip install -e /tmp/superbpe/tokenizers_superbpe/bindings/python/
```

Use Python 3.10-3.12. O fork SuperBPE depende de PyO3 0.21, que não compila
com Python 3.13+.

## Configuração Principal

A configuração principal é `pre-train/configs/train_200m_fineweb_edu.yml`. Os scripts
modernos recebem esse arquivo pelo argumento compartilhado `--run-config`:

```bash
python scripts/count_parameters.py --run-config pre-train/configs/train_200m_fineweb_edu.yml
```

Para checagens pequenas, use `pre-train/configs/train_200m_fineweb_edu_debug.yml`.

## Pipeline Completo

Execute o fluxo completo com:

```bash
python pre-train/scripts/run_all.py --run-config pre-train/configs/train_200m_fineweb_edu.yml
```

Esse comando tokeniza os dados, verifica parâmetros, lança treino DDP quando
CUDA/NCCL estão disponíveis, avalia, gera plots e imprime uma amostra de texto.
Se o DDP não puder ser iniciado com segurança pelo script, ele mostra o comando
manual de treino.

## Etapas Independentes

Execute cada etapa separadamente quando quiser controlar ou depurar o fluxo:

```bash
python pre-train/scripts/tokenize_dataset.py --run-config pre-train/configs/train_200m_fineweb_edu.yml
python scripts/count_parameters.py --run-config pre-train/configs/train_200m_fineweb_edu.yml
torchrun --standalone --nproc_per_node=2 pre-train/scripts/train.py --run-config pre-train/configs/train_200m_fineweb_edu.yml
python benchmarks/scripts/evaluate.py --run-config pre-train/configs/train_200m_fineweb_edu.yml
python scripts/plot_training.py --run-config pre-train/configs/train_200m_fineweb_edu.yml
python scripts/sample_checkpoint.py --run-config pre-train/configs/train_200m_fineweb_edu.yml --checkpoint checkpoints/llm_200m_fineweb_edu/latest.pt --prompt "Scientific progress depends on"
```

O comando DDP esperado para o treino principal de 2 GPUs é:

```bash
torchrun --standalone --nproc_per_node=2 pre-train/scripts/train.py --run-config pre-train/configs/train_200m_fineweb_edu.yml
```

## Comparação com GPT-2 Byte-Level BPE

Retokenize o corpus processado com SuperBPE para o tokenizador GPT-2
byte-level BPE:

```bash
python scripts/retokenize_superbpe_to_byte_bpe.py \
  --run-config pre-train/configs/train_200m_fineweb_edu.yml \
  --output-dir data/processed_byte_bpe_gpt2
```

Treine usando a configuração correspondente:

```bash
torchrun --standalone --nproc_per_node=2 pre-train/scripts/train.py --run-config pre-train/configs/train_200m_fineweb_edu_byte_bpe_gpt2.yml
```

O GPT-2 byte-level BPE usa `50257` IDs, então a configuração byte-BPE mantém
`model.vocab_size: 50257`.

## Métricas e Plots

As métricas do treino são gravadas em:

```text
outputs/llm_200m_fineweb_edu/logs/metrics.jsonl
```

Os metadados da execução são gravados em:

```text
outputs/llm_200m_fineweb_edu/logs/training_metadata.json
```

Os plots são gravados em:

```text
outputs/llm_200m_fineweb_edu/plots/
```

Gere ou regenere os plots com:

```bash
python scripts/plot_training.py --run-config pre-train/configs/train_200m_fineweb_edu.yml
```

O notebook [docs/notebooks/processed_metadata_training_plots.ipynb](notebooks/processed_metadata_training_plots.ipynb)
usa as mesmas especificações do script para visualizar os gráficos inline.

## Geração de Texto

Gere uma conclusão qualitativa a partir de um checkpoint treinado:

```bash
python scripts/sample_checkpoint.py --run-config pre-train/configs/train_200m_fineweb_edu.yml --checkpoint checkpoints/llm_200m_fineweb_edu/latest.pt --prompt "Scientific progress depends on"
```

## App Streamlit

Instale as dependências do app e inicie a interface local:

```bash
python -m pip install -e ".[app]"
streamlit run apps/streamlit_chatbot.py
```

O app usa os mesmos artefatos de tokenizador e checkpoint do treinamento. Se os
checkpoints tiverem sido extraídos em `checkpoints/checkpoints/`, a interface
tenta detectar esse caminho aninhado.

## Script Legado de Plot

O script `scripts/plot_train_loss.py` continua disponível para fluxos antigos
que esperam apenas um gráfico de perda de treino/validação.
