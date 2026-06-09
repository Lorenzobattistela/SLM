# App Streamlit

O app em `apps/streamlit_chatbot.py` oferece uma interface local para gerar
texto a partir de checkpoints treinados pelo projeto. Ele usa o mesmo runtime de
inferência de `scripts/sample_checkpoint.py`, mas expõe os controles principais
em uma tela Streamlit.

## Instalação

Instale as dependências do app dentro do ambiente do projeto:

```bash
python -m pip install -e ".[app]"
```

Se o checkpoint usar o tokenizador SuperBPE local, instale também o backend
SuperBPE descrito em [docs/tokenizer.md](tokenizer.md).

## Execução

Inicie a interface a partir da raiz do repositório:

```bash
streamlit run apps/streamlit_chatbot.py
```

Por padrão, o app tenta usar:

```text
configs/train_200m_fineweb_edu.yml
checkpoints/llm_200m_fineweb_edu/latest.pt
```

Se os checkpoints tiverem sido extraídos sob `checkpoints/checkpoints/`, o app
tenta detectar automaticamente esse caminho aninhado.

## Controles da Interface

A barra lateral permite configurar:

- `Run config`: YAML usado para montar o modelo, tokenizador e parâmetros de inferência.
- `Checkpoint`: arquivo `.pt` com os pesos salvos do modelo.
- `Device`: seleção automática, CUDA, CPU ou MPS.
- `Max new tokens`: limite de tokens gerados.
- `Temperature`: intensidade de amostragem.
- `Top-k`: recorte do vocabulário considerado a cada passo.
- `Formato`: saída completa ou apenas continuação.
- `Mostrar comando equivalente`: exibe o comando `scripts/sample_checkpoint.py` correspondente.

O painel principal mostra o prompt, o botão de geração, métricas da amostra e
um histórico curto das gerações recentes.

## Relação com o Script de Inferência

Para reproduzir uma geração fora da interface, copie o comando exibido pelo app
ou use diretamente:

```bash
python scripts/sample_checkpoint.py --run-config configs/train_200m_fineweb_edu.yml --checkpoint checkpoints/llm_200m_fineweb_edu/latest.pt --prompt "Scientific progress depends on"
```

Essa equivalência ajuda a depurar diferenças entre interface e linha de comando.

## Problemas Comuns

- `FileNotFoundError` para o checkpoint: confira o caminho no campo `Checkpoint`.
- Erro ao carregar tokenizador SuperBPE: instale o backend SuperBPE e confirme os artefatos em `artifacts/tokenizer_superbpe_50k_olmo_p99/`.
- CUDA indisponível: selecione `cpu` no campo `Device` para uma checagem funcional menor.
- Modelo incompatível com checkpoint: use o mesmo `--run-config` usado no treinamento daquele checkpoint.
