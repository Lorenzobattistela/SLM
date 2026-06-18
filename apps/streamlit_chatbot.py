from __future__ import annotations

import shlex
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.runtime import (  # noqa: E402
    default_prompt,
    generate_text,
    load_decoder_runtime,
    load_inference_config,
)

# CONFIG_DEFAULT_CHECKPOINT = "checkpoints/llm_200m_fineweb_edu/latest.pt"

# DEFAULT_RUN_CONFIG = "pre-train/configs/train_200m_fineweb_edu.yml"
#DEFAULT_RUN_CONFIG = "pre-train/configs/train_200m_fineweb_edu_byte_bpe_gpt2.yml"
#CONFIG_DEFAULT_CHECKPOINT = "checkpoints/pre_train_finals/bbpe_pre_train.pt"

DEFAULT_RUN_CONFIG = "/home/gabrielstefanello/faculdade/SLM/apps/models/bbpe_sft/sft_200m_byte_bpe_gpt2.yml"
CONFIG_DEFAULT_CHECKPOINT = "/home/gabrielstefanello/faculdade/SLM/apps/models/bbpe_sft/final.pt"
def default_checkpoint_path() -> str:
    candidates = [
        CONFIG_DEFAULT_CHECKPOINT,
        "checkpoints/checkpoints/llm_200m_fineweb_edu/latest.pt",
        "checkpoints/llm_200m_fineweb_edu/final.pt",
        "checkpoints/checkpoints/llm_200m_fineweb_edu/final.pt",
    ]
    for candidate in candidates:
        if (PROJECT_ROOT / candidate).exists():
            return candidate
    return CONFIG_DEFAULT_CHECKPOINT


@st.cache_resource(show_spinner=False)
def cached_runtime(run_config_path: str, checkpoint_path: str, device_name: str):
    return load_decoder_runtime(
        run_config_path=run_config_path.strip() or None,
        checkpoint_path=checkpoint_path.strip() or None,
        device_name=device_name,
    )


@st.cache_data(show_spinner=False)
def cached_default_prompt(run_config_path: str, checkpoint_path: str) -> str:
    config = load_inference_config(
        run_config_path.strip() or None,
        checkpoint_path.strip() or None,
    )
    return default_prompt(config)


def shell_command(
    *,
    run_config_path: str,
    checkpoint_path: str,
    prompt: str,
    temperature: float,
    top_k: int,
    max_new_tokens: int,
) -> str:
    parts = [
        "python",
        "scripts/sample_checkpoint.py",
        "--run-config",
        run_config_path,
        "--checkpoint",
        checkpoint_path,
        "--prompt",
        prompt,
        "--temperature",
        str(temperature),
        "--top-k",
        str(top_k),
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    return " ".join(shlex.quote(part) for part in parts)


def init_state() -> None:
    st.session_state.setdefault("samples", [])
    st.session_state.setdefault("prompt_text", None)


def render_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
        }
        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.7rem 0.85rem;
        }
        div[data-testid="stTextArea"] textarea {
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_controls() -> dict[str, object]:
    with st.sidebar:
        st.header("Checkpoint")
        run_config_path = st.text_input("Run config", value=DEFAULT_RUN_CONFIG)
        checkpoint_path = st.text_input("Checkpoint", value=default_checkpoint_path())
        device_name = st.selectbox("Device", options=["auto", "cuda", "cpu", "mps"], index=0)

        st.header("Decoding")
        max_new_tokens = st.slider("Max new tokens", min_value=1, max_value=1024, value=80)
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.8,
            step=0.05,
        )
        top_k = st.number_input("Top-k", min_value=0, max_value=50000, value=40)

        st.header("Saida")
        output_mode = st.radio(
            "Formato",
            options=["Completa", "Somente continuacao"],
            horizontal=True,
        )
        show_command = st.checkbox("Mostrar comando equivalente", value=True)

        if st.button("Limpar amostras", use_container_width=True):
            st.session_state.samples = []

    return {
        "run_config_path": run_config_path,
        "checkpoint_path": checkpoint_path,
        "device_name": device_name,
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_k": int(top_k),
        "output_mode": output_mode,
        "show_command": bool(show_command),
    }


def render_runtime_summary(runtime) -> None:
    model_cfg = runtime.model_config
    cols = st.columns(5)
    cols[0].metric("Device", str(runtime.device))
    cols[1].metric("Contexto", model_cfg.context_length)
    cols[2].metric("Camadas", model_cfg.n_layers)
    cols[3].metric("Dimensao", model_cfg.d_model)
    cols[4].metric("Vocabulario", model_cfg.vocab_size)
    st.caption(f"Checkpoint carregado: {runtime.checkpoint_path}")


def render_result(result, *, output_text: str) -> None:
    cols = st.columns(4)
    cols[0].metric("Prompt tokens", result.prompt_tokens)
    cols[1].metric("Context tokens", result.context_tokens)
    cols[2].metric("Generated tokens", result.generated_tokens)
    cols[3].metric("Truncated", "yes" if result.truncated else "no")
    st.text_area("Resultado", value=output_text, height=380)


def render_history() -> None:
    samples = st.session_state.get("samples", [])
    if not samples:
        return

    st.subheader("Amostras recentes")
    for index, sample in enumerate(reversed(samples[-5:]), start=1):
        label = (
            f"{index}. {sample['generated_tokens']} tokens | "
            f"T={sample['temperature']} | top-k={sample['top_k']}"
        )
        with st.expander(label):
            st.text_area(
                "Prompt",
                value=sample["prompt"],
                height=110,
                key=f"history_prompt_{index}_{len(samples)}",
            )
            st.text_area(
                "Saida",
                value=sample["output"],
                height=220,
                key=f"history_output_{index}_{len(samples)}",
            )


def main() -> None:
    st.set_page_config(page_title="SLM Sample Checkpoint", layout="wide")
    render_style()
    init_state()

    controls = sidebar_controls()

    st.title("SLM Sample Checkpoint")

    if st.session_state.prompt_text is None:
        try:
            st.session_state.prompt_text = cached_default_prompt(
                str(controls["run_config_path"]),
                str(controls["checkpoint_path"]),
            )
        except Exception:
            st.session_state.prompt_text = "Scientific progress depends on"

    prompt_button_col, _ = st.columns([1, 5])
    if prompt_button_col.button("Prompt padrao", use_container_width=True):
        try:
            st.session_state.prompt_text = cached_default_prompt(
                str(controls["run_config_path"]),
                str(controls["checkpoint_path"]),
            )
        except Exception:
            st.session_state.prompt_text = "Scientific progress depends on"
        st.rerun()

    st.text_area(
        "Prompt",
        height=180,
        key="prompt_text",
    )
    prompt = st.session_state.prompt_text

    left, _ = st.columns([1, 5])
    generate_clicked = left.button("Gerar", type="primary", use_container_width=True)

    if controls["show_command"]:
        command = shell_command(
            run_config_path=str(controls["run_config_path"]),
            checkpoint_path=str(controls["checkpoint_path"]),
            prompt=prompt,
            temperature=float(controls["temperature"]),
            top_k=int(controls["top_k"]),
            max_new_tokens=int(controls["max_new_tokens"]),
        )
        st.code(command, language="bash")

    if generate_clicked:
        with st.spinner("Carregando checkpoint e gerando amostra..."):
            try:
                runtime = cached_runtime(
                    str(controls["run_config_path"]),
                    str(controls["checkpoint_path"]),
                    str(controls["device_name"]),
                )
                result = generate_text(
                    runtime,
                    prompt,
                    max_new_tokens=int(controls["max_new_tokens"]),
                    temperature=float(controls["temperature"]),
                    top_k=int(controls["top_k"]),
                    stop_sequences=(),
                )
                output_text = (
                    result.output_text
                    if controls["output_mode"] == "Completa"
                    else result.full_text
                )
                render_runtime_summary(runtime)
                render_result(result, output_text=output_text)
                st.session_state.samples.append(
                    {
                        "prompt": prompt,
                        "output": output_text,
                        "temperature": controls["temperature"],
                        "top_k": controls["top_k"],
                        "generated_tokens": result.generated_tokens,
                    }
                )
            except Exception as exc:  # pragma: no cover - Streamlit displays runtime errors.
                st.error(f"{type(exc).__name__}: {exc}")

    render_history()


if __name__ == "__main__":
    main()
