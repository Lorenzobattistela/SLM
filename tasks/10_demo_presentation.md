# Task 10: Interactive Chat Demo and Presentation Preparation

## Objective

Develop an interactive Streamlit-based chat interface to showcase the trained Small Language Model (SLM) and prepare the final presentation deliverables summarizing architectural choices, training dynamics, evaluations, and insights.

---

## Required Features

### 1. Interactive Chat Interface (Streamlit Demo)
Build a clean, local web application that allows real-time interaction with the final SFT model checkpoint.
- **Model Loader**: Load the final SFT model weights and tokenizer dynamically.
- **User Interface**: Present a chat-style screen allowing message inputs and displaying generated responses.
- **Hyperparameter Tuning**: Expose adjustable sliders and fields in the sidebar to control generation options:
  - **Temperature**: Control randomness (e.g., scale from 0.1 to 1.5).
  - **Top-K**: Truncate generation to the top-K most likely next tokens.
  - **Max Tokens**: Limit the length of generated responses.
- **System Prompt / Chat Formatting**: Ensure inputs are properly tokenized and wrapped in the chat template identical to the SFT training stage.

### 2. Presentation Preparation (10-15 Minutes)
Prepare slides and demo materials covering:
1. **Architecture & Design Decisions**: Detail and justify the model architecture choices (e.g., RoPE, GQA, Flash Attention, SwiGLU, RMSNorm).
2. **Scaling Laws Analysis**: Explain how the optimal training token count was calculated in relation to the model parameter count (using Chinchilla Scaling Laws).
3. **Loss Curves**: Include visual charts showing training/validation loss curves for all stages: pretraining, mid-training, and SFT.
4. **Evaluation Metrics**: Report validation perplexity and benchmark accuracies (HellaSwag, ARC, PIQA, WinoGrande, and GSM8K), including the tokenizer comparison results.
5. **Insights & Discussion**: Detail what worked, what did not, and what adjustments would be made given more time and computational resources.
6. **Live Demo**: Demonstrate the model generating responses to standard instruction prompts live.

### 3. Final Deliverables
Ensure the repository contains all elements required for final submission:
- **Clean Codebase**: Organised scripts for pretraining, mid-training, SFT, evaluation, plotting, and the app.
- **Checkpoints**: Host SFT, mid-training, and pretraining checkpoints on HuggingFace Hub or Google Drive, and provide links.
- **Documentation**: Provide a detailed `README` guiding local execution for all steps, plus slides (as a PDF or link).

---

## Required Command

Run the Streamlit application using:

```bash
streamlit run apps/chat_app.py -- --run-config configs/sft_200m.yml --checkpoint outputs/llm_200m_sft/checkpoints/final.pt
```

---

## Required Source Structure

Create or adapt the following files:

```text
apps/chat_app.py
docs/presentation_outline.md
```

### Chat Application (`apps/chat_app.py`)
A Streamlit script that:
1. Parses config and model checkpoint arguments from the command line.
2. Initializes the model and tokenizer and loads weights.
3. Sets up standard Streamlit state management (`st.session_state`) for conversation history.
4. Provides side panels for temperature, top-k, and max-token parameters.
5. Implements the chat input/output message container view.

---

## Testing

Ensure the application runs locally using standard streamlit commands:

```bash
streamlit run apps/chat_app.py -- --run-config configs/sft_200m_debug.yml
```

Expected outcomes:
- Streamlit launches a local web server (usually at `http://localhost:8501`).
- The user can select generation parameters in the UI.
- Entering a prompt generates a response from the loaded model checkpoint.

---

## Acceptance Criteria

This task is complete when:
- `apps/chat_app.py` is fully functional and interactive.
- Generation hyperparameters are adjustable in the UI.
- The model successfully utilizes the chat template to communicate.
- The presentation slides outline is prepared in `docs/presentation_outline.md`.
