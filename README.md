# 🧮 AI Math Tutor

An interactive math tutor that generates step-by-step solutions with executable Python code and visualizations. Use the Streamlit UI for a friendly web app or the CLI for a terminal-based experience.

-   Web app entrypoint: [app.py](app.py)
-   Core logic and LLM integration: [tutor.py](tutor.py)
-   Dependencies: [requirements.txt](requirements.txt)
-   Environment template: [.env.example](.env.example)

## Features

-   Step-by-step math explanations with inline and block math rendering.
-   Automatically generates Python code (NumPy, SymPy, Matplotlib) to verify computations and produce plots.
-   Safely executes generated code, captures output, and renders plots.
-   Streamlit UI with example prompts and history.
-   CLI mode with optional saving to solution.md.

## Project Structure

```
AI Math Tutor
├─ app.py               # Streamlit UI
├─ tutor.py             # Math tutor logic + LLM + code execution
├─ requirements.txt     # Python dependencies
├─ .env.example         # Environment variable template
└─ .gitignore
```

## Prerequisites

-   Python 3.10+
-   A Groq API key with access to the specified model(s)

## Setup

1. Clone or open the project in VS Code.

2. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (Command Prompt)
.venv\Scripts\activate.bat
# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure environment variables:

-   Copy .env.example to .env and set your key:
-   Get your Groq API key from [here](https://console.groq.com/keys).

```bash
# Edit .env and set:
# GROQ_API_KEY="your_groq_api_key_here"
```

## Running

### Streamlit Web App

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (http://localhost:8501).

-   Enter a math problem or click an example in the sidebar.
-   The app displays:
    -   A step-by-step solution (with math rendering).
    -   Each generated code block.
    -   Code output and any generated plot images.

### CLI

```bash
python tutor.py
```

-   Runs a quick connectivity test ("What is 2 + 2?").
-   Starts an interactive session for entering problems.
-   Saves solutions to solution.md with embedded code and base64 images.

## Configuration

You can tweak defaults in [tutor.py](tutor.py):

Check the `safe_globals` dict inside [`tutor.MathTutor.execute_python_code`](tutor.py) for exposed functions and libraries.

---

## License

MIT
