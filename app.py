from tutor import MathTutor
import base64

# ======================
# Streamlit app

import streamlit as st

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="🧮 AI Math Tutor",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="auto",
)

# ------------------------------------------------------------------
# Session-state helpers
# ------------------------------------------------------------------
if "tutor" not in st.session_state:
    st.session_state.tutor = MathTutor()

if "history" not in st.session_state:
    st.session_state.history = []

if "last_run" not in st.session_state:
    st.session_state.last_run = None

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:

    st.header("📋 Examples")
    examples = [
        "Solve the quadratic equation: 2x^2 - 5x + 3 = 0",
        "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1",
        "Calculate the integral of sin(x) from 0 to π",
        "Factor the polynomial: x^4 - 16",
        "Solve the system: 2x + 3y = 7, x - y = 1",
        "Find the limit of (sin(x))/x as x approaches 0",
        "Plot the function y = x^2 - 4x + 3 and find its roots",
        "Calculate the area under the curve y = x^2 from x=0 to x=3",
        "Find the slope of the line passing through (1,2) and (5,8)",
        "Convert 45 degrees to radians and find sin(45°)",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state.problem_input = ex

# ------------------------------------------------------------------
# Main UI
# ------------------------------------------------------------------
st.title("🧮 AI Math Tutor")
st.markdown("Ask any math question, get step-by-step solutions with code and plots!")

problem = str(
    st.text_area(
        "Enter your math problem:",
        value=st.session_state.get("problem_input", ""),
        height=100,
    )
).strip()

col1, col2 = st.columns([1, 4])
with col1:
    solve_btn = st.button("Solve", type="primary")
with col2:
    clear_btn = st.button("Clear History")

# ------------------------------------------------------------------
# Event handlers
# ------------------------------------------------------------------
if clear_btn:
    st.session_state.history.clear()
    st.rerun()

if solve_btn and problem.strip():
    with st.spinner("Solving..."):
        result = st.session_state.tutor.solve_problem(problem)

    st.session_state.last_run = result
    st.session_state.history.insert(0, result)

# ------------------------------------------------------------------
# Display results
# ------------------------------------------------------------------
if st.session_state.history:
    st.divider()
    for idx, res in enumerate(st.session_state.history):
        l = len(st.session_state.history)
        st.markdown(f"# Problem {l - idx}")
        st.write(f"**Question:** {res['problem']}")
        if res["success"]:
            st.markdown(res["solution"])

            # Plot any generated figures
            for block_idx, exec_res in enumerate(res["execution_results"]):
                st.markdown(
                    f"### Code Block {block_idx+1}\n```python\n{res['code_blocks'][block_idx]}\n```"
                )
                if exec_res["success"]:
                    if exec_res.get("output"):
                        st.markdown(
                            f"##### **Output:**\n```\n{exec_res['output']}\n```"
                        )
                    if exec_res.get("plot_data"):
                        image = base64.b64decode(exec_res["plot_data"])
                        st.image(
                            image,
                            width=600,
                            caption=f"Plot from code block {block_idx+1}",
                        )
                else:
                    st.error(
                        f"Error in code block {block_idx+1}: {exec_res.get('error')}"
                    )
        else:
            st.error(f"Error: {res.get('error')}")
        st.divider()
