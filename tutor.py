import re
import os
import io
import base64
import openai
import numpy as np
import sympy as sp
from textwrap import dedent
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from sympy import symbols, solve, diff, integrate, simplify, expand, factor

load_dotenv()

api_key: str = str(os.getenv("GROQ_API_KEY"))
base_url: str = "https://api.groq.com/openai/v1"
model: str = "llama-3.3-70b-versatile"
small_model: str = "moonshotai/kimi-k2-instruct"


class MathTutor:

    def __init__(
        self,
        api_key: str = api_key,
        base_url: str = base_url,
        model: str = model,
        small_model: str = small_model,
    ):
        """
        Initialize the Math Tutor with OpenAI API key

        Args:
            api_key (str): OpenAI API key
        """
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.small_model = small_model
        self.conversation_history = []

    def _llm_call(
        self, user_prompt: str, system_prompt: str, model: Optional[str] = None
    ) -> str:
        """
        Call the OpenAI API with the user and system prompts
        """
        if model is None:
            model = self.model

        response = str(
            self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            .choices[0]
            .message.content
        )
        return response

    def execute_python_code(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code safely and return results with detailed output

        Args:
            code (str): Python code to execute

        Returns:
            Dict containing execution results, output, and any generated plots
        """
        # Capture stdout
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        result = {
            "success": True,
            "output": "",
            "result": None,
            "error": None,
            "plot_data": None,
        }

        try:
            # Create a safe execution environment
            safe_globals = {
                "__builtins__": __builtins__,
                "np": np,
                "plt": plt,
                "sp": sp,
                "symbols": symbols,
                "solve": solve,
                "diff": diff,
                "integrate": integrate,
                "simplify": simplify,
                "expand": expand,
                "factor": factor,
                "print": print,
                "range": range,
                "len": len,
                "abs": abs,
                "round": round,
                "sum": sum,
                "max": max,
                "min": min,
                "pow": pow,
                "sqrt": np.sqrt,
                "sin": np.sin,
                "cos": np.cos,
                "tan": np.tan,
                "log": np.log,
                "exp": np.exp,
                "pi": np.pi,
                "e": np.e,
            }

            # Execute the code
            exec(code, safe_globals)

            # Capture any printed output
            result["output"] = captured_output.getvalue()

            # Check if a plot was created
            if plt.get_fignums():
                # Save plot as base64 string
                buffer = io.BytesIO()
                plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.getvalue()).decode()
                result["plot_data"] = plot_data
                plt.close("all")  # Clean up

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)

        finally:
            sys.stdout = old_stdout

        return result

    def extract_python_code(self, text: str) -> List[str]:
        """
        Extract Python code blocks from the response text
        """
        pattern = r"```python\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches

    def check_code(self, problem: str, code: str) -> str:
        """
        Check the validity and correctness of the provided Python code
        """

        system_prompt = "Check the code for errors and return only the corrected code. No need for robust error handling, just check for any ayntax errors and unassigned variables. \nUse Python code blocks (```python) when you need to perform calculations, display graphs, or demonstrate concepts visually using numpy, sympy, or matplotlib as appropriate. Do not use plt.show"

        prompt = dedent(
            f"""
            Problem:\n---\n{problem}\n---
            
            Check the following code for errors:\n---\n{code}\n---
            """
        )

        try:
            code_result = self._llm_call(
                user_prompt=prompt, system_prompt=system_prompt, model=self.small_model
            )

            return self.extract_python_code(code_result)[0]

        except Exception as e:
            return f"Error occurred: {e}"

    def solve_problem(self, problem: str) -> Dict[str, Any]:
        """
        Solve a math problem using OpenAI API and Python code execution

        Args:
            problem (str): The math problem to solve

        Returns:
            Dict containing the solution, code execution results, and any visualizations
        """

        try:
            # Create the prompt
            system_prompt = dedent(
                """
                You are an expert mathematics tutor who provides clear, rigorous, and structured solutions. 
                Your responsibilities:
                - Analyze each problem carefully before solving.
                - Present solutions step by step, with logical explanations.
                - When calculations are needed, include Python code that can be executed.
                - Use `numpy`, `sympy`, or `matplotlib` for computation and visualization.
                - Format mathematical expressions correctly:
                - Use $...$ for inline math.
                - Use $$...$$ for block math.
                - Explanations must be precise, concise, and conceptually accurate.
                - Include intermediate steps and reasoning, not just final answers.
                - Avoid unnecessary text or commentary beyond the solution structure.
                """
            )

            user_prompt = dedent(
                f"""
                You are an expert math tutor. Solve the following math problem with detailed reasoning.

                Problem: {problem}

                Provide your solution in this format:
                1. **Problem Analysis**: Identify the type of problem and outline the method of solution.
                2. **Step-by-Step Solution**: Show each step in logical order with explanations.
                3. **Python Code**: When calculations are required, include executable Python code 
                (in ```python blocks) using numpy, sympy, or matplotlib for verification.
                4. **Key Concepts**: Summarize the core mathematical concepts involved.

                Strict Rules:
                - Use $...$ for inline math and $$...$$ for block math.
                - Ensure accuracy, completeness, and readability.
                - Show intermediate steps clearly; do not skip reasoning.
                - Keep responses professional, concise, and well-formatted.
                """
            )

            # Get response from OpenAI
            solution_text = self._llm_call(
                user_prompt=user_prompt, system_prompt=system_prompt
            )

            # replace \[ and \] with $$ for block math
            solution_text = re.sub(
                r"\\\[(.*?)\\\]", r"$$\1$$", solution_text, flags=re.DOTALL
            )

            # replace \( and \) with $ for inline math
            solution_text = re.sub(
                r"\\\((.*?)\\\)", r"$\1$", solution_text, flags=re.DOTALL
            )

            # Extract and execute Python code
            code_blocks = self.extract_python_code(solution_text)
            execution_results = []

            for i, code in enumerate(code_blocks):
                print(f"\n--- Executing Code Block {i+1} ---")
                code = self.check_code(problem, code)
                print(code)

                print("--- Output ---")

                result = self.execute_python_code(code)
                execution_results.append(result)

                if result["success"]:
                    if result["output"]:
                        print(result["output"])
                    if result["plot_data"]:
                        print("📊 Plot generated successfully!")
                else:
                    print(f"❌ Error: {result['error']}")

            return {
                "problem": problem,
                "solution": solution_text,
                "code_blocks": code_blocks,
                "execution_results": execution_results,
                "success": True,
            }

        except Exception as e:
            return {
                "problem": problem,
                "solution": None,
                "error": str(e),
                "success": False,
            }

    def interactive_session(self):
        """
        Start an interactive math tutoring session
        """
        print("🧮 Welcome to the AI Math Tutor!")
        print("I can help you solve various math problems step by step.")
        print(
            "Type 'quit' to exit, 'help' for examples, or just enter your math problem.\n"
        )

        while True:
            try:
                user_input = input("📝 Enter your math problem: ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("👋 Thank you for using the Math Tutor! Goodbye!")
                    break

                elif user_input.lower() == "help":
                    self.show_examples()
                    continue

                elif not user_input:
                    continue

                print(f"\n🔍 Solving: {user_input}")
                print("=" * 60)

                # Solve the problem
                result = self.solve_problem(user_input)

                if result["success"]:
                    print("\n📚 **SOLUTION:**")
                    print(result["solution"])

                    # Show execution results summary
                    if result["execution_results"]:
                        print(
                            f"\n🔧 Executed {len(result['code_blocks'])} code blocks:"
                        )
                        for i, exec_result in enumerate(result["execution_results"]):
                            status = "✅" if exec_result["success"] else "❌"
                            print(f"  Code block {i+1}: {status}")

                    # save the result to a file
                    with open("solution.md", "a", encoding="utf-8") as f:
                        f.write(f"# {user_input}\n\n")
                        f.write("\n## 📚 **SOLUTION:**\n\n")
                        f.write(result["solution"] + "\n\n")
                        for i, exec_result in enumerate(result["execution_results"]):
                            f.write(f"\n### Code Block {i+1}\n")
                            f.write("\n```python\n")
                            f.write(result["code_blocks"][i])
                            f.write("\n```\n")
                            if exec_result["success"]:
                                if exec_result["output"]:
                                    f.write("\n```\n")
                                    f.write(exec_result["output"])
                                    f.write("```\n")
                                if exec_result["plot_data"]:
                                    f.write(
                                        f"![Plot](data:image/png;base64,{exec_result['plot_data']})\n\n\n"
                                    )
                            else:
                                f.write(f"Error: {exec_result['error']}\n")

                else:
                    print(f"❌ Error solving problem: {result['error']}")

                print("\n" + "=" * 60)

            except KeyboardInterrupt:
                print("\n\n👋 Session ended. Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")

    def show_examples(self):
        """
        Show example problems the tutor can solve
        """
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

        print("\n📋 **Example Problems I Can Solve:**")
        for i, example in enumerate(examples, 1):
            print(f"{i:2d}. {example}")
        print()


def cli_app():
    """
    Main function to run the Math Tutor
    """

    try:
        # Initialize the tutor
        tutor = MathTutor()

        # Test with a simple problem first
        print("🧪 Testing connection with a simple problem...")
        test_result = tutor.solve_problem("What is 2 + 2?")

        if test_result["success"]:
            print("✅ Connection successful!" + "\n" * 3)
            # Start interactive session
            print("🧪 Starting interactive session...\n\n\n")
            tutor.interactive_session()
        else:
            print(f"❌ Connection failed: {test_result['error']}")

    except Exception as e:
        print(f"❌ Failed to initialize tutor: {e}")


if __name__ == "__main__":
    cli_app()
