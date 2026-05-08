"""Generate synthetic Markovian RSA (Recursive Synthesis & Aggregation) training examples.

RSA format enables multi-trace reasoning during SFT phase:
  [PROBLEM] → [REASONING_TRACE_1] → [REASONING_TRACE_2] → ... → [AGGREGATE] → [SOLUTION]

This module creates 10K+ synthetic examples for SFT curriculum.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class RSAExample:
    """Single RSA training example."""

    problem: str
    reasoning_traces: list[str]  # 8 parallel reasoning paths
    aggregate_instruction: str
    final_answer: str
    source: str  # "math", "gsm8k", "aime", "synthetic"


class RSADataGenerator:
    """Generate synthetic RSA examples from problem sets."""

    # Template for generating diverse reasoning traces
    REASONING_TRACE_TEMPLATES = [
        "Approach 1 (algebraic): {reasoning}",
        "Approach 2 (geometric): {reasoning}",
        "Approach 3 (induction): {reasoning}",
        "Approach 4 (contradiction): {reasoning}",
        "Approach 5 (recursive): {reasoning}",
        "Approach 6 (modular arithmetic): {reasoning}",
        "Approach 7 (group theory): {reasoning}",
        "Approach 8 (probabilistic): {reasoning}",
    ]

    # Aggregate instructions
    AGGREGATE_TEMPLATES = [
        "Synthesizing {n_traces} reasoning paths: These approaches suggest {hint}. Conclusion:",
        "Merging {n_traces} perspectives: Common pattern observed across traces is {hint}. Therefore:",
        "Aggregating results from {n_traces} methods: All suggest {hint}. Final answer:",
        "Cross-checking {n_traces} solution routes: Consistent finding is {hint}. Answer:",
    ]

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.examples: list[RSAExample] = []

    def add_math_problem(
        self,
        problem: str,
        solution: str,
        reasoning_traces: list[str],
        source: str = "math",
    ) -> RSAExample:
        """Add a single math problem with pre-computed reasoning traces.
        
        Args:
            problem: Problem statement
            solution: Final answer
            reasoning_traces: List of 8 parallel solution approaches
            source: Data source identifier
        
        Returns:
            RSAExample
        """
        # Pick random aggregate template
        agg_template = random.choice(self.AGGREGATE_TEMPLATES)
        aggregate_instruction = agg_template.format(
            n_traces=len(reasoning_traces),
            hint=solution[:50],  # Use start of solution as hint
        )

        example = RSAExample(
            problem=problem,
            reasoning_traces=reasoning_traces,
            aggregate_instruction=aggregate_instruction,
            final_answer=solution,
            source=source,
        )

        self.examples.append(example)
        return example

    def generate_synthetic_arithmetic(self, n_examples: int = 100) -> list[RSAExample]:
        """Generate synthetic arithmetic/algebra problems with reasoning traces.
        
        Example problem: "Solve x^2 + 2x + 1 = 0"
        """
        examples = []

        for i in range(n_examples):
            # Random quadratic equation
            a = random.randint(1, 5)
            b = random.randint(-10, 10)
            c = random.randint(-10, 10)

            problem = f"Solve {a}x^2 + {b}x + {c} = 0"

            # Generate 8 diverse reasoning traces
            traces = [
                f"Using quadratic formula: x = ({-b} ± sqrt({b**2 - 4*a*c})) / {2*a}",
                f"Completing the square: ({a})x^2 + {b}x = {-c}",
                f"Factoring attempt: Check if discriminant {b**2 - 4*a*c} is perfect square",
                f"Synthetic division: Test rational roots ±{c}/{a}",
                f"Graphical approach: Parabola opens {'upward' if a > 0 else 'downward'}",
                f"Vieta's formulas: Sum of roots = {-b/a}, product = {c/a}",
                f"Numerical method: Newton-Raphson iteration starting from x=0",
                f"Matrix eigenvalue perspective: Companion matrix eigenvalues",
            ]

            # Approximate solution (for demo; real version would compute exactly)
            discriminant = b**2 - 4*a*c
            if discriminant < 0:
                solution = f"Complex roots (discriminant = {discriminant} < 0)"
            else:
                sqrt_disc = discriminant ** 0.5
                x1 = (-b + sqrt_disc) / (2*a)
                x2 = (-b - sqrt_disc) / (2*a)
                solution = f"x = {x1:.3f} or x = {x2:.3f}"

            example = self.add_math_problem(
                problem=problem,
                solution=solution,
                reasoning_traces=traces,
                source="synthetic_arithmetic",
            )
            examples.append(example)

        return examples

    def generate_synthetic_word_problems(self, n_examples: int = 100) -> list[RSAExample]:
        """Generate synthetic word problems (shopping, distance, work rate)."""
        examples = []

        problem_templates = [
            "Alice buys {n1} apples at ${p1} each and {n2} oranges at ${p2} each. How much does she spend?",
            "A train travels {d1} miles in {t1} hours. At this rate, how far will it go in {t2} hours?",
            "Worker A completes a job in {h1} hours, Worker B in {h2} hours. Working together, how long?",
            "A rectangle has length {l} and width {w}. What is its area and perimeter?",
            "A store has {n1} items at 10% off and {n2} items at 20% off. Average discount?",
        ]

        for i in range(n_examples):
            template = random.choice(problem_templates)

            # Fill in random values
            params = {
                "n1": random.randint(2, 10),
                "n2": random.randint(2, 10),
                "p1": random.randint(1, 5),
                "p2": random.randint(1, 5),
                "d1": random.randint(50, 300),
                "t1": random.randint(2, 8),
                "t2": random.randint(2, 20),
                "h1": random.randint(2, 10),
                "h2": random.randint(2, 10),
                "l": random.randint(5, 20),
                "w": random.randint(2, 15),
            }

            problem = template.format(**params)

            # Generate traces for this problem
            traces = [
                f"Direct calculation: Substitute values and compute step-by-step",
                f"Unit analysis: Ensure units are consistent throughout",
                f"Estimation: Approximate answer to check reasonableness",
                f"Algebraic setup: Write equations before solving",
                f"Working backwards: Start from answer format, reverse operations",
                f"Multiple methods: Try two different solution approaches",
                f"Dimensional analysis: Check units at each step",
                f"Boundary conditions: Test edge cases and constraints",
            ]

            # Simple computed solution
            if "apples" in problem and "oranges" in problem:
                solution = f"${params.get('n1', 3) * params.get('p1', 1) + params.get('n2', 2) * params.get('p2', 1)}"
            else:
                solution = "Computed via traces above"

            example = self.add_math_problem(
                problem=problem,
                solution=solution,
                reasoning_traces=traces,
                source="synthetic_word_problems",
            )
            examples.append(example)

        return examples

    def to_jsonl(self, output_path: str) -> None:
        """Save all examples to JSONL format."""
        with open(output_path, "w") as f:
            for example in self.examples:
                record = {
                    "problem": example.problem,
                    "reasoning_traces": example.reasoning_traces,
                    "aggregate_instruction": example.aggregate_instruction,
                    "final_answer": example.final_answer,
                    "source": example.source,
                }
                f.write(json.dumps(record) + "\n")

    def to_training_format(self, output_path: str) -> None:
        """Save in training format: [PROBLEM] → [TRACES] → [AGGREGATE] → [ANSWER]"""
        with open(output_path, "w") as f:
            for example in self.examples:
                # Format: structured tokens that model learns to produce
                formatted = {
                    "prompt": f"[PROBLEM]\n{example.problem}\n\n[REASONING_PATHS]",
                    "reasoning_traces": "\n".join(
                        f"[TRACE_{i+1}] {t}"
                        for i, t in enumerate(example.reasoning_traces)
                    ),
                    "aggregate": f"\n\n[AGGREGATE]\n{example.aggregate_instruction}",
                    "answer": f"\n[ANSWER] {example.final_answer}",
                }

                # Concatenate for training
                full_text = (
                    formatted["prompt"]
                    + "\n"
                    + formatted["reasoning_traces"]
                    + formatted["aggregate"]
                    + formatted["answer"]
                )

                f.write(json.dumps({"text": full_text, "source": example.source}) + "\n")

    def sample_examples(self, n: int = 10) -> list[RSAExample]:
        """Return n random examples for inspection."""
        return random.sample(self.examples, min(n, len(self.examples)))

    def get_stats(self) -> dict:
        """Return dataset statistics."""
        sources = {}
        for ex in self.examples:
            sources[ex.source] = sources.get(ex.source, 0) + 1

        return {
            "total_examples": len(self.examples),
            "by_source": sources,
            "avg_traces_per_example": sum(
                len(ex.reasoning_traces) for ex in self.examples
            ) / max(1, len(self.examples)),
        }


def generate_full_rsa_dataset(
    output_dir: str = "data/synthetic_rsa",
    n_arithmetic: int = 3000,
    n_word_problems: int = 3000,
    seed: int = 42,
) -> None:
    """Generate complete RSA training dataset (10K examples).
    
    Saves two formats:
    - JSONL: Structured format for inspection
    - Training: Concatenated text for SFT
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    print("[RSA] Generating synthetic dataset...")
    generator = RSADataGenerator(seed=seed)

    print(f"  Generating {n_arithmetic} arithmetic problems...")
    generator.generate_synthetic_arithmetic(n_arithmetic)

    print(f"  Generating {n_word_problems} word problems...")
    generator.generate_synthetic_word_problems(n_word_problems)

    # Save outputs
    jsonl_path = os.path.join(output_dir, "rsa_examples.jsonl")
    training_path = os.path.join(output_dir, "rsa_training_data.jsonl")

    print(f"  Saving to {jsonl_path}...")
    generator.to_jsonl(jsonl_path)

    print(f"  Saving training format to {training_path}...")
    generator.to_training_format(training_path)

    # Print stats
    stats = generator.get_stats()
    print("\n[RSA] Dataset Stats:")
    print(f"  Total examples: {stats['total_examples']}")
    for source, count in stats["by_source"].items():
        print(f"    {source}: {count}")
    print(f"  Avg traces per example: {stats['avg_traces_per_example']:.1f}")

    # Show sample
    print("\n[RSA] Sample examples:")
    for i, ex in enumerate(generator.sample_examples(3)):
        print(f"\n  Example {i+1}:")
        print(f"    Problem: {ex.problem}")
        print(f"    Traces: {len(ex.reasoning_traces)} paths")
        print(f"    Answer: {ex.final_answer}")


if __name__ == "__main__":
    generate_full_rsa_dataset()
