"""
evaluate.py
===========
Standalone evaluation script for the Course Content Simplification Agent.

Runs 5 hardcoded dense academic paragraphs through the full Beginner-level
pipeline (RAG retrieval → IBM Granite simplification) and prints a table
of Flesch Reading Ease (FRE) scores before and after simplification.

Suitable for including in a project report to demonstrate measurable
simplification quality.

Usage:
    python evaluate.py

Prerequisites:
    - .env file with valid watsonx.ai credentials (see .env.example)
    - knowledge_base/ folder with .txt glossary files
    - pip install -r requirements.txt
"""

import sys
import textstat
import rag_utils
import granite_client

# ---------------------------------------------------------------------------
# Five dense academic paragraphs (mix of CS and general science)
# ---------------------------------------------------------------------------
PARAGRAPHS = [
    # 1 — Computer Science: algorithmic complexity
    (
        "The computational complexity of an algorithm is characterised by its "
        "asymptotic behaviour as input size n approaches infinity. Big-O notation "
        "abstracts away constant factors and lower-order terms, expressing the "
        "dominant growth rate of time or space requirements. An algorithm exhibiting "
        "O(n log n) complexity — as is the case with comparison-based optimal sorting "
        "algorithms such as MergeSort and HeapSort — is substantially more scalable "
        "than a naive O(n²) approach such as Bubble Sort, particularly for large "
        "datasets. Understanding worst-case, average-case, and best-case complexity "
        "is essential for selecting appropriate algorithms under resource constraints."
    ),
    # 2 — Biology: DNA replication
    (
        "DNA replication is a semiconservative process in which the double-stranded "
        "helix is unwound by helicase at the replication fork, exposing complementary "
        "template strands. DNA polymerase III synthesises new strands in the 5′ to 3′ "
        "direction by catalysing phosphodiester bond formation between incoming "
        "deoxyribonucleotide triphosphates and the 3′-hydroxyl terminus of the "
        "growing chain. Because of directional constraints, one strand — the leading "
        "strand — is synthesised continuously, while the lagging strand is assembled "
        "discontinuously as a series of Okazaki fragments, subsequently joined by "
        "DNA ligase."
    ),
    # 3 — Physics: entropy and thermodynamics
    (
        "The Second Law of Thermodynamics stipulates that for any spontaneous "
        "process occurring within an isolated system, the total entropy ΔS must be "
        "greater than or equal to zero. Entropy, a state function quantifying the "
        "degree of microscopic disorder, is related to the number of available "
        "microstates Ω by Boltzmann's equation S = k_B ln Ω, where k_B is the "
        "Boltzmann constant. Consequently, macroscopic processes are irreversible "
        "because the probability of spontaneous reversion to a lower-entropy "
        "configuration is astronomically small, establishing a thermodynamic arrow "
        "of time."
    ),
    # 4 — Computer Science: neural networks
    (
        "A feedforward artificial neural network consists of an input layer, one or "
        "more hidden layers of parameterised affine transformations followed by "
        "element-wise non-linear activation functions, and an output layer that "
        "produces a probability distribution over class labels via the softmax "
        "function. Training proceeds by minimising a differentiable loss function — "
        "typically cross-entropy — through stochastic gradient descent and "
        "backpropagation, which applies the chain rule of calculus to compute "
        "analytical gradients with respect to all network parameters. Regularisation "
        "techniques such as dropout and weight decay are employed to mitigate "
        "overfitting on finite training datasets."
    ),
    # 5 — Chemistry: catalysis and activation energy
    (
        "Catalysis accelerates chemical reactions by providing an alternative "
        "mechanistic pathway characterised by a lower activation energy (Eₐ) "
        "compared to the uncatalysed reaction. According to Arrhenius kinetics, "
        "a reduction in Eₐ produces an exponential increase in the reaction rate "
        "constant k at a given temperature T. Heterogeneous catalysts — typically "
        "transition-metal surfaces — function by adsorbing reactant molecules, "
        "weakening intramolecular bonds through electronic interactions with the "
        "surface d-orbitals, and facilitating bond rearrangement before product "
        "desorption. The catalyst itself is regenerated and not consumed in the "
        "overall stoichiometric reaction."
    ),
]

# Delimiter used by the model to start the Key Terms section
KEY_TERMS_DELIMITER = "Key Terms Explained:"


def extract_main_text(full_output: str) -> str:
    """Strip the 'Key Terms Explained:' section before FRE scoring."""
    idx = full_output.find(KEY_TERMS_DELIMITER)
    if idx != -1:
        return full_output[:idx].strip()
    return full_output.strip()


def print_table(rows: list) -> None:
    """Print a formatted evaluation results table."""
    col_w = [6, 52, 10, 12, 10]
    header = ["#", "Paragraph (first 50 chars)", "Orig FRE", "Simpl FRE", "Delta"]
    sep = "  ".join("-" * w for w in col_w)

    print()
    print("=" * 96)
    print("  COURSE CONTENT SIMPLIFICATION AGENT — Evaluation Results (Beginner Level)")
    print("=" * 96)
    print(
        f"  {header[0]:<{col_w[0]}}"
        f"  {header[1]:<{col_w[1]}}"
        f"  {header[2]:<{col_w[2]}}"
        f"  {header[3]:<{col_w[3]}}"
        f"  {header[4]:<{col_w[4]}}"
    )
    print("  " + sep)

    for row in rows:
        num, snippet, orig, simpl, delta, status = row
        if status == "ok":
            delta_str = f"{delta:+.1f}"
            print(
                f"  {str(num):<{col_w[0]}}"
                f"  {snippet:<{col_w[1]}}"
                f"  {orig:<{col_w[2]}.1f}"
                f"  {simpl:<{col_w[3]}.1f}"
                f"  {delta_str:<{col_w[4]}}"
            )
        else:
            # Error row
            print(
                f"  {str(num):<{col_w[0]}}"
                f"  {snippet:<{col_w[1]}}"
                f"  {'ERROR':<{col_w[2]}}"
                f"  {'ERROR':<{col_w[3]}}"
                f"  {status:<{col_w[4]}}"
            )

    print()
    print(
        "  Flesch Reading Ease scale: "
        "90–100 Very Easy | 80–89 Easy | 70–79 Fairly Easy | 60–69 Standard\n"
        "                             "
        "50–59 Fairly Difficult | 30–49 Difficult | 0–29 Very Confusing"
    )
    print("=" * 96)
    print()


def main():
    print("\nBuilding RAG index from knowledge_base/…")
    try:
        rag_utils.build_index("knowledge_base")
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print("RAG index ready. Running evaluation on 5 paragraphs…\n")

    rows = []
    for i, paragraph in enumerate(PARAGRAPHS, start=1):
        snippet = (paragraph[:49] + "…") if len(paragraph) > 50 else paragraph
        print(f"  [{i}/5] Processing paragraph {i}…", end=" ", flush=True)

        try:
            # RAG retrieval
            context_chunks = rag_utils.retrieve(
                query=paragraph, top_k=4, min_score=0.15, folder="knowledge_base"
            )

            # Granite simplification (Beginner level)
            simplified_output = granite_client.simplify(
                original_text=paragraph,
                level="Beginner",
                context_chunks=context_chunks,
            )

            # FRE scoring — exclude Key Terms section from simplified text
            main_simplified = extract_main_text(simplified_output)
            orig_fre  = textstat.flesch_reading_ease(paragraph)
            simpl_fre = textstat.flesch_reading_ease(main_simplified)
            delta     = simpl_fre - orig_fre

            rows.append((i, snippet, orig_fre, simpl_fre, delta, "ok"))
            print(f"done  (orig={orig_fre:.1f}, simpl={simpl_fre:.1f}, Δ={delta:+.1f})")

        except EnvironmentError as exc:
            rows.append((i, snippet, 0.0, 0.0, 0.0, "CRED ERR"))
            print(f"CREDENTIAL ERROR: {exc}")
            # Credential errors affect all paragraphs — stop early
            break
        except Exception as exc:
            rows.append((i, snippet, 0.0, 0.0, 0.0, "FAIL"))
            print(f"FAILED: {exc}")

    print_table(rows)


if __name__ == "__main__":
    main()
