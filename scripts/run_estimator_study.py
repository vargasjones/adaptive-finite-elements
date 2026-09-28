from pathlib import Path
import matplotlib.pyplot as plt

from adaptive_fem.experiments import adaptive_convergence


def main():
    out = Path(__file__).resolve().parents[1] / "figures"
    out.mkdir(exist_ok=True)
    alpha = 5.0 / 3.0

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for degree, marker in ((1, "o"), (2, "s")):
        _, ndof, errors, estimators, _ = adaptive_convergence(alpha, degree, iterations=9)
        ax.loglog(ndof, errors, marker + "-", label=f"P{degree} error")
        ax.loglog(ndof, estimators, marker + "--", label=f"P{degree} estimator")
    ax.set_xlabel("degrees of freedom")
    ax.set_ylabel("error / estimator")
    ax.set_title("Residual estimator reliability")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "estimator_reliability.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
