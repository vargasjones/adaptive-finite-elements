from pathlib import Path
import matplotlib.pyplot as plt

from adaptive_fem.experiments import uniform_convergence, adaptive_convergence


def main():
    out = Path(__file__).resolve().parents[1] / "figures"
    out.mkdir(exist_ok=True)

    alpha = 5.0 / 3.0
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for degree, ax in zip((1, 2), axes):
        nd_u, err_u, slope_u = uniform_convergence(alpha, degree)
        _, nd_a, err_a, _, slope_a = adaptive_convergence(alpha, degree, iterations=9)
        ax.loglog(nd_u, err_u, "o-", label=f"Uniform (slope {slope_u:.2f})")
        ax.loglog(nd_a, err_a, "s-", label=f"Adaptive (slope {slope_a:.2f})")
        theory = -1 if degree == 1 else -2
        ref = err_a[-1] * (nd_a / nd_a[-1]) ** theory
        ax.loglog(nd_a, ref, "--", label=f"Reference slope {theory}")
        ax.set_title(f"P{degree}, alpha = 5/3")
        ax.set_xlabel("degrees of freedom")
        ax.set_ylabel("H1 seminorm error")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
    fig.suptitle("Uniform versus adaptive finite elements")
    fig.tight_layout()
    fig.savefig(out / "uniform_vs_adaptive.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
