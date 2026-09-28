from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from adaptive_fem.adaptivity import adaptive_history


def main():
    out = Path(__file__).resolve().parents[1] / "figures"
    out.mkdir(exist_ok=True)
    alpha = 5.0 / 3.0
    hist = adaptive_history(alpha, degree=2, iterations=9)

    fig, ax = plt.subplots(figsize=(9.5, 2.8))
    levels = [0, 3, 6, 9]
    for row, k in enumerate(levels):
        mesh = hist[k]["mesh"]
        y = np.full_like(mesh, len(levels) - row, dtype=float)
        ax.plot(mesh, y, "|", markersize=13, markeredgewidth=1.4)
        ax.text(1.01, len(levels) - row, f"iter {k}", va="center")
    ax.set_xlim(-0.01, 1.07)
    ax.set_ylim(0.4, 4.6)
    ax.set_yticks([])
    ax.set_xlabel("x")
    ax.set_title("Adaptive mesh concentration near the singularity x = 0")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "adaptive_mesh.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
