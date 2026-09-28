from adaptive_fem.experiments import uniform_convergence, adaptive_convergence


def main():
    print("alpha  degree  strategy   fitted_slope   final_ndof   final_H1_error   eta/error")
    for alpha in (5.0 / 3.0, 10.0):
        for degree in (1, 2):
            nd_u, err_u, slope_u = uniform_convergence(alpha, degree)
            print(f"{alpha:5.3f}  P{degree}      uniform    {slope_u:11.3f}   {int(nd_u[-1]):10d}   {err_u[-1]:14.6e}   {'-':>9}")
            _, nd_a, err_a, eta_a, slope_a = adaptive_convergence(alpha, degree, iterations=9)
            print(f"{alpha:5.3f}  P{degree}      adaptive   {slope_a:11.3f}   {int(nd_a[-1]):10d}   {err_a[-1]:14.6e}   {eta_a[-1]/err_a[-1]:9.3f}")


if __name__ == "__main__":
    main()
