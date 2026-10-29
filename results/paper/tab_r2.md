<!-- fontes: curvature_vs_hparams -->

**Table 7. How much of each geometric measure is a function of the five hyperparameters: R² of the log measure regressed on (log lr, log wd, dropout, width, depth), linear and with quadratic/cross terms, per dataset (range over runs).**

| dataset | measure | runs | R² linear (min–max) | R² quadratic (min–max) | dominant single factor (R² range) |
|:--|:--|--:|:--|:--|:--|
| CIFAR-10 | Tr(H²) M=5 | 3 | 0.76–0.77 | 0.79–0.82 | log lr (0.35–0.47, 3 runs) |
| CIFAR-10 | sharpness ρ=0.1 | 3 | 0.30–0.34 | 0.40–0.40 | width (0.27–0.29, 3 runs) |
| CIFAR-10 | sharpness ρ=0.5 | 3 | 0.63–0.65 | 0.73–0.76 | width (0.42–0.44, 3 runs) |
| DREBIN cluster | Tr(H²) M=5 | 30 | 0.45–0.62 | 0.50–0.68 | depth (0.33–0.49, 30 runs) |
| DREBIN cluster | λmax | 15 | 0.93–0.95 | 0.97–0.99 | depth (0.68–0.77, 15 runs) |
| DREBIN cluster | sharpness ρ=0.1 | 15 | 0.48–0.73 | 0.63–0.84 | width (0.38–0.71, 15 runs) |
| DREBIN cluster | sharpness ρ=0.5 | 15 | 0.57–0.67 | 0.66–0.79 | width (0.29–0.64, 14 runs); dropout (0.22–0.22, 1 run) |
| KronoDroid | Tr(H²) M=5 | 6 | 0.77–0.81 | 0.82–0.85 | depth (0.39–0.43, 4 runs); dropout (0.42–0.42, 2 runs) |
| KronoDroid | λmax | 3 | 0.89–0.90 | 0.95–0.95 | dropout (0.44–0.52, 3 runs) |
| KronoDroid | sharpness ρ=0.1 | 3 | 0.52–0.57 | 0.60–0.63 | depth (0.23–0.32, 3 runs) |
| KronoDroid | sharpness ρ=0.5 | 3 | 0.63–0.65 | 0.73–0.76 | dropout (0.25–0.31, 3 runs) |
| NSL-KDD | Tr(H²) M=5 | 4 | 0.64–0.76 | 0.66–0.78 | depth (0.50–0.62, 4 runs) |
| NSL-KDD | λmax | 1 | 0.78–0.78 | 0.80–0.80 | depth (0.60–0.60, 1 run) |
| NSL-KDD | sharpness ρ=0.1 | 1 | 0.52–0.52 | 0.64–0.64 | width (0.31–0.31, 1 run) |
| NSL-KDD | sharpness ρ=0.5 | 1 | 0.62–0.62 | 0.72–0.72 | width (0.38–0.38, 1 run) |
