# scripts/verify_setup.py
"""
Verificador de setup do Tune3. Roda do basico ao avancado e diz EXATAMENTE
o que esta pronto e o que falta. Uso:
    python scripts/verify_setup.py
"""
import importlib
import sys
from pathlib import Path

OK = "[ OK ]"
NO = "[FALTA]"
WARN = "[AVISO]"


def check_import(modpath, names):
    try:
        mod = importlib.import_module(modpath)
        for n in names:
            if not hasattr(mod, n):
                return NO, f"{modpath}: falta '{n}'"
        return OK, modpath
    except Exception as e:
        return NO, f"{modpath}: {type(e).__name__}: {e}"


def main():
    print("=" * 64)
    print(" VERIFICADOR DE SETUP DO TUNE3")
    print("=" * 64)

    print("\n-- Dependencias --")
    deps = ["numpy", "torch", "pandas", "sklearn", "structlog",
            "botorch", "gpytorch", "optuna", "hydra", "wandb", "pytest"]
    faltam_deps = []
    for d in deps:
        try:
            importlib.import_module(d)
            print(f"  {OK} {d}")
        except Exception:
            print(f"  {NO} {d}  -> pip install {d}")
            faltam_deps.append(d)

    print("\n-- Fase 0: estimador de curvatura --")
    print("  " + " ".join(check_import("tune3.curvature", ["HutchinsonEstimator", "HutchinsonConfig"])))

    print("\n-- Fase 1: dados DREBIN --")
    print("  " + " ".join(check_import("tune3.data.drebin", ["DrebinLoader", "DrebinConfig"])))

    print("\n-- Fase 2: MICRO + safety --")
    print("  " + " ".join(check_import("tune3.micro", ["DDKFController", "DDKFConfig"])))
    print("  " + " ".join(check_import("tune3.safety", ["CantelliGuard", "CantelliConfig"])))

    print("\n-- Fase 2b: gating, EoS, GSNR --")
    print("  " + " ".join(check_import("tune3.micro", ["RegimeGate", "RegimeGatingConfig"])))
    print("  " + " ".join(check_import("tune3.safety", ["EoSDetector", "EoSConfig"])))
    print("  " + " ".join(check_import("tune3.curvature", ["GSNREstimator", "GSNRConfig"])))

    print("\n-- Fase 3: CVaR + MACRO --")
    print("  " + " ".join(check_import("tune3.core", ["cvar", "value_at_risk"])))
    print("  " + " ".join(check_import("tune3.macro", ["Tune3MacroLoop", "MacroConfig"])))

    print("\n-- Dados: arquivo DREBIN-215 --")
    candidates = ["data/drebin215.csv", "data/drebin-215-dataset-5560malware-9476-benign.csv"]
    found = None
    for c in candidates:
        if Path(c).exists():
            found = c
            break
    if found:
        print(f"  {OK} encontrado: {found}")
        try:
            from tune3.data.drebin import DrebinLoader, DrebinConfig
            loader = DrebinLoader(DrebinConfig(csv_path=found))
            Xtr, Xv, Xte, ytr, yv, yte = loader.load_splits()
            print(f"  {OK} load_splits(): treino={Xtr.shape}, val={Xv.shape}, teste={Xte.shape}")
            frac = float(ytr.mean())
            print(f"  {OK} features={Xtr.shape[1]} (esperado 215), frac_malware={frac:.3f} (esperado ~0.37)")
            if Xtr.shape[1] != 215:
                print(f"  {WARN} numero de features != 215; confira a coluna de classe")
        except Exception as e:
            print(f"  {NO} load_splits() falhou: {type(e).__name__}: {e}")
    else:
        print(f"  {NO} data/drebin215.csv NAO encontrado")
        print("        Baixe o DREBIN-215 (figshare/Kaggle) e salve em data/drebin215.csv")

    print("\n" + "=" * 64)
    if faltam_deps:
        print(f" ACAO: instale dependencias faltantes: pip install {' '.join(faltam_deps)}")
    if not found:
        print(" ACAO: baixe o DREBIN-215 para data/drebin215.csv (gate da Fase 4)")
    if not faltam_deps and found:
        print(" TUDO PRONTO para a Fase 4 (integracao).")
    print("=" * 64)


if __name__ == "__main__":
    main()
