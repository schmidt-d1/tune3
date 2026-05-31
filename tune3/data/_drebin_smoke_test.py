# scripts/smoke_test_fase0_1.py -- valida fundacao sem precisar do CSV real
"""Rode: python -m tune3.data._drebin_smoke_test
Gera um DREBIN-215 sintetico, salva CSV, e testa o loader ponta a ponta."""
import numpy as np, pandas as pd, tempfile, os

def make_fake_drebin(path, n=600, d=215, frac_mal=0.37, seed=42):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, d)) < 0.15).astype(int)   # features binarias esparsas
    y = (rng.random(n) < frac_mal)
    cols = [f"feat_{i}" for i in range(d)]
    df = pd.DataFrame(X, columns=cols)
    df["class"] = np.where(y, "S", "B")            # formato S/B como o real
    # injeta alguns '?' para testar a limpeza
    df["feat_0"] = df["feat_0"].astype(object); df.loc[0, "feat_0"] = "?"
    df.to_csv(path, index=False)

if __name__ == "__main__":
    from tune3.data.drebin import DrebinLoader, DrebinConfig
    tmp = os.path.join(tempfile.gettempdir(), "fake_drebin215.csv")
    make_fake_drebin(tmp)
    loader = DrebinLoader(DrebinConfig(csv_path=tmp))
    X_tr, X_val, X_te, y_tr, y_val, y_te = loader.load_splits()
    print("shapes:", X_tr.shape, X_val.shape, X_te.shape)
    print("fracao malware (treino):", round(float(y_tr.mean()), 3))
    assert X_tr.shape[1] == 215
    assert set(np.unique(y_tr)) <= {0, 1}
    print("SMOKE TEST FASE 0/1: OK")
