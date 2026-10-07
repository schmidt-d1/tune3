#!/usr/bin/env python
# scripts/download_cifar.py
"""
Baixa o CIFAR-10 (versao python) e o CIFAR-10.1 v6 para data/ (fora do git).

  data/cifar10/cifar-10-batches-py/...      ~170 MB, MD5 conferido
  data/cifar10_1/cifar10.1_v6_{data,labels}.npy   ~6 MB, formato conferido

Se algum download falhar (rede, URL mudou), o script diz qual arquivo faltou e onde coloca-lo a
mao -- nada mais depende de URL depois disso.

Uso:
    python scripts/download_cifar.py
"""
from __future__ import annotations

import hashlib, os, sys, tarfile, urllib.request

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR101_BASE = "https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/master/datasets/"
CIFAR101_FILES = ["cifar10.1_v6_data.npy", "cifar10.1_v6_labels.npy"]


def _download(url, dest):
    print(f"  baixando {url}")
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ok = True
    os.makedirs("data/cifar10", exist_ok=True); os.makedirs("data/cifar10_1", exist_ok=True)

    print("[CIFAR-10]")
    if os.path.isdir("data/cifar10/cifar-10-batches-py"):
        print("  ja' extraido em data/cifar10/cifar-10-batches-py")
    else:
        tgz = "data/cifar10/cifar-10-python.tar.gz"
        try:
            if not os.path.exists(tgz):
                _download(CIFAR10_URL, tgz)
            md5 = _md5(tgz)
            if md5 != CIFAR10_MD5:
                print(f"  ERRO: MD5 {md5} != {CIFAR10_MD5} -- arquivo corrompido; apague e rode de novo")
                ok = False
            else:
                with tarfile.open(tgz, "r:gz") as t:
                    t.extractall("data/cifar10", filter="data")
                print("  MD5 conferido e extraido")
        except Exception as e:  # noqa: BLE001
            print(f"  FALHOU ({e}). Baixe {CIFAR10_URL} e coloque em {tgz}; rode de novo.")
            ok = False

    print("[CIFAR-10.1 v6]")
    for f in CIFAR101_FILES:
        dest = os.path.join("data/cifar10_1", f)
        if os.path.exists(dest):
            print(f"  {f} ja' existe"); continue
        try:
            _download(CIFAR101_BASE + f, dest)
        except Exception as e:  # noqa: BLE001
            print(f"  FALHOU ({e}). Baixe {CIFAR101_BASE + f} e coloque em {dest}.")
            ok = False
    try:
        import numpy as np
        X = np.load("data/cifar10_1/cifar10.1_v6_data.npy"); y = np.load("data/cifar10_1/cifar10.1_v6_labels.npy")
        good = X.shape == (2000, 32, 32, 3) and y.shape == (2000,) and set(np.unique(y)) <= set(range(10))
        print(f"  formato {X.shape} {X.dtype}, rotulos {y.shape}: {'OK' if good else 'INESPERADO'}")
        ok = ok and good
    except Exception as e:  # noqa: BLE001
        print(f"  nao foi possivel conferir o CIFAR-10.1: {e}"); ok = False

    print("\nPRONTO" if ok else "\nINCOMPLETO -- veja as mensagens acima")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
