import libero
import pkgutil

import libero.libero as libero_core
import libero.lifelong as lifelong


def main():
    print("LIBERO import OK")
    print("libero file:", libero.__file__)

    print("libero.libero modules:")
    print([m.name for m in pkgutil.iter_modules(libero_core.__path__)])

    print("libero.lifelong modules:")
    print([m.name for m in pkgutil.iter_modules(lifelong.__path__)])

    from libero.libero import benchmark

    print("benchmark import OK")
    print("benchmark module:", benchmark)

    print("LIBERO smoke test passed.")


if __name__ == "__main__":
    main()
