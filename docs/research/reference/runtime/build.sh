#!/bin/sh
# Builds the C-ABI dylib (for ctypes) and the nanobind extension with plain clang.
set -eu
cd "$(dirname "$0")"
mkdir -p build
PY=.venv/bin/python
PYINC=$($PY -c "import sysconfig;print(sysconfig.get_paths()['include'])")
EXT=$($PY -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")
NBINC=$($PY -c "import nanobind;print(nanobind.include_dir())")
NBSRC=$($PY -c "import nanobind;print(nanobind.source_dir())")
CFLAGS="-O2 -fobjc-arc -mmacosx-version-min=15.0"
FW="-framework Metal -framework Foundation"

# The dylib also carries the Metal 4 experiment, which needs macOS 26 APIs.
time clang++ -O2 -fobjc-arc -mmacosx-version-min=26.0 -std=c++17 -dynamiclib native/forge_rt.mm native/forge_mtl4.mm $FW -o build/libforge_rt.dylib

# nanobind: compile nb_combined.cpp once (slowest part), then link.
[ -f build/nb_combined.o ] || time clang++ -O2 -mmacosx-version-min=15.0 -std=c++17 -fPIC -fvisibility=hidden \
    -I"$PYINC" -I"$NBINC" -I"$NBSRC/../ext/robin_map/include" \
    -c "$NBSRC/nb_combined.cpp" -o build/nb_combined.o
time clang++ $CFLAGS -std=c++17 -fvisibility=hidden -shared -undefined dynamic_lookup \
    -I"$PYINC" -I"$NBINC" -I"$NBSRC/../ext/robin_map/include" \
    native/forge_nb.mm native/forge_rt.mm build/nb_combined.o $FW -o "build/forge_nb$EXT"
ls -la build
