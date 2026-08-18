# Building the client-side analysis engine (analysis_engine.{js,wasm})

Source: `wasm/analysis_engine.cpp` (research-owned; includes `../HeartsEnv.hpp`).
The built artifacts are served by the site and live in the PRIVATE site
repo (`perilune-site/hearts_web/static/`); they reach this repo's
`hearts_web/` mirror via the site's sync. Never hand-edit the glue.
This recipe was recovered 2026-08-17 — it had not been written down
anywhere before.

Toolchain: emsdk at `C:\Users\patom\emsdk` (Emscripten 6.0.6 as of
2026-08-17). `emsdk_env.bat` does not put emcc on PATH under Git Bash;
call the tools directly with the config in the environment:

```sh
export EMSDK=/c/Users/patom/emsdk EM_CONFIG=/c/Users/patom/emsdk/.emscripten \
       EMSDK_PYTHON=/c/Users/patom/emsdk/python/3.13.3_64bit/python.exe
export PATH="/c/Users/patom/emsdk/upstream/emscripten:/c/Users/patom/emsdk/node/24.19.0_64bit/bin:/c/Users/patom/emsdk/python/3.13.3_64bit:$PATH"
em++.exe wasm/analysis_engine.cpp -O3 -std=c++17 -fexceptions \
  -sMODULARIZE=1 -sEXPORT_NAME=AnalysisEngine -sALLOW_MEMORY_GROWTH=1 \
  -sEXPORTED_FUNCTIONS=_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAP32,HEAPF32 \
  -o build/wasm_dist/analysis_engine.js
```

Notes:
- Use `em++` (not `emcc`): the C++ link needs `operator delete`.
- `-O3` minifies wasm import/export names automatically; the glue binds
  `_an_*` from the mangled names, so js and wasm MUST be built together
  and shipped as a pair (commit 6c59d29 was a mismatched-pair incident).
- Exceptions on (`-fexceptions`): the engine wraps every export in
  `Trap` to turn C++ throws into `-1` returns.
- Output goes to `build/wasm_dist/` (gitignored). The site side copies
  the pair from there into `perilune-site/hearts_web/static/` and commits.
- Verify before handing off: the exported `_an_*` set covers the worker's
  needs (`grep -o "_an_[a-z_]*" hearts_web/static/analysis_worker.js | sort -u`)
  and a node A/A against the previous pair on `an_debug_selfplay`
  (instantiate with an `instantiateWasm` hook — the glue is web/worker-only).
