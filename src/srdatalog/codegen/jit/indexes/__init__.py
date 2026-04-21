'''Index-plugin implementations for non-default GPU index types.

Port of src/srdatalog/indexes/. Each submodule here constructs an
IndexPlugin instance and registers it with `register_index_plugin()`
at module import time — mirroring Nim's `static: registerIndexPlugin(...)`
compile-time registration.

Import submodules you want active; for now:

  two_level  — Device2LevelIndex (HEAD + FULL for FULL reads)
'''
