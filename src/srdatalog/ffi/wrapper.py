'''This is the definition of the lightweight wrapper class created when using FFI with a SRDatalog program.'''

from cffi import FFI


class DatalogFFI:
  def __init__(self, header_path: str, binary_path: str):

    self.ffi = FFI()

    # load the header and define the C API
    self.ffi.cdef(open(header_path).read())
    self.lib = self.ffi.dlopen(binary_path)

    self._h = None  # set a default in case db_new fails
    self._h = self.lib.db_new()

  def load_data(self, root_dir: str):
    '''Loads all data within the specified directory to the program. The directory should contain files with names matching the INPUT pragmas specified in the program schema.'''
    self.lib.load(self._h, root_dir.encode("utf-8"))

  def run(self, max_iters: int = 2**63 - 1):
    '''Runs the program.'''
    self.lib.run(self._h, max_iters)

  def __del__(self):
    if self._h:
      self.lib.db_free(self._h)
      self._h = self.ffi.NULL
