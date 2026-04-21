def generate_extern_c(name):
  return (
    """

  extern "C" {

  using DBHandle = void *;

  DBHandle db_new() {
    return new SemiNaiveDatabase<"""
    + name
    + """DB>();
  }

  void db_free(DBHandle h) {
    // must be called by program whenever the database is no longer needed to avoid memory leaks
    auto ptr = static_cast<SemiNaiveDatabase<"""
    + name
    + """DB> *>(h);
    delete ptr;
  }

  void load(DBHandle h, const char *root_dir) {
    auto &db = *static_cast<SemiNaiveDatabase<"""
    + name
    + """DB> *>(h);
    """
    + name
    + """_Runner::load_data(db, std::string(root_dir));
  }

  void run(DBHandle h, size_t max_iters) {
    auto &db = *static_cast<SemiNaiveDatabase<"""
    + name
    + """DB> *>(h);
    """
    + name
    + """_Runner::run(db, max_iters);
  }

  } // extern "C"

        """
  )
