from srdatalog.mir.commands import MirInstructions
from srdatalog.mir.schema import Pragma, SchemaDefinition


def _include_load_pragmas(schema):
  pragmas = ""
  for f in schema.facts:
    if Pragma.INPUT in f.pragmas:
      pragmas += (
        "\n" + f'SRDatalog::load_from_file<{f.name}>(db, root_dir + "/{f.pragmas[Pragma.INPUT]}");'
      )
  load_data_str = f'''
    static void load_data(DB& db, std::string root_dir) {{
    {pragmas}
    }}'''
  return load_data_str


def _include_printsize_pragmas(schema):
  printsize_pragmas = ""
  for f in schema.facts:
    if Pragma.PRINT_SIZE in f.pragmas:
      printsize_pragmas += (
        "\n"
        + f'''
        std::cout << " >>>>>>>>>>>>>>>>> {f.name} : "
        << get_relation_by_schema<{f.name}, FULL_VER>(db).interned_size()
        << std::endl;'''
      )

  return printsize_pragmas


def generate_runner(name: str, instructions: MirInstructions, schema: SchemaDefinition):
  '''Generates the runner program. The database schema is included as it's needed for including pragmas,
  and the instructions are used for generating the steps of the program.'''

  load_data_str = _include_load_pragmas(schema)
  section_steps = _generate_section_steps(name, instructions)
  runner = _generate_run_func(instructions, schema)
  return load_data_str + '\n' + section_steps + '\n' + runner


def _generate_section_steps(name: str, instructions: MirInstructions):

  section_steps = ""
  for section_i in range(len(instructions.structure)):
    rec = "max_iterations" if instructions.structure[section_i].recursive else "1"  # is recursive?
    section_steps += f'''
      template <typename DB>
      static void step_{section_i}(DB& db, std::size_t max_iterations) {{
        SRDatalog::GPU::execute_gpu_mir_query<typename {name}_Plans::step_{section_i}_t::type>(db, {rec});
      }}'''

  return section_steps


def _generate_run_func(instructions: MirInstructions, schema: SchemaDefinition):

  body = ""

  for section_i in range(len(instructions.structure)):
    rec = "recursive" if instructions.structure[section_i].recursive else "simple"
    rel = ", ".join(spec.fact.name for spec in instructions.structure[section_i].dests)
    body += f'''
      auto step_{section_i}_start = std::chrono::high_resolution_clock::now();
      step_{section_i}(db, max_iterations);
      auto step_{section_i}_end = std::chrono::high_resolution_clock::now();
      auto step_{section_i}_duration = std::chrono::duration_cast<std::chrono::milliseconds>(step_{section_i}_end - step_{section_i}_start);
      std::cout << "[Step {section_i} ({rec})] " << "Relations: {rel}" << " completed in " << step_{section_i}_duration.count() << " ms" << std::endl;
    '''

  body += _include_printsize_pragmas(schema)

  return f'''
    template <typename DB> static void run(DB& db, std::size_t max_iterations = std::numeric_limits<int>::max()) {{
    {body}
    }}
  '''
