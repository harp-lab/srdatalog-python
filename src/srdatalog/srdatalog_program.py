import subprocess
from pathlib import Path
import json
from dataclasses import dataclass

from srdatalog.mir.schema import SchemaDefinition
from srdatalog.mir.commands import MirInstructions
from srdatalog.mir.runner import generate_runner
from srdatalog.ffi_header import generate_extern_c
from srdatalog.ffi_wrapper import DatalogFFI


PRELUDE = '''
#include "srdatalog.h"
#include "runtime/io.h"
using namespace SRDatalog;
using namespace SRDatalog::AST::Literals;
using string = std::string;
using Arena = boost::container::pmr::monotonic_buffer_resource;

#include "gpu/runtime/query.h"
#include "gpu/gpu_api.h"
#include "gpu/init.h"
#include "gpu/runtime/gpu_fixpoint_executor.h"
#include <chrono>
'''

HEADER = '''
  
  /* Defines API for cffi functions
 */

typedef void *DBHandle;

DBHandle db_new();
void db_free(DBHandle h);
void load(DBHandle h, const char *root_dir);
void run(DBHandle h, size_t max_iters);

'''

@dataclass
class SRDatalogProgram:
  '''
  Each SRDatalog program consists of two main parts: the database schema blueprint, provided as a SchemaDefinition 
  (which contains an array of FactDefinitions), and the instructions, which describe the Datalog operations
  performed. For compilation reasons, instructions must be passed as a MirInstructions, which contains an array of code blocks
  and necessary bookkeeping for the compilation of this code. See generate_runner and MirInstructions for generation details.
  '''

  name: str
  database: SchemaDefinition
  instructions: MirInstructions
  prelude: str = PRELUDE
  source_loc: str | None = None
  binary_loc: str | None = None
  header_loc: str | None = None
  existing_compile_type: str | None = None

  def __init__(self, name:str, database:SchemaDefinition, instructions:MirInstructions):
    self.name = name
    self.database = database
    self.instructions = instructions
    self.instructions.name = self.name

  def generate(self, main:str = "", include_cffi_api:bool = True) -> str:
    ''' Returns C++ code from self.database and self.instructions. Does NOT include the contents of the .h file (see generate_header_to_file and/or HEADER).
    @param main: the code to be included in the main function of the generated C++ file. This should include code for inserting test data and for running the runner (i.e. Name_Runner::run(device_db);). Use if you're planning on running as an executable (not default)
    @param include_cffi_api: whether to include the api required for CFFI functionality. If False, the generated code will not include it. Leave as True if you're planning on using open_cffi.'''

    code = self.prelude + '\n'+self._generate_schema() +"\n"+self._generate_fixpoint_plans() + "\n"+self._generate_fixpoint_runner() + self._generate_main(main)
    if include_cffi_api:
      code += self._generate_cffi_api()
    return code
  
  def generate_to_file(self, main:str = "", dest_file=None, generate_cffi:bool = True, generate_header:bool = True) -> str | None:
    '''
      Generate the C++ code to a file and return the location. The defaults will generate code suitable for compiling for CFFI. 
      @param main: the code to be included in the main function of the generated C++ file. This can be left blank if you plan to use CFFI.
      @param dest_file: the destination file for the generated C++ code. If None, defaults to python/output/NAME.cpp, python/output/NAME.h.
      @param generate_cffi: whether to include the api required for CFFI functionality. Including the API will not affect the ability to compile to an executable, but is necessary for using open_cffi.
      @param generate_header: whether to generate the header file required for CFFI functionality. Including the header will not affect the ability to compile to an executable, but is necessary for using open_cffi.
    '''

    # default path is python/output/NAME.cpp
    if dest_file is None:
      script_dir = Path(__file__).resolve().parent / "output"
      script_dir.mkdir(exist_ok=True)
      dest_file = str(script_dir) + "/" + self.name + ".cpp"
      # in case we also want to generate the header, set that path as well
      dest_header_file = str(script_dir) + "/" + self.name + ".h"

    with open(dest_file, 'w') as file:
      file.write(self.generate(main, include_cffi_api=generate_cffi))

    if generate_header:
      self.generate_header_to_file(dest_header_file)
    
    self.source_loc = dest_file
    print(f"Successfully generated {dest_file}")

    # Run formatter
    command = ["clang-format", "-style=llvm", "-i", dest_file]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
        return dest_file
    except subprocess.CalledProcessError as e:
        print(f"Error formatting file: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Error: clang-format executable not found. Make sure it is installed and in your PATH.")
        return None


  def set_prelude(self, prelude:str):
    ''' Use this function to alternatively overwrite the default prelude '''
    self.prelude = PRELUDE if prelude == "" else prelude


  def compile_to_file(self, main:str="", output_filename=None, output_cpp_filename=None, compile_type="shared", clean_previous=False, timeout=60):
    '''
    Compile the program for either use as an executable or use with CFFI. 
    @param main: the code to be included in the main function of the generated C++ file. This can be left blank if you plan to use CFFI.
    @param compile_type: either "executable" or "shared". If "executable", compiles to a binary that can be run directly. If "shared", compiles to a .so file that can be loaded with cffi.
    @param clean_previous: whether to regenerate the source file before compiling. If False, will use the existing source file at self.source_loc iff self.source_loc is not None. If True, will generate a new source file even if one already exists at self.source_loc.
    @param timeout: the timeout for the compilation process in seconds. May need to be increased for larger programs.
    @param output_filename: the destination file for the compiled binary (.so or no extension). If None, defaults to the same location as the source file.
    @param output_cpp_filename: the destination file for the generated C++ source code, if it needs to be (re)generated. 
    '''
    if compile_type not in ["executable", "shared"]:
      print(f"Error: Invalid compile_type '{compile_type}'. Must be 'executable' or 'shared'.")
      return None
    if clean_previous or self.source_loc is None:
      # (re)gen cpp if necessary
      self.source_loc = self.generate_to_file(main, output_cpp_filename)
      if self.source_loc is None:
          print("Error generating source file. Compilation aborted.")
          return None
    
    # determine proper output location (same as source if not given, otherwise as named)
    if output_filename is None:
      if compile_type == "shared":
        dest_filename = self.source_loc.replace(".cpp", ".so")
      else:
        dest_filename = self.source_loc.replace(".cpp", "")
    else:
      dest_filename = output_filename
      dest_folder = Path(dest_filename).parent  # ensure the destination folder exists
      Path(dest_folder).mkdir(exist_ok=True)

    # Creates the compilation command from compile_args.json
    script_dir = str(Path(__file__).resolve().parent)
    root_dir = str(Path(__file__).resolve().parent.parent.parent)
    args = json.load(open(script_dir+"/compile_args.json"))
    cmd = []
    cmd.append(args["compile_program"])
    cmd.extend(args["compile_args"])
    if compile_type == "shared":
      cmd.extend(args["api_args"])
    cmd.append(self.source_loc)
    cmd.extend(args["link_args"])
    cmd.extend(["-o", dest_filename])
    for i in range(len(cmd)):
      cmd[i] = cmd[i].replace("{root}", root_dir)
    print("Compiling to "+dest_filename)
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
        if str(result.returncode) == "0":
          print(f"Successfully compiled to {dest_filename}")
          self.existing_compile_type = compile_type
          self.binary_loc = dest_filename
          return dest_filename
        else:
          print(f"Compilation failed with return code {result.returncode}")
          print(f"Compiler output: {result.stdout}")
          print(f"Compiler error output: {result.stderr}")
          return None
    except subprocess.CalledProcessError as e:
        print(f"Error compiling file: {e.stderr}")


  def generate_header_to_file(self, dest_file=None):
    ''' Generate the header file required for CFFI functionality in the indicated location.'''
    if dest_file is None:
      script_dir = Path(__file__).resolve().parent / "output"
      script_dir.mkdir(exist_ok=True)
      dest_file = str(script_dir) + "/" + self.name + ".h"

    with open(dest_file, 'w') as file:
      file.write(HEADER)
    
    print(f"Successfully generated header {dest_file}")
    self.header_loc = dest_file
    return dest_file
  

  def open_ffi(self):
    if self.header_loc is None or self.binary_loc is None:
      print("Error: Header file or binary file not found. Please ensure both are generated before opening FFI, i.e. by running .compile_to_file() with argument compile_type='shared'. If you've already compiled the file and don't want to recompile it, set arg recompile=False")
      return None
    
    return DatalogFFI(self.header_loc, self.binary_loc)


  def set_file_location(self, source_loc:str = None, binary_loc:str = None, header_loc:str = None, compile_type:str = None):
    '''Use this function to set the file locations for the source, binary, and header files if you've already generated/compiled them in a previous run.
       Useful for setting locations for using open_cffi without having to recompile the program.
       Ensure that any passed file locations point to the intended files.'''
    if source_loc is not None:
      self.source_loc = source_loc
    if binary_loc is not None:
      self.binary_loc = binary_loc
    if header_loc is not None:
      self.header_loc = header_loc
    if compile_type is not None:
      if compile_type not in ["executable", "shared"]:
        print(f"Error: Invalid compile_type '{compile_type}'. Must be 'executable' or 'shared'. Ensure that it matches the type of the provided binary location")
      else:
        self.existing_compile_type = compile_type


  def run(self):
    ''' A shortcut function to run a compiled executable. '''
    if self.binary_loc is None:
      print("Error: No compiled binary found. Please compile the program before running.")
      return None
    if self.existing_compile_type == "shared":
      print("Warning: Running a shared library as an executable may not work as expected. Make sure to compile with compile_type='executable' if you intend to run directly.")
    
    try:
        result = subprocess.run([self.binary_loc], check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running program: {e.stderr}")
        return None


  def __str__(self):
    return self.generate()
  

  def _generate_schema(self):
    res = str(self.database)
    relations = ", ".join(str(fact.name) for fact in self.database.facts)
    res += f"\nusing {self.name}DB = AST::Database<{relations}>;"
    res += f"\nusing {self.name}Plan_DB = AST::Database<{relations}>;"
    res += f"\nusing namespace SRDatalog::mir::dsl;"
    return res
  

  def _generate_fixpoint_runner(self):
    return 'struct '+self.name+'_Runner {template <typename DB>\n'+generate_runner(self.name, self.instructions, self.database) +'};'
  

  def _generate_fixpoint_plans(self):
    return "namespace "+self.name+"_Plans {" +self.instructions.cpp()+"}"


  def _generate_main(self, main_code):
    '''If you're planning on running the program as an executable, ensure that you pass code for inserting test data AND for running the runner (i.e. Name_Runner::run(device_db);)'''

    return 'int main(int argc, char** args, char** env) {' + main_code + '}'

  def _generate_cffi_api(self):
    return generate_extern_c(self.name)
  



# Example main function:
'''
    using namespace SRDatalog;
    // Instantiate Host DB
    auto db = SemiNaiveDatabase<TriangleDB>();
    
    // Insert test data: triangle 1->2->3->1
    auto& rrel = get_relation_by_schema<RRel, FULL_VER>(db);
    rrel.push_row({1, 2}, BooleanSR::one());
    
    auto& srel = get_relation_by_schema<SRel, FULL_VER>(db);
    srel.push_row({2, 3, 0}, BooleanSR::one());
    
    auto& trel = get_relation_by_schema<TRel, FULL_VER>(db);
    trel.push_row({3, 1, 0}, BooleanSR::one());
    
    std::cout << "Running Triangle Counting (GPU)..." << std::endl;
    
    // Prepare Device DB (copy from host)
    auto device_db = SRDatalog::GPU::copy_host_to_device(db);
    
    // Time the entire execution
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Execute using the generated Runner
    Triangle_Runner::run(device_db);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Total execution time: " << duration.count() << " ms" << std::endl;'''