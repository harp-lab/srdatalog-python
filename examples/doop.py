"""Auto-generated from /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/doop/doop_bg_2level.nim by tools/nim_to_dsl.py.
Do not edit manually — regenerate via:

    python tools/nim_to_dsl.py /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/doop/doop_bg_2level.nim --out <this file>
"""

from __future__ import annotations

from srdatalog.dataset_const import load_meta, resolve_program_consts
from srdatalog.dsl import SPLIT, Filter, Program, Relation, Var

# ----- Relations ----------------------------------------------

DirectSuperclass = Relation(
  "DirectSuperclass",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="DirectSuperclass.csv",
)
DirectSuperinterface = Relation(
  "DirectSuperinterface",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="DirectSuperinterface.csv",
)
MainClass = Relation("MainClass", 1, column_types=(int,), input_file="MainClass.csv")
FormalParam = Relation(
  "FormalParam",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  input_file="FormalParam.csv",
)
ComponentType = Relation(
  "ComponentType",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="ComponentType.csv",
)
AssignReturnValue = Relation(
  "AssignReturnValue",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="AssignReturnValue.csv",
)
ActualParam = Relation(
  "ActualParam",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  input_file="ActualParam.csv",
)
Method_Modifier = Relation(
  "Method_Modifier",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="Method_Modifier.csv",
)
Var_Type = Relation(
  "Var_Type",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="Var_Type.csv",
)
HeapAllocation_Type = Relation(
  "HeapAllocation_Type",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="HeapAllocation_Type.csv",
)
Method_Descriptor = Relation(
  "Method_Descriptor",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="Method_Descriptor.csv",
)
ClassType = Relation("ClassType", 1, column_types=(int,), input_file="ClassType.csv")
ArrayType = Relation("ArrayType", 1, column_types=(int,), input_file="ArrayType.csv")
InterfaceType = Relation("InterfaceType", 1, column_types=(int,), input_file="InterfaceType.csv")
Var_DeclaringMethod = Relation(
  "Var_DeclaringMethod",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="Var_DeclaringMethod.csv",
)
ApplicationClass = Relation(
  "ApplicationClass", 1, column_types=(int,), input_file="ApplicationClass.csv"
)
ThisVar = Relation(
  "ThisVar",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="ThisVar.csv",
)
Field_DeclaringType = Relation(
  "Field_DeclaringType",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="Field_DeclaringType.csv",
)
Method_SimpleName = Relation(
  "Method_SimpleName",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="Method_SimpleName.csv",
)
Method_DeclaringType = Relation(
  "Method_DeclaringType",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="Method_DeclaringType.csv",
)
Instruction_Method = Relation(
  "Instruction_Method",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="Instruction_Method.csv",
)
isVirtualMethodInvocation_Insn = Relation(
  "isVirtualMethodInvocation_Insn",
  1,
  column_types=(int,),
  input_file="isVirtualMethodInvocation_Insn.csv",
)
isStaticMethodInvocation_Insn = Relation(
  "isStaticMethodInvocation_Insn",
  1,
  column_types=(int,),
  input_file="isStaticMethodInvocation_Insn.csv",
)
MethodInvocation_Method = Relation(
  "MethodInvocation_Method",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="MethodInvocation_Method.csv",
)
VirtualMethodInvocation_Base = Relation(
  "VirtualMethodInvocation_Base",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="VirtualMethodInvocation_Base.csv",
)
SpecialMethodInvocation_Base = Relation(
  "SpecialMethodInvocation_Base",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="SpecialMethodInvocation_Base.csv",
)
LoadInstanceField = Relation(
  "LoadInstanceField",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  input_file="LoadInstanceField.csv",
)
StoreInstanceField = Relation(
  "StoreInstanceField",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  input_file="StoreInstanceField.csv",
)
LoadStaticField = Relation(
  "LoadStaticField",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  input_file="LoadStaticField.csv",
)
StoreStaticField = Relation(
  "StoreStaticField",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  input_file="StoreStaticField.csv",
)
LoadArrayIndex = Relation(
  "LoadArrayIndex",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  input_file="LoadArrayIndex.csv",
)
StoreArrayIndex = Relation(
  "StoreArrayIndex",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  input_file="StoreArrayIndex.csv",
)
AssignCast = Relation(
  "AssignCast",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  input_file="AssignCast.csv",
)
AssignLocal = Relation(
  "AssignLocal",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  input_file="AssignLocal.csv",
)
AssignHeapAllocation = Relation(
  "AssignHeapAllocation",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  input_file="AssignHeapAllocation.csv",
)
ReturnVar = Relation(
  "ReturnVar",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="ReturnVar.csv",
)
StaticMethodInvocation = Relation(
  "StaticMethodInvocation",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  input_file="StaticMethodInvocation.csv",
)
VirtualMethodInvocation_SimpleName = Relation(
  "VirtualMethodInvocation_SimpleName",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="VirtualMethodInvocation_SimpleName.csv",
)
VirtualMethodInvocation_Descriptor = Relation(
  "VirtualMethodInvocation_Descriptor",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="VirtualMethodInvocation_Descriptor.csv",
)
isType = Relation("isType", 1, column_types=(int,))
isReferenceType = Relation("isReferenceType", 1, column_types=(int,))
isArrayType_IDB = Relation("isArrayType_IDB", 1, column_types=(int,))
isClassType_IDB = Relation("isClassType_IDB", 1, column_types=(int,))
isInterfaceType_IDB = Relation("isInterfaceType_IDB", 1, column_types=(int,))
MethodLookup = Relation(
  "MethodLookup",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)
MethodImplemented = Relation(
  "MethodImplemented",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)
DirectSubclass = Relation(
  "DirectSubclass",
  2,
  column_types=(
    int,
    int,
  ),
)
Subclass = Relation(
  "Subclass",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)
Superclass = Relation(
  "Superclass",
  2,
  column_types=(
    int,
    int,
  ),
)
Superinterface = Relation(
  "Superinterface",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)
SubtypeOf = Relation(
  "SubtypeOf",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)
SupertypeOf = Relation(
  "SupertypeOf",
  2,
  column_types=(
    int,
    int,
  ),
)
SubtypeOfDifferent = Relation(
  "SubtypeOfDifferent",
  2,
  column_types=(
    int,
    int,
  ),
)
MainMethodDeclaration = Relation("MainMethodDeclaration", 1, column_types=(int,), print_size=True)
ClassInitializer = Relation(
  "ClassInitializer",
  2,
  column_types=(
    int,
    int,
  ),
)
InitializedClass = Relation("InitializedClass", 1, column_types=(int,), print_size=True)
Assign = Relation(
  "Assign",
  2,
  column_types=(
    int,
    int,
  ),
)
VarPointsTo = Relation(
  "VarPointsTo",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
  index_type="SRDatalog::GPU::Device2LevelIndex",
)
InstanceFieldPointsTo = Relation(
  "InstanceFieldPointsTo",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  print_size=True,
  index_type="SRDatalog::GPU::Device2LevelIndex",
)
StaticFieldPointsTo = Relation(
  "StaticFieldPointsTo",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)
CallGraphEdge = Relation(
  "CallGraphEdge",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)
ArrayIndexPointsTo = Relation(
  "ArrayIndexPointsTo",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)
Reachable = Relation("Reachable", 1, column_types=(int,), print_size=True)
MethodInvocation_Base = Relation(
  "MethodInvocation_Base",
  2,
  column_types=(
    int,
    int,
  ),
)
CastTo = Relation(
  "CastTo",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)
HeapHelper = Relation(
  "HeapHelper",
  5,
  column_types=(
    int,
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)
HeapHelperNoThis = Relation(
  "HeapHelperNoThis",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)
VirtualMethodInvocation = Relation(
  "VirtualMethodInvocation",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)
HeapAllocSuperType = Relation(
  "HeapAllocSuperType",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)
ReachableInstruction = Relation("ReachableInstruction", 1, column_types=(int,), print_size=True)
ReachableSortedIndex = Relation(
  "ReachableSortedIndex",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)
ReachableLoadInstanceField = Relation(
  "ReachableLoadInstanceField",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  print_size=True,
)
ArrayTypeCompat = Relation(
  "ArrayTypeCompat",
  2,
  column_types=(
    int,
    int,
  ),
)
IsObjectArrayHeap = Relation("IsObjectArrayHeap", 1, column_types=(int,))
IsStringHeap = Relation("IsStringHeap", 1, column_types=(int,))
IsCastableToString = Relation("IsCastableToString", 1, column_types=(int,))

# ----- dataset_const declarations -----------------------------

DATASET_CONST_DECLS = {
  "ABSTRACT": "abstract",
  "PUBLIC": "public",
  "STATIC": "static",
  "MAIN": "main",
  "MAIN_DESCRIPTOR": "main_descriptor",
  "JAVA_LANG_OBJECT": "java_lang_Object",
  "JAVA_LANG_CLONEABLE": "java_lang_Cloneable",
  "JAVA_IO_SERIALIZABLE": "java_io_Serializable",
  "CLINIT": "clinit",
  "CLINIT_DESCRIPTOR": "clinit_descriptor",
  "CLASS_INIT_METHOD": "class_init_method",
  "REGISTER_NATIVES_METHOD": "register_natives_method",
  "DESIRED_ASSERTION_STATUS_METHOD": "desiredAssertionStatus_method",
  "JAVA_LANG_STRING_TYPE": "java_lang_String_type",
  "JAVA_LANG_CLASS_TYPE": "java_lang_Class_type",
  "JAVA_LANG_OBJECT_ARRAY": "java_lang_Object_array",
}

# ----- Rules: DoopDB -----


def build_doopdb_program() -> Program:
  ABSTRACT = Var("ABSTRACT")
  CLINIT = Var("CLINIT")
  CLINIT_DESCRIPTOR = Var("CLINIT_DESCRIPTOR")
  JAVA_IO_SERIALIZABLE = Var("JAVA_IO_SERIALIZABLE")
  JAVA_LANG_CLONEABLE = Var("JAVA_LANG_CLONEABLE")
  JAVA_LANG_OBJECT = Var("JAVA_LANG_OBJECT")
  JAVA_LANG_OBJECT_ARRAY = Var("JAVA_LANG_OBJECT_ARRAY")
  JAVA_LANG_STRING_TYPE = Var("JAVA_LANG_STRING_TYPE")
  MAIN = Var("MAIN")
  MAIN_DESCRIPTOR = Var("MAIN_DESCRIPTOR")
  PUBLIC = Var("PUBLIC")
  STATIC = Var("STATIC")
  a = Var("a")
  actual = Var("actual")
  arr = Var("arr")
  b = Var("b")
  base = Var("base")
  basecomptype = Var("basecomptype")
  baseheap = Var("baseheap")
  baseheaptype = Var("baseheaptype")
  c = Var("c")
  casttype = Var("casttype")
  class_ = Var("class")
  classD = Var("classD")
  classOrIface = Var("classOrIface")
  clinit = Var("clinit")
  comptype = Var("comptype")
  descriptor = Var("descriptor")
  fld = Var("fld")
  formal = Var("formal")
  frm = Var("frm")
  heap = Var("heap")
  heaptype = Var("heaptype")
  idx = Var("idx")
  iface = Var("iface")
  inMeth = Var("inMeth")
  inmeth = Var("inmeth")
  inv = Var("inv")
  invocation = Var("invocation")
  k = Var("k")
  local = Var("local")
  meth = Var("meth")
  mtype = Var("mtype")
  ret = Var("ret")
  s = Var("s")
  sc = Var("sc")
  sig = Var("sig")
  simplename = Var("simplename")
  superclass = Var("superclass")
  superiface = Var("superiface")
  supertype = Var("supertype")
  t = Var("t")
  tc = Var("tc")
  thisP = Var("thisP")
  to = Var("to")
  toMeth = Var("toMeth")
  tometh = Var("tometh")
  vr = Var("vr")
  vtype = Var("vtype")

  # dataset_consts appear as Var(UPPER_NAME); substitute via resolve_program_consts.
  ABSTRACT = Var("ABSTRACT")
  CLASS_INIT_METHOD = Var("CLASS_INIT_METHOD")
  CLINIT = Var("CLINIT")
  CLINIT_DESCRIPTOR = Var("CLINIT_DESCRIPTOR")
  DESIRED_ASSERTION_STATUS_METHOD = Var("DESIRED_ASSERTION_STATUS_METHOD")
  JAVA_IO_SERIALIZABLE = Var("JAVA_IO_SERIALIZABLE")
  JAVA_LANG_CLASS_TYPE = Var("JAVA_LANG_CLASS_TYPE")
  JAVA_LANG_CLONEABLE = Var("JAVA_LANG_CLONEABLE")
  JAVA_LANG_OBJECT = Var("JAVA_LANG_OBJECT")
  JAVA_LANG_OBJECT_ARRAY = Var("JAVA_LANG_OBJECT_ARRAY")
  JAVA_LANG_STRING_TYPE = Var("JAVA_LANG_STRING_TYPE")
  MAIN = Var("MAIN")
  MAIN_DESCRIPTOR = Var("MAIN_DESCRIPTOR")
  PUBLIC = Var("PUBLIC")
  REGISTER_NATIVES_METHOD = Var("REGISTER_NATIVES_METHOD")
  STATIC = Var("STATIC")

  return Program(
    relations=[
      DirectSuperclass,
      DirectSuperinterface,
      MainClass,
      FormalParam,
      ComponentType,
      AssignReturnValue,
      ActualParam,
      Method_Modifier,
      Var_Type,
      HeapAllocation_Type,
      Method_Descriptor,
      ClassType,
      ArrayType,
      InterfaceType,
      Var_DeclaringMethod,
      ApplicationClass,
      ThisVar,
      Field_DeclaringType,
      Method_SimpleName,
      Method_DeclaringType,
      Instruction_Method,
      isVirtualMethodInvocation_Insn,
      isStaticMethodInvocation_Insn,
      MethodInvocation_Method,
      VirtualMethodInvocation_Base,
      SpecialMethodInvocation_Base,
      LoadInstanceField,
      StoreInstanceField,
      LoadStaticField,
      StoreStaticField,
      LoadArrayIndex,
      StoreArrayIndex,
      AssignCast,
      AssignLocal,
      AssignHeapAllocation,
      ReturnVar,
      StaticMethodInvocation,
      VirtualMethodInvocation_SimpleName,
      VirtualMethodInvocation_Descriptor,
      isType,
      isReferenceType,
      isArrayType_IDB,
      isClassType_IDB,
      isInterfaceType_IDB,
      MethodLookup,
      MethodImplemented,
      DirectSubclass,
      Subclass,
      Superclass,
      Superinterface,
      SubtypeOf,
      SupertypeOf,
      SubtypeOfDifferent,
      MainMethodDeclaration,
      ClassInitializer,
      InitializedClass,
      Assign,
      VarPointsTo,
      InstanceFieldPointsTo,
      StaticFieldPointsTo,
      CallGraphEdge,
      ArrayIndexPointsTo,
      Reachable,
      MethodInvocation_Base,
      CastTo,
      HeapHelper,
      HeapHelperNoThis,
      VirtualMethodInvocation,
      HeapAllocSuperType,
      ReachableInstruction,
      ReachableSortedIndex,
      ReachableLoadInstanceField,
      ArrayTypeCompat,
      IsObjectArrayHeap,
      IsStringHeap,
      IsCastableToString,
    ],
    rules=[
      (isType(class_) <= ClassType(class_)).named('isType_Class'),
      (isReferenceType(class_) <= ClassType(class_)).named('isRefType_Class'),
      (isClassType_IDB(class_) <= ClassType(class_)).named('isClassType'),
      (isType(arr) <= ArrayType(arr)).named('isType_Array'),
      (isReferenceType(arr) <= ArrayType(arr)).named('isRefType_Array'),
      (isArrayType_IDB(arr) <= ArrayType(arr)).named('isArrayType'),
      (isType(iface) <= InterfaceType(iface)).named('isType_Interface'),
      (isReferenceType(iface) <= InterfaceType(iface)).named('isRefType_Interface'),
      (isInterfaceType_IDB(iface) <= InterfaceType(iface)).named('isInterfaceType'),
      (isType(t) <= ApplicationClass(t)).named('isType_AppClass'),
      (isReferenceType(t) <= ApplicationClass(t)).named('isRefType_AppClass'),
      (DirectSubclass(a, c) <= DirectSuperclass(a, c)).named('DirectSubclass'),
      (Superinterface(k, c) <= DirectSuperinterface(c, k)).named('Superinterface_Base'),
      (Superinterface(k, c) <= Superinterface(k, b) & DirectSuperinterface(c, b)).named(
        'Superinterface_Trans'
      ),
      (Superinterface(k, c) <= Superinterface(k, b) & DirectSuperclass(c, b)).named(
        'Superinterface_Inherit'
      ),
      (
        MethodImplemented(simplename, descriptor, mtype, meth)
        <= Method_SimpleName(meth, simplename)
        & Method_Descriptor(meth, descriptor)
        & Method_DeclaringType(meth, mtype)
        & ~Method_Modifier(ABSTRACT, meth)
      ).named('MethodImplemented'),
      (
        MainMethodDeclaration(meth)
        <= MainClass(mtype)
        & Method_DeclaringType(meth, mtype)
        & Method_SimpleName(meth, MAIN)
        & Method_Descriptor(meth, MAIN_DESCRIPTOR)
        & Method_Modifier(PUBLIC, meth)
        & Method_Modifier(STATIC, meth)
        & Filter(
          ('meth',),
          "return meth != CLASS_INIT_METHOD && meth != REGISTER_NATIVES_METHOD && meth != DESIRED_ASSERTION_STATUS_METHOD;",
        )
      ).named('MainMethodDecl_Base'),
      (MethodInvocation_Base(inv, base) <= VirtualMethodInvocation_Base(inv, base)).named(
        'MIBase_Virtual'
      ),
      (MethodInvocation_Base(inv, base) <= SpecialMethodInvocation_Base(inv, base)).named(
        'MIBase_Special'
      ),
      (Subclass(c, a) <= DirectSubclass(a, c)).named('Subclass_Base'),
      (Subclass(c, a) <= Subclass(b, a) & DirectSubclass(b, c)).named('Subclass_Trans'),
      (Superclass(c, a) <= Subclass(a, c)).named('Superclass'),
      (SubtypeOf(s, s) <= isClassType_IDB(s)).named('SubtypeOf_Refl_Class'),
      (SubtypeOf(t, t) <= isType(t)).named('SubtypeOf_Refl_Type'),
      (SubtypeOf(s, s) <= isInterfaceType_IDB(s)).named('SubtypeOf_Refl_Interface'),
      (SubtypeOf(s, t) <= Subclass(t, s)).named('SubtypeOf_Subclass'),
      (SubtypeOf(s, t) <= isClassType_IDB(s) & Superinterface(t, s)).named('SubtypeOf_Class_Iface'),
      (SubtypeOf(s, t) <= isInterfaceType_IDB(s) & Superinterface(t, s)).named(
        'SubtypeOf_Iface_Iface'
      ),
      (SubtypeOf(s, JAVA_LANG_OBJECT) <= isInterfaceType_IDB(s)).named('SubtypeOf_Iface_Object'),
      (SubtypeOf(s, JAVA_LANG_OBJECT) <= isArrayType_IDB(s)).named('SubtypeOf_Array_Object'),
      (
        SubtypeOf(s, t)
        <= SubtypeOf(sc, tc)
        & ComponentType(s, sc)
        & ComponentType(t, tc)
        & isReferenceType(sc)
        & isReferenceType(tc)
      ).named('SubtypeOf_Array'),
      (SubtypeOf(s, JAVA_LANG_CLONEABLE) <= isArrayType_IDB(s)).named('SubtypeOf_Array_Cloneable'),
      (SubtypeOf(s, JAVA_IO_SERIALIZABLE) <= isArrayType_IDB(s)).named(
        'SubtypeOf_Array_Serializable'
      ),
      (SupertypeOf(s, t) <= SubtypeOf(t, s)).named('SupertypeOf'),
      (
        SubtypeOfDifferent(s, t)
        <= SubtypeOf(s, t)
        & Filter(
          (
            's',
            't',
          ),
          "return s != t;",
        )
      ).named('SubtypeOfDiff_Base'),
      (
        MethodLookup(simplename, descriptor, mtype, meth)
        <= MethodImplemented(simplename, descriptor, mtype, meth)
      ).named('MethodLookup_Base'),
      (
        MethodLookup(simplename, descriptor, mtype, meth)
        <= DirectSuperclass(mtype, supertype)
        & MethodLookup(simplename, descriptor, supertype, meth)
        & ~MethodImplemented(simplename, descriptor, mtype, Var("_"))
      ).named('MethodLookup_Super'),
      (
        MethodLookup(simplename, descriptor, mtype, meth)
        <= DirectSuperinterface(mtype, supertype)
        & MethodLookup(simplename, descriptor, supertype, meth)
        & ~MethodImplemented(simplename, descriptor, mtype, Var("_"))
      ).named('MethodLookup_Iface'),
      (
        ClassInitializer(mtype, meth) <= MethodImplemented(CLINIT, CLINIT_DESCRIPTOR, mtype, meth)
      ).named('ClassInitializer'),
      (
        CastTo(frm, to, inmeth, heap)
        <= AssignCast(casttype, frm, to, inmeth)
        & SupertypeOf(casttype, heaptype)
        & HeapAllocation_Type(heap, heaptype)
        & Filter(('heaptype',), "return heaptype != JAVA_LANG_STRING_TYPE;")
      )
      .named('Precompute0')
      .with_plan(
        var_order=['casttype', 'heaptype', 'frm', 'to', 'inmeth', 'heap'], block_group=True
      ),
      (
        CastTo(frm, to, inmeth, heap)
        <= AssignCast(casttype, frm, to, inmeth) & IsCastableToString(casttype) & IsStringHeap(heap)
      )
      .named('Precompute0_String')
      .with_plan(var_order=['casttype', 'frm', 'to', 'inmeth', 'heap'], block_group=True),
      (
        HeapHelper(simplename, descriptor, toMeth, thisP, heap)
        <= MethodLookup(simplename, descriptor, heaptype, toMeth)
        & HeapAllocation_Type(heap, heaptype)
        & ThisVar(toMeth, thisP)
      )
      .named('Precompute1')
      .with_plan(
        var_order=['heaptype', 'toMeth', 'simplename', 'descriptor', 'heap', 'thisP'],
        block_group=True,
      ),
      (
        HeapHelperNoThis(simplename, descriptor, toMeth, heap)
        <= MethodLookup(simplename, descriptor, heaptype, toMeth)
        & HeapAllocation_Type(heap, heaptype)
      )
      .named('Precompute1b')
      .with_plan(
        var_order=['heaptype', 'toMeth', 'simplename', 'descriptor', 'heap'], block_group=True
      ),
      (
        VirtualMethodInvocation(invocation, base, simplename, descriptor)
        <= VirtualMethodInvocation_Base(invocation, base)
        & VirtualMethodInvocation_SimpleName(invocation, simplename)
        & VirtualMethodInvocation_Descriptor(invocation, descriptor)
      ).named('Precompute2'),
      (
        HeapAllocSuperType(heap, baseheap)
        <= HeapAllocation_Type(heap, heaptype)
        & HeapAllocation_Type(baseheap, baseheaptype)
        & ComponentType(baseheaptype, comptype)
        & SupertypeOf(comptype, heaptype)
        & Filter(('baseheaptype',), "return baseheaptype != JAVA_LANG_OBJECT_ARRAY;")
      )
      .named('HeapAllocHelper')
      .with_plan(
        var_order=['heaptype', 'comptype', 'baseheaptype', 'heap', 'baseheap'], block_group=True
      ),
      (IsObjectArrayHeap(baseheap) <= HeapAllocation_Type(baseheap, JAVA_LANG_OBJECT_ARRAY)).named(
        'IsObjectArrayHeap_rule'
      ),
      (IsStringHeap(heap) <= HeapAllocation_Type(heap, JAVA_LANG_STRING_TYPE)).named(
        'IsStringHeap_rule'
      ),
      (IsCastableToString(casttype) <= SupertypeOf(casttype, JAVA_LANG_STRING_TYPE)).named(
        'IsCastableToString_rule'
      ),
      (
        ArrayTypeCompat(baseheap, vtype)
        <= HeapAllocation_Type(baseheap, baseheaptype)
        & ComponentType(baseheaptype, basecomptype)
        & SupertypeOf(vtype, basecomptype)
      )
      .named('ArrayTypeCompatPrecompute')
      .with_plan(var_order=['baseheaptype', 'basecomptype', 'vtype', 'baseheap'], block_group=True),
      (
        InitializedClass(superclass)
        <= InitializedClass(classD) & DirectSuperclass(classD, superclass)
      ).named('InitClass_Super'),
      (
        InitializedClass(superiface)
        <= InitializedClass(classOrIface) & DirectSuperinterface(classOrIface, superiface)
      ).named('InitClass_Iface'),
      (
        InitializedClass(classD) <= MainMethodDeclaration(meth) & Method_DeclaringType(meth, classD)
      ).named('InitClass_Main'),
      (
        InitializedClass(classD)
        <= Reachable(inmeth)
        & AssignHeapAllocation(heap, Var("_"), inmeth)
        & HeapAllocation_Type(heap, classD)
      ).named('InitClass_Heap'),
      (
        InitializedClass(classD)
        <= Reachable(inmeth)
        & Instruction_Method(invocation, inmeth)
        & MethodInvocation_Method(invocation, sig)
        & Method_DeclaringType(sig, classD)
        & isStaticMethodInvocation_Insn(invocation)
      ).named('InitClass_StaticInvoke'),
      (
        InitializedClass(classOrIface)
        <= Reachable(inmeth)
        & StoreStaticField(Var("_"), sig, inmeth)
        & Field_DeclaringType(sig, classOrIface)
      ).named('InitClass_StoreStatic'),
      (
        InitializedClass(classOrIface)
        <= Reachable(inmeth)
        & LoadStaticField(sig, Var("_"), inmeth)
        & Field_DeclaringType(sig, classOrIface)
      ).named('InitClass_LoadStatic'),
      (Reachable(clinit) <= InitializedClass(classD) & ClassInitializer(classD, clinit)).named(
        'Reachable_Clinit'
      ),
      (Reachable(meth) <= MainMethodDeclaration(meth)).named('Reachable_Main'),
      (
        (Reachable(tometh) | CallGraphEdge(invocation, tometh))
        <= Reachable(inmeth) & StaticMethodInvocation(invocation, tometh, inmeth)
      ).named('Static_MultiHead'),
      (
        Assign(actual, formal)
        <= CallGraphEdge(invocation, meth)
        & FormalParam(idx, meth, formal)
        & ActualParam(idx, invocation, actual)
      ).named('Assign_Param'),
      (
        Assign(ret, local)
        <= CallGraphEdge(invocation, meth)
        & ReturnVar(ret, meth)
        & AssignReturnValue(invocation, local)
      ).named('Assign_Return'),
      (VarPointsTo(heap, vr) <= AssignHeapAllocation(heap, vr, inMeth) & Reachable(inMeth)).named(
        'VPT_HeapAlloc'
      ),
      (VarPointsTo(heap, to) <= Assign(frm, to) & VarPointsTo(heap, frm)).named('VPT_Assign'),
      (
        VarPointsTo(heap, to)
        <= Reachable(inmeth) & AssignLocal(frm, to, inmeth) & VarPointsTo(heap, frm)
      ).named('VPT_Local'),
      (
        VarPointsTo(heap, to)
        <= Reachable(inmeth) & CastTo(frm, to, inmeth, heap) & VarPointsTo(heap, frm)
      )
      .named('VPT_Cast')
      .with_plan(delta=2, var_order=['heap', 'frm', 'inmeth', 'to']),
      (
        ReachableLoadInstanceField(base, sig, to)
        <= Reachable(inmeth) & LoadInstanceField(base, sig, to, inmeth)
      ).named('ReachableLoadInstanceField_rule'),
      (
        VarPointsTo(heap, to)
        <= ReachableLoadInstanceField(base, sig, to)
        & VarPointsTo(baseheap, base)
        & InstanceFieldPointsTo(heap, sig, baseheap)
      )
      .named('VPT_LoadField')
      .with_plan(delta=2, var_order=['baseheap', 'sig', 'base', 'heap', 'to'], block_group=True),
      (
        VarPointsTo(heap, to)
        <= Reachable(inmeth) & LoadStaticField(fld, to, inmeth) & StaticFieldPointsTo(heap, fld)
      ).named('VPT_LoadStatic'),
      (
        VarPointsTo(heap, to)
        <= Reachable(inmeth)
        & LoadArrayIndex(base, to, inmeth)
        & VarPointsTo(baseheap, base)
        & ArrayIndexPointsTo(baseheap, heap)
        & Var_Type(to, vtype)
        & ArrayTypeCompat(baseheap, vtype)
      ).named('VPT_LoadArray'),
      (
        ReachableInstruction(invocation)
        <= Reachable(inmeth) & Instruction_Method(invocation, inmeth)
      ).named('ReachableInstructionHelper'),
      (
        (VarPointsTo(heap, thisP) | CallGraphEdge(invocation, toMeth) | Reachable(toMeth))
        <= ReachableInstruction(invocation)
        & VirtualMethodInvocation(invocation, base, simplename, descriptor)
        & VarPointsTo(heap, base)
        & HeapHelper(simplename, descriptor, toMeth, thisP, heap)
      )
      .named('VirtualDispatch_MultiHead')
      .with_plan(
        delta=2,
        var_order=['base', 'heap', 'invocation', 'simplename', 'descriptor', 'toMeth', 'thisP'],
        block_group=True,
      ),
      (
        (Reachable(toMeth) | CallGraphEdge(invocation, toMeth))
        <= ReachableInstruction(invocation)
        & VirtualMethodInvocation(invocation, base, simplename, descriptor)
        & VarPointsTo(heap, base)
        & HeapHelperNoThis(simplename, descriptor, toMeth, heap)
      )
      .named('Reachable_CGE_Virtual')
      .with_plan(
        delta=2,
        var_order=['base', 'heap', 'invocation', 'simplename', 'descriptor', 'toMeth'],
        block_group=True,
      ),
      (
        (VarPointsTo(heap, thisP) | CallGraphEdge(invocation, tometh) | Reachable(tometh))
        <= ReachableInstruction(invocation)
        & SpecialMethodInvocation_Base(invocation, base)
        & VarPointsTo(heap, base)
        & MethodInvocation_Method(invocation, tometh)
        & ThisVar(tometh, thisP)
      ).named('SpecialDispatch_MultiHead'),
      (
        InstanceFieldPointsTo(heap, fld, baseheap)
        <= Reachable(inmeth)
        & StoreInstanceField(frm, base, fld, inmeth)
        & VarPointsTo(heap, frm)
        & VarPointsTo(baseheap, base)
      )
      .named('IFPT_Store')
      .with_plan(delta=2, var_order=['frm', 'inmeth', 'base', 'heap', 'fld', 'baseheap'])
      .with_plan(delta=3, var_order=['base', 'inmeth', 'frm', 'heap', 'fld', 'baseheap']),
      (
        StaticFieldPointsTo(heap, fld)
        <= Reachable(inmeth) & StoreStaticField(frm, fld, inmeth) & VarPointsTo(heap, frm)
      ).named('SFPT_Store'),
      (
        ReachableSortedIndex(frm, base) <= Reachable(inmeth) & StoreArrayIndex(frm, base, inmeth)
      ).named('ReachableSortedIndex_rule'),
      (
        ArrayIndexPointsTo(baseheap, heap)
        <= ReachableSortedIndex(frm, base)
        & VarPointsTo(baseheap, base)
        & IsObjectArrayHeap(baseheap)
        & VarPointsTo(heap, frm)
      )
      .named('AIPT_Store_ObjectArray')
      .with_plan(delta=1, var_order=['base', 'frm', 'baseheap', 'heap'], block_group=True),
      (
        ArrayIndexPointsTo(baseheap, heap)
        <= ReachableSortedIndex(frm, base)
        & VarPointsTo(baseheap, base)
        & VarPointsTo(heap, frm)
        & HeapAllocSuperType(heap, baseheap)
      )
      .named('AIPT_Store')
      .with_plan(delta=1, var_order=['base', 'frm', 'baseheap', 'heap'], block_group=True),
    ],
  )


def build_doopdb(meta_json_path: str) -> tuple[Program, dict[str, int]]:
  """Convenience: build the program, load dataset_consts, substitute."""
  consts = load_meta(meta_json_path, DATASET_CONST_DECLS)
  return resolve_program_consts(build_doopdb_program(), consts), consts
