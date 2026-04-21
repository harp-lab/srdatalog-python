
#include <tuple>
#include <type_traits>

struct MySchema {
  struct relation_type {
    relation_type(int) {}
  };
};

template <typename Schema>
using SchemaToRelation = typename Schema::relation_type;

template <typename DB, template <typename> typename RelationMapper = SchemaToRelation>
struct TestStruct {
  RelationMapper<DB> field;
  TestStruct() : field(0) {}
};

int main() {
  TestStruct<MySchema> t;
  return 0;
}
