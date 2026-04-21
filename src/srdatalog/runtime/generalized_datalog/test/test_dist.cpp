#include <mpi.h>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <vector>
#include <cstdint> // uint8_t
#include <cstddef> // std::byte

#include "query.h"
#include "relation_col.h"
#include "dist_relation_io.h"


// semiring
struct NatSemiring
{
    using value_type = std::uint64_t;
    static constexpr value_type add(const value_type &a, const value_type &b) { return a + b; }
    static constexpr value_type mul(const value_type &a, const value_type &b) { return a * b; }
    static constexpr value_type zero() { return 0ull; }
    static constexpr value_type one() { return 1ull; }
};

int main(int argc, char **argv)
{   
    using namespace SRDatalog;
    MPI_Init(&argc, &argv);

    int rk = 0, nprocs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rk);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // check if enough ranks
    if (nprocs < 2 && rk == 0)
    {
        std::cerr << "Warning: distribution tests (3,4) need >= 2 ranks.\n";
    }

    // define the relation on all ranks
    // but only rank 0 will actually populate it
    // using rel_atype = Relation<NatSemiring, int, std::uint8_t, int>; rel_atype rela;
    DEFINE_RELATION(rel, NatSemiring, int, int, int);
    rel.set_column_names({"id", "tag", "val"});

    // variables for testing pack function
    std::vector<std::byte> packed_buf; // we'll reuse it in step 2
    std::size_t packed_nbytes = 0;

    if (rk == 0)
    {
        // insert 8 rows of data
        // (id, tag, val)
        FACT(rel, NatSemiring::one(), 1, 1, 10);
        FACT(rel, NatSemiring::one(), 2, 2, 20);
        FACT(rel, 3, 3, 2, 30);
        FACT(rel, 4, 4, 3, 40);
        FACT(rel, NatSemiring::one(), 5, 3, 50);
        FACT(rel, NatSemiring::one(), 6, 4, 60);
        FACT(rel, 7, 7, 4, 70);
        FACT(rel, 8, 8, 5, 80);

        std::cout << "[1][rk0] original relation size = " << rel.size() << "\n";
        rel.head(20, std::cout);

        // pick a subset to pack
        std::vector<std::size_t> row_ids = {0, 2, 4, 6}; // 4 rows
        pack_relation_rows(rel, row_ids, packed_buf);
        packed_nbytes = packed_buf.size();

        std::cout << "[1][rk0] packed " << row_ids.size()
                  << " rows into " << packed_nbytes << " bytes\n";
    }

    // still on the rank 0, we gonna unpack the packed buffer
    // into another buffer and check if it's correct
    if (rk == 0)
    {
        DEFINE_RELATION(rel_unpacked, NatSemiring, int, int, int);
        rel_unpacked.set_column_names({"id", "tag", "val"});

        unpack_append_relation(packed_buf, rel_unpacked);

        std::cout << "[2][rk0] after unpack, relation size = "
                  << rel_unpacked.size() << "\n";
        rel_unpacked.head(20, std::cout);

        if (rel_unpacked.size() != 4)
        {
            std::cerr << "[2][rk0] ERROR: expected 4 rows after unpack, got "
                      << rel_unpacked.size() << "\n";
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Test the mpi distribution function of version using all ranks as sources

    if (nprocs >= 2)
    {
        rel.clear();
        rel.set_column_names({"id", "tag", "val"});

        // create data on each rank
        int base_id = rk * 100 + 1;
        FACT(rel, NatSemiring::one(), base_id + 0, (rk + 1), 10 * (rk + 1));
        FACT(rel, NatSemiring::one(), base_id + 1, (rk + 2), 10 * (rk + 1) + 5);
        FACT(rel, NatSemiring::one(), base_id + 2, (rk + 3), 10 * (rk + 1) + 10);
        FACT(rel, NatSemiring::one(), base_id + 3, (rk + 4), 10 * (rk + 1) + 15);

        std::cout << "[3][rk" << rk << "] BEFORE distribute_relation, size = "
                  << rel.size() << " rows\n";
        rel.head(10, std::cout);

        // choose column 0 ("id") as distribution key
        SRDatalog::distribute_relation(rel, /*column_idx=*/0, MPI_COMM_WORLD);

        // after this, each rank should hold only the rows that map to it
        // (according to whatever rule distribute_relation uses)
        std::cout << "[3][rk" << rk << "] AFTER distribute_relation, size = "
                  << rel.size() << " rows\n";
        rel.head(20, std::cout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Test the mpi distribution function of version using rank 0 as source
    // distribute the rank 0 data to all other ranks
    if (nprocs >= 2)
    {
        if (rk == 0)
        {
            rel.clear();
            // rebuild the same 8 rows
            FACT(rel, NatSemiring::one(), 1, 1, 10);
            FACT(rel, NatSemiring::one(), 2, 2, 20);
            FACT(rel, 3, 3, 2, 30);
            FACT(rel, 4, 4, 3, 40);
            FACT(rel, NatSemiring::one(), 5, 3, 50);
            FACT(rel, NatSemiring::one(), 6, 4, 60);
            FACT(rel, 7, 7, 4, 70);
            FACT(rel, 8, 8, 5, 80);
        }
        else
        {
            rel.clear();
            rel.set_column_names({"id", "tag", "val"});
        }

        MPI_Barrier(MPI_COMM_WORLD);

        SRDatalog::distribute_relation_rk0(rel, /*column_idx=*/0, MPI_COMM_WORLD);

        std::cout << "[4][rk" << rk << "] after distribute_relation_rk0, size = "
                  << rel.size() << "\n";
    }

    MPI_Finalize();
    return 0;
}
