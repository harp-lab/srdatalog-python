#include <cstdint>
#include <vector>
#include <tuple>
#include <string>
#include <type_traits>
#include <cstring>  // std::memcpy 
#include <cstddef>
#include <mpi.h>
#include "relation_col.h"

namespace SRDatalog {
/***
 * buffer composition:
 * [# of rows (size_t) | column1: # of rows * sizeof(column1 datatype) | 
 *              column2 # of rows * sizeof(column2 datatype)| # of rows * sizeof(annotation) ]
 */
template<Semiring SR, ColumnElementTuple AttrTuple>
void pack_relation_rows(
    const SRDatalog::Relation<SR, AttrTuple>& rel,
    const std::vector<std::size_t>& row_ids,
    std::vector<std::byte>& buffer
) {
    constexpr std::size_t ARITY = std::tuple_size_v<AttrTuple>;
    using AnnotationVal = typename SR::value_type;

    const std::uint32_t nrows = static_cast<std::uint32_t>(row_ids.size());

    // calculating the total size
    // size for number of rows, columns' size and the semiring size 
    std::size_t total_size = sizeof(std::uint32_t);
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        (( total_size += nrows * sizeof(std::tuple_element_t<I, AttrTuple>) ), ...);
    }(std::make_index_sequence<ARITY>{});   
    total_size += nrows * sizeof(AnnotationVal);       

    buffer.resize(total_size);
    std::byte* ptr = buffer.data();

    // write in the size of nrows
    std::memcpy(ptr, &nrows, sizeof(nrows));
    ptr += sizeof(nrows);

    // write out each column 
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        (([&]{
            using ColT = std::tuple_element_t<I, AttrTuple>;
            const auto& col = rel.template column<I>();
            for (std::size_t k = 0; k < row_ids.size(); ++k) {
                const ColT& v = col[row_ids[k]];
                std::memcpy(ptr, &v, sizeof(ColT));
                ptr += sizeof(ColT);
            }
        }()), ...);
    }(std::make_index_sequence<ARITY>{});

    // write in the content of semiring column 
    const auto& ann = rel.provenance();   
    for (std::size_t k = 0; k < row_ids.size(); ++k) {
        const AnnotationVal& a = ann[row_ids[k]];
        std::memcpy(ptr, &a, sizeof(AnnotationVal));
        ptr += sizeof(AnnotationVal);
    }
}



#include <cstring> // for std::memcpy

template<Semiring SR, ColumnElementTuple AttrTuple>
void unpack_append_relation(
    const std::vector<std::byte>& buffer,
    SRDatalog::Relation<SR, AttrTuple>& rel
) {
    constexpr std::size_t ARITY = std::tuple_size_v<AttrTuple>;
    using AnnotationVal = typename SR::value_type;

    const std::byte* ptr = buffer.data();

    // read number of rows
    std::uint32_t nrows = 0;
    std::memcpy(&nrows, ptr, sizeof(nrows));
    ptr += sizeof(nrows);

    // figure out where to append
    const std::size_t old_rows = rel.size();
    const std::size_t new_rows = old_rows + nrows;

    // resize all attribute columns and annotation column(mirror of what we read)
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        // resize each attribute column
        ((
            rel.template column<I>().reserve(new_rows)
        ), ...);
    }(std::make_index_sequence<ARITY>{});

    // resize annotation vector
    auto& ann = rel.provenance();
    ann.resize(new_rows);

    // read each column, in the same order we packed
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        (([&]{
            using ColT = std::tuple_element_t<I, AttrTuple>;
            auto& col = rel.template column<I>();
            for (std::size_t k = 0; k < nrows; ++k) {
                ColT v;
                std::memcpy(&v, ptr, sizeof(ColT));
                ptr += sizeof(ColT);
                col.push_back(v);   // append position
            }
        }()), ...);
    }(std::make_index_sequence<ARITY>{});

    // read annotation column
    for (std::size_t k = 0; k < nrows; ++k) {
        AnnotationVal v;
        std::memcpy(&v, ptr, sizeof(AnnotationVal));
        ptr += sizeof(AnnotationVal);
        ann.push_back(v);
    } 
}
  
/// @Brief: distribute the relation to all ranks based on the value of the indexing column
/// @Param: rel: the relation to distribute
/// @Param: column_idx: the index of the column to distribute the relation based on
/// @Param: comm: the MPI communicator to use, default to MPI_COMM_WORLD
/// @Return: a new relation that is distributed to all ranks
template<Semiring SR, ColumnElementTuple AttrTuple> 
void distribute_relation(
    const Relation<SR, AttrTuple>& rel,
    std::size_t column_idx,
    MPI_Comm comm
){
    using Rel = SRDatalog::Relation<SR, AttrTuple>;
    constexpr std::size_t ARITY = std::tuple_size_v<AttrTuple>;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // get the size of the relation, more specifically, the number of local rows
    const std::size_t nlocal = rel.size();
    // each rank have what rows to send 
    std::vector<std::vector<std::size_t>> rows_for_rank(size);

    for (std::size_t r = 0; r < nlocal; ++r) {

        // this is a lambda function
        // (row id and column index) -> rank destination
        std::size_t owner = 0;
        bool found = false;

        [&]<std::size_t... I>(std::index_sequence<I...>) {
            // expand I columns 
            (
                (column_idx == I
                 ? ( [&]{
                        using ColT = std::tuple_element_t<I, AttrTuple>;
                        // getting the refernece for the indexing column
                        const auto& col = rel.template column<I>();
                        // compute the owner rank based on the value of the indexing column
                        owner = static_cast<std::size_t>(col[r]) % static_cast<std::size_t>(size);
                        found = true;
                    }(), 0 )
                 : 0),
                ...
            );
        }(std::make_index_sequence<ARITY>{});

        if (!found) {
            // the row doesn't belong to any rank, assign it to rank 0
            owner = 0;
        }

        rows_for_rank[owner].push_back(r);
    }

    // pack each ranks data to send into buffer and flatten the buffer 
    std::vector<int> send_sizes(size);
    std::vector<std::byte> send_linear;  // store data to send here
    send_linear.reserve(1024);           // reserve some space for the buffer

    std::vector<int> sdispls(size);
    for (int p = 0; p < size; ++p) {
        std::vector<std::byte> tmp;
        pack_relation_rows(rel, rows_for_rank[p], tmp);
        send_sizes[p] = static_cast<int>(tmp.size());
        sdispls[p] = static_cast<int>(send_linear.size());
        // attached the data into the buffer 
        send_linear.insert(send_linear.end(), tmp.begin(), tmp.end());
    }

    // exchange the size of data to send for each rank 
    std::vector<int> recv_sizes(size);
    MPI_Alltoall(send_sizes.data(), 1, MPI_INT,
                 recv_sizes.data(), 1, MPI_INT,
                 comm);

    // based on the recv_sizes, calculate the recv_displs
    std::vector<int> rdispls(size);
    int recv_total = 0;
    for (int p = 0; p < size; ++p) {
        rdispls[p] = recv_total;
        recv_total += recv_sizes[p];
    }
    std::vector<std::byte> recv_linear(recv_total);

    // then finally each rank will send indefinite amount of data to other ranks 
    MPI_Alltoallv(
        send_linear.data(), send_sizes.data(), sdispls.data(), MPI_BYTE,
        recv_linear.data(), recv_sizes.data(), rdispls.data(), MPI_BYTE,
        comm
    );

    // each rank will received information and then 
    // unpack the inforamtion into the realtions 
    Rel local_new;
    for (int p = 0; p < size; ++p) {
        if (recv_sizes[p] == 0) continue;
        std::vector<std::byte> one_msg(
            recv_linear.begin() + rdispls[p],
            recv_linear.begin() + rdispls[p] + recv_sizes[p]
        );
        unpack_append_relation(one_msg, local_new);
    }

    //return local_new;
}

// @Brief: distribute the relation to all ranks based on the value of the indexing column
// @Param: rel: the relation to distribute
// @Param: column_idx: the index of the column to distribute the relation based on
// @Param: comm: the MPI communicator to use, default to MPI_COMM_WORLD
// @Return: None
// @Detail: This is a specialization of distribute_relation for the case where the indexing column is the first column.
//          In this case, we can directly distribute the relation to all ranks without any additional computation.
template<Semiring SR, ColumnElementTuple AttrTuple> 
void distribute_relation_rk0(
    const Relation<SR, AttrTuple>& rel,
    std::size_t column_idx,
    MPI_Comm comm
) {
    using Rel = SRDatalog::Relation<SR, AttrTuple>;
    constexpr std::size_t ARITY = std::tuple_size_v<AttrTuple>;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // each rank 
    Rel local_new;

    // calculating the rank destination of for each row 
    // and put them into corresponding rank's rows_for_rank
    std::vector<int> send_sizes;   // bytes_2_receive for each rank 
    std::vector<int> sdispls;      // offsets for each rank's data in the send_linear buffer
    std::vector<std::byte> send_linear;

    if (rank == 0) {
        const std::size_t nlocal = rel.size();
        std::vector<std::vector<std::size_t>> rows_for_rank(size);

        // rank 0 iterate over the rows 
        for (std::size_t r = 0; r < nlocal; ++r) {
            std::size_t owner = 0;
            bool found = false;

            // using index_sequence to expand I columns
            [&]<std::size_t... I>(std::index_sequence<I...>) {
                (
                    (column_idx == I
                     ? ( [&]{
                            using ColT = std::tuple_element_t<I, AttrTuple>;
                            const auto& col = rel.template column<I>();
                            owner = static_cast<std::size_t>(col[r]) % static_cast<std::size_t>(size);
                            found = true;
                        }(), 0 )
                     : 0),
                    ...
                );
            }(std::make_index_sequence<ARITY>{});

            if (!found) {
                owner = 0;
            }
            rows_for_rank[owner].push_back(r);
        }

        // pack data send to each rank and put them into send_linear buffer 
        send_sizes.resize(size);
        sdispls.resize(size);
        send_linear.reserve(1024);

        for (int p = 0; p < size; ++p) {
            std::vector<std::byte> tmp;
            pack_relation_rows(rel, rows_for_rank[p], tmp);  // 你之前写的那个 pack
            send_sizes[p] = static_cast<int>(tmp.size());
            sdispls[p]    = static_cast<int>(send_linear.size());
            send_linear.insert(send_linear.end(), tmp.begin(), tmp.end());
        }
    }

    // each rank get size to receive from rank 0 
    int my_recv_size = 0;
    MPI_Scatter(
        rank == 0 ? send_sizes.data() : nullptr,   
        1, MPI_INT,
        &my_recv_size, 1, MPI_INT,                 
        0, comm
    );

    // allocate the buffer for receiving data 
    std::vector<std::byte> my_recv_buf(my_recv_size);

    // using scatterv to get the data from rank 0 
    MPI_Scatterv(
        rank == 0 ? send_linear.data() : nullptr,             
        rank == 0 ? send_sizes.data()   : nullptr,
        rank == 0 ? sdispls.data()      : nullptr,
        MPI_BYTE,
        my_recv_buf.data(), my_recv_size, MPI_BYTE,          
        0, comm
    );

    // each rank will unpack the data into the local relation
    if (my_recv_size > 0) {
        unpack_append_relation(my_recv_buf, local_new);
    }

    //return local_new;
}

}