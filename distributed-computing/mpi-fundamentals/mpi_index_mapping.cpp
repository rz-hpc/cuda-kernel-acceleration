// Compile and run with:
// !g++ mpi_index_mapping.cpp -o mpi_index_mapping
// !./mpi_index_mapping

#include <iostream>
#include <cassert>

// Index Mapping Struct
struct LocalCoord {
    int rank_coord; // Process row or column index (0 to P-1)
    int local_idx; // Index inside that process's local array
};

struct Local2DCoord {
    int rank_row, rank_col; // Owning process in the 2D process grid
    int local_row, local_col; // Local memory array coordinates on that process
};

struct Global2DCoord {
    int g_row, g_col; // global matrix coordinates
};

// 1D Helper Functions
// Map global matrix index to process owner and local memory offset
// Nb: block size
inline LocalCoord global_to_local(int global_idx, int nb, int p_dim) {
    int block_idx = global_idx / nb;
    int owner_coord = block_idx % p_dim;
    int local_block = block_idx / p_dim;
    int offset = global_idx % nb;

    return {owner_coord, local_block * nb + offset};
}

// Map local offset on a specific process back to global matrix index
inline int local_to_global(int local_idx, int rank_coord, int nb, int p_dim) {
    int local_block = local_idx / nb;
    int offset = local_idx % nb;
    int global_block = local_block * p_dim + rank_coord;

    return global_block * nb + offset;
}

bool verify_1d_index_mapping() {
    int nb = 2; // Block size
    int p_dim = 2; // processes in this dimension
    int global_N = 8; // 8 x 8 axis

    for (int g = 0; g < global_N; g++) {
        LocalCoord target = global_to_local(g, nb, p_dim);
        int reconstructed_g = local_to_global(target.local_idx, target.rank_coord, nb, p_dim);
        
        // Round-trip conversion should be perfectly lossless
        assert(g == reconstructed_g);
    }

    std::cout << "Index Mapping 1D round trip verified." << std::endl;
    
    return true;
}

// 2D Transformations
inline Local2DCoord global_to_local_2d(int g_row, int g_col, int nb, int P_r, int P_c) {
    LocalCoord r = global_to_local(g_row, nb, P_r);
    LocalCoord c = global_to_local(g_col, nb, P_c);
    return {r.rank_coord, c.rank_coord, r.local_idx, c.local_idx};
}

inline Global2DCoord local_to_global_2d(int local_row, int local_col, int rank_row, int rank_col, int nb, int P_r, int P_c) {
    int g_row = local_to_global(local_row, rank_row, nb, P_r);
    int g_col = local_to_global(local_col, rank_col, nb, P_c);

    return {g_row, g_col};
}

bool verify_2d_index_mapping() {
    int nb = 2; // Block size (2x2 tiles)
    int P_r = 2; // 2 process rows
    int P_c = 2; // 2 process cols
    int M = 8, N = 8; // Global 8x8 matrix

    for (int r = 0; r < M; r++) {
        for (int c = 0; c < N; c++) {
            // Step 1: Map global (r, c) to 2D process owner and local offsets
            Local2DCoord target = global_to_local_2d(r, c, nb, P_r, P_c);

            // Step 2: Map local coordinates back to global (r, c)
            Global2DCoord recon = local_to_global_2d(target.local_row, target.local_col,
                                                    target.rank_row, target.rank_col,
                                                    nb, P_r, P_c);

            // Step 3: Assert round-trip equivalence
            assert(r == recon.g_row);
            assert(c == recon.g_col);
        }
    }
    
    std::cout << "2D Block-Cyclic Index Mapping round trip verified for "
              << M << "x" << N << " matrix on "
              << P_r << "x" << P_c << " process grid!\n";

    return true;
}

int main() {

    verify_1d_index_mapping();

    verify_2d_index_mapping();

    return 0;
}
