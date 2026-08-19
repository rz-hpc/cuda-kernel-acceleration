//////////////////////////////////////////////////////////////////////////////////
// Install OpenMPI
// !apt-get install -y openmpi-bin libopenmpi-dev
// !pip install mpi4py
//
// # 1. Compile the code
// !mpicxx mpi_grid_setup.cpp -o mpi_grid_setup
//
// # 2. Verify it compiled successfully (this will print the file details)
// !ls -l mpi_grid_setup
//
// # 3. Run the executable with the oversubscribe flag
// !mpirun --allow-run-as-root --oversubscribe -n 4 ./mpi_grid_setup
//
// Example output -n 4 (Google Colab T4)
// -rwxr-xr-x 1 root root 117208 Aug 19 06:24 mpi_grid_setup
// Rank 0 out of 4
// Rank 2 out of 4
// Rank 3 out of 4
// Rank 1 out of 4
// [Grid 2x2]World Rank: 0 -> Cart Rank: 0 is at (Row: 0, Col: 0) Mapped Rank: 0)
// [Grid 2x2]World Rank: 2 -> Cart Rank: 2 is at (Row: 1, Col: 0) Mapped Rank: 2)
// [Grid 2x2]World Rank: 1 -> Cart Rank: 1 is at (Row: 0, Col: 1) Mapped Rank: 1)
// [Grid 2x2]World Rank: 3 -> Cart Rank: 3 is at (Row: 1, Col: 1) Mapped Rank: 3)
//
// Example output -n 6 (Google Colab T4)
// -rwxr-xr-x 1 root root 117208 Aug 19 06:36 mpi_grid_setup
// Rank 1 out of 6
// Rank 0 out of 6
// Rank 2 out of 6
// Rank 3 out of 6
// Rank 4 out of 6
// Rank 5 out of 6
// [Grid 3x2]World Rank: 0 -> Cart Rank: 0 is at (Row: 0, Col: 0) Mapped Rank: 0)
// [Grid 3x2]World Rank: 4 -> Cart Rank: 4 is at (Row: 2, Col: 0) Mapped Rank: 4)
// [Grid 3x2]World Rank: 1 -> Cart Rank: 1 is at (Row: 0, Col: 1) Mapped Rank: 1)
// [Grid 3x2]World Rank: 2 -> Cart Rank: 2 is at (Row: 1, Col: 0) Mapped Rank: 2)
// [Grid 3x2]World Rank: 3 -> Cart Rank: 3 is at (Row: 1, Col: 1) Mapped Rank: 3)
// [Grid 3x2]World Rank: 5 -> Cart Rank: 5 is at (Row: 2, Col: 1) Mapped Rank: 5)
//////////////////////////////////////////////////////////////////////////////////

#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // 1. Basic Hello World
    printf("Rank %d out of %d\n", world_rank, world_size);

    // 2. 2D Process Grid Setup
    int dims[2] = {0, 0};

    // MPI_Dims_create automatically factors the total process count into a balanced 2D grid
    // e.g., 4 processes -> 2x2, 6 processes -> 3x2
    MPI_Dims_create(world_size, 2, dims);

    int periods[2] = {0, 0}; // 0 means no wrap-around (non-periodic boundaries)
    int reorder = 1; // 1 allows MPI to reorder ranks for optimal hardware mapping
    
    // Create the Cartesian topology communicator (grid-based computations etc)
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD,
                    2,
                    dims,
                    periods,
                    reorder,
                    &cart_comm);

    // 3. Coordinate Mapping
    if (cart_comm != MPI_COMM_NULL) {
        int cart_rank;
        int coords[2];

        // Get the rank in the new communicator
        MPI_Comm_rank(cart_comm, &cart_rank);

        // Translate that rank into 2D (row, col) coordinates
        // rank -> (row, col)
        MPI_Cart_coords(cart_comm, cart_rank, 2, coords);
        int row = coords[0], col = coords[1];

        // (row, col) -> rank
        int mapped_rank;
        MPI_Cart_rank(cart_comm, coords, &mapped_rank);

        std::cout << "[Grid " << dims[0] << "x" << dims[1] << "]"
                  << "World Rank: " << world_rank
                  << " -> Cart Rank: " << cart_rank
                  << " is at (Row: " << coords[0] << ", Col: " << coords[1] << ") "
                  << "Mapped Rank: " << mapped_rank << ")\n";
    }

    MPI_Finalize();
    return 0;
}
