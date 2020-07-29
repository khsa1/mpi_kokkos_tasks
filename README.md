# README

## Getting Trilinos

```
git clone https://github.com/trilinos/Trilinos.git
git checkout b783a65
```

## Building Kokkos

Run the following cmake command to build Kokkos
```
cmake -D CMAKE_INSTALL_PREFIX=$INSTALL_DIR -D \
TPL_ENABLE_MPI:BOOL=OFF -D Trilinos_ENABLE_Fortran:BOOL=OFF -D TPL_ENABLE_Pthread:BOOL=OFF \
-D Kokkos_ENABLE_Pthread:BOOL=OFF -D Trilinos_ENABLE_OpenMP:BOOL=ON \
-D Kokkos_ENABLE_OpenMP:BOOL=ON -D Trilinos_ENABLE_ALL_PACKAGES:BOOL=OFF \
-D Trilinos_ENABLE_EXAMPLES:BOOL=ON -D Trilinos_ENABLE_TESTS:BOOL=ON \
-D Trilinos_ENABLE_KokkosCore:BOOL=ON -D Trilinos_ENABLE_KokkosContainers:BOOL=ON \
-D Trilinos_ENABLE_KokkosExample:BOOL=OFF -D Trilinos_ENABLE_KokkosAlgorithms:BOOL=ON \
-D Trilinos_ENABLE_CXX11:BOOL=ON -D Kokkos_ENABLE_CXX11:BOOL=ON \
-D Kokkos_ENABLE_Serial:BOOL=ON -D Trilinos_ENABLE_ShyLU:BOOL=OFF \
-D Trilinos_ENABLE_ShyLU_DDCore:BOOL=OFF -D Trilinos_ENABLE_ShyLU_NodeTacho:BOOL=ON \
-D Trilinos_ENABLE_Teuchos:BOOL=ON -D Teuchos_ENABLE_TESTS:BOOL=OFF \
-D TPL_ENABLE_Cholmod:BOOL=OFF -D TPL_ENABLE_METIS:BOOL=ON \
-D METIS_INCLUDE_DIRS:FILEPATH=/ apps/metis/intel/18.0/5.1.0/include \
-D METIS_LIBRARY_DIRS:FILEPATH=/apps/metis/intel/18.0/5.1.0/lib \
-D CMAKE_BUILD_TYPE:STRING=RELEASE -D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
-D CMAKE_C_COMPILER=icc -D CMAKE_CXX_COMPILER=icpc \
-D CMAKE_CXX_FLAGS:STRING=-DKOKKOS_OPT_RANGE_AGGRESSIVE_VECTORIZATION \
-D CMAKE_EXE_LINKER_FLAGS:STRING=-lnuma -lrt -ldl -lgfortran \
-D TPL_ENABLE_MKL:BOOL=ON  -D TPL_MKL_LIBRARIES:FILEPATH=-mkl \
-D TPL_ENABLE_BLAS:BOOL=ON -D TPL_BLAS_LIBRARIES:FILEPATH=-mkl \
-D TPL_ENABLE_LAPACK:BOOL=ON  -D TPL_LAPACK_LIBRARIES:FILEPATH=-mkl \
-D KOKKOS_ARCH=BDW $SRC_DIR
```

`$SRC_DIR` should be replaced with the location where you extracted Kokkos and `$INSTALL_DIR` should be replaced with the installation directory.

## Building MPI+Kokkos codes

The included Makefile will build three applications:

1. MPI+Kokkos GEMM code
2. MPI+Kokkos locking Graph500
3. MPI+Kokkos non-locking Graph500

Make sure that you have already built Trilinos. The following changes will need to be made to build correctly:

1. Replace `$TRILINOS_OPT` with `$INSTALL_DIR`
2. Replace `$TRILINOS_SRC` with `$SRC_DIR`

Then, run `make all` to generate the following executables:
1. `Tacho_ExampleDenseByBlocks`
2. `graph500/kokkos/graph500_locking`
3. `graph500/kokkos/graph500_nonlocking`
