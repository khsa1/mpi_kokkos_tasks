CXX=mpic++
CXXFLAGS=-qopenmp -mkl

TRILINOS_OPT=/fs/project/PZS0530/skhuvis/src/trilinos-build
TRILINOS_SRC=/fs/project/PZS0530/skhuvis/src/Trilinos
KOKKOS_PATH=${TRILINOS_SRC}/packages/kokkos
SHYLU_SRC=${TRILINOS_SRC}/packages/shylu/shylu_node/tacho/src
SHYLU_OPT=${TRILINOS_OPT}/packages/shylu/shylu_node/tacho/src
KOKKOS_OPT=${TRILINOS_OPT}/packages/kokkos
KOKKOS_DEVICES="OpenMP"
KOKKOS_ARCH="BDW"

include ${KOKKOS_PATH}/Makefile.kokkos

INCS=-I${SHYLU_OPT} -I${SHYLU_SRC}

LDFLAGS=-lmetis $(KOKKOS_LDFLAGS) -L${KOKKOS_OPT}/core/src -L${KOKKOS_OPT}/containers/src -L${KOKKOS_OPT}/core/src -L${TRILINOS_OPT}/packages/common/auxiliarySoftware/SuiteSparse/src -L${SHYLU_OPT} -L${KOKKOS_OPT}/algorithms/src -L${TRILINOS_OPT}/commonTools/gtest -L./ -lkokkoscore -lkokkoscontainers -lshylu_nodetacho -ltrilinosss -lkokkosalgorithms -lgtest


EXE=Tacho_ExampleDenseByBlocks graph500_nonlocking graph500_locking

all: $(EXE)

Tacho_ExampleDenseByBlocks: Tacho_ExampleDenseByBlocks.o $(KOKKOS_LINK_DEPENDS)
	$(CXX) $(CXXFLAGS) -o Tacho_ExampleDenseByBlocks Tacho_ExampleDenseByBlocks.o $(LDFLAGS) $(KOKKOS_LIBS)

graph500_nonlocking:
	make -C graph500/kokkos graph500_nonlocking INCS="${INCS}" KOKKOS_PATH="${KOKKOS_PATH}" KOKKOS_DEVICES="$(KOKKOS_DEVICES)" KLDFLAGS="${LDFLAGS}" 
graph500_locking:
	make -C graph500/kokkos graph500_locking INCS="${INCS}" KOKKOS_PATH="${KOKKOS_PATH}" KOKKOS_DEVICES="$(KOKKOS_DEVICES)" KLDFLAGS="${LDFLAGS}" 

%.o : %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCS) $(KOKKOS_CXXFLAGS) $< -o $@

clean:
	rm -f *.o *.a *.tmp $(EXE)
	-make -C graph500/kokkos clean KOKKOS_PATH="${KOKKOS_PATH}"
