include ../make.inc
include $(KOKKOS_PATH)/Makefile.kokkos
CXX = icpc
MPICXX = mpic++

BIN = graph500_locking graph500_nonlocking

all:	$(BIN)

GENLIB=../generator/libgraph_generator_mpi.a

graph500_locking: main.cpp bfs_simple_locking.cpp convert_to_csr.c find_roots.c utils.c validate.c $(GENLIB) $(KOKKOS_LINK_DEPENDS)
	$(MPICXX) $(CFLAGS) $(LDFLAGS) $(KLDFLAGS) -o graph500_locking bfs_simple_locking.cpp convert_to_csr.c find_roots.c main.cpp utils.c validate.c $(GENLIB) -lm ${INCS} $(KOKKOS_CXXFLAGS) $(KOKKOS_LIBS)

graph500_nonlocking: main.cpp bfs_simple.cpp convert_to_csr.c find_roots.c utils.c validate.c $(GENLIB) $(KOKKOS_LINK_DEPENDS)
	$(MPICXX) $(CXXFLAGS) $(LDFLAGS) $(KLDFLAGS) -o graph500_nonlocking bfs_simple.cpp convert_to_csr.c find_roots.c main.cpp utils.c validate.c $(GENLIB) -lm ${INCS} $(KOKKOS_CXXFLAGS) $(KOKKOS_LIBS)
 
$(GENLIB):
	$(MAKE) -C ../generator -f ../generator/Makefile.mpi

clean:
	-rm -f graph500_* *.o *.a KokkosCore_config.*
	-$(MAKE) -C ../generator -f ../generator/Makefile.mpi clean
