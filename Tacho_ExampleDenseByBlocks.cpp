#include "ShyLU_NodeTacho_config.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <impl/Kokkos_Timer.hpp>

#include "Tacho_Util.hpp"
#include "Tacho_DenseMatrixView.hpp"
#include "Tacho_DenseFlopCount.hpp"

#include "Tacho_Chol_ByBlocks.hpp"
#include "Tacho_Gemm_ByBlocks.hpp"
#include "Tacho_Herk_ByBlocks.hpp"
#include "Tacho_Trsm_ByBlocks.hpp"

#include "Tacho_CommandLineParser.hpp" 

#ifdef TACHO_HAVE_MKL
#include "mkl_service.h"
#endif
#include <mpi.h>

using namespace Tacho;

#define PRINT_TIMER                                                     \
  printf("  Time \n");                                                  \
  printf("       byblocks/reference (speedup):                   %10.6f\n", t_reference/t_byblocks); \
  printf("\n");                                                         

int main (int argc, char *argv[]) {
  CommandLineParser opts("This example program measure the performance of dense-by-blocks on Kokkos::OpenMP");  

  bool serial = false;
  int nthreads = 1;
  bool verbose = true;
  int mb = 128;
  int rank, np;
  ordinal_type m = 1000;
  ordinal_type l_m = 1000;
  const double alpha = 1.0, beta = 1.0;
  bool local = false;
  MPI_Status statuses[4];
  MPI_Request send_requests[2], recv_requests[2], requests[4];
  int ierr;

  opts.set_option<bool>("serial", "Flag for invoking serial algorithm", &serial);
  opts.set_option<int>("kokkos-threads", "Number of threads", &nthreads);
  opts.set_option<bool>("verbose", "Flag for verbose printing", &verbose);
  opts.set_option<ordinal_type>("local-size", "Test problem local size", &l_m);
  opts.set_option<ordinal_type>("size", "Test problem step size", &m);  
  opts.set_option<int>("mb", "Blocksize", &mb);
  opts.set_option<bool>("local", "Set local size", &local);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse) return 0; // print help return 
  int r_val = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  printf("Process %d of %d\n", rank, np);
  if(sqrt(np) != int(sqrt(np))) {printf("Number of ranks must be a perfect square\n"); return 0; }
  int p = sqrt(np);

  if(local) {
    m = l_m * p;
  }
  else {
    l_m = m / p;
    ordinal_type rem = m % p;
    if(rem != 0) {if(rank==0) printf("Number of ranks must divide into size\n"); return 0; }
  }
  int periods[]={1,1}; //both vertical and horizontal movement; 
  int dims[]={p,p};
  int coords[2]; /* 2 Dimension topology so 2 coordinates */
  int right=0, left=0, down=0, up=0;    // neighbor ranks
  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD,2,dims,periods,1,&cart_comm );

  MPI_Comm_rank(cart_comm,&rank);
  MPI_Cart_coords(cart_comm,rank,2,coords);
  MPI_Cart_shift(cart_comm, 1, coords[0], &left,&right);
  MPI_Cart_shift(cart_comm, 0, coords[1], &up,&down);

  Kokkos::initialize(argc, argv);

  typedef double value_type;
  typedef Kokkos::pair<ordinal_type,ordinal_type> range_type;
  typedef Kokkos::DefaultExecutionSpace exec_space;
  //typedef Kokkos::DefaultHostExecutionSpace exec_space;
  typedef Kokkos::DefaultHostExecutionSpace host_exec_space;

  printExecSpaceConfiguration<host_exec_space>("Default HostSpace");
  printExecSpaceConfiguration<     exec_space>("Default DeviceSpace");

  const double eps = std::numeric_limits<double>::epsilon()*10000;  
  {
    typedef DenseMatrixView<value_type,exec_space>               DenseMatrixViewType;
    typedef DenseMatrixView<DenseMatrixViewType,exec_space>      DenseMatrixOfBlocksType;

    typedef DenseMatrixView<value_type,host_exec_space>          DenseMatrixViewHostType;
    typedef DenseMatrixView<DenseMatrixViewType,host_exec_space> DenseMatrixOfBlocksHostType;

    Kokkos::Impl::Timer timer;

    typedef Kokkos::TaskScheduler<exec_space> sched_type;
    sched_type sched;

    typedef TaskFunctor_Gemm<sched_type,double,DenseMatrixOfBlocksType,
      Trans::NoTranspose,Trans::NoTranspose,Algo::ByBlocks> task_functor_gemm;

    const ordinal_type max_functor_size = 4*sizeof(task_functor_gemm);
    
    Kokkos::DualView<value_type*,exec_space> 
      l_a("l_a", l_m*l_m),
      l_c("l_c", l_m*l_m), 
      l_b("l_b", l_m*l_m);
    
    Kokkos::DualView<value_type*,exec_space> 
      l_a2("l_a2", l_m*l_m),
      l_b2("l_b2", l_m*l_m);
    
    const ordinal_type l_bmend = (l_m/mb) + 1;
    Kokkos::DualView<DenseMatrixViewType*,exec_space> 
      ha("ha", l_bmend*l_bmend), hb("hb", l_bmend*l_bmend), hc("hc", l_bmend*l_bmend);
    Kokkos::DualView<DenseMatrixViewType*,exec_space> 
      ha2("ha2", l_bmend*l_bmend), hb2("hb2", l_bmend*l_bmend);

    {    
      const ordinal_type
        task_queue_capacity_tmp = 2*l_bmend*l_bmend*l_bmend*max_functor_size,
        min_block_size  = 16,
        max_block_size  = 4*max_functor_size,
        num_superblock  = 4,
        superblock_size = std::max(task_queue_capacity_tmp/num_superblock,max_block_size),
        task_queue_capacity = std::max(task_queue_capacity_tmp,superblock_size*num_superblock);
      
      if(rank==0) {
      std::cout << "capacity = " << task_queue_capacity << "\n";
      std::cout << "min_block_size = " << min_block_size << "\n";
      std::cout << "max_block_size = " << max_block_size << "\n";
      std::cout << "superblock_size = " << superblock_size << "\n";
      std::cout << "local_array_size = " << l_m*l_m << "\n";
      std::cout << "array_size = " << m*m << "\n";
      }
      
      sched = sched_type(typename sched_type::memory_space(),
                         (size_t)task_queue_capacity,
                         (unsigned)min_block_size,
                         (unsigned)max_block_size,
                         (unsigned)superblock_size);
    }

    const ordinal_type dry = -2, niter = 3;

    double t_reference = 0, t_byblocks = 0;

    Random<value_type> random;
    auto randomize = [&](const DenseMatrixViewHostType &mat) {
      const ordinal_type m = mat.extent(0), n = mat.extent(1);
      for (ordinal_type j=0;j<n;++j)
        for (ordinal_type i=0;i<m;++i)
          mat(i,j) = random.value()*(rank+1);
    };
    auto zeroize= [&](const DenseMatrixViewHostType &mat) {
      const ordinal_type m = mat.extent(0), n = mat.extent(1);
      for (ordinal_type j=0;j<n;++j)
        for (ordinal_type i=0;i<m;++i)
          mat(i,j) = 0;
    };
    ///
    /// Gemm
    ///
    double t1, t2;
    t1 = MPI_Wtime();
    //for (ordinal_type m=mbeg;m<=mend;m+=step) {
      t_reference = 0; t_byblocks = 0;

      auto l_sub_a  = Kokkos::subview(l_a,  range_type(0,l_m*l_m));
      auto l_sub_b  = Kokkos::subview(l_b,  range_type(0,l_m*l_m));
      auto l_sub_c = Kokkos::subview(l_c, range_type(0,l_m*l_m));

      auto l_sub_a2  = Kokkos::subview(l_a2,  range_type(0,l_m*l_m));
      auto l_sub_b2  = Kokkos::subview(l_b2,  range_type(0,l_m*l_m));

      {
        l_sub_b. modify<host_exec_space>();
        l_sub_b2. modify<host_exec_space>();
        
        DenseMatrixViewHostType l_A, l_B, l_C;
        DenseMatrixViewHostType l_A2, l_B2;
        l_A.set_view(l_m, l_m);
        l_A.attach_buffer(1, l_m, l_sub_a.h_view.data());
        l_A2.set_view(l_m, l_m);
        l_A2.attach_buffer(1, l_m, l_sub_a.h_view.data());

        l_B.set_view(l_m, l_m);
        l_B.attach_buffer(1, l_m, l_sub_b.h_view.data());
        l_B2.set_view(l_m, l_m);
        l_B2.attach_buffer(1, l_m, l_sub_b.h_view.data());

        l_C.set_view(l_m, l_m);
        l_C.attach_buffer(1, l_m, l_sub_c.h_view.data());

        randomize(l_A);
        randomize(l_B);
        zeroize(l_C);


        const ordinal_type bm = (l_m/mb) + (l_m%mb>0);

        ha.modify<host_exec_space>();
        hb.modify<host_exec_space>();
        hc.modify<host_exec_space>();

        DenseMatrixOfBlocksHostType HA, HB, HC;
        DenseMatrixOfBlocksHostType HA2, HB2;

        HA.set_view(bm, bm);
        HA.attach_buffer(1, bm, ha.h_view.data());
        HA2.set_view(bm, bm);
        HA2.attach_buffer(1, bm, ha2.h_view.data());

        HB.set_view(bm, bm);
        HB.attach_buffer(1, bm, hb.h_view.data());
        HB2.set_view(bm, bm);
        HB2.attach_buffer(1, bm, hb2.h_view.data());

        HC.set_view(bm, bm);
        HC.attach_buffer(1, bm, hc.h_view.data());

        setMatrixOfBlocks(HA, l_m, l_m, mb);
        attachBaseBuffer(HA, l_A.data(), l_A.stride_0(), l_A.stride_1());
        setMatrixOfBlocks(HA2, l_m, l_m, mb);
        attachBaseBuffer(HA2, l_A2.data(), l_A2.stride_0(), l_A2.stride_1());
        
        setMatrixOfBlocks(HB, l_m, l_m, mb);
        attachBaseBuffer(HB, l_B.data(), l_B.stride_0(), l_B.stride_1());
        setMatrixOfBlocks(HB2, l_m, l_m, mb);
        attachBaseBuffer(HB2, l_B2.data(), l_B2.stride_0(), l_B2.stride_1());
        
        setMatrixOfBlocks(HC, l_m, l_m, mb);
        attachBaseBuffer(HC, l_C.data(), l_C.stride_0(), l_C.stride_1());

        ha.sync<exec_space>();
        hb.sync<exec_space>();
        hc.sync<exec_space>();
        ha2.sync<exec_space>();
        hb2.sync<exec_space>();
        
        DenseMatrixOfBlocksType DA, DB, DC;
        DenseMatrixOfBlocksType DA2, DB2;
        
        DA.set_view(bm, bm);
        DA.attach_buffer(1, bm, ha.d_view.data());
        DA2.set_view(bm, bm);
        DA2.attach_buffer(1, bm, ha2.d_view.data());

        DB.set_view(bm, bm);
        DB.attach_buffer(1, bm, hb.d_view.data());
        DB2.set_view(bm, bm);
        DB2.attach_buffer(1, bm, hb2.d_view.data());

        DC.set_view(bm, bm);
        DC.attach_buffer(1, bm, hc.d_view.data());

        MPI_Sendrecv_replace(&(l_A(0,0)),l_m*l_m,MPI_DOUBLE,left,11,right,11,cart_comm,MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(&(l_B(0,0)),l_m*l_m,MPI_DOUBLE,up,11,down,11,cart_comm,MPI_STATUS_IGNORE);
        {
          double loop1_start = MPI_Wtime();
            timer.reset();
            Kokkos::host_spawn(Kokkos::TaskSingle(sched, Kokkos::TaskPriority::High),
                               task_functor_gemm(sched, alpha, DA, DB, beta, DC));
            Kokkos::wait(sched);
          double loop1_end = MPI_Wtime();
        }
        MPI_Cart_shift(cart_comm, 1, 1, &left,&right);
        MPI_Cart_shift(cart_comm, 0, 1, &up,&down);
        MPI_Sendrecv_replace(&(l_A(0,0)),l_m*l_m,MPI_DOUBLE,left,11,right,11,cart_comm,MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(&(l_B(0,0)),l_m*l_m,MPI_DOUBLE,up,11,down,11,cart_comm,MPI_STATUS_IGNORE);
        for(int i=1; i<p; i++) {
          MPI_Isend(&(l_A(0,0)), l_m*l_m, MPI_DOUBLE, left, 0, cart_comm, &requests[0]);
          MPI_Irecv(&(l_A2(0,0)), l_m*l_m, MPI_DOUBLE, right, MPI_ANY_TAG, cart_comm, &requests[1]);
          MPI_Isend(&(l_B(0,0)), l_m*l_m, MPI_DOUBLE, up, 0, cart_comm, &requests[2]);
          MPI_Irecv(&(l_B2(0,0)), l_m*l_m, MPI_DOUBLE, down, MPI_ANY_TAG, cart_comm, &requests[3]);
          double loop_start = MPI_Wtime();
          for (ordinal_type iter=dry;iter<niter;++iter) {
            timer.reset();
            Kokkos::host_spawn(Kokkos::TaskSingle(sched, Kokkos::TaskPriority::High),
                               task_functor_gemm(sched, alpha, DA, DB, beta, DC));
            Kokkos::wait(sched);
          }
          double loop_end = MPI_Wtime();
          MPI_Waitall(4, requests, statuses);
          for(int ii=0; ii<l_m; ii++) {
            for(int jj=0; jj<l_m; jj++) {
              l_A(ii,jj)=l_A2(ii,jj);
              l_B(ii,jj)=l_B2(ii,jj);
            }
          }
        }
      }
      
    t2 = MPI_Wtime();
    if(rank==0) printf("Time spent in GEMM code: %f\n", t2 - t1);
  }
  Kokkos::finalize();
  MPI_Finalize();

  return r_val;
}
