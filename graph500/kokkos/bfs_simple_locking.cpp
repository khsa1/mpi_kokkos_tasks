/* Copyright (C) 2010 The Trustees of Indiana University.                  */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

#include "common.h"
#include <mpi.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>
#include <assert.h>
#include "ShyLU_NodeTacho_config.h"
#include <Kokkos_Core.hpp>
#include <atomic>
#include <unistd.h>
#include <mutex>

int while_iter;
#define MAX_SIZE 40
int total_num_tasks;
std::atomic_int64_t newq_count;
int bfs_id=-1;

#define coalescing_size 512
#define SET_VISITED(v) do {visited[VERTEX_LOCAL((v)) / ulong_bits] |= (1UL << (VERTEX_LOCAL((v)) % ulong_bits));} while (0)
#define TEST_VISITED(v) ((visited[VERTEX_LOCAL((v)) / ulong_bits] & (1UL << (VERTEX_LOCAL((v)) % ulong_bits))) != 0)
std::mutex m;

using ExecSpace = Kokkos::DefaultExecutionSpace;

template<class Space>
struct RecvFunctor {
  using Scheduler   = Kokkos::TaskScheduler< Space > ;
  using MemorySpace = typename Scheduler::memory_space ;
  using MemberType  = typename Scheduler::member_type ;
  using FutureType  = Kokkos::Future< int64_t, Space > ;

    typedef int64_t value_type;
    unsigned long *visited;
    int ulong_bits;
    int64_t *pred;
    int64_t *newq;
    int64_t *recvbuf;
    int num_tasks;

    Scheduler sched;

  KOKKOS_INLINE_FUNCTION
  RecvFunctor(const Scheduler & _sched, unsigned long *_visited, int _ulong_bits,
    int64_t *_pred, int64_t *_newq,
    int64_t *_recvbuf, int _num_tasks)
  : sched( _sched ), visited( _visited ), ulong_bits( _ulong_bits ), pred( _pred ),
  newq ( _newq ), recvbuf ( _recvbuf ),
  num_tasks( _num_tasks ){}

  KOKKOS_INLINE_FUNCTION
  void operator()( const MemberType & member, value_type & result) noexcept
  {
    result = 0;
    int is_active = 1;
    int num_tasks_done = 0;
    int flag=0;
    int count;
    MPI_Status st;
    MPI_Request recvreq;
    int64_t src, tgt;

    while(num_tasks_done < (size-1)*total_num_tasks) {
      count = MPI_UNDEFINED;
      recvbuf[0] = -123; //
      MPI_Irecv(recvbuf, coalescing_size * 2, INT64_T_MPI_TYPE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &recvreq);
      flag = 0;
      while(!flag) {
        MPI_Test(&recvreq, &flag, &st);
      }
      MPI_Get_count(&st, INT64_T_MPI_TYPE, &count);
      if(count == MPI_UNDEFINED) { }
      else if(count==0) {
        ++num_tasks_done;
      }
      else {
        // Move recvbuf
        for(int i = 0; i < count; i += 2) {
          src = recvbuf[i];
          tgt = recvbuf[i+1];
          m.lock();
          if (!TEST_VISITED(tgt)) {
            SET_VISITED(tgt);
            pred[VERTEX_LOCAL(tgt)] = src;
            newq[newq_count++] = tgt;
          }
          m.unlock();
        }
      }
    }
    result = 1;
  }
};

template<class Space>
struct QueueFunctor {
  using Scheduler   = Kokkos::TaskScheduler< Space > ;
  using MemorySpace = typename Scheduler::memory_space ;
  using MemberType  = typename Scheduler::member_type ;
  using FutureType  = Kokkos::Future< int64_t, Space > ;

    typedef int64_t value_type;

    Scheduler sched;
    int64_t src;
    const csr_graph *const g;
    int ulong_bits;
    int64_t *pred;
    int64_t *newq;
    int64_t *oldq;
    unsigned long *visited;
    int64_t visited_size;
    size_t src_start;
    size_t src_end;
    int taskid;
    

  KOKKOS_INLINE_FUNCTION
  QueueFunctor(const Scheduler & _sched,
      const csr_graph* const _g,
      int _ulong_bits,
      int64_t *_pred, int64_t *_newq,
      int64_t *_oldq,
      unsigned long *_visited,
      int64_t _visited_size,
      size_t _src_start, size_t _src_end, int _taskid)
  : sched( _sched ), g( _g ),
    ulong_bits( _ulong_bits), pred( _pred ), newq( _newq ), oldq( _oldq ),
    visited( _visited ), visited_size( _visited_size ),
    src_start( _src_start ), src_end( _src_end ), taskid ( _taskid ){}

  KOKKOS_INLINE_FUNCTION
  void operator()( const MemberType & member, value_type & result) noexcept
  {
    int64_t *message_queue = (int64_t*) calloc(2 * coalescing_size * size, sizeof(int64_t));
    int *queue_counts = (int*) calloc(size, sizeof(int));
    /* Iterate through its incident edges. */
    for(size_t i = src_start; i < src_end; i++) {
    src = oldq[i];
    assert (VERTEX_OWNER(src) == rank);
    assert (pred[VERTEX_LOCAL(src)] >= 0 && pred[VERTEX_LOCAL(src)] < g->nglobalverts);
    size_t j;
    size_t j_start =  g->rowstarts[VERTEX_LOCAL(src)];
    size_t j_end = g->rowstarts[VERTEX_LOCAL(src)+1];
    for (j = j_start; j < j_end; ++j) {
      int64_t tgt = g->column[j];
      int owner = VERTEX_OWNER(tgt);
      // If the other endpoint is mine, update the visited map, predecessor
      //  map, and next-level queue locally; otherwise, send the target and
      //  the current vertex (its possible predecessor) to the target's owner.
      if (owner == rank) {
        m.lock();
        if (!TEST_VISITED(tgt)) {
          SET_VISITED(tgt);
          pred[VERTEX_LOCAL(tgt)] = src;
          newq[newq_count++] = tgt;
        }
        m.unlock();
      } else {
        if(queue_counts[owner] >= coalescing_size*2) {
          MPI_Send(&message_queue[owner*2*coalescing_size], queue_counts[owner], INT64_T_MPI_TYPE, owner, 0, MPI_COMM_WORLD);
          queue_counts[owner] = 0;
        }
        int cur_len = queue_counts[owner];
        if(cur_len >=2*coalescing_size) {
          MPI_Abort(MPI_COMM_WORLD, 7654);
        }
        message_queue[owner*2*coalescing_size+cur_len] = src;
        message_queue[owner*2*coalescing_size+cur_len+1] = tgt;
        queue_counts[owner] += 2;
      }
    }
    }
    int is_done = -1;
    for(int rnk=0; rnk < size; rnk++) {
      if(rank != rnk) {
        if(queue_counts[rnk] > 0) {
        MPI_Send(&message_queue[rnk*2*coalescing_size], queue_counts[rnk], INT64_T_MPI_TYPE, rnk, 0, MPI_COMM_WORLD);
        queue_counts[rnk] = 0;
        }
        MPI_Send(&is_done, 0, MPI_INT, rnk, 0, MPI_COMM_WORLD);
      }
    }
    result = 1;
  }
};

typedef QueueFunctor<ExecSpace> queueFunctor;
typedef RecvFunctor<ExecSpace> recvFunctor;

  int64_t while_loop(int64_t* outgoing,
      MPI_Request* outgoing_reqs, int* outgoing_reqs_active,
      int64_t* recvbuf, MPI_Request recvreq, int recvreq_active,
      int64_t oldq_count, int64_t *oldq, int64_t *newq,
      int ulong_bits, int64_t visited_size, unsigned long *visited,
      int64_t *pred, const csr_graph* const g)
  {
#pragma omp parallel
    {
      // subtract 3 for send, receive, and flush tasks
      total_num_tasks = omp_get_num_threads() - 1;
    }
    int64_t result = 0;
    size_t i;
    long total_alloc_size = 10000000;
    int min_superblock_size = 100000;
    const unsigned min_block_size =  320 ;
    const unsigned max_block_size = 1280 ;
    Kokkos::TaskScheduler<ExecSpace> sched(Kokkos::TaskScheduler<ExecSpace>::memory_space(),
                                           total_alloc_size,
                                           min_block_size,
                                           max_block_size,
                                           min_superblock_size);
    while_iter=0;
    int is_active;
    while (1) {
    is_active = 1;
      /* Step through the current level's queue. */
      {
      int src_start = 0;
      int src_end = 0;
      int block_size = (oldq_count % total_num_tasks) ? oldq_count / total_num_tasks + 1 : oldq_count / total_num_tasks;
      queueFunctor::FutureType qFutures[total_num_tasks];
      int task_id=0;
      while(src_end < oldq_count) {
        src_start = src_end;
        src_end = src_start + block_size < oldq_count ? src_start + block_size : oldq_count;
        qFutures[task_id++] = Kokkos::host_spawn(Kokkos::TaskSingle(sched), queueFunctor(sched, g, ulong_bits, pred, newq, oldq, visited, visited_size, src_start, src_end, task_id));
        result+=src_end-src_start;
      }
      int is_done = -1;
      for(int i=task_id; i< total_num_tasks; i++) {
        for(int rnk = 0; rnk < size; rnk++) {
          if(rank != rnk) {
            MPI_Send(&is_done, 0, MPI_INT, rnk, 0, MPI_COMM_WORLD);
          }
        }
      }
      recvFunctor::FutureType rfuture = Kokkos::task_spawn(Kokkos::TaskSingle(sched, Kokkos::TaskPriority::High), recvFunctor(sched, visited, ulong_bits, pred, newq, recvbuf, task_id));
    Kokkos::wait(sched);
      }
      MPI_Barrier(MPI_COMM_WORLD);

      // Test globally if all queues are empty.
      int64_t global_newq_count;
      MPI_Allreduce(&newq_count, &global_newq_count, 1, INT64_T_MPI_TYPE, MPI_SUM, MPI_COMM_WORLD);

      // Quit if they all are empty.
      if (global_newq_count == 0) break;

      /* Swap old and new queues; clear new queue for next level. */
      {int64_t* temp = oldq; oldq = newq; newq = temp;}
      oldq_count = newq_count;
      newq_count = 0;
      MPI_Barrier(MPI_COMM_WORLD);
      while_iter++;
    }
    return result;
  }

/* This version is the traditional level-synchronized BFS using two queues.  A
 * bitmap is used to indicate which vertices have been visited.  Messages are
 * sent and processed asynchronously throughout the code to hopefully overlap
 * communication with computation. */
void run_mpi_bfs(const csr_graph* const g, int64_t root, int64_t* pred) {
  long total_alloc_size = 10000000;
  int min_superblock_size = 100000;
  const unsigned min_block_size =  320 ;
  const unsigned max_block_size = 1280 ;
  const size_t nlocalverts = g->nlocalverts;

  /* Set up the queues. */
  int64_t* oldq = (int64_t*)xmalloc(nlocalverts * sizeof(int64_t));
  int64_t* newq = (int64_t*)xmalloc(nlocalverts * sizeof(int64_t));
  int64_t* predq = (int64_t*)xmalloc(nlocalverts * sizeof(int64_t));
  size_t oldq_count = 0;
  size_t newq_count = 0;

  /* Set up the visited bitmap. */
  const int ulong_bits = sizeof(unsigned long) * CHAR_BIT;
  int64_t visited_size = (nlocalverts + ulong_bits - 1) / ulong_bits;
  unsigned long* visited = (unsigned long*)xcalloc(visited_size, sizeof(unsigned long)); /* Uses zero-init */

  /* Set up buffers for message coalescing, MPI requests, etc. for
   * communication. */
  int64_t* outgoing = (int64_t*)xMPI_Alloc_mem(coalescing_size * size * 2 * sizeof(int64_t));
  MPI_Request* outgoing_reqs = (MPI_Request*)xmalloc(size * sizeof(MPI_Request));
  int* outgoing_reqs_active = (int*)xcalloc(size, sizeof(int)); /* Uses zero-init */
  int64_t* recvbuf = (int64_t*)xMPI_Alloc_mem(coalescing_size * 2 * sizeof(int64_t));
  MPI_Request recvreq;
  int recvreq_active = 0;

  /* Termination counter for each level: this variable counts the number of
   * ranks that have said that they are done sending to me in the current
   * level.  This rank can stop listening for new messages when it reaches
   * size. */
  int num_ranks_done;

  /* Set all vertices to "not visited." */
  {size_t i; for (i = 0; i < nlocalverts; ++i) predq[i] = -1;}

  /* Mark the root and put it into the queue. */
  if (VERTEX_OWNER(root) == rank) {
    m.lock();
    SET_VISITED(root);
    predq[VERTEX_LOCAL(root)] = root;
    oldq[oldq_count++] = root;
    m.unlock();
  }

  int64_t nvisited_local = while_loop(outgoing, outgoing_reqs, outgoing_reqs_active, recvbuf, recvreq, recvreq_active, oldq_count, oldq, newq, ulong_bits, visited_size, visited, predq, g);

  for(size_t i = 0; i < nlocalverts; i++) pred[i]= predq[i];

  free(oldq);
  free(newq);
  free( outgoing );
  free(outgoing_reqs);
  free(outgoing_reqs_active);
  free( recvbuf );
  free(visited);
  free(predq);
}
#undef CHECK_MPI_REQS
