// Example by Nathan Bell and Jared Hoberock
// (modified by Mihail Ivakhnenko)
//
// This example demonstrates how to intercept calls to get_temporary_buffer
// and return_temporary_buffer to control how Thrust allocates temporary storage
// during algorithms such as thrust::reduce. The idea will be to create a simple
// cache of allocations to search when temporary storage is requested. If a hit
// is found in the cache, we quickly return the cached allocation instead of
// resorting to the more expensive thrust::cuda::malloc.
//
// Note: this implementation cached_allocator is not thread-safe. If multiple
// (host) threads use the same cached_allocator then they should gain exclusive
// access to the allocator before accessing its methods.

#pragma once

#include "utils/logger.h"
#include <iostream>
#include <map>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/pair.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/vector.h>

namespace Aperture {

// cached_allocator: a simple allocator for caching allocation requests
class cached_allocator {
 public:
  // just allocate bytes
  typedef char value_type;

  cached_allocator() {}

  ~cached_allocator() {
    // free all allocations when cached_allocator goes out of scope
    free_all();
  }

  char* allocate(std::ptrdiff_t num_bytes) {
    char* result = 0;

    // search the cache for a free block
    free_blocks_type::iterator free_block = free_blocks.find(num_bytes);

    if (free_block != free_blocks.end()) {
      // std::cout << "cached_allocator::allocator(): found a hit" << std::endl;
      Logger::print_detail("cached_allocator::allocator(): found a hit");

      // get the pointer
      result = free_block->second;

      // erase from the free_blocks map
      free_blocks.erase(free_block);
    } else {
      // no allocation of the right size exists
      // create a new one with cuda::malloc
      // throw if cuda::malloc can't satisfy the request
      try {
        // std::cout << "cached_allocator::allocator(): no free block found; "
        //              "calling cuda::malloc"
        //           << std::endl;
        Logger::print_detail(
            "cached_allocator::allocator(): no free block found; "
            "calling cuda::malloc");

        // allocate memory and convert cuda::pointer to raw pointer
        result = thrust::cuda::malloc<char>(num_bytes).get();
      } catch (std::runtime_error& e) {
        throw;
      }
    }

    // insert the allocated pointer into the allocated_blocks map
    allocated_blocks.insert(std::make_pair(result, num_bytes));

    return result;
  }

  void deallocate(char* ptr, size_t n) {
    // erase the allocated block from the allocated blocks map
    allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);
    std::ptrdiff_t num_bytes = iter->second;
    allocated_blocks.erase(iter);

    // insert the block into the free blocks map
    free_blocks.insert(std::make_pair(num_bytes, ptr));
  }

 private:
  typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
  typedef std::map<char*, std::ptrdiff_t> allocated_blocks_type;

  free_blocks_type free_blocks;
  allocated_blocks_type allocated_blocks;

  void free_all() {
    std::cout << "cached_allocator::free_all(): cleaning up after ourselves..."
              << std::endl;

    // deallocate all outstanding blocks in both lists
    for (free_blocks_type::iterator i = free_blocks.begin();
         i != free_blocks.end(); i++) {
      // transform the pointer to cuda::pointer before calling cuda::free
      thrust::cuda::free(thrust::cuda::pointer<char>(i->second));
    }

    for (allocated_blocks_type::iterator i = allocated_blocks.begin();
         i != allocated_blocks.end(); i++) {
      // transform the pointer to cuda::pointer before calling cuda::free
      thrust::cuda::free(thrust::cuda::pointer<char>(i->first));
    }
  }
};

extern cached_allocator alloc_global;

}  // namespace Aperture
