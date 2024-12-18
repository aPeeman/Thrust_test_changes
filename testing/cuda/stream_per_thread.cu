#include <unittest/unittest.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/detail/util.h>

#include <thread>

void verify_stream()
{
  auto exec = thrust::device;
  auto stream = thrust::cuda_cub::stream(exec);
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  ASSERT_EQUAL(stream, cudaStreamPerThread);
#else
  ASSERT_EQUAL(stream, (cudaStream_t)cudaStreamDefault);
#endif
}

void TestPerThreadDefaultStream()
{
  verify_stream();

  std::thread t(verify_stream);
  t.join();
}
DECLARE_UNITTEST(TestPerThreadDefaultStream);
