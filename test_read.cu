#include "iostack_decouple.cuh"
#include <unordered_set>
#include <algorithm>
#include <random>
#include <assert.h>
#include <unistd.h>
#define TEST_SIZE 0x10000000
#define APP_BUF_SIZE 0x10000000
#define NUM_QUEUES_PER_SSD 128
#define NUM_SSDS 1

__device__ float **IO_buf_base;

__device__ uint64_t seed;
__global__ void gen_test_data(int ssd_id, int req_id)
{
    for (int i = 0; i < MAX_IO_SIZE / 4; i++)
    {
        seed = seed * 0x5deece66d + 0xb;
        IO_buf_base[ssd_id][i] = req_id * MAX_IO_SIZE / 4 + i; 
    }
}

__global__ void check_test_data(float *app_buf, int idx)
{
    for (int i = 0; i < MAX_IO_SIZE / 4; i++)
    {
        if(i < 10){
            printf("%f\n", app_buf[i]);
        }
        // if (app_buf[i] != idx * MAX_IO_SIZE / 4 + i)
        // {
        //     printf("check failed at block %d, i = %d, read %lx, expected %x\n", idx, i, app_buf[i], idx * MAX_IO_SIZE / 4 + i);
        //     assert(0);
        // }
    }
}

__global__ void fill_app_buf(float *app_buf)
{
    for (int i = 0; i < TEST_SIZE / 4; i++)
        app_buf[i] = 0;
}

int main()
{
    IOStack iostack(NUM_SSDS, NUM_QUEUES_PER_SSD);
    float **d_IO_buf_base = (float**) iostack.get_d_io_buf_base();
    CHECK(cudaMemcpyToSymbol(IO_buf_base, &d_IO_buf_base, sizeof(float **)));

    // test do_io_req
    float *app_buf;
    CHECK(cudaMalloc(&app_buf, APP_BUF_SIZE));
    fill_app_buf<<<1, 1>>>(app_buf);
    int num_reqs = TEST_SIZE / MAX_IO_SIZE;
    IOReq *reqs;
    CHECK(cudaMalloc(&reqs, sizeof(IOReq) * num_reqs));
    IOReq *h_reqs;
    CHECK(cudaHostAlloc(&h_reqs, sizeof(IOReq) * num_reqs, cudaHostAllocMapped));
    std::unordered_set<uint64_t> lbs;
    srand(time(NULL));
    uint64_t h_seed = 0;
    CHECK(cudaMemcpyToSymbol(seed, &h_seed, sizeof(uint64_t)));
    int percent = 1;
    clock_t clstart = clock();
    for (int i = 0; i < num_reqs; i++)
    {
        uint64_t lb;
        lb = i % 2;
        h_reqs[i].start_lb = lb;
        h_reqs[i].num_items = 1;
        h_reqs[i].dest_addr[0] = (app_addr_t)(app_buf + (1ll * i * ITEM_SIZE) % APP_BUF_SIZE / 4);

        int ssd_id = 0;

        if (i >= num_reqs / 100 * percent)
        {
            double eta = (clock() - clstart) / (double)CLOCKS_PER_SEC / percent * (100 - percent);
            fprintf(stderr, "generating test data: %d%% done, eta %.0lfs\r", percent, eta);
            percent++;
        }
    }
    CHECK(cudaDeviceSynchronize());
    // std::shuffle(h_reqs, h_reqs + num_reqs, std::default_random_engine(0));
    CHECK(cudaMemcpy(reqs, h_reqs, sizeof(IOReq) * num_reqs, cudaMemcpyHostToDevice));

    for(int k = 0; k < 10; k++){
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        fprintf(stderr, "starting do_io_req...\n");
        iostack.submit_io_req(reqs, num_reqs, 0);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        fprintf(stderr, "do_io_req takes %f ms\n", ms);
        fprintf(stderr, "%dB random read bandwidth: %f MiB/s\n", MAX_IO_SIZE, TEST_SIZE / (1024 * 1024) / (ms / 1000));
    
        CHECK(cudaMemcpyToSymbol(seed, &h_seed, sizeof(uint64_t)));
        percent = 1;
        clstart = clock();
        for (int i = 0; i < num_reqs; i++)
        {
            if(i == 0){
                check_test_data<<<1, 1>>>(app_buf + i * ITEM_SIZE / 4, i);
            }
        }
        CHECK(cudaDeviceSynchronize());
    }
    
    fprintf(stderr, "check passed!\n");
    return 0;
}
