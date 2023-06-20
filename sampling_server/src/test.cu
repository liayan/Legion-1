#include "iostack.cuh"
#include <unordered_set>
#include <algorithm>
#include <random>
#include <assert.h>
#include <unistd.h>
#define TEST_SIZE 0x10000000
#define MAX_LBS 0x100000000
#define APP_BUF_SIZE 0x10000000
#define NUM_WARPS_PER_SSD 32
#define NUM_SSDS 8

__device__ uint64_t **IO_buf_base;

__device__ uint64_t seed;
__global__ void gen_test_data(int ssd_id, int req_id)
{
    for (int i = 0; i < MAX_IO_SIZE / 8; i++)
    {
        seed = seed * 0x5deece66d + 0xb;
        IO_buf_base[ssd_id][i] = req_id * MAX_IO_SIZE / 8 + i;
    }
}

__global__ void check_test_data(uint64_t *app_buf, int idx)
{
    for (int i = 0; i < MAX_IO_SIZE / 8; i++)
    {
        seed = seed * 0x5deece66d + 0xb;
        if (app_buf[i] != idx * MAX_IO_SIZE / 8 + i)
        {
            printf("check failed at block %d, i = %d, read %lx, expected %x\n", idx, i, app_buf[i], idx * MAX_IO_SIZE / 8 + i);
            assert(0);
        }
    }
}

__global__ void fill_app_buf(uint64_t *app_buf)
{
    for (int i = 0; i < TEST_SIZE / 8; i++)
        app_buf[i] = i;
}

int main()
{
    IOStack iostack(NUM_SSDS, NUM_WARPS_PER_SSD);
    uint64_t **ptr = iostack.get_d_io_buf_base(); // rename to avoid confusion
    CHECK(cudaMemcpyToSymbol(IO_buf_base, &ptr, sizeof(uint64_t **)));
    uint64_t **hptr = iostack.get_h_io_buf_base();

    // test do_io_req
    uint64_t *app_buf;
    CHECK(cudaMalloc(&app_buf, APP_BUF_SIZE));
    fill_app_buf<<<1, 1>>>(app_buf);
    int num_reqs = TEST_SIZE / MAX_IO_SIZE;
    IOReq *reqs;
    CHECK(cudaMalloc(&reqs, sizeof(IOReq) * num_reqs));
    IOReq *h_reqs;
    CHECK(cudaHostAlloc(&h_reqs, sizeof(IOReq) * num_reqs, cudaHostAllocMapped));
    std::unordered_set<uint64_t> lbs;
    srand(0);
    uint64_t h_seed = 0;
    CHECK(cudaMemcpyToSymbol(seed, &h_seed, sizeof(uint64_t)));
    int percent = 1;
    clock_t clstart = clock();
    for (int i = 0; i < num_reqs; i++)
    {
        uint64_t lb;
        do
        {
            lb = rand() % (MAX_LBS / MAX_ITEMS);
        } while (lbs.find(lb) != lbs.end());
        lbs.insert(lb);
        h_reqs[i].start_lb = lb * MAX_ITEMS;
        h_reqs[i].num_items = MAX_ITEMS;
        for (int j = 0; j < MAX_ITEMS; j++)
            // h_reqs[i].dest_addr[j] = app_buf + i * MAX_IO_SIZE / 8 + j * ITEM_SIZE / 8;
            h_reqs[i].dest_addr[j] = app_buf + (1ll * i * MAX_IO_SIZE + j * ITEM_SIZE) % APP_BUF_SIZE / 8;
        int ssd_id = (i / WARP_SIZE / NUM_WARPS_PER_SSD) % NUM_SSDS;
        // printf("%d %d\n", i, ssd_id);
        // CHECK(cudaMemcpyFromSymbol(&h_seed, seed, sizeof(uint64_t)));
        gen_test_data<<<1, 1>>>(ssd_id, i);
        iostack.write_data(ssd_id, h_reqs[i].start_lb, MAX_IO_SIZE / LB_SIZE);
        // CHECK(cudaMemset(hptr[ssd_id], 0, MAX_IO_SIZE));
        // iostack.read_data(ssd_id, h_reqs[i].start_lb, MAX_IO_SIZE / LB_SIZE);
        // CHECK(cudaMemcpyToSymbol(seed, &h_seed, sizeof(uint64_t)));
        // check_test_data<<<1, 1>>>(hptr[ssd_id], i);
        if (i >= num_reqs / 100 * percent)
        {
            double eta = (clock() - clstart) / (double)CLOCKS_PER_SEC / percent * (100 - percent);
            printf("generating test data: %d%% done, eta %.0lfs\r", percent, eta);
            percent++;
        }
    }
    CHECK(cudaDeviceSynchronize());
    // std::shuffle(h_reqs, h_reqs + num_reqs, std::default_random_engine(0));
    // for (int i = 0; i < 10; i++)
    //     printf("%lx\n", h_reqs[i].start_lb);
    CHECK(cudaMemcpy(reqs, h_reqs, sizeof(IOReq) * num_reqs, cudaMemcpyHostToDevice));

    // printf("Press enter to start testing...\n");
    // getchar();
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    printf("starting do_io_req...\n");
    iostack.do_io_req(reqs, num_reqs, 0);
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float ms;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("do_io_req takes %f ms\n", ms);
    printf("%dB random read bandwidth: %f MiB/s\n", MAX_IO_SIZE, TEST_SIZE / (1024 * 1024) / (ms / 1000));

    // uint32_t *cmd_id_to_req_id = (uint32_t *)malloc(65536 * 4);
    // std::unordered_set<int> req_ids;
    // for (int i = 0; i < NUM_SSDS * NUM_WARPS_PER_SSD; i++)
    // {
    //     SSDQueuePair qp;
    //     CHECK(cudaMemcpy(&qp, iostack.d_ssdqp + i, sizeof(SSDQueuePair), cudaMemcpyDeviceToHost));
    //     CHECK(cudaMemcpy(cmd_id_to_req_id, qp.cmd_id_to_req_id, 65536 * 4, cudaMemcpyDeviceToHost));
    //     for (int j = 0; j < qp.cmd_id; j++)
    //     {
    //         if (req_ids.find(cmd_id_to_req_id[j]) != req_ids.end())
    //             printf("duplicate req_id %d at queue %d cid %d\n", cmd_id_to_req_id[j], i, j);
    //         req_ids.insert(cmd_id_to_req_id[j]);
    //     }
    // }
    // for (int i = 0; i < num_reqs; i++)
    //     if (req_ids.find(i) == req_ids.end())
    //         printf("req_id %d not found\n", i);

    CHECK(cudaMemcpyToSymbol(seed, &h_seed, sizeof(uint64_t)));
    percent = 1;
    clstart = clock();
    for (int i = 0; i < num_reqs; i++)
    {
        check_test_data<<<1, 1>>>(app_buf + i * MAX_IO_SIZE / 8, i);
        if (i >= num_reqs / 100 * percent)
        {
            double eta = (clock() - clstart) / (double)CLOCKS_PER_SEC / percent * (100 - percent);
            printf("checking: %d%% done, eta %.0lfs\r", percent, eta);
            percent++;
        }
    }
    CHECK(cudaDeviceSynchronize());
    printf("check passed!\n");
    return 0;
}