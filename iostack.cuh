#pragma once
#include "ssdqp.cuh"
#include "common.cuh"
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#define _CUDA
#include "ioctl.h"
#include <sys/ioctl.h>
#include <assert.h>

typedef uint64_t app_addr_t;
struct IOReq
{
    uint64_t start_lb;
    app_addr_t dest_addr[MAX_ITEMS];
    int num_items;

    __forceinline__ __host__ __device__ IOReq(){};

    __forceinline__
        __host__ __device__
        IOReq(uint64_t ssd, uint64_t length) : start_lb(ssd), num_items(length)
    {
        for (int i = 0; i < num_items; i++)
            dest_addr[i] = ~0ULL;
    }

    __host__ __device__ bool operator<(const IOReq &lhs) const
    {
        return this->start_lb < lhs.start_lb;
    }

    __forceinline__
        __host__ __device__ ~IOReq()
    {
        // delete[] gpu_addr;
    }
};

__global__ static void do_io_req_kernel(IOReq *reqs, int num_reqs, int num_ssds, int num_warps_per_ssd, SSDQueuePair *ssdqp, uint64_t *prp1, uint64_t **IO_buf_base, uint64_t *prp2, bool *req_processed)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int lane_id = thread_id % WARP_SIZE;
    int ssd_id = warp_id / num_warps_per_ssd;
    if (ssd_id >= num_ssds)
        return;

    // submit first page of double buffer
    assert(thread_id < num_reqs);
    int base_req_id = thread_id - lane_id;
    int sq_pos = (ssdqp[warp_id].sq_tail + lane_id) % QUEUE_DEPTH;
    uint64_t io_addr = prp1[ssd_id] + warp_id % num_warps_per_ssd * QUEUE_IOBUF_SIZE + sq_pos * MAX_IO_SIZE; // assume contiguous!
    uint64_t io_addr2 = io_addr / HOST_PGSZ * HOST_PGSZ + HOST_PGSZ;
    if (reqs[thread_id].num_items * ITEM_SIZE > HOST_PGSZ * 2)
    {
        int prp_size_per_ssd = PRP_SIZE * QUEUE_DEPTH * num_warps_per_ssd / HOST_PGSZ * HOST_PGSZ + HOST_PGSZ;
        uint64_t offset = warp_id % num_warps_per_ssd * PRP_SIZE * QUEUE_DEPTH + sq_pos * PRP_SIZE;
        io_addr2 = prp2[ssd_id * prp_size_per_ssd / HOST_PGSZ + offset / HOST_PGSZ] + offset % HOST_PGSZ;
    }
    if (lane_id == 0)
    {
        // ssdqp[warp_id].cmd_id = 0;
        // printf("queue %d cmd_id %d\n", warp_id, ssdqp[warp_id].cmd_id);
        for (int i = 0; i < QUEUE_DEPTH; i++)
            ssdqp[warp_id].sq_entry_busy[i] = false;
    }
    int num_lbs = reqs[thread_id].num_items ? reqs[thread_id].num_items * ITEM_SIZE / LB_SIZE - 1 : 0;
    ssdqp[warp_id].fill_sq(
        ssdqp[warp_id].cmd_id + lane_id,               // command id
        sq_pos,                                        // position in SQ
        OPCODE_READ,                                   // opcode
        io_addr,                                       // prp1
        io_addr2,                                      // prp2
        reqs[thread_id].start_lb & 0xffffffff,         // start lb low
        (reqs[thread_id].start_lb >> 32) & 0xffffffff, // start lb high
        RW_RETRY_MASK | num_lbs,                       // number of LBs
        thread_id                                      // req id
    );
    // printf("thread %d req_id %d cmd_id %d num_completed %d sq_pos %d\n", thread_id, thread_id, ssdqp[warp_id].cmd_id + lane_id, ssdqp[warp_id].num_completed, sq_pos);

    __threadfence_system();
    // __syncwarp();
    if (lane_id == 0)
    {
        ssdqp[warp_id].cmd_id += WARP_SIZE;
        ssdqp[warp_id].sq_tail = (ssdqp[warp_id].sq_tail + WARP_SIZE) % QUEUE_DEPTH;
        // printf("Warp %d, sq_tail is %p, set sqtdbl to %d\n", warp_id, ssdqp[warp_id].sqtdbl, ssdqp[warp_id].sq_tail);
        *ssdqp[warp_id].sqtdbl = ssdqp[warp_id].sq_tail;
    }

    int stride = num_ssds * num_warps_per_ssd * WARP_SIZE;
    for (int i = thread_id + stride; i < num_reqs + stride; i += stride)
    {
        int prev_sq_tail = ssdqp[warp_id].sq_tail;
        base_req_id = i - lane_id; // first req_id in warp
        if (i < num_reqs)
        {
            // submit second page of double buffer
            int sq_pos = (ssdqp[warp_id].sq_tail + lane_id) % QUEUE_DEPTH;
            uint64_t io_addr = prp1[ssd_id] + warp_id % num_warps_per_ssd * QUEUE_IOBUF_SIZE + sq_pos * MAX_IO_SIZE; // assume contiguous!
            uint64_t io_addr2 = io_addr / HOST_PGSZ * HOST_PGSZ + HOST_PGSZ;
            if (reqs[thread_id].num_items * ITEM_SIZE > HOST_PGSZ * 2)
            {
                int prp_size_per_ssd = PRP_SIZE * QUEUE_DEPTH * num_warps_per_ssd / HOST_PGSZ * HOST_PGSZ + HOST_PGSZ;
                uint64_t offset = warp_id % num_warps_per_ssd * PRP_SIZE * QUEUE_DEPTH + sq_pos * PRP_SIZE;
                io_addr2 = prp2[ssd_id * prp_size_per_ssd / HOST_PGSZ + offset / HOST_PGSZ] + offset % HOST_PGSZ;
            }
            assert(ssdqp[warp_id].sq_entry_busy[sq_pos] == false);
            // if (i >= stride * 4 && !req_processed[i - stride * 4])
            // {
            //     printf("thread %d req_id %d not processed\n", thread_id, i - stride * 4);
            //     for (int i = 0; i < ssdqp[warp_id].cmd_id; i++)
            //     {
            //         int req_id = ssdqp[warp_id].cmd_id_to_req_id[i];
            //         int sq_pos = ssdqp[warp_id].cmd_id_to_sq_pos[i];
            //         if (req_id != 0xffffffff)
            //             printf("thread %d cmd_id %d req_id %d processed %d sq_pos %d busy %d\n", thread_id, i, req_id, req_processed[req_id], sq_pos, ssdqp[warp_id].sq_entry_busy[sq_pos]);
            //     }
            //     assert(0);
            // }
            int num_lbs = reqs[i].num_items ? reqs[i].num_items * ITEM_SIZE / LB_SIZE - 1 : 0;
            ssdqp[warp_id].fill_sq(
                ssdqp[warp_id].cmd_id + lane_id,       // command id
                sq_pos,                                // position in SQ
                OPCODE_READ,                           // opcode
                io_addr,                               // prp1
                io_addr2,                              // prp2
                reqs[i].start_lb & 0xffffffff,         // start lb low
                (reqs[i].start_lb >> 32) & 0xffffffff, // start lb high
                RW_RETRY_MASK | num_lbs,               // number of LBs
                i                                      // req id
            );
            // printf("thread %d req_id %d cmd_id %d num_completed %d sq_pos %d\n", thread_id, i, ssdqp[warp_id].cmd_id + lane_id, ssdqp[warp_id].num_completed, sq_pos);

            __threadfence_system();
            // __syncwarp();
            if (lane_id == 0)
            {
                int cnt = num_reqs - base_req_id < WARP_SIZE ? num_reqs - base_req_id : WARP_SIZE;
                assert(cnt == WARP_SIZE);
                ssdqp[warp_id].cmd_id += cnt;
                ssdqp[warp_id].sq_tail = (ssdqp[warp_id].sq_tail + cnt) % QUEUE_DEPTH;
                // printf("Warp %d, sq_tail is %p, set sqtdbl to %d\n", warp_id, ssdqp[warp_id].sqtdbl, ssdqp[warp_id].sq_tail);
                *ssdqp[warp_id].sqtdbl = ssdqp[warp_id].sq_tail;
            }
        }

        // poll and copy the *previous* page of double buffer
        int prev_cq_head = ssdqp[warp_id].cq_head;
        if (lane_id == 0)
        {
            uint32_t code = ssdqp[warp_id].poll_range(prev_sq_tail, i < num_reqs);
            assert(code == 0);
            if (i + stride < num_reqs)
            {
                base_req_id += stride;
                int next_cnt = num_reqs - base_req_id < WARP_SIZE ? num_reqs - base_req_id : WARP_SIZE;
                for (int j = 0; j < next_cnt; j++)
                {
                    int sq_pos = (ssdqp[warp_id].sq_tail + j) % QUEUE_DEPTH;
                    if (ssdqp[warp_id].sq_entry_busy[sq_pos])
                    {
                        code = ssdqp[warp_id].poll_until_sq_entry_free(sq_pos);
                        assert(code == 0);
                    }
                }
            }
        }

        // copy data from IO buffer to app buffer
        for (int j = prev_cq_head; j != ssdqp[warp_id].cq_head; j = (j + 1) % QUEUE_DEPTH)
        {
            int cmd_id = (ssdqp[warp_id].cq[j * 4 + 3] & CID_MASK) % QUEUE_DEPTH;
            int req_id = ssdqp[warp_id].cmd_id_to_req_id[cmd_id];
            int sq_pos = ssdqp[warp_id].cmd_id_to_sq_pos[cmd_id];
            if (lane_id == 0)
            {
                // uint64_t first_data = IO_buf_base[ssd_id][(warp_id % num_warps_per_ssd * QUEUE_IOBUF_SIZE + (sq_pos * MAX_ITEMS + 8) * ITEM_SIZE) / 8];
                // if (first_data != (req_id * MAX_IO_SIZE + 4096) / 8)
                // {
                //     printf("thread %d read %lx, expected %x\n", thread_id, first_data, (req_id * MAX_IO_SIZE + 4096) / 8);
                //     printf("cmd_id %d, req_id %d, sq_pos %d, processed %d\n", cmd_id, req_id, sq_pos, req_processed[req_id]);
                //     assert(0);
                // }
                // if (req_processed[req_id])
                // {
                //     printf("req_id %d already processed\n", req_id);
                //     assert(0);
                // }
                // req_processed[req_id] = true;
            }
            for (int k = 0; k < reqs[req_id].num_items; k++)
                for (int l = lane_id; l < ITEM_SIZE / 8; l += WARP_SIZE)
                    ((uint64_t *)reqs[req_id].dest_addr[k])[l] = IO_buf_base[ssd_id][(warp_id % num_warps_per_ssd * QUEUE_IOBUF_SIZE + (sq_pos * MAX_ITEMS + k) * ITEM_SIZE) / 8 + l];
        }
    }
}

__global__ static void rw_data_kernel(uint32_t opcode, int ssd_id, uint64_t start_lb, uint64_t num_lb, int num_warps_per_ssd, SSDQueuePair *ssdqp, uint64_t *prp1, uint64_t **IO_buf_base, uint64_t *prp2)
{
    uint32_t cid;
    int prp_size_per_ssd = PRP_SIZE * QUEUE_DEPTH * num_warps_per_ssd / HOST_PGSZ * HOST_PGSZ + HOST_PGSZ;
    ssdqp[ssd_id * num_warps_per_ssd].submit(cid, opcode, prp1[ssd_id], LB_SIZE * num_lb <= HOST_PGSZ * 2 ? prp1[ssd_id] + HOST_PGSZ : prp2[ssd_id * prp_size_per_ssd / HOST_PGSZ], start_lb & 0xffffffff, (start_lb >> 32) & 0xffffffff, RW_RETRY_MASK | (num_lb - 1));
    uint32_t status;
    ssdqp[ssd_id * num_warps_per_ssd].poll(status, cid);
    if (status != 0)
    {
        printf("read/write failed with status 0x%x\n", status);
        assert(0);
    }
}

__device__ int buckets[MAX_SSDS_SUPPORTED], num_distributed_reqs;
__global__ static void preprocess_io_req_1(IOReq *reqs, int num_reqs, int num_ssds, int num_warps_per_ssd)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int i = tid; i < num_reqs; i += num_threads)
    {
        int ssd_id = reqs[i].start_lb / NUM_LBS_PER_SSD;
        assert(ssd_id < num_ssds);
        atomicAdd(&buckets[ssd_id], 1);
    }
}

__global__ static void preprocess_io_req_2(IOReq *reqs, int num_reqs, int num_ssds, int num_warps_per_ssd)
{
    int max_bucket = 0;
    for (int i = 0; i < num_ssds; i++)
        if (buckets[i] > max_bucket)
            max_bucket = buckets[i];
    int num_reqs_per_chunk = num_warps_per_ssd * WARP_SIZE;
    max_bucket = (max_bucket + num_reqs_per_chunk - 1) / num_reqs_per_chunk * num_reqs_per_chunk;
    num_distributed_reqs = max_bucket * num_ssds;
}

__device__ int req_ids[MAX_SSDS_SUPPORTED];
__global__ static void distribute_io_req_1(IOReq *reqs, int num_reqs, int num_ssds, int num_warps_per_ssd, IOReq *distributed_reqs)
{
    int num_reqs_per_chunk = num_warps_per_ssd * WARP_SIZE;
    for (int i = 0; i < num_ssds; i++)
        req_ids[i] = i * num_reqs_per_chunk;
}

__global__ static void distribute_io_req_2(IOReq *reqs, int num_reqs, int num_ssds, int num_warps_per_ssd, IOReq *distributed_reqs)
{
    int num_reqs_per_chunk = num_warps_per_ssd * WARP_SIZE;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int i = tid; i < num_reqs; i += num_threads)
    {
        int ssd_id = reqs[i].start_lb / NUM_LBS_PER_SSD;
        assert(ssd_id < num_ssds);
        int req_id;
        for (;;)
        {
            req_id = req_ids[ssd_id];
            int next_req_id = req_id + 1;
            if (next_req_id % num_reqs_per_chunk == 0)
                next_req_id += num_reqs_per_chunk * (num_ssds - 1);
            if (atomicCAS(&req_ids[ssd_id], req_id, next_req_id) == req_id)
                break;
        }
        distributed_reqs[req_id] = reqs[i];
        distributed_reqs[req_id].start_lb %= NUM_LBS_PER_SSD;
    }
}

__global__ static void distribute_io_req_3(IOReq *reqs, int num_reqs, int num_ssds, int num_warps_per_ssd, IOReq *distributed_reqs)
{
    int num_reqs_per_chunk = num_warps_per_ssd * WARP_SIZE;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int i = tid; i < num_ssds; i += num_threads)
        for (int j = req_ids[i]; j < num_distributed_reqs;)
        {
            distributed_reqs[j].num_items = 0;
            distributed_reqs[j++].start_lb = 0;
            if (j % num_reqs_per_chunk == 0)
                j += num_reqs_per_chunk * (num_ssds - 1);
        }
}

class IOStack
{
public:
    IOStack(int num_ssds, int num_warps_per_ssd) : num_ssds(num_ssds), num_warps_per_ssd(num_warps_per_ssd)
    {
        // alloc device variables
        CHECK(cudaMalloc(&d_ssdqp, num_ssds * num_warps_per_ssd * sizeof(SSDQueuePair)));
        CHECK(cudaMalloc(&d_prp1, num_ssds * sizeof(uint64_t)));
        int prp_size_per_ssd = PRP_SIZE * QUEUE_DEPTH * num_warps_per_ssd / HOST_PGSZ * HOST_PGSZ + HOST_PGSZ;
        CHECK(cudaMalloc(&d_prp2, num_ssds * prp_size_per_ssd / HOST_PGSZ * sizeof(uint64_t)));
        CHECK(cudaMalloc(&d_IO_buf_base, num_ssds * sizeof(uint64_t *)));
        h_IO_buf_base = (uint64_t **)malloc(sizeof(uint64_t *) * num_ssds);

        // init ssds
        for (int i = 0; i < num_ssds; i++)
            init_ssd(i);
    }

    ~IOStack()
    {
        CHECK(cudaFree(d_ssdqp));
        CHECK(cudaFree(d_prp1));
        CHECK(cudaFree(d_IO_buf_base));
        munmap(reg_ptr, REG_SIZE);
        free(h_admin_queue);
        // CHECK(cudaFree(d_io_queue));
        // CHECK(cudaFree(d_io_buf));
        free(h_IO_buf_base);
    }

    void do_io_req(IOReq *reqs, int num_reqs, cudaStream_t stream)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        int h_num_distributed_reqs;
        preprocess_io_req_1<<<100, NUM_THREADS_PER_BLOCK, 0, stream>>>(reqs, num_reqs, num_ssds, num_warps_per_ssd);
        preprocess_io_req_2<<<1, 1, 0, stream>>>(reqs, num_reqs, num_ssds, num_warps_per_ssd);
        CHECK(cudaMemcpyFromSymbol(&h_num_distributed_reqs, num_distributed_reqs, sizeof(int)));
        fprintf(stderr, "num_reqs %d to %d\n", num_reqs, h_num_distributed_reqs);
        IOReq *distributed_reqs;
        CHECK(cudaMalloc(&distributed_reqs, h_num_distributed_reqs * sizeof(IOReq)));
        distribute_io_req_1<<<1, 1, 0, stream>>>(reqs, num_reqs, num_ssds, num_warps_per_ssd, distributed_reqs);
        distribute_io_req_2<<<100, NUM_THREADS_PER_BLOCK, 0, stream>>>(reqs, num_reqs, num_ssds, num_warps_per_ssd, distributed_reqs);
        distribute_io_req_3<<<100, NUM_THREADS_PER_BLOCK, 0, stream>>>(reqs, num_reqs, num_ssds, num_warps_per_ssd, distributed_reqs);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        fprintf(stderr, "distribute takes %f ms\n", ms);
        int num_threads = num_ssds * num_warps_per_ssd * WARP_SIZE;
        int num_blocks = (num_threads + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
        do_io_req_kernel<<<num_blocks, NUM_THREADS_PER_BLOCK, 0, stream>>>(distributed_reqs, h_num_distributed_reqs, num_ssds, num_warps_per_ssd, d_ssdqp, d_prp1, d_IO_buf_base, d_prp2, req_processed);
        CHECK(cudaFree(distributed_reqs));
    }

    void read_data(int ssd_id, uint64_t start_lb, uint64_t num_lb)
    {
        rw_data_kernel<<<1, 1>>>(OPCODE_READ, ssd_id, start_lb, num_lb, num_warps_per_ssd, d_ssdqp, d_prp1, d_IO_buf_base, d_prp2);
    }

    void write_data(int ssd_id, uint64_t start_lb, uint64_t num_lb)
    {
        rw_data_kernel<<<1, 1>>>(OPCODE_WRITE, ssd_id, start_lb, num_lb, num_warps_per_ssd, d_ssdqp, d_prp1, d_IO_buf_base, d_prp2);
    }

    uint64_t **get_d_io_buf_base()
    {
        return d_IO_buf_base;
    }

    uint64_t **get_h_io_buf_base()
    {
        return h_IO_buf_base;
    }

    // private:
    int num_ssds;
    int num_warps_per_ssd;
    SSDQueuePair *d_ssdqp;
    uint64_t *d_prp1, *d_prp2;
    uint64_t **d_IO_buf_base, **h_IO_buf_base;
    void *reg_ptr;
    void *h_admin_queue;
    void *d_io_queue;
    void *d_io_buf;
    bool *req_processed;

    void init_ssd(int ssd_id)
    {
        // open file and map BAR
        fprintf(stderr, "setting up SSD %d\n", ssd_id);
        char fname[20];
        sprintf(fname, "/dev/libnvm%d", ssd_id);
        int fd = open(fname, O_RDWR);
        if (fd < 0)
        {
            fprintf(stderr, "Failed to open: %s\n", strerror(errno));
            exit(1);
        }
        reg_ptr = mmap(NULL, REG_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_LOCKED, fd, 0);
        if (reg_ptr == MAP_FAILED)
        {
            fprintf(stderr, "Failed to mmap: %s\n", strerror(errno));
            exit(1);
        }
        CHECK(cudaHostRegister(reg_ptr, REG_SIZE, cudaHostRegisterIoMemory));

        // reset controller
        uint64_t h_reg_ptr = (uint64_t)reg_ptr;
        *(uint32_t *)(h_reg_ptr + REG_CC) &= ~REG_CC_EN;
        while (*(uint32_t volatile *)(h_reg_ptr + REG_CSTS) & REG_CSTS_RDY)
            ;
        fprintf(stderr, "reset done\n");

        // set admin_qp queue attributes
        *(uint32_t *)(h_reg_ptr + REG_AQA) = ((ADMIN_QUEUE_DEPTH - 1) << 16) | (ADMIN_QUEUE_DEPTH - 1);
        posix_memalign(&h_admin_queue, HOST_PGSZ, HOST_PGSZ * 2);
        memset(h_admin_queue, 0, HOST_PGSZ * 2);
        nvm_ioctl_map req; // convert to physical address
        req.vaddr_start = (uint64_t)h_admin_queue;
        req.n_pages = 2;
        req.ioaddrs = (uint64_t *)malloc(sizeof(uint64_t) * 2);
        int err = ioctl(fd, NVM_MAP_HOST_MEMORY, &req);
        if (err)
        {
            fprintf(stderr, "Failed to map admin_qp queue: %s\n", strerror(errno));
            exit(1);
        }
        uint64_t asq = (uint64_t)h_admin_queue;
        *(uint64_t *)(h_reg_ptr + REG_ASQ) = req.ioaddrs[0];
        uint64_t acq = (uint64_t)h_admin_queue + HOST_PGSZ;
        *(uint64_t *)(h_reg_ptr + REG_ACQ) = req.ioaddrs[1];
        SSDQueuePair admin_qp((volatile uint32_t *)asq, (volatile uint32_t *)acq, BROADCAST_NSID, (uint32_t *)(h_reg_ptr + REG_SQTDBL), (uint32_t *)(h_reg_ptr + REG_CQHDBL), ADMIN_QUEUE_DEPTH);
        fprintf(stderr, "set admin_qp queue attributes done\n");

        // enable controller
        *(uint32_t *)(h_reg_ptr + REG_CC) |= REG_CC_EN;
        while (!(*(uint32_t volatile *)(h_reg_ptr + REG_CSTS) & REG_CSTS_RDY))
            ;
        fprintf(stderr, "enable controller done\n");

        // set number of I/O queues
        uint32_t cid;
        admin_qp.submit(cid, OPCODE_SET_FEATURES, 0x0, 0x0, FID_NUM_QUEUES,
                        ((num_warps_per_ssd - 1) << 16) | (num_warps_per_ssd - 1));
        uint32_t status;
        admin_qp.poll(status, cid);
        if (status != 0)
        {
            fprintf(stderr, "set number of queues failed with status 0x%x\n", status);
            exit(1);
        }
        fprintf(stderr, "set number of queues done!\n");

        // create I/O queues
        int sq_size = QUEUE_DEPTH * SQ_ITEM_SIZE;
        assert(sq_size % HOST_PGSZ == 0);
        uint64_t *phys = cudaMallocAlignedMapped(d_io_queue, sq_size * 2 * num_warps_per_ssd, fd); // 2 stands for SQ and CQ
        CHECK(cudaMemset(d_io_queue, 0, sq_size * 2 * num_warps_per_ssd));
        for (int i = 0; i < num_warps_per_ssd; i++)
        {
            uint64_t sq = (uint64_t)d_io_queue + sq_size * (2 * i);
            uint64_t cq = (uint64_t)d_io_queue + sq_size * (2 * i + 1);
            int qid = i + 1;
            int offset = sq_size * (2 * i + 1);
            uint64_t prp1 = phys[offset / DEVICE_PGSZ] + offset % DEVICE_PGSZ;
            admin_qp.submit(cid, OPCODE_CREATE_IO_CQ, prp1, 0x0, ((QUEUE_DEPTH - 1) << 16) | qid, 0x1);
            admin_qp.poll(status, cid);
            if (status != 0)
            {
                fprintf(stderr, "create I/O CQ failed with status 0x%x\n", status);
                exit(1);
            }
            offset = sq_size * (2 * i);
            prp1 = phys[offset / DEVICE_PGSZ] + offset % DEVICE_PGSZ;
            admin_qp.submit(cid, OPCODE_CREATE_IO_SQ, prp1, 0x0, ((QUEUE_DEPTH - 1) << 16) | qid, (qid << 16) | 0x1);
            admin_qp.poll(status, cid);
            if (status != 0)
            {
                fprintf(stderr, "create I/O SQ failed with status 0x%x\n", status);
                exit(1);
            }
            uint32_t *cmd_id_to_req_id;
            CHECK(cudaMalloc(&cmd_id_to_req_id, QUEUE_DEPTH * 4));
            uint32_t *cmd_id_to_sq_pos;
            CHECK(cudaMalloc(&cmd_id_to_sq_pos, QUEUE_DEPTH * 4));
            bool *sq_entry_busy;
            CHECK(cudaMalloc(&sq_entry_busy, QUEUE_DEPTH));
            CHECK(cudaMemset(sq_entry_busy, 0, QUEUE_DEPTH));
            SSDQueuePair current_qp((volatile uint32_t *)sq, (volatile uint32_t *)cq, 0x1, (uint32_t *)(h_reg_ptr + REG_SQTDBL + DBL_STRIDE * qid), (uint32_t *)(h_reg_ptr + REG_CQHDBL + DBL_STRIDE * qid), QUEUE_DEPTH, cmd_id_to_req_id, cmd_id_to_sq_pos, sq_entry_busy);
            CHECK(cudaMemcpy(d_ssdqp + ssd_id * num_warps_per_ssd + i, &current_qp, sizeof(SSDQueuePair), cudaMemcpyHostToDevice));
        }
        // free(phys);
        fprintf(stderr, "create I/O queues done!\n");

        // alloc IO buffer
        phys = cudaMallocAlignedMapped(d_io_buf, QUEUE_IOBUF_SIZE * num_warps_per_ssd, fd);
        CHECK(cudaMemcpy(d_prp1 + ssd_id, phys, sizeof(uint64_t), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_IO_buf_base + ssd_id, &d_io_buf, sizeof(uint64_t), cudaMemcpyHostToDevice));
        h_IO_buf_base[ssd_id] = (uint64_t *)d_io_buf;
        // for (int i = 0; i < QUEUE_IOBUF_SIZE * num_warps_per_ssd / DEVICE_PGSZ; i++)
        //     printf("%lx\n", phys[i]);

        // build PRP list
        assert(PRP_SIZE <= HOST_PGSZ);
        int prp_size_per_ssd = PRP_SIZE * QUEUE_DEPTH * num_warps_per_ssd / HOST_PGSZ * HOST_PGSZ + HOST_PGSZ; // 1 table per ssd in host memory
        // printf("prp_size_per_ssd: %d\n", prp_size_per_ssd);
        void *tmp;
        posix_memalign(&tmp, HOST_PGSZ, prp_size_per_ssd);
        memset(tmp, 0, prp_size_per_ssd);
        uint64_t *prp = (uint64_t *)tmp;
        for (int i = 0; i < QUEUE_DEPTH * num_warps_per_ssd; i++)
            for (int j = 1; j < NUM_PRP_ENTRIES; j++)
            {
                int prp_idx = i * NUM_PRP_ENTRIES + j;
                int offset = i * MAX_IO_SIZE + j * HOST_PGSZ;
                prp[prp_idx - 1] = phys[offset / DEVICE_PGSZ] + offset % DEVICE_PGSZ;
            }
        // Fill in each PRP table
        // free(phys);
        // for (int i = 0; i < QUEUE_DEPTH * num_warps_per_ssd * NUM_PRP_ENTRIES; i++)
        //     printf("%lx\n", prp[i]);
        req.vaddr_start = (uint64_t)prp;
        req.n_pages = prp_size_per_ssd / HOST_PGSZ;
        req.ioaddrs = (uint64_t *)malloc(sizeof(uint64_t) * req.n_pages);
        // req.ioaddrs[0] is a physical pointer to PRP table
        err = ioctl(fd, NVM_MAP_HOST_MEMORY, &req);
        if (err)
        {
            fprintf(stderr, "Failed to map: %s\n", strerror(errno));
            exit(1);
        }
        // for (int i = 0; i < req.n_pages; i++)
        //     printf("%lx ", req.ioaddrs[i]);
        // printf("\n");
        CHECK(cudaMemcpy(d_prp2 + ssd_id * req.n_pages, req.ioaddrs, req.n_pages * sizeof(uint64_t), cudaMemcpyHostToDevice));
        // d_prp2 is an array of physical pointer to PRP table
    }

    uint64_t *cudaMallocAlignedMapped(void *&vaddr, size_t size, int fd)
    {
        size = size / DEVICE_PGSZ * DEVICE_PGSZ + DEVICE_PGSZ;
        uint64_t *ptr;
        CHECK(cudaMalloc(&ptr, size + DEVICE_PGSZ));
        vaddr = (void *)((uint64_t)ptr / DEVICE_PGSZ * DEVICE_PGSZ + DEVICE_PGSZ);
        int flag = 0;
        if ((uint64_t)vaddr != (uint64_t)ptr)
        {
            flag = 1;
        }
        nvm_ioctl_map req;
        req.ioaddrs = (uint64_t *)malloc(sizeof(uint64_t) * (size / DEVICE_PGSZ + flag));
        req.n_pages = size / DEVICE_PGSZ + flag;
        req.vaddr_start = (uint64_t)ptr;
        int err = ioctl(fd, NVM_MAP_DEVICE_MEMORY, &req);
        if (err)
        {
            printf("Failed to map: %s\n", strerror(errno));
            return nullptr;
        }
        return req.ioaddrs + flag;
    }
};
