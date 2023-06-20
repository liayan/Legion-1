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

struct IOReq
{
    uint64_t start_lb;
    uint64_t *dest_addr[MAX_ITEMS];
    int num_items;
};

__global__ static void do_io_req_kernel(IOReq *reqs, int num_reqs, int num_ssds, int num_warps_per_ssd, SSDQueuePair *ssdqp, uint64_t *prp1, uint64_t **IO_buf_base)
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
    uint64_t io_addr = prp1[ssd_id] + warp_id % num_warps_per_ssd * WARP_IOBUF_SIZE + sq_pos * MAX_IO_SIZE; // assume contiguous!
    uint64_t io_addr2 = io_addr / HOST_PGSZ * HOST_PGSZ + HOST_PGSZ;
    if (lane_id == 0)
    {
        ssdqp[warp_id].cmd_id = 0;
        // printf("queue %d cmd_id %d\n", warp_id, ssdqp[warp_id].cmd_id);
    }
    __syncwarp();
    ssdqp[warp_id].fill_sq(
        ssdqp[warp_id].cmd_id + lane_id,                                       // command id
        sq_pos,                                                                // position in SQ
        OPCODE_READ,                                                           // opcode
        io_addr,                                                               // prp1
        io_addr2,                                                              // prp2
        reqs[thread_id].start_lb & 0xffffffff,                                 // start lb low
        (reqs[thread_id].start_lb >> 32) & 0xffffffff,                         // start lb high
        RW_RETRY_MASK | (reqs[thread_id].num_items * ITEM_SIZE / LB_SIZE - 1), // number of LBs
        thread_id                                                              // req id
    );
    if (lane_id == 0)
    {
        ssdqp[warp_id].cmd_id += WARP_SIZE;
        ssdqp[warp_id].sq_tail = (ssdqp[warp_id].sq_tail + WARP_SIZE) % QUEUE_DEPTH;
        *ssdqp[warp_id].sqtdbl = ssdqp[warp_id].sq_tail;
    }

    int stride = blockDim.x * gridDim.x;
    for (int i = thread_id + stride; i < num_reqs + stride; i += stride)
    {
        base_req_id = i - lane_id; // first req_id in warp
        if (i < num_reqs)
        {
            // submit second page of double buffer
            int sq_pos = (ssdqp[warp_id].sq_tail + lane_id) % QUEUE_DEPTH;
            uint64_t io_addr = prp1[ssd_id] + warp_id % num_warps_per_ssd * WARP_IOBUF_SIZE + sq_pos * MAX_IO_SIZE; // assume contiguous!
            uint64_t io_addr2 = io_addr / HOST_PGSZ * HOST_PGSZ + HOST_PGSZ;
            ssdqp[warp_id].fill_sq(
                ssdqp[warp_id].cmd_id + lane_id,                               // command id
                sq_pos,                                                        // position in SQ
                OPCODE_READ,                                                   // opcode
                io_addr,                                                       // prp1
                io_addr2,                                                      // prp2
                reqs[i].start_lb & 0xffffffff,                                 // start lb low
                (reqs[i].start_lb >> 32) & 0xffffffff,                         // start lb high
                RW_RETRY_MASK | (reqs[i].num_items * ITEM_SIZE / LB_SIZE - 1), // number of LBs
                i                                                              // req id
            );
            int cnt = num_reqs - base_req_id < WARP_SIZE ? num_reqs - base_req_id : WARP_SIZE;
            if (lane_id == 0)
            {
                ssdqp[warp_id].cmd_id += cnt;
                ssdqp[warp_id].sq_tail = (ssdqp[warp_id].sq_tail + cnt) % QUEUE_DEPTH;
                *ssdqp[warp_id].sqtdbl = ssdqp[warp_id].sq_tail;
            }
        }

        // poll and copy the *previous* page of double buffer
        int cnt = num_reqs - (base_req_id - stride) < WARP_SIZE ? num_reqs - (base_req_id - stride) : WARP_SIZE;
        if (lane_id == 0)
        {
            uint32_t code = ssdqp[warp_id].poll_range(cnt, i < num_reqs);
            if (code)
            {
                // printf("read failed with status 0x%x\n", code);
                assert(0);
            }
        }
        // __syncwarp();
        for (int j = (ssdqp[warp_id].cq_head - cnt + QUEUE_DEPTH) % QUEUE_DEPTH; j != ssdqp[warp_id].cq_head; j = (j + 1) % QUEUE_DEPTH)
        {
            int cmd_id = (ssdqp[warp_id].cq[j * 4 + 3] & CID_MASK);
            int req_id = ssdqp[warp_id].cmd_id_to_req_id[cmd_id];
            int sq_pos = ssdqp[warp_id].cmd_id_to_sq_pos[cmd_id];
            // if (lane_id == 0)
            // {
            //     if (visit[req_id])
            //     {
            //         printf("req_id %d already visited\n", req_id);
            //         assert(0);
            //     }
            //     visit[req_id] = 1;
            // }
            // if (lane_id == 0 && warp_id == 0 && req_id < 32)
            //     printf("found req_id %d at cmd_id %d\n", req_id, cmd_id);
            for (int k = 0; k < reqs[req_id].num_items; k++)
                for (int l = lane_id; l < ITEM_SIZE / 8; l += WARP_SIZE)
                    reqs[req_id].dest_addr[k][l] = IO_buf_base[ssd_id][(warp_id % num_warps_per_ssd * WARP_IOBUF_SIZE + (sq_pos * MAX_ITEMS + k) * ITEM_SIZE) / 8 + l];
        }
    }
    if (thread_id % (num_warps_per_ssd * WARP_SIZE) == 0)
        printf("timestamp %ld ssd %d done\n", clock64(), ssd_id);
}

__global__ static void rw_data_kernel(uint32_t opcode, int ssd_id, uint64_t start_lb, uint64_t num_lb, int num_warps_per_ssd, SSDQueuePair *ssdqp, uint64_t *prp1, uint64_t **IO_buf_base)
{
    assert(LB_SIZE * num_lb <= HOST_PGSZ * 2);
    uint32_t cid;
    ssdqp[ssd_id * num_warps_per_ssd].submit(cid, opcode, prp1[ssd_id], prp1[ssd_id] + HOST_PGSZ, start_lb & 0xffffffff, (start_lb >> 32) & 0xffffffff, RW_RETRY_MASK | (num_lb - 1));
    uint32_t status;
    ssdqp[ssd_id * num_warps_per_ssd].poll(status, cid);
    if (status != 0)
    {
        printf("read/write failed with status 0x%x\n", status);
        assert(0);
    }
}

__global__ static void check_visit(bool *visit, int num_reqs)
{
    for (int i = 0; i < num_reqs; i++)
        if (!visit[i])
        {
            printf("req %d not visited\n", i);
            assert(0);
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

    void do_io_req(IOReq *req, int num_reqs, cudaStream_t stream)
    {
        int num_threads = num_ssds * num_warps_per_ssd * WARP_SIZE;
        int num_blocks = (num_threads + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
        do_io_req_kernel<<<num_blocks, NUM_THREADS_PER_BLOCK, 0, stream>>>(req, num_reqs, num_ssds, num_warps_per_ssd, d_ssdqp, d_prp1, d_IO_buf_base);
        // check_visit<<<1, 1>>>(d_visit, num_reqs);
    }

    void read_data(int ssd_id, uint64_t start_lb, uint64_t num_lb)
    {
        rw_data_kernel<<<1, 1>>>(OPCODE_READ, ssd_id, start_lb, num_lb, num_warps_per_ssd, d_ssdqp, d_prp1, d_IO_buf_base);
    }

    void write_data(int ssd_id, uint64_t start_lb, uint64_t num_lb)
    {
        rw_data_kernel<<<1, 1>>>(OPCODE_WRITE, ssd_id, start_lb, num_lb, num_warps_per_ssd, d_ssdqp, d_prp1, d_IO_buf_base);
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
    uint64_t *d_prp1;
    uint64_t **d_IO_buf_base, **h_IO_buf_base;
    void *reg_ptr;
    void *h_admin_queue;
    void *d_io_queue;
    void *d_io_buf;

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
        int qsize = QUEUE_DEPTH * SQ_ITEM_SIZE;
        assert(qsize % HOST_PGSZ == 0);
        uint64_t *phys = cudaMallocAlignedMapped(d_io_queue, qsize * 2 * num_warps_per_ssd, fd); // 2 stands for SQ and CQ
        CHECK(cudaMemset(d_io_queue, 0, qsize * 2 * num_warps_per_ssd));
        for (int i = 0; i < num_warps_per_ssd; i++)
        {
            uint64_t sq = (uint64_t)d_io_queue + qsize * (2 * i);
            uint64_t cq = (uint64_t)d_io_queue + qsize * (2 * i + 1);
            int qid = i + 1;
            int offset = qsize * (2 * i + 1);
            uint64_t prp1 = phys[offset / DEVICE_PGSZ] + offset % DEVICE_PGSZ;
            admin_qp.submit(cid, OPCODE_CREATE_IO_CQ, prp1, 0x0, ((QUEUE_DEPTH - 1) << 16) | qid, 0x1);
            admin_qp.poll(status, cid);
            if (status != 0)
            {
                fprintf(stderr, "create I/O CQ failed with status 0x%x\n", status);
                exit(1);
            }
            offset = qsize * (2 * i);
            prp1 = phys[offset / DEVICE_PGSZ] + offset % DEVICE_PGSZ;
            admin_qp.submit(cid, OPCODE_CREATE_IO_SQ, prp1, 0x0, ((QUEUE_DEPTH - 1) << 16) | qid, (qid << 16) | 0x1);
            admin_qp.poll(status, cid);
            if (status != 0)
            {
                fprintf(stderr, "create I/O SQ failed with status 0x%x\n", status);
                exit(1);
            }
            uint32_t *cmd_id_to_req_id;
            CHECK(cudaMalloc(&cmd_id_to_req_id, 65536 * 4));
            uint32_t *cmd_id_to_sq_pos;
            CHECK(cudaMalloc(&cmd_id_to_sq_pos, 65536 * 4));
            SSDQueuePair current_qp((volatile uint32_t *)sq, (volatile uint32_t *)cq, 0x1, (uint32_t *)(h_reg_ptr + REG_SQTDBL + DBL_STRIDE * qid), (uint32_t *)(h_reg_ptr + REG_CQHDBL + DBL_STRIDE * qid), QUEUE_DEPTH, cmd_id_to_req_id, cmd_id_to_sq_pos);
            CHECK(cudaMemcpy(d_ssdqp + ssd_id * num_warps_per_ssd + i, &current_qp, sizeof(SSDQueuePair), cudaMemcpyHostToDevice));
        }
        free(phys);
        fprintf(stderr, "create I/O queues done!\n");

        // alloc IO buffer
        phys = cudaMallocAlignedMapped(d_io_buf, WARP_IOBUF_SIZE * num_warps_per_ssd, fd);
        CHECK(cudaMemcpy(d_prp1 + ssd_id, phys, sizeof(uint64_t), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_IO_buf_base + ssd_id, &d_io_buf, sizeof(uint64_t), cudaMemcpyHostToDevice));
        h_IO_buf_base[ssd_id] = (uint64_t *)d_io_buf;
        free(phys);
    }

    uint64_t *cudaMallocAlignedMapped(void *&vaddr, size_t size, int fd)
    {
        size = size / DEVICE_PGSZ * DEVICE_PGSZ + DEVICE_PGSZ;
        uint64_t *ptr;
        CHECK(cudaMalloc(&ptr, size + DEVICE_PGSZ));
        vaddr = (void *)((uint64_t)ptr / DEVICE_PGSZ * DEVICE_PGSZ + DEVICE_PGSZ);
        nvm_ioctl_map req;
        req.ioaddrs = (uint64_t *)malloc(sizeof(uint64_t) * size / DEVICE_PGSZ);
        req.n_pages = size / DEVICE_PGSZ;
        req.vaddr_start = (uint64_t)vaddr;
        int err = ioctl(fd, NVM_MAP_DEVICE_MEMORY, &req);
        if (err)
        {
            printf("Failed to map: %s\n", strerror(errno));
            return nullptr;
        }
        return req.ioaddrs;
    }
};
