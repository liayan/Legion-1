#pragma once
#include <stdint.h>
#include <assert.h>
#include "common.cuh"
class SSDQueuePair
{
public:
    volatile uint32_t *sq;
    volatile uint32_t *cq;
    uint32_t sq_tail;
    uint32_t cq_head;
    uint32_t cmd_id;
    uint32_t namespace_id;
    uint32_t *sqtdbl, *cqhdbl;
    uint32_t *cmd_id_to_req_id;
    uint32_t *cmd_id_to_sq_pos;
    uint32_t queue_depth;

    __host__ __device__ SSDQueuePair()
    {
    }

    __host__ __device__ SSDQueuePair(volatile uint32_t *sq, volatile uint32_t *cq, uint32_t namespace_id, uint32_t *sqtdbl, uint32_t *cqhdbl, uint32_t queue_depth, uint32_t *cmd_id_to_req_id = nullptr, uint32_t *cmd_id_to_sq_pos = nullptr)
        : sq(sq), cq(cq), sq_tail(0), cq_head(0), cmd_id(0), namespace_id(namespace_id), sqtdbl(sqtdbl), cqhdbl(cqhdbl), cmd_id_to_req_id(cmd_id_to_req_id), cmd_id_to_sq_pos(cmd_id_to_sq_pos), queue_depth(queue_depth)
    {
    }

    __host__ __device__ void submit(uint32_t &cid, uint32_t opcode, uint64_t prp1, uint64_t prp2, uint32_t dw10, uint32_t dw11, uint32_t dw12 = 0)
    {
        // printf("%lx %lx %x %x %x %x %x\n", prp1, prp2, dw10, dw11, dw12, opcode, cmd_id);
        fill_sq(cmd_id, sq_tail, opcode, prp1, prp2, dw10, dw11, dw12);
        sq_tail = (sq_tail + 1) % queue_depth;
        *sqtdbl = sq_tail;
        cid = cmd_id;
        cmd_id = (cmd_id + 1) & CID_MASK;
    }

    __host__ __device__ void fill_sq(uint32_t cid, uint32_t pos, uint32_t opcode, uint64_t prp1, uint64_t prp2, uint32_t dw10, uint32_t dw11, uint32_t dw12 = 0, uint32_t req_id = 0)
    {
        // if (req_id == 1152)
        //     printf("%lx %lx %x %x %x %x %x %x\n", prp1, prp2, dw10, dw11, dw12, opcode, cid, namespace_id);
        sq[pos * 16 + 0] = opcode | (cid << 16);
        sq[pos * 16 + 1] = namespace_id;
        sq[pos * 16 + 6] = prp1 & 0xffffffff;
        sq[pos * 16 + 7] = prp1 >> 32;
        sq[pos * 16 + 8] = prp2 & 0xffffffff;
        sq[pos * 16 + 9] = prp2 >> 32;
        sq[pos * 16 + 10] = dw10;
        sq[pos * 16 + 11] = dw11;
        sq[pos * 16 + 12] = dw12;
        if (cmd_id_to_req_id)
            cmd_id_to_req_id[cid] = req_id;
        if (cmd_id_to_sq_pos)
            cmd_id_to_sq_pos[cid] = pos;
    }

    __host__ __device__ void poll(uint32_t &code, uint32_t cid)
    {
        uint32_t current_phase = ((cmd_id - 1) / queue_depth) & 1;
        uint32_t status = cq[cq_head * 4 + 3];
        while (((status & PHASE_MASK) >> 16) == current_phase)
            status = cq[cq_head * 4 + 3];
        if ((status & CID_MASK) != cid)
        {
            printf("expected cid: %d, actual cid: %d\n", cid, status & CID_MASK);
            assert(0);
        }
        cq_head = (cq_head + 1) % queue_depth;
        *cqhdbl = cq_head;
        code = (status >> 17) & SC_MASK;
    }

    __device__ uint32_t poll_range(int size, bool double_buffer)
    {
        uint32_t current_phase = ((cmd_id - size - double_buffer * WARP_SIZE) / queue_depth) & 1;
        // printf("cmd_id: %d, size: %d, current_phase: %d\n", cmd_id, size, current_phase);
        for (int i = cq_head; i != (cq_head + size) % queue_depth; i = (i + 1) % queue_depth)
        {
            uint32_t status = cq[i * 4 + 3];
            while (((status & PHASE_MASK) >> 16) == current_phase)
                status = cq[i * 4 + 3];
            if ((status >> 17) & SC_MASK != 0)
            {
                printf("cq[%d] status: 0x%x, cid: %d\n", i, (status >> 17) & SC_MASK, status & CID_MASK);
                int cmd_id = status & CID_MASK;
                int req_id = cmd_id_to_req_id[cmd_id];
                int sq_pos = cmd_id_to_sq_pos[cmd_id];
                printf("req_id: %d, sq_pos: %d\n", req_id, sq_pos);
                // for (int i = 0; i < 16; i++)
                //     printf("%08x ", sq[sq_pos * 16 + i]);
                // printf("\n");
                return (status >> 17) & SC_MASK;
            }
        }
        cq_head = (cq_head + size) % queue_depth;
        *cqhdbl = cq_head;
        return 0;
    }
};
