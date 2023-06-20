#ifndef OPERATOR_H
#define OPERATOR_H

struct OpParams {
    int device_id;
    cudaStream_t stream;
    cudaEvent_t event;
    void* memorypool;
    void* cache;
    void* graph;
    void* feature;
    void* env;
    int neighbor_count;
    bool is_presc;
    bool in_memory;
};

class Operator {
public:
    virtual void run(OpParams* params) = 0;
};

Operator* NewBatchGenerateOP(int op_id);
Operator* NewRandomSampler(int op_id);
Operator* NewCacheLookupOP(int op_id);
Operator* NewSSDIOSubmitOP(int op_id);
Operator* NewSSDIOCompleteOP(int op_id);

#endif