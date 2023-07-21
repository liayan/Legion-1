#include <iostream>
#include <climits>
// #include <iomanip>
#include <stdlib.h>
#include <cuda.h>
#include <vector>
#include <algorithm>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/shuffle.h>
#include <unistd.h>
#include <cub/cub.cuh>
// #include

#define my_ULLONG_MAX (~0ULL)
#define MAX_ITEMS 16//when setting to 32 will cause compilation error
using namespace std;

typedef uint64_t app_addr_t;


// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }


struct cache_id{
    uint64_t buffer_id;
    uint64_t ssd_id;
    __forceinline__
    __host__ __device__
    cache_id(){
        buffer_id=my_ULLONG_MAX;
        ssd_id=my_ULLONG_MAX;
    }
    __forceinline__
    __host__ __device__
    cache_id(uint64_t ssd,uint64_t buffer):ssd_id(ssd),buffer_id(buffer){}
    
    __host__ __device__
    bool operator < (const cache_id& lhs)const{
        return this->ssd_id<lhs.ssd_id;
    }
};

struct Compare{
    __device__ 
    bool operator()(const IOReq& a, const IOReq& b) const
        {
            return a.start_lb < b.start_lb;
        }

    __device__ 
    bool operator()(const cache_id& a,const cache_id& b) const{
        return a.ssd_id < b.ssd_id;
    }
};

__global__
void print_cacheid(cache_id *D_array,uint64_t len){
    for(uint64_t i=0;i<len;i++){
        printf("D_array[%lu]=%lu\n",i,D_array[i].ssd_id);
    }
}

__global__
void print_uint64(uint64_t *D_array,uint64_t len){
    for(uint64_t i=0;i<len;i++){
        printf("D_array[%lu]=%lu\n",i,D_array[i]);
    }
}

__global__
void print_Dret(IOReq* d_ret,uint64_t len,void *ssd_start_addr,void *dst_float_buffer,uint64_t ssd_block_size){
    printf("print_Dret\n");
    for(uint64_t i=0;i<len;i++){
        printf("cache:%lu\n",((uint64_t)d_ret[i].start_lb-(uint64_t)ssd_start_addr)/ssd_block_size);
        printf("gpu0:%lu\n",((uint64_t)d_ret[i].dest_addr[0]-(uint64_t)dst_float_buffer)/ssd_block_size);
        printf("gpu1:%lu\n",((uint64_t)d_ret[i].dest_addr[1]-(uint64_t)dst_float_buffer)/ssd_block_size);
        printf("gpu2:%lu\n",((uint64_t)d_ret[i].dest_addr[2]-(uint64_t)dst_float_buffer)/ssd_block_size);
        printf("gpu3:%lu\n",((uint64_t)d_ret[i].dest_addr[3]-(uint64_t)dst_float_buffer)/ssd_block_size);
        printf("gpu4:%lu\n",((uint64_t)d_ret[i].dest_addr[4]-(uint64_t)dst_float_buffer)/ssd_block_size);
        printf("gpu5:%lu\n",((uint64_t)d_ret[i].dest_addr[5]-(uint64_t)dst_float_buffer)/ssd_block_size);
        printf("gpu6:%lu\n",((uint64_t)d_ret[i].dest_addr[6]-(uint64_t)dst_float_buffer)/ssd_block_size);
        printf("gpu7:%lu\n\n",((uint64_t)d_ret[i].dest_addr[7]-(uint64_t)dst_float_buffer)/ssd_block_size);
    }
}

__global__
void print_Dret_ssd(uint64_t* d_ret_ssd,uint64_t len,void *ssd_start_addr,uint64_t ssd_block_size){
    printf("print_Dret_ssd\n");
    for(uint64_t i=0;i<len;i++){
        // if(i==0){
        //     printf("the first origin ssd:%lu\n",(uint64_t)d_ret_ssd[i]);
        // }
        printf("ssd:%lu\n",((uint64_t)d_ret_ssd[i]-(uint64_t)ssd_start_addr)/ssd_block_size);
    }
}
__global__
void get_miss(cache_id *d_ssd_plus_buffer_,int32_t *cache_index,int32_t *input_ids,uint64_t input_num);

__global__
void no_merge_kernel(IOReq *d_ret,int32_t *cache_index,int32_t *input_ids,uint64_t input_num,
                    void *dst_float_buffer,uint64_t ssd_block_size);

__global__
void merge_kernel(cache_id *d_ssd_plus_buffer_,IOReq *d_ret,uint64_t *d_ret_ssd,uint64_t* d_split_flag_,uint64_t lenth,
                    void *dst_float_buffer,uint64_t merge_lenth,uint64_t ssd_block_size);

__global__
void split_flag(uint64_t *d_split_flag_,cache_id *d_ssd_plus_buffer_,uint64_t lenth,uint64_t merge_lenth);   

// __host__ __device__
// bool cmp(IOReq a,IOReq b){
//     return a.ssd_addr<b.ssd_addr;
// }

struct IOMerge{
    void *ssd_start_addr;
    uint64_t* p_miss_cnt_;
    uint64_t grid_size{1};//cuda param
    uint64_t block_size{1024};

    uint64_t merge_lenth_{8};
    uint64_t ssd_block_size{ITEM_SIZE};//block size
    uint64_t init_buffer_cnt_;
    
    cache_id* d_ssd_plus_buffer_;
    size_t temp_storage_bytes_{102400000};//if the input size is too big, should set it when constructing
    void *d_temp_storage_;
    // int32_t *cache_index{nullptr};
    // int32_t *input_ids{nullptr};
    uint64_t* d_split_flag_;
    IOReq *d_ret_;
    uint64_t *d_ret_ssd_;

    // __forceinline__
    __host__ __device__
    IOMerge(uint64_t grid_size,uint64_t block_size,uint64_t merge_lenth,uint64_t ssd_block_size,uint64_t init_buffer_cnt,size_t temp_storage_bytes):
            grid_size(grid_size),block_size(block_size),merge_lenth_(merge_lenth),
            ssd_block_size(ssd_block_size),init_buffer_cnt_(init_buffer_cnt),temp_storage_bytes_(temp_storage_bytes){
            cudaMalloc((void **)&p_miss_cnt_,sizeof(uint64_t));

            cudaMalloc((void **)&d_ssd_plus_buffer_,sizeof(cache_id)*init_buffer_cnt_);
            cudaMalloc((void **)&d_temp_storage_,temp_storage_bytes_);
            cudaMalloc((void **)&d_split_flag_,sizeof(uint64_t)*init_buffer_cnt_);
            cudaMalloc((void **)&d_ret_,sizeof(IOReq)*init_buffer_cnt_);
            cudaMalloc((void **)&d_ret_ssd_,sizeof(uint64_t)*init_buffer_cnt_); 
            // cudaMalloc((void **)&input_ids,sizeof(uint64_t)*init_buffer_cnt_);
            // cudaMalloc((void **)&cache_index,sizeof(int32_t)*init_buffer_cnt_);               
            };
    
    // __forceinline__
    __host__ __device__
    IOMerge(uint64_t grid_size,uint64_t block_size,uint64_t merge_lenth,uint64_t ssd_block_size,uint64_t init_buffer_cnt):
            grid_size(grid_size),block_size(block_size),merge_lenth_(merge_lenth),
            ssd_block_size(ssd_block_size),init_buffer_cnt_(init_buffer_cnt){
            cudaMalloc((void **)&p_miss_cnt_,sizeof(uint64_t));

            cudaMalloc((void **)&d_ssd_plus_buffer_,sizeof(cache_id)*init_buffer_cnt_);
            cudaMalloc((void **)&d_temp_storage_,temp_storage_bytes_);
            cudaMalloc((void **)&d_split_flag_,sizeof(uint64_t)*init_buffer_cnt_);
            cudaMalloc((void **)&d_ret_,sizeof(IOReq)*init_buffer_cnt_);
            cudaMalloc((void **)&d_ret_ssd_,sizeof(uint64_t)*init_buffer_cnt_); 
            
            // cudaMalloc((void **)&input_ids,sizeof(uint64_t)*init_buffer_cnt_);
            // cudaMalloc((void **)&cache_index,sizeof(int32_t)*init_buffer_cnt_);               
            };

    __forceinline__
    __host__ __device__
    ~IOMerge(){
        //maybe free the cuda memory
    }

    __host__
    IOReq* naive_merge(int32_t* cache_index, int32_t* input_ids, int32_t& input_num, float* dst_float_buffer, cudaStream_t stream){

        uint64_t miss_cnt;
        uint64_t *p_miss_cnt_;
        cub::DeviceReduce::Sum(NULL,temp_storage_bytes_,cache_index,p_miss_cnt_,input_num,stream);
        cub::DeviceReduce::Sum(d_temp_storage_,temp_storage_bytes_,cache_index,p_miss_cnt_,input_num,stream);
        cudaMemcpy(&miss_cnt,p_miss_cnt_,sizeof(uint64_t),cudaMemcpyDeviceToHost);
        
        get_miss<<<grid_size,block_size,0,stream>>>(d_ssd_plus_buffer_,cache_index,input_ids,input_num);
        
        cub::DeviceRadixSort::SortPairs(NULL,temp_storage_bytes_,input_ids,input_ids,
                                        d_ssd_plus_buffer_,d_ssd_plus_buffer_,miss_cnt,
                                        0,sizeof(uint64_t)*8,stream);

        cub::DeviceRadixSort::SortPairs(d_temp_storage_,temp_storage_bytes_,input_ids,input_ids,
                                        d_ssd_plus_buffer_,d_ssd_plus_buffer_,miss_cnt,
                                        0,sizeof(uint64_t)*8,stream);

        split_flag<<<grid_size,block_size,0,stream>>>(d_split_flag_,d_ssd_plus_buffer_,miss_cnt,merge_lenth_);

        merge_kernel<<<grid_size,block_size,0,stream>>>(d_ssd_plus_buffer_,d_ret_,d_ret_ssd_,d_split_flag_,
                                                miss_cnt,dst_float_buffer,merge_lenth_,ssd_block_size);
            
        cub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes_, d_ret_ssd_, d_ret_ssd_, 
                                        d_ret_, d_ret_, miss_cnt,
                                        0,sizeof(uint64_t)*8,stream);
        // printf("temp_storage_bytes_:%d\n",temp_storage_bytes_);
        cub::DeviceRadixSort::SortPairs(d_temp_storage_, temp_storage_bytes_, d_ret_ssd_, d_ret_ssd_, 
                                        d_ret_, d_ret_, miss_cnt,
                                        0,sizeof(uint64_t)*8,stream);
        
        cout<<"miss_cnt:"<<miss_cnt<<endl;
        return d_ret_;
    }

    __host__
    IOReq* no_merge(int32_t* cache_index, int32_t* input_ids, int32_t& input_num, float* dst_float_buffer, cudaStream_t stream){

        no_merge_kernel<<<grid_size,block_size,0,stream>>>(d_ret_,cache_index,input_ids,input_num,
                                                            dst_float_buffer,ssd_block_size);
        return d_ret_;
    }

};

__global__
void get_miss(cache_id *d_ssd_plus_buffer_,int32_t *cache_index,int32_t *input_ids,uint64_t input_num){
    int thread_id=blockIdx.x*blockDim.x+threadIdx.x;
    for(int i=thread_id;i<input_num;i+=blockDim.x*gridDim.x){
        // printf("cache_index[%d]=%d input_ids[%d]=%lu\n",i,cache_index[i],i,input_ids[i]);
        cache_id tmp(my_ULLONG_MAX,my_ULLONG_MAX);
        if(cache_index[i]==1){
            tmp.ssd_id=input_ids[i];
            tmp.buffer_id=i;
            d_ssd_plus_buffer_[i]=tmp;
        }
        else{
            d_ssd_plus_buffer_[i]=tmp;
            input_ids[i]=my_ULLONG_MAX;
        }
    }
    return;
}

__global__
void merge_kernel(cache_id *d_ssd_plus_buffer_,IOReq *d_ret,uint64_t *d_ret_ssd,uint64_t* d_split_flag_,uint64_t lenth,
                    void *dst_float_buffer,uint64_t merge_lenth,uint64_t ssd_block_size){
    uint64_t thread_id=blockIdx.x*blockDim.x+threadIdx.x;
    for(uint64_t i=thread_id;i<lenth;i+=blockDim.x*gridDim.x){
        // printf("d_ssd_plus_buffer_[%lu].ssd_id=%lu\n",i,d_ssd_plus_buffer_[i].ssd_id);
        if(d_split_flag_[i]==1){
            // printf("d_ssd_plus_buffer_[%lu].ssd_id=%lu\n",i,d_ssd_plus_buffer_[i].ssd_id);
            uint64_t base_id=d_ssd_plus_buffer_[i].ssd_id/merge_lenth*merge_lenth;
            IOReq req(base_id,merge_lenth);
            d_ret_ssd[i]=base_id;
            uint64_t start=i+1;
            req.dest_addr[d_ssd_plus_buffer_[i].ssd_id-base_id]=(uint64_t)dst_float_buffer+ssd_block_size*d_ssd_plus_buffer_[i].buffer_id;
            while(d_ssd_plus_buffer_[start].ssd_id!=my_ULLONG_MAX&&d_split_flag_[start]==0){
                req.dest_addr[d_ssd_plus_buffer_[start].ssd_id-base_id]=(uint64_t)dst_float_buffer+ssd_block_size*d_ssd_plus_buffer_[start].buffer_id;
                start++;
            }
            d_ret[i]=req;
        }else{
            IOReq req(my_ULLONG_MAX,merge_lenth);
            d_ret_ssd[i]=my_ULLONG_MAX;
            d_ret[i]=req;
        }
        
    }
}


__global__
void no_merge_kernel(IOReq *d_ret,int32_t *cache_index,int32_t *input_ids,uint64_t input_num,
                    void *dst_float_buffer,uint64_t ssd_block_size){
    uint64_t thread_id=blockIdx.x*blockDim.x+threadIdx.x;              
    for(uint64_t i=thread_id;i<input_num;i+=blockDim.x*gridDim.x){
        int32_t cache_idx = cache_index[i];
            d_ret[i].start_lb=input_ids[i];
            d_ret[i].dest_addr[0]=(uint64_t)dst_float_buffer+ssd_block_size*i;
            d_ret[i].num_items=1;
            // if(thread_id == 0){
            //     printf("d_ret[%lu].start_lb=%lu, %d, %lu\n",i,d_ret[i].start_lb, cache_index[i], d_ret[i].dest_addr[0]);
            // }
        // if(cache_index[i] == CACHEMISS_FLAG){
        //     d_ret[i].start_lb=input_ids[i];
        //     d_ret[i].dest_addr[0]=(uint64_t)dst_float_buffer+ssd_block_size*i;
        // }else{
        //     d_ret[i].start_lb=my_ULLONG_MAX;
        //     d_ret[i].dest_addr[0]=my_ULLONG_MAX;
        //     printf("d_ret[%lu].start_lb=%lu, %d\n",i,d_ret[i].start_lb, cache_index[i]);
        // }

    }
}

__global__
void split_flag(uint64_t *d_split_flag_,cache_id *d_ssd_plus_buffer_,uint64_t lenth,uint64_t merge_lenth){
    uint64_t thread_id=blockIdx.x*blockDim.x+threadIdx.x;
    if(thread_id==0){
        d_split_flag_[0]=1;
        thread_id+=blockDim.x*gridDim.x;
    }
    for(uint64_t i=thread_id;i<lenth;i+=blockDim.x*gridDim.x){
        if(d_ssd_plus_buffer_[i].ssd_id/merge_lenth==d_ssd_plus_buffer_[i-1].ssd_id/merge_lenth){
            d_split_flag_[i]=0;
        }
        else{
            d_split_flag_[i]=1;
        }
        // printf("D_ssds_plus_buffer[%lu].ssd_id=%lu  d_split_flag_[%lu]=%lu\n ",i,d_ssd_plus_buffer_[i].ssd_id,i,d_split_flag_[i]);
    }
}

