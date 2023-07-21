#include<iostream>
#include<climits>
// #include <iomanip>
#include<stdlib.h>
#include<cuda.h>
#include<vector>
#include<algorithm>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include<thrust/sort.h>
#include<thrust/copy.h>
#include<thrust/reduce.h>
#include<thrust/shuffle.h>
#include<unistd.h>
#include<cub/cub.cuh>
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
struct IOReq{
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

    __host__ __device__ 
    bool operator<(const IOReq &lhs) const
    {
        return this->start_lb < lhs.start_lb;
    }

    __forceinline__
    __host__ __device__ 
    ~IOReq()
    {
        // delete[] gpu_addr;
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
void print_Dret(IOReq* D_ret,uint64_t len,void *ssd_start_addr,void *gpu_start_addr,uint64_t typesize){
    printf("print_Dret\n");
    for(uint64_t i=0;i<len;i++){
        printf("cache:%lu\n",((uint64_t)D_ret[i].start_lb-(uint64_t)ssd_start_addr)/typesize);
        printf("gpu0:%lu\n",((uint64_t)D_ret[i].dest_addr[0]-(uint64_t)gpu_start_addr)/typesize);
        printf("gpu1:%lu\n",((uint64_t)D_ret[i].dest_addr[1]-(uint64_t)gpu_start_addr)/typesize);
        printf("gpu2:%lu\n",((uint64_t)D_ret[i].dest_addr[2]-(uint64_t)gpu_start_addr)/typesize);
        printf("gpu3:%lu\n",((uint64_t)D_ret[i].dest_addr[3]-(uint64_t)gpu_start_addr)/typesize);
        printf("gpu4:%lu\n",((uint64_t)D_ret[i].dest_addr[4]-(uint64_t)gpu_start_addr)/typesize);
        printf("gpu5:%lu\n",((uint64_t)D_ret[i].dest_addr[5]-(uint64_t)gpu_start_addr)/typesize);
        printf("gpu6:%lu\n",((uint64_t)D_ret[i].dest_addr[6]-(uint64_t)gpu_start_addr)/typesize);
        printf("gpu7:%lu\n\n",((uint64_t)D_ret[i].dest_addr[7]-(uint64_t)gpu_start_addr)/typesize);
    }
}

__global__
void print_Dret_ssd(uint64_t* D_ret_ssd,uint64_t len,void *ssd_start_addr,uint64_t typesize){
    printf("print_Dret_ssd\n");
    for(uint64_t i=0;i<len;i++){
        // if(i==0){
        //     printf("the first origin ssd:%lu\n",(uint64_t)D_ret_ssd[i]);
        // }
        printf("ssd:%lu\n",((uint64_t)D_ret_ssd[i]-(uint64_t)ssd_start_addr)/typesize);
    }
}
__global__
void get_miss(cache_id *D_ssd_plus_buffer,uint32_t *D_cache_miss,uint64_t *D_ids,uint64_t D_buffer_cnt);

__global__
void merge_kernel(cache_id *D_ssd_plus_buffer,IOReq *D_ret,uint64_t *D_ret_ssd,uint64_t* D_split_flag,uint64_t lenth,
                    void *ssd_start_addr,void *gpu_start_addr,uint64_t merge_lenth,uint64_t typesize);

__global__
void split_flag(uint64_t *D_split_flag,cache_id *D_ssd_plus_buffer,uint64_t lenth,uint64_t merge_lenth);   

// __host__ __device__
// bool cmp(IOReq a,IOReq b){
//     return a.ssd_addr<b.ssd_addr;
// }

struct IO_Merge{
    void *ssd_start_addr;
    void *gpu_start_addr;

    uint64_t grid_size{1};//cuda param
    uint64_t block_size{1024};

    uint64_t merge_lenth{8};
    uint64_t typesize{512};//block size
    uint64_t buffer_cnt;
    
    cache_id* D_ssd_plus_buffer;
    size_t temp_storage_bytes{102400000};//if the input size is too big, should set it when constructing
    void *d_temp_storage;
    // uint32_t *D_cache_miss{nullptr};
    // uint64_t *D_ids{nullptr};
    uint64_t* D_split_flag;
    IOReq *D_ret;
    uint64_t *D_ret_ssd;

    // __forceinline__
    __host__ __device__
    IO_Merge(void *ssd,void *gpu,uint64_t grid_size,uint64_t block_size,uint64_t merge_lenth,uint64_t typesize,uint64_t buffer_cnt,size_t temp_storage_bytes):
            ssd_start_addr(ssd),gpu_start_addr(gpu),grid_size(grid_size),block_size(block_size),merge_lenth(merge_lenth),
            typesize(typesize),buffer_cnt(buffer_cnt),temp_storage_bytes(temp_storage_bytes){

            cudaMalloc((void **)&D_ssd_plus_buffer,sizeof(cache_id)*buffer_cnt);
            cudaMalloc((void **)&d_temp_storage,temp_storage_bytes);
            cudaMalloc((void **)&D_split_flag,sizeof(uint64_t)*buffer_cnt);
            cudaMalloc((void **)&D_ret,sizeof(IOReq)*buffer_cnt);
            cudaMalloc((void **)&D_ret_ssd,sizeof(uint64_t)*buffer_cnt); 
            // cudaMalloc((void **)&D_ids,sizeof(uint64_t)*buffer_cnt);
            // cudaMalloc((void **)&D_cache_miss,sizeof(uint32_t)*buffer_cnt);               
            };
    
    // __forceinline__
    __host__ __device__
    IO_Merge(void *ssd,void *gpu,uint64_t grid_size,uint64_t block_size,uint64_t merge_lenth,uint64_t typesize,uint64_t buffer_cnt):
            ssd_start_addr(ssd),gpu_start_addr(gpu),grid_size(grid_size),block_size(block_size),merge_lenth(merge_lenth),
            typesize(typesize),buffer_cnt(buffer_cnt){

            cudaMalloc((void **)&D_ssd_plus_buffer,sizeof(cache_id)*buffer_cnt);
            cudaMalloc((void **)&d_temp_storage,temp_storage_bytes);
            cudaMalloc((void **)&D_split_flag,sizeof(uint64_t)*buffer_cnt);
            cudaMalloc((void **)&D_ret,sizeof(IOReq)*buffer_cnt);
            cudaMalloc((void **)&D_ret_ssd,sizeof(uint64_t)*buffer_cnt); 
            
            // cudaMalloc((void **)&D_ids,sizeof(uint64_t)*buffer_cnt);
            // cudaMalloc((void **)&D_cache_miss,sizeof(uint32_t)*buffer_cnt);               
            };

    __forceinline__
    __host__ __device__
    ~IO_Merge(){
        //maybe free the cuda memory
    }

    __host__
    IOReq* call_kernel(uint32_t *D_cache_miss,uint64_t *D_ids,cudaStream_t stream){

        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        // print_uint64<<<1,1,0,stream>>>(D_ids,buffer_cnt);
        cudaEventRecord(start,0);
        uint64_t miss_cnt;
        uint64_t *p_miss_cnt;
        cudaMalloc((void **)&p_miss_cnt,sizeof(uint64_t));
        cub::DeviceReduce::Sum(NULL,temp_storage_bytes,D_cache_miss,p_miss_cnt,buffer_cnt,stream);
        // printf("temp_storage_bytes:%d\n",temp_storage_bytes);
        cub::DeviceReduce::Sum(d_temp_storage,temp_storage_bytes,D_cache_miss,p_miss_cnt,buffer_cnt,stream);
        cudaMemcpy(&miss_cnt,p_miss_cnt,sizeof(uint64_t),cudaMemcpyDeviceToHost);
        
        get_miss<<<grid_size,block_size,0,stream>>>(D_ssd_plus_buffer,D_cache_miss,D_ids,buffer_cnt);
        // cudaDeviceSynchronize();
        // cudaCheckError();
        

        cub::DeviceRadixSort::SortPairs(NULL,temp_storage_bytes,D_ids,D_ids,
                                        D_ssd_plus_buffer,D_ssd_plus_buffer,miss_cnt,
                                        0,sizeof(uint64_t)*8,stream);
        // printf("temp_storage_bytes:%d\n",temp_storage_bytes);

        cub::DeviceRadixSort::SortPairs(d_temp_storage,temp_storage_bytes,D_ids,D_ids,
                                        D_ssd_plus_buffer,D_ssd_plus_buffer,miss_cnt,
                                        0,sizeof(uint64_t)*8,stream);

        // print<<<1,1>>>(D_ssd_plus_buffer,buffer_cnt);
        // cudaCheckError();
        split_flag<<<grid_size,block_size,0,stream>>>(D_split_flag,D_ssd_plus_buffer,miss_cnt,merge_lenth);
        // cudaDeviceSynchronize();
        // cudaCheckError();
        merge_kernel<<<grid_size,block_size,0,stream>>>(D_ssd_plus_buffer,D_ret,D_ret_ssd,D_split_flag,
                                                miss_cnt,ssd_start_addr,gpu_start_addr,merge_lenth,typesize);
        // cudaCheckError();
        // cudaDeviceSynchronize();
        // print_Dret_ssd<<<1,1>>>(D_ret_ssd,miss_cnt,ssd_start_addr);
        // cudaDeviceSynchronize();
            
        cub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, D_ret_ssd, D_ret_ssd, 
                                        D_ret, D_ret, miss_cnt,
                                        0,sizeof(uint64_t)*8,stream);
        // printf("temp_storage_bytes:%d\n",temp_storage_bytes);
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, D_ret_ssd, D_ret_ssd, 
                                        D_ret, D_ret, miss_cnt,
                                        0,sizeof(uint64_t)*8,stream);
        // cudaDeviceSynchronize();
        
        // cudaCheckError();
        //////////////test{
        // printf("buffer_cnt:%lu\n",buffer_cnt);
        // printf("\n\n\n\nafter sorted\n\n\n\n");
        // cout<<my_ULLONG_MAX<<endl;
        // printf("%lu\n",my_ULLONG_MAX);
        
        ////////////}
       
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime,start,stop);
        printf("GPU_part time:%f ms\n",elapsedTime);
        
        cout<<"miss_cnt:"<<miss_cnt<<endl;
        // print_Dret_ssd<<<1,1,0,stream>>>(D_ret_ssd,miss_cnt,ssd_start_addr,typesize);
        // print_Dret<<<1,1,0,stream>>>(D_ret,miss_cnt,ssd_start_addr,gpu_start_addr,typesize);
        cudaDeviceSynchronize();
        return D_ret;
        //the following is for debug
        // IOReq *H_ret=(IOReq *)malloc(sizeof(IOReq)*miss_cnt);
        // cudaMemcpy(H_ret,D_ret,sizeof(IOReq)*miss_cnt,cudaMemcpyDeviceToHost);
        // return H_ret;
    }

    

};

__global__
void get_miss(cache_id *D_ssd_plus_buffer,uint32_t *D_cache_miss,uint64_t *D_ids,uint64_t D_buffer_cnt){
    int thread_id=blockIdx.x*blockDim.x+threadIdx.x;
    for(int i=thread_id;i<D_buffer_cnt;i+=blockDim.x*gridDim.x){
        // printf("D_cache_miss[%d]=%d D_ids[%d]=%lu\n",i,D_cache_miss[i],i,D_ids[i]);
        cache_id tmp(my_ULLONG_MAX,my_ULLONG_MAX);
        if(D_cache_miss[i]==1){
            tmp.ssd_id=D_ids[i];
            tmp.buffer_id=i;
            D_ssd_plus_buffer[i]=tmp;
        }
        else{
            D_ssd_plus_buffer[i]=tmp;
            D_ids[i]=my_ULLONG_MAX;
        }
    }
    return;
}

__global__
void merge_kernel(cache_id *D_ssd_plus_buffer,IOReq *D_ret,uint64_t *D_ret_ssd,uint64_t* D_split_flag,uint64_t lenth,
                    void *ssd_start_addr,void *gpu_start_addr,uint64_t merge_lenth,uint64_t typesize){
    uint64_t thread_id=blockIdx.x*blockDim.x+threadIdx.x;
    for(uint64_t i=thread_id;i<lenth;i+=blockDim.x*gridDim.x){
        // printf("D_ssd_plus_buffer[%lu].ssd_id=%lu\n",i,D_ssd_plus_buffer[i].ssd_id);
        if(D_split_flag[i]==1){
            // printf("D_ssd_plus_buffer[%lu].ssd_id=%lu\n",i,D_ssd_plus_buffer[i].ssd_id);
            uint64_t base_id=D_ssd_plus_buffer[i].ssd_id/merge_lenth*merge_lenth;
            IOReq req((uint64_t)ssd_start_addr+typesize*base_id,merge_lenth);
            D_ret_ssd[i]=(uint64_t)ssd_start_addr+typesize*base_id;
            uint64_t start=i+1;
            req.dest_addr[D_ssd_plus_buffer[i].ssd_id-base_id]=(uint64_t)gpu_start_addr+typesize*D_ssd_plus_buffer[i].buffer_id;
            while(D_ssd_plus_buffer[start].ssd_id!=my_ULLONG_MAX&&D_split_flag[start]==0){
                req.dest_addr[D_ssd_plus_buffer[start].ssd_id-base_id]=(uint64_t)gpu_start_addr+typesize*D_ssd_plus_buffer[start].buffer_id;
                start++;
            }
            D_ret[i]=req;
            // printf("cache:%lu\n",((uint64_t)D_ret[i].ssd_addr-(uint64_t)ssd_start_addr)/typesize);
            // printf("gpu0:%lu\n",((uint64_t)D_ret[i].gpu_addr[0]-(uint64_t)gpu_start_addr)/typesize);
            // printf("gpu1:%lu\n",((uint64_t)D_ret[i].gpu_addr[1]-(uint64_t)gpu_start_addr)/typesize);
            // printf("gpu2:%lu\n",((uint64_t)D_ret[i].gpu_addr[2]-(uint64_t)gpu_start_addr)/typesize);
            // printf("gpu3:%lu\n",((uint64_t)D_ret[i].gpu_addr[3]-(uint64_t)gpu_start_addr)/typesize);
            // printf("gpu4:%lu\n",((uint64_t)D_ret[i].gpu_addr[4]-(uint64_t)gpu_start_addr)/typesize);
            // printf("gpu5:%lu\n",((uint64_t)D_ret[i].gpu_addr[5]-(uint64_t)gpu_start_addr)/typesize);
            // printf("gpu6:%lu\n",((uint64_t)D_ret[i].gpu_addr[6]-(uint64_t)gpu_start_addr)/typesize);
            // printf("gpu7:%lu\n\n",((uint64_t)D_ret[i].gpu_addr[7]-(uint64_t)gpu_start_addr)/typesize);
            
        }else{
            IOReq req(my_ULLONG_MAX,merge_lenth);
            D_ret_ssd[i]=my_ULLONG_MAX;
            D_ret[i]=req;
        }
        
    }
}

__global__
void split_flag(uint64_t *D_split_flag,cache_id *D_ssd_plus_buffer,uint64_t lenth,uint64_t merge_lenth){
    uint64_t thread_id=blockIdx.x*blockDim.x+threadIdx.x;
    if(thread_id==0){
        D_split_flag[0]=1;
        thread_id+=blockDim.x*gridDim.x;
    }
    for(uint64_t i=thread_id;i<lenth;i+=blockDim.x*gridDim.x){
        if(D_ssd_plus_buffer[i].ssd_id/merge_lenth==D_ssd_plus_buffer[i-1].ssd_id/merge_lenth){
            D_split_flag[i]=0;
        }
        else{
            D_split_flag[i]=1;
        }
        // printf("D_ssds_plus_buffer[%lu].ssd_id=%lu  D_split_flag[%lu]=%lu\n ",i,D_ssd_plus_buffer[i].ssd_id,i,D_split_flag[i]);
    }
}





