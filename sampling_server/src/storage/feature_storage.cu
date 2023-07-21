#include "feature_storage.cuh"
#include "feature_storage_impl.cuh"

#include <iostream>

class CompleteFeatureStorage : public FeatureStorage{
public: 
    CompleteFeatureStorage(){
    }

    virtual ~CompleteFeatureStorage(){};

    void Build(BuildInfo* info) override {
        iostack_ = new IOStack(info->num_ssd, info->num_queues_per_ssd);
        iomerge_ = new IOMerge(64, 1024, 8, 512, 4000000, 1000000000);

        int32_t partition_count = info->partition_count;
        total_num_nodes_ = info->total_num_nodes;
        float_feature_len_ = info->float_feature_len;
        float* host_float_feature = info->host_float_feature;

        if(float_feature_len_ > 0){
            cudaHostGetDevicePointer(&float_feature_, host_float_feature, 0);
        }
        cudaCheckError();

        cudaSetDevice(0);

        training_set_num_.resize(partition_count);
        training_set_ids_.resize(partition_count);
        training_labels_.resize(partition_count);

        validation_set_num_.resize(partition_count);
        validation_set_ids_.resize(partition_count);
        validation_labels_.resize(partition_count);

        testing_set_num_.resize(partition_count);
        testing_set_ids_.resize(partition_count);
        testing_labels_.resize(partition_count);

        partition_count_ = partition_count;

        for(int32_t i = 0; i < info->shard_to_partition.size(); i++){
            int32_t part_id = info->shard_to_partition[i];
            int32_t device_id = info->shard_to_device[i];
            /*part id = 0, 1, 2...*/

            training_set_num_[part_id] = info->training_set_num[part_id];
            // std::cout<<"Training set count "<<training_set_num_[part_id]<<" "<<info->training_set_num[part_id]<<"\n";

            validation_set_num_[part_id] = info->validation_set_num[part_id];
            testing_set_num_[part_id] = info->testing_set_num[part_id];

            cudaSetDevice(device_id);
            cudaCheckError();

            // std::cout<<"Training set on device "<<part_id<<" "<<training_set_num_[part_id]<<"\n";
            // std::cout<<"Testing set on device "<<part_id<<" "<<testing_set_num_[part_id]<<"\n";

            int32_t* train_ids;
            cudaMalloc(&train_ids, training_set_num_[part_id] * sizeof(int32_t));
            cudaMemcpy(train_ids, info->training_set_ids[part_id].data(), training_set_num_[part_id] * sizeof(int32_t), cudaMemcpyHostToDevice);
            training_set_ids_[part_id] = train_ids;
            cudaCheckError();

            int32_t* valid_ids;
            cudaMalloc(&valid_ids, validation_set_num_[part_id] * sizeof(int32_t));
            cudaMemcpy(valid_ids, info->validation_set_ids[part_id].data(), validation_set_num_[part_id] * sizeof(int32_t), cudaMemcpyHostToDevice);
            validation_set_ids_[part_id] = valid_ids;
            cudaCheckError();

            int32_t* test_ids;
            cudaMalloc(&test_ids, testing_set_num_[part_id] * sizeof(int32_t));
            cudaMemcpy(test_ids, info->testing_set_ids[part_id].data(), testing_set_num_[part_id] * sizeof(int32_t), cudaMemcpyHostToDevice);
            testing_set_ids_[part_id] = test_ids;
            cudaCheckError();

            int32_t* train_labels;
            cudaMalloc(&train_labels, training_set_num_[part_id] * sizeof(int32_t));
            cudaMemcpy(train_labels, info->training_labels[part_id].data(), training_set_num_[part_id] * sizeof(int32_t), cudaMemcpyHostToDevice);
            training_labels_[part_id] = train_labels;
            cudaCheckError();

            int32_t* valid_labels;
            cudaMalloc(&valid_labels, validation_set_num_[part_id] * sizeof(int32_t));
            cudaMemcpy(valid_labels, info->validation_labels[part_id].data(), validation_set_num_[part_id] * sizeof(int32_t), cudaMemcpyHostToDevice);
            validation_labels_[part_id] = valid_labels;
            cudaCheckError();

            int32_t* test_labels;
            cudaMalloc(&test_labels, testing_set_num_[part_id] * sizeof(int32_t));
            cudaMemcpy(test_labels, info->testing_labels[part_id].data(), testing_set_num_[part_id] * sizeof(int32_t), cudaMemcpyHostToDevice);
            testing_labels_[part_id] = test_labels;
            cudaCheckError();

        }

        cudaMalloc(&d_req_count_, sizeof(unsigned long long));
        cudaMemset(d_req_count_, 0, sizeof(unsigned long long));
        cudaCheckError();

    };

    void Finalize() override {
        cudaFreeHost(float_feature_);
        for(int32_t i = 0; i < partition_count_; i++){
            cudaSetDevice(i);
            cudaFree(training_set_ids_[i]);
            cudaFree(validation_set_ids_[i]);
            cudaFree(testing_set_ids_[i]);
            cudaFree(training_labels_[i]);
            cudaFree(validation_labels_[i]);
            cudaFree(testing_labels_[i]);
        }
    }

    int32_t* GetTrainingSetIds(int32_t part_id) const override {
        return training_set_ids_[part_id];
    }
    int32_t* GetValidationSetIds(int32_t part_id) const override {
        return validation_set_ids_[part_id];
    }
    int32_t* GetTestingSetIds(int32_t part_id) const override {
        return testing_set_ids_[part_id];
    }

	int32_t* GetTrainingLabels(int32_t part_id) const override {
        return training_labels_[part_id];
    };
    int32_t* GetValidationLabels(int32_t part_id) const override {
        return validation_labels_[part_id];
    }
    int32_t* GetTestingLabels(int32_t part_id) const override {
        return testing_labels_[part_id];
    }

    int32_t TrainingSetSize(int32_t part_id) const override {
        return training_set_num_[part_id];
    }
    int32_t ValidationSetSize(int32_t part_id) const override {
        return validation_set_num_[part_id];
    }
    int32_t TestingSetSize(int32_t part_id) const override {
        return testing_set_num_[part_id];
    }

    int32_t TotalNodeNum() const override {
        return total_num_nodes_;
    }

    float* GetAllFloatFeature() const override {
        return float_feature_;
    }
    int32_t GetFloatFeatureLen() const override {
        return float_feature_len_;
    }

    void Print(BuildInfo* info) override {
    }

    void IOSubmit(int32_t* sampled_ids, int32_t* cache_index,
                  int32_t* node_counter, float* dst_float_buffer,
                  int32_t op_id, int32_t dev_id, cudaStream_t strm_hdl) override {
		
        int32_t* h_node_counter = (int32_t*)malloc(16*sizeof(int32_t));
		cudaMemcpy(h_node_counter, node_counter, 64, cudaMemcpyDeviceToHost);
		cudaCheckError();
        int32_t node_off = 0;
        int32_t batch_size = 0;
    
        node_off   = h_node_counter[(op_id % INTRABATCH_CON) * 2];
        batch_size = h_node_counter[(op_id % INTRABATCH_CON) * 2 + 1];
        if(batch_size > 0){
            IOReq* req = iomerge_->no_merge(cache_index + node_off, sampled_ids + node_off, batch_size, dst_float_buffer + (int64_t(node_off) * float_feature_len_), strm_hdl);
            cudaCheckError();
            iostack_->submit_io_req(req, batch_size, strm_hdl);
            cudaCheckError();
        }
    }

private:
    std::vector<int> training_set_num_;
    std::vector<int> validation_set_num_;
    std::vector<int> testing_set_num_;

    std::vector<int32_t*> training_set_ids_;
    std::vector<int32_t*> validation_set_ids_;
    std::vector<int32_t*> testing_set_ids_;

    std::vector<int32_t*> training_labels_;
    std::vector<int32_t*> validation_labels_;
    std::vector<int32_t*> testing_labels_;

    int32_t partition_count_;
    int32_t total_num_nodes_;
    float* float_feature_;
    int32_t float_feature_len_;

    unsigned long long* d_req_count_;

    IOStack* iostack_;//single GPU multi-SSD
    IOMerge* iomerge_;

    friend FeatureStorage* NewCompleteFeatureStorage();
};

extern "C" 
FeatureStorage* NewCompleteFeatureStorage(){
    CompleteFeatureStorage* ret = new CompleteFeatureStorage();
    return ret;
}
