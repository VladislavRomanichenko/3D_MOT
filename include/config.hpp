struct Config 
{                
    double state_func_covariance;               
    double measure_func_covariance;             
    double prediction_score_decay;              
    double LiDAR_scanning_frequency;
    
    // max prediction number of state function
    int max_prediction_num;                     
    int max_prediction_num_for_new_object;      
    double association_threshold;

    //detection score threshold
    double input_score;
    double init_score;                          
    double update_score;                        
    double post_score;
    
    // tracking latency (s)
    // -1: global tracking
    // 0.->500: online or near online tracking
    double latency;                             
};