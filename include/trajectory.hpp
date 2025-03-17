#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <map>
#include <cmath>

struct Config {
    double LiDAR_scanning_frequency;   
    double state_func_covariance;      
    double measure_func_covariance;    
    double prediction_score_decay;     
    double latency;                    
};

struct Object {
    Eigen::VectorXd updated_state;        
    Eigen::VectorXd predicted_state;      
    Eigen::VectorXd detected_state;       
    Eigen::MatrixXd updated_covariance;   
    Eigen::MatrixXd predicted_covariance; 
    double prediction_score;              
    double score;                         
    Eigen::VectorXd features;             

    Object() : prediction_score(0.0), score(-1.0) {}
};

class Trajectory {
public:
    Trajectory(const Eigen::VectorXd& init_bb,           
               const Eigen::VectorXd* init_features,     
               double init_score,                        
               int init_timestamp,                       
               int label,                                
               bool tracking_features = true,            
               bool bb_as_features = false,              
               const Config& config = Config());        

    int size() const;

    void state_prediction(int timestamp);

    void state_update(const Eigen::VectorXd& bb,         
                      const Eigen::VectorXd* features,   
                      double score,                      
                      int timestamp);                    

    void filtering(const Config& config);

private:
    Eigen::VectorXd init_bb;              
    Eigen::VectorXd init_features;        
    double init_score;                    
    int init_timestamp;                   
    int label;                            
    bool tracking_features;               
    bool bb_as_features;                  
    bool tracking_bb_size;                
    Config config;                        
    double scanning_interval;             
    int track_dim;                        
    Eigen::MatrixXd A, Q, P, B, H, K;     
    std::map<int, Object> trajectory;     
    int consecutive_missed_num;           
    int first_updated_timestamp;          
    int last_updated_timestamp;           

    int compute_track_dim();              
    void init_parameters();               
    void init_trajectory();               
    double sigmoid(double x);             
};
