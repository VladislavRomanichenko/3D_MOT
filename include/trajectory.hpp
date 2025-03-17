#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <map>
#include <cmath>
#include <config.hpp>

struct Object 
{
    Eigen::VectorXd updated_state;        
    Eigen::VectorXd predicted_state;      
    Eigen::VectorXd detected_state;       
    Eigen::MatrixXd updated_covariance;   
    Eigen::MatrixXd predicted_covariance; 
    double prediction_score;              
    double score;                         

    Object() : prediction_score(0.0), score(-1.0) {}
};

//TODO: переделать права доступа к полям

class Trajectory 
{
public:
    Trajectory(const Eigen::VectorXd& init_bb,           
               double init_score,                        
               int init_timestamp,                       
               int label,                                
               const Config& config = Config());        

    int size() const;

    void state_prediction(int timestamp);

    void state_update(const Eigen::VectorXd& bb,         
                      double score,                      
                      int timestamp);                    

    void filtering(const Config& config);

    Eigen::VectorXd init_bb;              
    double init_score;                    
    int init_timestamp;                   
    int label;                            
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
