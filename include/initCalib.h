#pragma once
#ifndef PROJECT_INITCALIB_H
#define PROJECT_INITCALIB_H

#include <vector>

#include <pcl/io/pcd_io.h>
#include <pcl/common/io.h>
#include <pcl/common/common.h>
#include <pcl/common/impl/io.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/conversions.h>
#include <pcl/console/time.h>
#include <pcl/surface/mls.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/extract_clusters.h>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv_modules.hpp>


#include <Eigen/Core>
#include "pso.h"
// #include "auto_calib.h"
#include <iostream>
#include <yaml.h>

#define PI (3.1415926535897932346f)

float countScore(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature, const cv::Mat distance_image,
                 Eigen::Matrix4f RT, Eigen::Matrix3f camera_param);

class InitCalib : public PsoAlgorithm {
private:
    YAML::Node config;  // 算法配置文件
    bool read_configs();
public:
    cv::Mat distance_image;
    Eigen::Matrix3f camera_param; // 相机内参
    Eigen::Matrix4f initial_T;  // 待优化的外参矩阵T
    Eigen::Matrix3f initial_R;  // 待优化的外参矩阵R
    Particle *initial_particle;  // 未优化的初始粒子
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature; // 点云特征

    Particle* get_initial_partical();

    // 从初始外参矩阵获取初始粒子
    void set_initial_particle();
    void set_initial_particle(Eigen::Matrix4f initial_T);

    // 给初始粒子加偏差(测试用函数)
    void bias_initial_particle();

    // 设置搜寻范围
    using PsoAlgorithm::setSearchScope;
    void setSearchScope(std::vector<double> search_scope, double maxspeedratio = 0.15);
    
    // 获得点云信息
    void set_pc_feature(pcl::PointCloud<pcl::PointXYZI>::Ptr _pc_feature);

    // 获得图片信息
    void set_distance_img(cv::Mat _distance_img);
    void set_distance_img(const std::string img_dir);

    // 从粒子信息提取RT矩阵
    static Eigen::Matrix4f particle2RT(Particle* p);
    static Eigen::Matrix4f particle2RT(double *p);
    static Eigen::Matrix4f particle2RT(std::vector<double> p);
    
    // 点云投影获取fitness值
    double fitnessFunction(Particle &p);

    // 构造函数
    InitCalib();
    InitCalib(int _dimension, int _particlenumber, const std::string config_dir,
                 double _result_threshold = 0.8, double _w = 0.9, double _cp = 1.6, double _cg = 2, double _wall = 0.8, int _time_to_end = 5);
    InitCalib(int _dimension, int _particlenumber, YAML::Node config, 
                 double _result_threshold = 0.8, double _w = 0.9, double _cp = 1.6, double _cg = 2, double _wall = 0.8, int _time_to_end = 5);
};

#endif