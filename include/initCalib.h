#pragma once
#ifndef PROJECT_INITCALIB_H
#define PROJECT_INITCALIB_H

#include "pso.h"
#include <yaml.h>
#include <pcl/common/common.h>
#include <opencv2/core.hpp>


// 读取指定路径的点云bin文件，并转换为pcl::PointCloud<pcl::PointXYZI>::Ptr
void getPointCloud(const std::string& path, pcl::PointCloud<pcl::PointXYZI>::Ptr& point_cloud);

// 限制点云范围到指定空间
void limitPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr& point_cloud, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max);

// 提取点云特征
void extractPCFeature(pcl::PointCloud<pcl::PointXYZI>::Ptr &pc, pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_feature, YAML::Node config);

float countScore(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature, const cv::Mat distance_image,
                 Eigen::Matrix4f RT, Eigen::Matrix3f camera_param);

// 粒子位置转换为RT矩阵
Eigen::Matrix4f position2RT(double* p);

// 粒子转换为RT矩阵
Eigen::Matrix4f particle2RT(const Particle& p);

// RT矩阵转换为粒子位置
void RT2position(const Eigen::Matrix4f& RT, double* p);

class CalibProblem : public BaseProblem{
public:
    // 计算fitness值所需的参数
    cv::Mat distance_image;
    Eigen::Matrix3f camera_param; // 相机内参
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature; // 点云特征

    // 重写fitness函数
    double calculateFitness (Particle &p) override {
        return countScore(pc_feature, distance_image, particle2RT(p), camera_param);
    };
    void set_pc_feature(pcl::PointCloud<pcl::PointXYZI>::Ptr _pc_feature){
        pc_feature = _pc_feature;
    };
    void set_distance_img(cv::Mat _distance_img){
        distance_image = _distance_img;
    };
    void set_camera_param(Eigen::Matrix3f _camera_param){
        camera_param = _camera_param;
    };
};

#endif