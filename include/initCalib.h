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

// 粒子位置转换为RT矩阵
Eigen::Matrix4f position2RT(double* p);

// 粒子转换为RT矩阵
Eigen::Matrix4f particle2RT(const Particle& p);

class CalibProblem : public BaseProblem{
public:
    // 计算fitness值所需的参数
    cv::Mat distance_image;
    Eigen::Matrix3f camera_param; // 相机内参
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature; // 点云特征

    // 重写fitness函数
    double CalculateFitness (Particle &p) override {
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