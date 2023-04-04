#include "initCalib.h"
#include <fstream>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <pcl/common/io.h>

// 重载<<运算符，方便输出std::vector
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); i++)
    {
        os << v[i];
        if (i != v.size() - 1)
            os << ", ";
    }
    os << "]";
    return os;
}

void project2image(pcl::PointCloud<pcl::PointXYZI>::Ptr pc, cv::Mat raw_image, cv::Mat &output_image, Eigen::Matrix4f RT, Eigen::Matrix3f camera_param)
{

    Eigen::Matrix<float, 3, 4> T_lidar2cam_top3_local, T_lidar2image_local; //lida2image=T_lidar2cam*(T_cam02cam2)*T_cam2image
    T_lidar2cam_top3_local = RT.topRows(3);                                 //R T��ǰ����
    T_lidar2image_local = camera_param * T_lidar2cam_top3_local;
    if (raw_image.channels() < 3 && raw_image.channels() >= 1)
    {

        cv::Mat output_image_3channels(raw_image.rows, raw_image.cols, CV_8UC3, cv::Scalar::all(0));
        for (int i = 0; i < raw_image.cols; i++)
        {
            for (int j = 0; j < raw_image.rows; j++)
            {
                output_image_3channels.at<cv::Vec3b>(j, i)[2] = output_image_3channels.at<cv::Vec3b>(j, i)[1] =
                    output_image_3channels.at<cv::Vec3b>(j, i)[0] = (int)raw_image.at<uchar>(j, i);

            }
        }
        output_image_3channels.copyTo(output_image);

    }
    else
    {
        raw_image.copyTo(output_image);
    }
    pcl::PointXYZI r;
    Eigen::Vector4f raw_point;
    Eigen::Vector3f trans_point;
    double deep, deep_config; //deep_config: normalize, max deep
    int point_r;
    deep_config = 80;
    point_r = 2;
    //std::cout << "image size; " << raw_image.cols << " * " << raw_image.rows << std::endl;
    for (int i = 0; i < pc->size(); i++)
    {
        r = pc->points[i];
        raw_point(0, 0) = r.x; //����˹�һ��ƽ���ϵĵ�??
        raw_point(1, 0) = r.y;
        raw_point(2, 0) = r.z;
        raw_point(3, 0) = 1;
        trans_point = T_lidar2image_local * raw_point;

        // delete the point behind
        if(trans_point(2, 0) < 0) continue;

        int x = (int)(trans_point(0, 0) / trans_point(2, 0));
        int y = (int)(trans_point(1, 0) / trans_point(2, 0));

        //cout<<"!!!@@@####"<<x<<" "<<y<<" ";

        if (x < 0 || x > (raw_image.cols - 1) || y < 0 || y > (raw_image.rows - 1))
            continue;
        deep = trans_point(2, 0) / deep_config;
        //deep = r.intensity / deep_config;
        int blue, red, green;
        if (deep <= 0.5)
        {
            green = (int)((0.5 - deep) / 0.5 * 255);
            red = (int)(deep / 0.5 * 255);
            blue = 0;
        }
        else if (deep <= 1)
        {
            green = 0;
            red = (int)((1 - deep) / 0.5 * 255);
            blue = (int)((deep - 0.5) / 0.5 * 255);
        }
        else
        {
            blue = 0;
            green = 0;
            red = 255;
        };
        //��ͼ���ϻ�С԰��������ɫ
        //cv::circle(output_image, cv::Point2f(x, y), point_r, cv::Scalar(255, 255, 0), -1);
        cv::circle(output_image, cv::Point2f(x, y), point_r, cv::Scalar(0, 255, 0), -1);
        // cv::circle(output_image, cv::Point2f(x, y), point_r, cv::Scalar(blue,green,red), -1);
    }
}

// 从配置文件中获取结果文件路径
inline std::string getResultPath(YAML::Node config) {
    std::string result_path = config["result_path"].as<std::string>();
    return result_path;
}

// 从配置文件中读取相机内参
inline Eigen::Matrix3f getCameraParam(YAML::Node config) {
    Eigen::Matrix3f camera_param;
    camera_param << config["fx"].as<float>(), 0, config["cx"].as<float>(),
        0, config["fy"].as<float>(), config["cy"].as<float>(),
        0, 0, 1;
    return camera_param;
}

// 从配置文件中读取点云路径
inline std::string getPointCloudPath(YAML::Node config) {
    std::string point_cloud_path = config["point_cloud_path"].as<std::string>();
    return point_cloud_path;
}

// 从配置文件中读取点云限制范围
inline void getPointCloudRange(YAML::Node config, float& x_min, float& x_max, float& y_min, float& y_max, float& z_min, float& z_max) {
    x_min = config["x_min"].as<float>();
    x_max = config["x_max"].as<float>();
    y_min = config["y_min"].as<float>();
    y_max = config["y_max"].as<float>();
    z_min = config["z_min"].as<float>();
    z_max = config["z_max"].as<float>();
}

// 从配置文件中读取图像路径
inline std::string getImagePath(YAML::Node config) {
    std::string image_path = config["image_path"].as<std::string>();
    return image_path;
}

// 从配置文件中读取外参真值
inline Eigen::Matrix4f getExtrinsicParam(YAML::Node config) {
    Eigen::Matrix4f extrinsic_param;
    
    Eigen::Matrix3f R_lidar2cam0_unbias;
    std::vector<float> ext = config["R_lidar2cam0_unbias"]["data"].as<std::vector<float>>();
    assert((int)ext.size() == 9);
    R_lidar2cam0_unbias << ext[0], ext[1], ext[2],
        ext[3], ext[4], ext[5],
        ext[6], ext[7], ext[8];
    
    Eigen::Matrix4f T_lidar2cam0_unbias;
    // 块操作可以被用作左值或右值
    T_lidar2cam0_unbias.block(0, 0, 3, 3) = R_lidar2cam0_unbias;
    T_lidar2cam0_unbias(0, 3) = config["t03"].as<float>();
    T_lidar2cam0_unbias(1, 3) = config["t13"].as<float>();
    T_lidar2cam0_unbias(2, 3) = config["t23"].as<float>();
    T_lidar2cam0_unbias.row(3) << 0, 0, 0, 1;

    Eigen::Matrix4f T_cam02cam2;
    std::vector<float> cam02cam2 = config["T_cam02cam2"]["data"].as<std::vector<float>>(); 
    assert((int)cam02cam2.size() == 16);
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            T_cam02cam2(row, col) = cam02cam2[row * 4 + col];
        }
    }

    extrinsic_param = T_cam02cam2 * T_lidar2cam0_unbias;

    return extrinsic_param;
}

const std::string config_path = "configs/config0.yaml";

int main() {
    
    YAML::Node config = YAML::LoadFile(config_path);

    // 设置随机数种子
    unsigned int seed = config["seed"].as<unsigned int>();
    if (seed == 0) {
        seed = (unsigned)time(NULL);
    }
    srand(seed);
    
    // 获取结果文件路径
    std::string result_path = getResultPath(config);
    std::ofstream result_txt(result_path, std::ios::app);
    result_txt << "The beginning of the PSO!!!!\n" << std::endl;

    // 获取相机内参
    Eigen::Matrix3f camera_param = getCameraParam(config);
    result_txt << "camera_param:\n" << camera_param << std::endl;

    // 获取点云路径
    std::string point_cloud_path = getPointCloudPath(config);
    result_txt << "point_cloud_path: " << point_cloud_path << std::endl;

    // 读取点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_source(new pcl::PointCloud<pcl::PointXYZI>);
    getPointCloud(point_cloud_path, pc_source);

    // 获取点云限制范围
    float x_min, x_max, y_min, y_max, z_min, z_max;
    getPointCloudRange(config, x_min, x_max, y_min, y_max, z_min, z_max);
    result_txt << "point_cloud_range: " << x_min << " " << x_max << " " << y_min << " " << y_max << " " << z_min << " " << z_max << std::endl;

    // 限制点云范围
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_filtered(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::copyPointCloud(*pc_source, *pc_filtered);
    limitPointCloud(pc_filtered, x_min, x_max, y_min, y_max, z_min, z_max);
    
    // 提取点云特征
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature(new pcl::PointCloud<pcl::PointXYZI>);
    extractPCFeature(pc_filtered, pc_feature, config);
    // // visualize the feature pointcloud
    // pcl::visualization::PCLVisualizer feature_viewer("pc_feature");
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> feature_color(pc_feature, 0, 255, 0);
    // feature_viewer.setBackgroundColor(0, 0, 0);
    // feature_viewer.addPointCloud(pc_feature, feature_color, "pc_feature");
    // feature_viewer.spin();

    // 获取原始语义图像
    std::string image_path = getImagePath(config);
    cv::Mat img_source = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    result_txt << "image_path: " << image_path << std::endl;
    // cv::imshow("img_source", img_source);
    // cv::waitKey();
    
    // 二值化
    cv::Mat img_thresh;
    cv::threshold(img_source, img_thresh, 0, 255, cv::THRESH_BINARY);
    // 内部距离变换
    cv::Mat distance_img = img_thresh.clone();
    cv::distanceTransform(distance_img, distance_img, 1, 3);
    cv::normalize(distance_img, distance_img, config["normalize_inside"].as<int>(), 0, cv::NORM_INF);
    distance_img.convertTo(distance_img, CV_8U);
    for( auto i = distance_img.begin<uchar>(); i != distance_img.end<uchar>(); ++i) {
        if(*i == 0) continue;
        *i = 255-*i;
    }
    // cv::imshow("distance_img", distance_img);
    // cv::waitKey();
    // 外部距离变换
    cv::Mat notmap = ~img_thresh;
    cv::Mat notdistimg;
    cv::distanceTransform(notmap, notdistimg, 1, 3);
    cv::normalize(notdistimg, notdistimg, config["normalize_outside"].as<int>(), 0, cv::NORM_INF);
    notdistimg.convertTo(notdistimg, CV_8U);
    for( auto i = notdistimg.begin<uchar>(); i != notdistimg.end<uchar>(); ++i) {
        if(*i == 0) continue;
        *i = 255-*i;
    }
    // 合并距离变换图
    cv::add(distance_img, notdistimg, distance_img);
    // cv::imshow("distance_img", distance_img);
    // cv::waitKey();

    // 获取外参真值
    Eigen::Matrix4f extrinsic_param = getExtrinsicParam(config);
    result_txt << "extrinsic_param:\n" << extrinsic_param << std::endl;
    double extrinsic_param_array[6];
    RT2position(extrinsic_param, extrinsic_param_array);
    result_txt << "extrinsic_param_array:\n" 
        << extrinsic_param_array[0] << " " 
        << extrinsic_param_array[1] << " "
        << extrinsic_param_array[2] << " "
        << extrinsic_param_array[3] << " "
        << extrinsic_param_array[4] << " "
        << extrinsic_param_array[5] << std::endl;
    
    // 获取真值投影图
    cv::Mat img_true;
    project2image(pc_feature, distance_img, img_true, extrinsic_param, camera_param);
    cv::imwrite("results/project_imgs/true.png", img_true);

    // 定义粒子空间
    int dimension = config["particle_space"]["dimension"].as<int>();
    std::vector<double> lower_bound = config["particle_space"]["lower_bound"].as<std::vector<double>>();
    std::vector<double> upper_bound = config["particle_space"]["upper_bound"].as<std::vector<double>>();
    ParticleSpace space{dimension, lower_bound, upper_bound};
    // 定义粒子群
    int swarm_size = config["particle_swarm"]["swarm_size"].as<int>();
    ParticleSwarm* swarm = new ParticleSwarm(swarm_size, space);
    // 创建一个位置速度更新器
    double inertia_weight = config["move_mode"]["inertia_weight"].as<double>();
    double personal_weight = config["move_mode"]["personal_weight"].as<double>();
    double social_weight = config["move_mode"]["social_weight"].as<double>();
    double max_velocity = config["move_mode"]["max_velocity"].as<double>();
    double rebound_decay = config["move_mode"]["rebound_decay"].as<double>();
    MoveMode* move_mode = new MoveMode(inertia_weight, personal_weight, social_weight, max_velocity, rebound_decay);
    // 创建一个要解决的问题
    CalibProblem* calib_problem = new CalibProblem();
    calib_problem->set_distance_img(distance_img);
    calib_problem->set_pc_feature(pc_feature);
    calib_problem->set_camera_param(camera_param);
    // 创建一个优化器
    BaseOptimizer* optimizer = new BaseOptimizer(calib_problem, move_mode);
    // 创建一个PSO优化器
    BasePSO* pso_calib = new BasePSO(swarm, optimizer);
    // PSO算法冷启动
    pso_calib->initPSO();
    std::cout << "init_fitness: " << pso_calib->getResultFitness() << std::endl;
    std::cout << "init_position: " << pso_calib->getResult() << std::endl;
    std::cout << "Init PSO Success!" << std::endl;
    
    result_txt << "init_fitness: " << pso_calib->getResultFitness() << std::endl;
    result_txt << "init_position: " << pso_calib->getResult() << std::endl;
    result_txt << "Init PSO Success!" << std::endl;

    // 获取真值得分
    Particle* true_particle = new Particle();
    true_particle->x = extrinsic_param_array;
    true_particle->pbest_fitness = calib_problem->calculateFitness(*true_particle);
    std::cout << "true_fitness: " << true_particle->pbest_fitness << std::endl;
    result_txt << "true_fitness: " << true_particle->pbest_fitness << std::endl;

    // 设置标准差收敛阈值
    double threshold = config["threshold"].as<double>();

    // PSO算法迭代优化
    double temp_fitness = pso_calib->getResultFitness(); // 有隐患，可能存在position不同但fitness相同的情况
    for(int i = 1; i <= config["search_loops"].as<int>(); ++i){
        pso_calib->step();
        // 每50轮判断标准差是否收敛
        if (i % 50 == 0) {
            std::cout << "ROUND " << i << "!" << "Judge whether the stddeviation is converged!" << std::endl;
            result_txt << "ROUND " << i << "!" << "Judge whether the stddeviation is converged!" << std::endl;
            std::vector<double> mean, stddeviation;
            swarm->getParticleMeanAndStd(mean, stddeviation);
            std::cout << "mean: " << mean << std::endl;
            result_txt << "mean: " << mean << std::endl;
            std::cout << "stddeviation: " << stddeviation << std::endl;
            result_txt << "stddeviation: " << stddeviation << std::endl;
            if (stddeviation[0] < threshold && stddeviation[1] < threshold && stddeviation[2] < threshold && stddeviation[3] < threshold && stddeviation[4] < threshold && stddeviation[5] < threshold) {
                std::cout << "stddeviation is converged!" << std::endl;
                result_txt << "stddeviation is converged!" << std::endl;
                break;
            }
        }

        std::cout << "ROUND " << i << "!" << std::endl;
        result_txt << "ROUND " << i << "!" << std::endl;
        if (pso_calib->particle_swarm_->gbest_fitness == temp_fitness) {
            std::cout << "result_fitness is not changed!" << std::endl;
            continue;
        }
        temp_fitness = pso_calib->getResultFitness();
        std::cout << "搜索后结果" << std::endl;
        result_txt << "搜索后结果" << std::endl;
        std::cout << "result_fitness:" << pso_calib->particle_swarm_->gbest_fitness << std::endl;
        result_txt << "result_fitness:" << pso_calib->particle_swarm_->gbest_fitness << std::endl;
        std::cout << "result_position:" << pso_calib->getResult() << std::endl;
        result_txt << "result_position:" << pso_calib->getResult() << std::endl;
        
        cv::Mat project_img;
        project2image(pc_feature,distance_img, project_img, 
                        position2RT(pso_calib->particle_swarm_->gbest_position), camera_param);
        cv::imwrite("results/project_imgs/" + std::to_string(i) + ".jpg", project_img);

    }

    // 获取最优解与真值的差值
    std::vector<double> diff;
    for(int i = 0; i < 6; ++i) diff.push_back(pso_calib->particle_swarm_->gbest_position[i] - extrinsic_param_array[i]);
    std::cout << "diff: " << diff << std::endl;
    result_txt << "diff: " << diff << std::endl;

    // 获取最优解，转为矩阵
    std::cout << "result_position: ";
    for (int i = 0; i < 6; ++i) std::cout << pso_calib->particle_swarm_->gbest_position[i] << " ";
    std::cout << std::endl;
    result_txt << "result_position: ";
    for (int i = 0; i < 6; ++i) result_txt << pso_calib->particle_swarm_->gbest_position[i] << " ";
    result_txt << std::endl;
    std::cout << "result_fitness: " << pso_calib->particle_swarm_->gbest_fitness << std::endl;
    result_txt << "result_fitness: " << pso_calib->particle_swarm_->gbest_fitness << std::endl;
    std::cout << "result_RT: " << position2RT(pso_calib->particle_swarm_->gbest_position) << std::endl;
    result_txt << "result_RT: " << position2RT(pso_calib->particle_swarm_->gbest_position) << std::endl;

    result_txt.close();
    
    return 0;
}