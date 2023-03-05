#include <initCalib.h>


float countScore(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature, const cv::Mat distance_image,
                 Eigen::Matrix4f RT, Eigen::Matrix3f camera_param){
    float score = 0;
    Eigen::Matrix<float, 3, 4> RT_TOP3, RT_X_CAM; //lida2image=T*(T_cam02cam2)*T_cam2image
    RT_TOP3 = RT.topRows(3);
    RT_X_CAM = camera_param * RT_TOP3;

    //count Score
    int edge_size = pc_feature->size();
    float one_score = 0;
    int points_num = 0;
    for (int j = 0; j < edge_size; j++)
    {
        pcl::PointXYZI r;
        Eigen::Vector4f raw_point;
        Eigen::Vector3f trans_point3;
        r = pc_feature->points[j];

        raw_point(0, 0) = r.x;
        raw_point(1, 0) = r.y;
        raw_point(2, 0) = r.z;
        raw_point(3, 0) = 1;
        trans_point3 = RT_X_CAM * raw_point;

        // ignore the point behind
        if(trans_point3(2, 0) < 0) 
            continue;

        int x = (int)(trans_point3(0, 0) / trans_point3(2, 0));
        int y = (int)(trans_point3(1, 0) / trans_point3(2, 0));
        // ignore the point outside
        if (x < 0 || x > (distance_image.cols - 1) || y < 0 || y > (distance_image.rows - 1)) 
            continue;
        // // ignore the point useless
        // if(distance_image.at<uchar>(y, x) < 50) 
        //     continue;
        
        // Error
        if (r.intensity < 0 || distance_image.at<uchar>(y, x) < 0)
        {
            std::cout << "\033[33mError: has intensity<0\033[0m" << std::endl;
            exit(0);
        }
    
        points_num++;

        double pt_dis = pow(r.x * r.x + r.y * r.y + r.z * r.z, double(1.0 / 2.0));
        //std::cout << r.x << "  " << r.y << "   " << r.z << "  " << pt_dis << std::endl;

        // add distance weight or not
        bool add_dis_weight = false;
        if (add_dis_weight)
        {
            // one_score +=  (distance_image.at<uchar>(y, x) * sqrt(pc_feature->points[j].intensity));
            if (abs(r.intensity - 0.1) < 0.2)
            {
                one_score += (distance_image.at<uchar>(y, x) / pt_dis * 2) * 1.2; 
            }
            else
            {
                one_score += (distance_image.at<uchar>(y, x) / pt_dis * 2);
            }
        }
        else
        {
            one_score += distance_image.at<uchar>(y, x);
        }
    }
    // 
    if (points_num < 200) return 0;
    // score = one_score;// / (float)data_num;
    // cout << "point_num: " << points_num << endl;
    score = one_score / 255.0;

    return score;
}




InitCalib::InitCalib(int _dimension, int _particlenumber, const string config_dir, double _result_threshold, double _w, double _cp, double _cg, double _wall, int _time_to_end)
    : PsoAlgorithm(_dimension, _particlenumber, _result_threshold, _w, _cp, _cg, _wall, _time_to_end) {
        cout << "Start reading configs: " << config_dir << endl;
        config = YAML::LoadFile(config_dir);
        read_configs();
    }

InitCalib::InitCalib(int _dimension, int _particlenumber, YAML::Node _config, double _result_threshold, double _w, double _cp, double _cg, double _wall, int _time_to_end)
    : PsoAlgorithm(_dimension, _particlenumber, _result_threshold, _w, _cp, _cg, _wall, _time_to_end), config(_config){
        read_configs();
    }

bool InitCalib::read_configs()
{
    cout << "Start reading camera_param" << endl;
    camera_param << config["fx"].as<float>(), 0.f, config["cx"].as<float>(), 
                    0.f, config["fy"].as<float>(), config["cy"].as<float>(),
                    0, 0, 1;

    cout << "Start reading and computing RT!" << endl;
    Eigen::Matrix3f R_lidar2cam0_unbias;
    std::vector<float> ext = config["R_lidar2cam0_unbias"]["data"].as<std::vector<float>>();
    assert((int)ext.size() == 9);
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            R_lidar2cam0_unbias(row, col) = ext[row * 3 + col];
        }
    }

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

    initial_T = T_cam02cam2 * T_lidar2cam0_unbias;
    initial_R = initial_T.block(0, 0, 3, 3);
    
    return true;
}

Particle *InitCalib::get_initial_partical()
{
    return initial_particle;
}

void InitCalib::set_initial_particle()
{
    initial_particle = new Particle;
    initial_particle->x = new double[dimension];
    initial_particle->v = new double[dimension];
    initial_particle->pbest = new double[dimension];

    initial_particle->x[0] = initial_T(0, 3);
    initial_particle->x[1] = initial_T(1, 3);
    initial_particle->x[2] = initial_T(2, 3);

    Eigen::Vector3f eulerAngle1 = initial_R.eulerAngles(0, 1, 2);
    initial_particle->x[3] = eulerAngle1[0];
    initial_particle->x[4] = eulerAngle1[1];
    initial_particle->x[5] = eulerAngle1[2];
    
    initial_particle->fitness = fitnessFunction((*initial_particle));

}
void InitCalib::set_initial_particle(Eigen::Matrix4f _initial_T)
{
    initial_particle = new Particle;
    initial_particle->x = new double[dimension];
    initial_particle->v = new double[dimension];
    initial_particle->pbest = new double[dimension];
    
    initial_T = _initial_T;
    initial_R = initial_T.block(0, 0, 3, 3);

    initial_particle->x[0] = initial_T(0, 3);
    initial_particle->x[1] = initial_T(1, 3);
    initial_particle->x[2] = initial_T(2, 3);

    Eigen::Vector3f eulerAngle1 = initial_R.eulerAngles(0, 1, 2);
    initial_particle->x[3] = eulerAngle1[0];
    initial_particle->x[4] = eulerAngle1[1];
    initial_particle->x[5] = eulerAngle1[2];
}


void InitCalib::bias_initial_particle()
{
    cout << "Start adding bias to initial particle!!" << endl;
    printParticle(initial_particle);
    vector<float> bias = config["bias_particle"].as<vector<float>>();
    for(int i = 0; i < dimension; ++i) initial_particle->x[i] += bias[i];
    printParticle(initial_particle);
}

void InitCalib::setSearchScope(double *bias, double _maxspeedratio)
{
    if (initial_particle == nullptr) {
        cerr << "Fatal error: Do not have initial particle to set search scope!!" << endl;
        exit(1);
    }
    positionMin = new double[dimension];
    positionMax = new double[dimension];
    maxspeedratio = _maxspeedratio;
    positionMin[0] = initial_particle->x[0] - bias[0];
    positionMin[1] = initial_particle->x[1] - bias[1];
    positionMin[2] = initial_particle->x[2] - bias[2];
    positionMin[3] = initial_particle->x[3] - bias[3];
    positionMin[4] = initial_particle->x[4] - bias[4];
    positionMin[5] = initial_particle->x[5] - bias[5];

    positionMax[0] = initial_particle->x[0] + bias[0];
    positionMax[1] = initial_particle->x[1] + bias[1];
    positionMax[2] = initial_particle->x[2] + bias[2];
    positionMax[3] = initial_particle->x[3] + bias[3];
    positionMax[4] = initial_particle->x[4] + bias[4];
    positionMax[5] = initial_particle->x[5] + bias[5];
}

void InitCalib::set_pc_feature(pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_source, pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_feature)
{
    // std::cout << "Start extract pc_feature" << std::endl;
    
    
    // float search_r, search_r2, search_r3;
    // float x_max, x_min, y_min, y_max, z_min, z_max;
    // int search_num, search_num2, search_num3;
    // x_max = config["x_max"].as<float>();
    // x_min = config["x_min"].as<float>();
    // y_min = config["y_min"].as<float>();
    // y_max = config["y_max"].as<float>();
    // z_min = config["z_min"].as<float>();
    // z_max = config["z_max"].as<float>();
    // search_r = config["search_r"].as<float>();
    // search_num = config["search_num"].as<int>();
    // search_r2 = config["search_r2"].as<float>();
    // search_num2 = config["search_num2"].as<int>();
    // search_r3 = config["search_r3"].as<float>();
    // search_num3 = config["search_num3"].as<int>();
    // dis_threshold = config["dis_threshold"].as<float>();
    // angle_threshold = config["angle_threshold"].as<float>();
    // canny_threshold_mini = config["canny_threshold_mini"].as<int>();
    // canny_threshold_max = config["canny_threshold_max"].as<int>();
    // normalize_config = config["normalize_config"].as<int>();
    // normalize_config_thin = config["normalize_config_thin"].as<int>();
    // factor = ((rings - 1) / (upperBound - lowerBound));

}
void InitCalib::set_pc_feature(pcl::PointCloud<pcl::PointXYZI>::Ptr _pc_feature)
{
    pc_feature = _pc_feature;
}

void InitCalib::set_distance_img(cv::Mat _distance_img)
{
    // 注意，浅拷贝！！若有需求之后可以改成深拷贝
    distance_image = _distance_img;
}
void InitCalib::set_distance_img(const string img_dir)
{
    distance_image = cv::imread(img_dir);
}

Eigen::Matrix4f InitCalib::particle2RT(Particle *p)
{
    Eigen::Matrix4f RT;
    
    Eigen::Vector3f ea(p->x[3], p->x[4], p->x[5]);
    Eigen::Matrix3f rotate_matrix3;
    rotate_matrix3 = Eigen::AngleAxisf(ea[0], Eigen::Vector3f::UnitX()) * 
                     Eigen::AngleAxisf(ea[1], Eigen::Vector3f::UnitY()) * 
                     Eigen::AngleAxisf(ea[2], Eigen::Vector3f::UnitZ());
    RT.block(0, 0, 3, 3) = rotate_matrix3;            
    RT(0, 3) = p->x[0];
    RT(1, 3) = p->x[1];
    RT(2, 3) = p->x[2];
    RT.row(3) << 0, 0, 0, 1;

    return RT;
}

Eigen::Matrix4f InitCalib::particle2RT(double *p)
{
    Eigen::Matrix4f RT;
    
    Eigen::Vector3f ea(p[3], p[4], p[5]);
    Eigen::Matrix3f rotate_matrix3;
    rotate_matrix3 = Eigen::AngleAxisf(ea[0], Eigen::Vector3f::UnitX()) * 
                     Eigen::AngleAxisf(ea[1], Eigen::Vector3f::UnitY()) * 
                     Eigen::AngleAxisf(ea[2], Eigen::Vector3f::UnitZ());
    RT.block(0, 0, 3, 3) = rotate_matrix3;            
    RT(0, 3) = p[0];
    RT(1, 3) = p[1];
    RT(2, 3) = p[2];
    RT.row(3) << 0, 0, 0, 1;

    return RT;
}

double InitCalib::fitnessFunction(Particle& p){
    float fitness = 0;
    Eigen::Matrix4f RT = particle2RT(&p);

    fitness = countScore(pc_feature, distance_image, RT, camera_param);

    return fitness;
}

double InitCalib::fitnessFunction(Particle *p)
{
    float fitness = 0;
    Eigen::Matrix4f RT = particle2RT(p);

    fitness = countScore(pc_feature, distance_image, RT, camera_param);

    return fitness;
}

InitCalib::InitCalib()
{
}

