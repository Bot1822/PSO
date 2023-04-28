#include "initCalib.h"
#include <vector>
#include <fstream>

#include <pcl/filters/passthrough.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

void getPointCloud(const std::string& path, pcl::PointCloud<pcl::PointXYZI>::Ptr& point_cloud)
{
    // Load the actual pointcloud.
    const size_t kMaxNumberOfPoints = 1e6; // From Readme for raw files.
    point_cloud->clear();
    point_cloud->reserve(kMaxNumberOfPoints);
    std::ifstream input(path, std::ios::in | std::ios::binary);
    if (!input) {
        std::cerr << "Could not open pointcloud file.\n";
        throw std::runtime_error("Could not open pointcloud file.");
    }
    for (size_t i = 0; input.good() && !input.eof(); i++) {
        pcl::PointXYZI point;
        input.read((char *)&point.x, 3 * sizeof(float));
        input.read((char *)&point.intensity, sizeof(float));
        point_cloud->push_back(point);
    }
    input.close();
    return;
}

void limitPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr& point_cloud, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max)
{
    pcl::PassThrough<pcl::PointXYZI> pass;
    pass.setInputCloud(point_cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(x_min, x_max);
    pass.filter(*point_cloud);
    pass.setInputCloud(point_cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(y_min, y_max);
    pass.filter(*point_cloud);
    pass.setInputCloud(point_cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(z_min, z_max);
    pass.filter(*point_cloud);
    return;
}

void extractPCFeature(pcl::PointCloud<pcl::PointXYZI>::Ptr &pc, pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_feature, YAML::Node config)
{
    float upperBound = config["upperBound"].as<float>();
    float lowerBound = config["lowerBound"].as<float>();
    int rings = config["rings"].as<int>();
    float factor_t = ((upperBound - lowerBound) / (rings - 1));
    float factor = ((rings - 1) / (upperBound - lowerBound));
    YAML::Node local_config = config["extract_pc_edges"];
    float dis_threshold = config["dis_threshold"].as<float>();

    std::vector<std::vector<float>> pc_image; //��ά��Vector
    std::vector<std::vector<float>> pc_image_copy;
    pc_image.resize(1000);
    pc_image_copy.resize(1000); //��һά��1000
    // resize img and set to -1
    for (int i = 0; i < pc_image.size(); i++) //ÿһ���״�����64����
    {
        pc_image[i].resize(rings);
        pc_image_copy[i].resize(rings);
        for (int j = 0; j < pc_image[i].size(); j++)
            pc_image[i][j] = -1;
    }
    // convert pointcloud from 3D to 2D img
    for (size_t i = 0; i < pc->size(); i++) //����ֻ�����˵���ת����ͼ��
    {
        float theta = 0;
        if (pc->points[i].y == 0)
            theta = 90.0;
        else if (pc->points[i].y > 0)
        {
            float tan_theta = pc->points[i].x / pc->points[i].y;
            theta = 180 * std::atan(tan_theta) / M_PI;
        }
        else
        {
            float tan_theta = -pc->points[i].y / pc->points[i].x;
            theta = 180 * std::atan(tan_theta) / M_PI;
            theta = 90 + theta;
        }
        int col = cvFloor(theta / 0.18); // theta [0, 180] ==> [0, 1000]
        if (col < 0 || col > 999)
            continue;
        float hypotenuse = std::sqrt(std::pow(pc->points[i].x, 2) + std::pow(pc->points[i].y, 2));
        float angle = std::atan(pc->points[i].z / hypotenuse);
        int ring_id = int(((angle * 180 / M_PI) - lowerBound) * factor + 0.5); 
        if (ring_id < 0 || ring_id > rings - 1)
            continue;
        float dist = std::sqrt(std::pow(pc->points[i].y, 2) + std::pow(pc->points[i].x, 2) + std::pow(pc->points[i].z, 2));
        if (dist < 2)
            continue; //10
        if (pc_image[col][ring_id] == -1)
        {
            pc_image[col][ring_id] = dist; //range
        }
        else if (dist < pc_image[col][ring_id])
        {
            pc_image[col][ring_id] = dist; //set the nearer point
        }
    }

    // copy
    for (int i = 0; i < pc_image.size(); i++) 
    {
        for (int j = 0; j < pc_image[i].size(); j++)
            pc_image_copy[i][j] = pc_image[i][j];
    }
    for (int i = 1; i < rings - 1; i++)
    {
        for (int j = 1; j < pc_image.size() - 1; j++)
        {
            float sum_dis = 0.0;
            int sum_n = 0;
            float far_sum_dis = 0.0;
            int far_sum_n = 0;
            float near_sum_dis = 0.0;
            int near_sum_n = 0;
            if (pc_image_copy[j - 1][i - 1] != -1)
            {
                if (pc_image[j][i] != -1)
                {
                    if (pc_image_copy[j - 1][i - 1] - pc_image[j][i] > dis_threshold)
                    { //������ڵ�ȴ˵�Զ��һ����ֵ
                        far_sum_n++;
                        far_sum_dis += pc_image_copy[j - 1][i - 1];
                    }
                    else if (pc_image[j][i] - pc_image_copy[j - 1][i - 1] > dis_threshold)
                    { //����˵�����ڵ�Զ��һ����ֵ
                        near_sum_n++;
                        near_sum_dis += pc_image_copy[j - 1][i - 1];
                    }
                }
                sum_dis += pc_image_copy[j - 1][i - 1];
                sum_n++;
            }
            if (pc_image_copy[j - 1][i] != -1)
            {
                if (pc_image[j][i] != -1)
                {
                    if (pc_image_copy[j - 1][i] - pc_image[j][i] > dis_threshold)
                    {
                        far_sum_n++;
                        far_sum_dis += pc_image_copy[j - 1][i];
                    }
                    else if (pc_image[j][i] - pc_image_copy[j - 1][i] > dis_threshold)
                    {
                        near_sum_n++;
                        near_sum_dis += pc_image_copy[j - 1][i];
                    }
                }
                sum_dis += pc_image_copy[j - 1][i];
                sum_n++;
            }
            if (pc_image_copy[j - 1][i + 1] != -1)
            {
                if (pc_image[j][i] != -1)
                {
                    if (pc_image_copy[j - 1][i + 1] - pc_image[j][i] > dis_threshold)
                    {
                        far_sum_n++;
                        far_sum_dis += pc_image_copy[j - 1][i + 1];
                    }
                    else if (pc_image[j][i] - pc_image_copy[j - 1][i + 1] > dis_threshold)
                    {
                        near_sum_n++;
                        near_sum_dis += pc_image_copy[j - 1][i + 1];
                    }
                }
                sum_dis += pc_image_copy[j - 1][i + 1];
                sum_n++;
            }
            if (pc_image_copy[j][i - 1] != -1)
            {
                if (pc_image[j][i] != -1)
                {
                    if (pc_image_copy[j][i - 1] - pc_image[j][i] > dis_threshold)
                    {
                        far_sum_n++;
                        far_sum_dis += pc_image_copy[j][i - 1];
                    }
                    else if (pc_image[j][i] - pc_image_copy[j][i - 1] > dis_threshold)
                    {
                        near_sum_n++;
                        near_sum_dis += pc_image_copy[j][i - 1];
                    }
                }
                sum_dis += pc_image_copy[j][i - 1];
                sum_n++;
            }
            if (pc_image_copy[j][i + 1] != -1)
            {
                if (pc_image[j][i] != -1)
                {
                    if (pc_image_copy[j][i + 1] - pc_image[j][i] > dis_threshold)
                    {
                        far_sum_n++;
                        far_sum_dis += pc_image_copy[j][i + 1];
                    }
                    else if (pc_image[j][i] - pc_image_copy[j][i + 1] > dis_threshold)
                    {
                        near_sum_n++;
                        near_sum_dis += pc_image_copy[j][i + 1];
                    }
                }
                sum_dis += pc_image_copy[j][i + 1];
                sum_n++;
            }
            if (pc_image_copy[j + 1][i - 1] != -1)
            {
                if (pc_image[j][i] != -1)
                {
                    if (pc_image_copy[j + 1][i - 1] - pc_image[j][i] > dis_threshold)
                    {
                        far_sum_n++;
                        far_sum_dis += pc_image_copy[j + 1][i - 1];
                    }
                    else if (pc_image[j][i] - pc_image_copy[j + 1][i - 1] > dis_threshold)
                    {
                        near_sum_n++;
                        near_sum_dis += pc_image_copy[j + 1][i - 1];
                    }
                }
                sum_dis += pc_image_copy[j + 1][i - 1];
                sum_n++;
            }
            if (pc_image_copy[j + 1][i] != -1)
            {
                if (pc_image[j][i] != -1)
                {
                    if (pc_image_copy[j + 1][i] - pc_image[j][i] > dis_threshold)
                    {
                        far_sum_n++;
                        far_sum_dis += pc_image_copy[j + 1][i];
                    }
                    else if (pc_image[j][i] - pc_image_copy[j + 1][i] > dis_threshold)
                    {
                        near_sum_n++;
                        near_sum_dis += pc_image_copy[j + 1][i];
                    }
                }
                sum_dis += pc_image_copy[j + 1][i];
                sum_n++;
            }
            if (pc_image_copy[j + 1][i + 1] != -1)
            {
                if (pc_image[j][i] != -1)
                {
                    if (pc_image_copy[j + 1][i + 1] - pc_image[j][i] > dis_threshold)
                    {
                        far_sum_n++;
                        far_sum_dis += pc_image_copy[j + 1][i + 1];
                    }
                    else if (pc_image[j][i] - pc_image_copy[j + 1][i + 1] > dis_threshold)
                    {
                        near_sum_n++;
                        near_sum_dis += pc_image_copy[j + 1][i + 1];
                    }
                }
                sum_dis += pc_image_copy[j + 1][i + 1];
                sum_n++;
            }
            if (sum_n >= 5 && pc_image[j][i] == -1)
            {                                     //>=5
                pc_image[j][i] = sum_dis / sum_n; //�����Χ�㶼�в��Ҵ˵��?-1������?ƽ��
                continue;
            }
            if (near_sum_n > sum_n / 2)
            {
                pc_image[j][i] = near_sum_dis / near_sum_n; //�����Χ����ȴ˵��??
            }
            if (far_sum_n > sum_n / 2)
            {
                pc_image[j][i] = far_sum_dis / far_sum_n; //�����Χ����ȴ˵�Զ
            }
        }
    }

    //pc_image data structure
    //  **
    //  **   1000*64
    //  **

    //�����Χ�ĵ㶼�?-1��Ϊ-1
    //   *
    //  *#*
    //   *
    for (int i = 0; i < rings; i++)
    {
        if (i == 0)
        { //������һ��
            for (int j = 0; j < pc_image.size(); j++)
            {
                if (j == 0)
                { //����pc_image��һ��
                    if (pc_image[j][i + 1] == -1 && pc_image[j + 1][i] == -1)
                        pc_image[j][i] = -1;              //�����һ�к���һ�ж��?-1����Ϊ-1
                    pc_image_copy[j][i] = pc_image[j][i]; //����Ϊpc_image
                }
                else if (j == pc_image.size() - 1)
                { //����pc_image���һ��??
                    if (pc_image[j][i + 1] == -1 && pc_image[j - 1][i] == -1)
                        pc_image[j][i] = -1; //�����һ�к���һ�ж��?-1����Ϊ-1
                    pc_image_copy[j][i] = pc_image[j][i];
                }
                else
                {
                    if (pc_image[j][i + 1] == -1 && pc_image[j - 1][i] == -1 && pc_image[j + 1][i] == -1)
                        pc_image[j][i] = -1; //�����һ�к���һ�к���һ�ж��?-1����Ϊ-1
                    pc_image_copy[j][i] = pc_image[j][i];
                }
            }
        }
        else if (i == rings - 1)
        { //�������һ��??
            for (int j = 0; j < pc_image.size(); j++)
            {
                if (j == 0)
                {
                    if (pc_image[j][i - 1] == -1 && pc_image[j + 1][i] == -1)
                        pc_image[j][i] = -1; //�����һ�к���һ�ж��?-1����Ϊ-1
                    pc_image_copy[j][i] = pc_image[j][i];
                }
                else if (j == pc_image.size() - 1)
                {
                    if (pc_image[j][i - 1] == -1 && pc_image[j - 1][i] == -1)
                        pc_image[j][i] = -1; //�����һ�к���һ�ж��?-1����Ϊ-1
                    pc_image_copy[j][i] = pc_image[j][i];
                }
                else
                {
                    if (pc_image[j][i - 1] == -1 && pc_image[j - 1][i] == -1 && pc_image[j + 1][i] == -1)
                        pc_image[j][i] = -1; //�����һ�к���һ�к���һ�ж��?-1����Ϊ-1
                    pc_image_copy[j][i] = pc_image[j][i];
                }
            }
        }
        else
        {
            for (int j = 0; j < pc_image.size(); j++)
            {
                if (j == 0)
                {
                    if (pc_image[j][i - 1] == -1 && pc_image[j + 1][i] == -1 && pc_image[j][i + 1] == -1)
                        pc_image[j][i] = -1; //�����һ�к���һ�к���һ�ж��?-1.��Ϊ-1
                    pc_image_copy[j][i] = pc_image[j][i];
                }
                else if (j == pc_image.size() - 1)
                {
                    if (pc_image[j][i - 1] == -1 && pc_image[j - 1][i] == -1 && pc_image[j][i + 1] == -1)
                        pc_image[j][i] = -1; //�����һ�к���һ�к���һ�ж��?-1����Ϊ-1
                    pc_image_copy[j][i] = pc_image[j][i];
                }
                else
                {
                    if (pc_image[j][i + 1] == -1 && pc_image[j][i - 1] == -1 && pc_image[j - 1][i] == -1 && pc_image[j + 1][i] == -1)
                        pc_image[j][i] = -1; //�����һ�к���һ�к���һ�к���һ�ж��?-1����Ϊ-1
                    pc_image_copy[j][i] = pc_image[j][i];
                }
            }
        }
    }

    for (int i = 0; i < rings; i++)
    {
        for (int j = 1; j < pc_image.size() - 1; j++)
        {
            if (pc_image[j][i] == -1)
            { //����˵��?-1������һ�к���һ�ж���Ϊ-1��ȡ��һ�л���һ�н�С���Ǹ�
                if (pc_image_copy[j - 1][i] != -1 && pc_image_copy[j + 1][i] != -1)
                {
                    pc_image[j][i] = pc_image_copy[j - 1][i] > pc_image_copy[j + 1][i] ? pc_image_copy[j + 1][i] : pc_image_copy[j - 1][i];
                }
            }
        }
    }

    // cv::Mat pc_img = cv::Mat::zeros();

    std::vector<std::vector<float>> mk_rings;
    for (int j = 0; j < pc_image.size(); j++)
    { //1000
        std::vector<float> mk_ring;
        mk_ring.clear();
        for (int i = 0; i < rings; i++)
        {
            // if(pc_image[j][i] != -1)//������в�����??-1�ĵ�
            {
                mk_ring.push_back(pc_image[j][i]); //�洢һ��rings����������
                // mk_ring.push_back(j);
            }
        }
        mk_rings.push_back(mk_ring); //�洢1000����
    }
    // std::cout<<"0"<<std::endl;

    //pc_image data structure  ��ֱ����
    //  **
    //  **   1000*64
    //  **
    for (int i = 0; i < rings; i++) //i<64
    {
        std::vector<float> mk;
        for (int j = 0; j < pc_image.size(); j++)
        { //j<1000
            if (pc_image[j][i] != -1)
            {                                 //������в�����??-1�ĵ�
                mk.push_back(pc_image[j][i]); //�����??(index: 0 2 4 6 ...)
                mk.push_back(j);              //�����?? 0-999(index: 1 3 5 7 ...)
            }
        }
        if (mk.size() < 6)
            continue;

#define DIS 0.1
        for (int j = 1; j < (mk.size() / 2) - 1; j++)
        { //mk��size����һ��(1000)���в�����-1�ĸ�����2
            // if(mk[(j-1)*2]!=-1 && (
            if (
                // mk[(j-1)*2] - mk[(j)*2] > config["dis_threshold"].as<float>() || //ˮƽ��ߵ����˵����һ����ֵ(��������)
                // mk[(j+1)*2] - mk[(j)*2] > config["dis_threshold"].as<float>() || //ˮƽ�ұߵ����˵����һ�����?(��������)
                ((mk[(j - 1) * 2] - mk[(j)*2] > mk[(j)*2] / dis_threshold)) || // && std::abs(mk[(j-1)*2+1]-mk[(j)*2+1])==1) || //ˮƽ��ߵ����˵����һ����ֵ(��������)
                ((mk[(j + 1) * 2] - mk[(j)*2] > mk[(j)*2] / dis_threshold)) || //&& std::abs(mk[(j+1)*2+1]-mk[(j)*2+1])==1) || //ˮƽ�ұߵ����˵����һ�����?(��������)
                // mk[j*2+1] - mk[(j-1)*2+1] > local_config["angle_pixel_dis"].as<int>() || //ˮƽ�ǶȾ������һ�����?
                // mk[(j+1)*2+1] - mk[j*2+1] > local_config["angle_pixel_dis"].as<int>() ||
                local_config["show_all"].as<bool>())
            {

                if (i == 0) // bottom
                {
                    // �м����� �� ���� ���� ������ֵ   ����    �м����� ���� ������ ������  ������ֵ
                    // ȥ����Ⱥ�㣬��֤��ǰ����Χ��Ȧ(��ȥ������)������һ����
                    // ��
                    //  **
                    // #** ����
                    //  **
                    float cen = mk_rings[mk[2 * j + 1]][i];
                    float up = mk_rings[mk[2 * j + 1]][i + 1];
                    float lu = mk_rings[mk[2 * j - 1]][i + 1];
                    float ru = mk_rings[mk[2 * j + 3]][i + 1];
                    float uu = mk_rings[mk[2 * j + 1]][i + 2];
                    float luu = mk_rings[mk[2 * j - 1]][i + 2];
                    float ruu = mk_rings[mk[2 * j + 3]][i + 2];

                    if ((abs(up - cen) > DIS && abs(lu - cen) > DIS && abs(ru - cen) > DIS) || (abs(uu - cen) > DIS && abs(luu - cen) > DIS && abs(ruu - cen) > DIS))
                    {
                        continue;
                    }
                }
                if (i == 1)
                {
                    float cen = mk_rings[mk[2 * j + 1]][i];
                    float up = mk_rings[mk[2 * j + 1]][i + 1];
                    float dw = mk_rings[mk[2 * j + 1]][i - 1];
                    float lu = mk_rings[mk[2 * j - 1]][i + 1];
                    float ru = mk_rings[mk[2 * j + 3]][i + 1];
                    float ld = mk_rings[mk[2 * j - 1]][i - 1];
                    float rd = mk_rings[mk[2 * j + 3]][i - 1];
                    float uu = mk_rings[mk[2 * j + 1]][i + 2];
                    float luu = mk_rings[mk[2 * j - 1]][i + 2];
                    float ruu = mk_rings[mk[2 * j + 3]][i + 2];
                    // �м����� �� �� ���� ���� ���� ���� ������ֵ   ����    �м����� ���� ������ ������  ������ֵ
                    // ȥ����Ⱥ�㣬��֤��ǰ����Χ��Ȧ(��ȥ������)������һ����
                    // * **
                    // *#**  ->��
                    // * **
                    if ((abs(up - cen) > DIS && abs(cen - dw) > DIS && abs(lu - cen) > DIS && abs(ru - cen) > DIS && abs(ld - cen) > DIS && abs(rd - cen) > DIS) || (abs(uu - cen) > DIS && abs(luu - cen) > DIS && abs(ruu - cen) > DIS))
                    {
                        continue;
                    }
                }
                if (i == rings - 1)
                {
                    float cen = mk_rings[mk[2 * j + 1]][i];
                    float dw = mk_rings[mk[2 * j + 1]][i - 1];
                    float ld = mk_rings[mk[2 * j - 1]][i - 1];
                    float rd = mk_rings[mk[2 * j + 3]][i - 1];
                    float dd = mk_rings[mk[2 * j + 1]][i - 2];
                    float ldd = mk_rings[mk[2 * j - 1]][i - 2];
                    float rdd = mk_rings[mk[2 * j + 3]][i - 2];

                    // �м����� �� ���� ���� ������ֵ   ����    �м����� ���� ������ ������  ������ֵ
                    // ȥ����Ⱥ�㣬��֤��ǰ����Χ��Ȧ(��ȥ������)������һ����
                    // **
                    // **#  ->��
                    // **
                    if ((abs(cen - dw) > DIS && abs(ld - cen) > DIS && abs(rd - cen) > DIS) || (abs(dd - cen) > DIS && abs(ldd - cen) > DIS && abs(rdd - cen) > DIS))
                    {
                        continue;
                    }
                }
                if (i == rings - 2)
                {
                    float cen = mk_rings[mk[2 * j + 1]][i];
                    float up = mk_rings[mk[2 * j + 1]][i + 1];
                    float dw = mk_rings[mk[2 * j + 1]][i - 1];
                    float lu = mk_rings[mk[2 * j - 1]][i + 1];
                    float ru = mk_rings[mk[2 * j + 3]][i + 1];
                    float ld = mk_rings[mk[2 * j - 1]][i - 1];
                    float rd = mk_rings[mk[2 * j + 3]][i - 1];
                    float dd = mk_rings[mk[2 * j + 1]][i - 2];
                    float ldd = mk_rings[mk[2 * j - 1]][i - 2];
                    float rdd = mk_rings[mk[2 * j + 3]][i - 2];

                    // �м����� �� �� ���� ���� ���� ���� ������ֵ   ����    �м����� ���� ������ ������  ������ֵ
                    // ȥ����Ⱥ�㣬��֤��ǰ����Χ��Ȧ(��ȥ������)������һ����
                    // ** *
                    // **#*  ->��
                    // ** *
                    if ((abs(up - cen) > DIS && abs(cen - dw) > DIS && abs(lu - cen) > DIS && abs(ru - cen) > DIS && abs(ld - cen) > DIS && abs(rd - cen) > DIS) || (abs(dd - cen) > DIS && abs(ldd - cen) > DIS && abs(rdd - cen) > DIS))
                    {
                        continue;
                    }
                }
                // std::cout<<"1"<<std::endl;
                if (i > 1 && i < rings - 2)
                {
                    // std::cout << "mk size = " << mk.size() << std::endl;
                    // std::cout << "mk[(j+1)*2+1]  = " << mk[(j+1)*2+1] << std::endl;
                    // std::cout << "rings size = " << mk_rings.size() << std::endl;
                    // std::cout<<"j i = " << j << " " << i << std::endl;
                    float cen = mk_rings[mk[2 * j + 1]][i];
                    float up = mk_rings[mk[2 * j + 1]][i + 1];
                    float dw = mk_rings[mk[2 * j + 1]][i - 1];
                    float lu = mk_rings[mk[2 * j - 1]][i + 1];
                    float ru = mk_rings[mk[2 * j + 3]][i + 1];
                    float ld = mk_rings[mk[2 * j - 1]][i - 1];
                    float rd = mk_rings[mk[2 * j + 3]][i - 1];
                    float uu = mk_rings[mk[2 * j + 1]][i + 2];
                    float dd = mk_rings[mk[2 * j + 1]][i - 2];
                    float luu = mk_rings[mk[2 * j - 1]][i + 2];
                    float ruu = mk_rings[mk[2 * j + 3]][i + 2];
                    float ldd = mk_rings[mk[2 * j - 1]][i - 2];
                    float rdd = mk_rings[mk[2 * j + 3]][i - 2];

                    // std::cout << "cen = " << cen<<" "<<up << " "<<dw<<" "<<lu<<" "<<ru<<" "<<ld<<" "<<rd<<std::endl;
                    // if(abs(mk_rings[mk[2*j+1]][i-1]-mk_rings[mk[2*j+1]][i])>0.2 || abs(mk_rings[mk[2*j+1]][i+1]-mk_rings[mk[2*j+1]][i])>0.2
                    // ||  abs(mk_rings[mk[2*j+1]][i-2]-mk_rings[mk[2*j+1]][i])>0.24 || abs(mk_rings[mk[2*j+1]][i+2]-mk_rings[mk[2*j+1]][i])>0.24
                    //    || abs(mk_rings[mk[2*j+1]][i+1]-mk_rings[mk[2*j+1]][i-1])>0.24 || abs(mk_rings[mk[2*j+1]][i+2]-mk_rings[mk[2*j+1]][i-2])>0.3
                    // �м����� �� �� ���� ���� ���� ���� ������ֵ   ����    �м����� ���� ������ ������ ���� ������ ������  ������ֵ
                    // ȥ����Ⱥ�㣬��֤��ǰ����Χ��Ȧ(��ȥ������)������һ����
                    // ** **
                    // **#**  ->��
                    // ** **
                    if ((abs(up - cen) > DIS && abs(cen - dw) > DIS && abs(lu - cen) > DIS && abs(ru - cen) > DIS && abs(ld - cen) > DIS && abs(rd - cen) > DIS) || (abs(uu - cen) > DIS && abs(luu - cen) > DIS && abs(ruu - cen) > DIS && abs(dd - cen) > DIS && abs(ldd - cen) > DIS && abs(rdd - cen) > DIS))
                    {
                        continue;
                    }
                }
                // recover the point from distance, the layer, and angle of the point
                pcl::PointXYZI p;
                p.x = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180.0) * std::cos((mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180.0);
                p.y = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180.0) * std::sin(-(mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180.0);
                p.z = mk[(j)*2] * std::sin((i * factor_t + lowerBound) * M_PI / 180.0);
                // i->64  j->1000   mk[dis,j, dis,j, ...]
                //�ж�ˮƽ�����Ƿ����һ����ֵ��intensityȡ��ֵ�����һ���������в�ֵԽ�����Խ������һ����
                if (mk[(j - 1) * 2] - mk[(j)*2] > mk[(j)*2] / dis_threshold || mk[(j + 1) * 2] - mk[(j)*2] > mk[(j)*2] / config["dis_threshold"].as<float>())
                {
                    // if(mk[(j-1)*2] - mk[(j)*2] > config["dis_threshold"].as<float>() || mk[(j+1)*2] - mk[(j)*2] > config["dis_threshold"].as<float>()){
                    p.intensity = (mk[(j - 1) * 2] - mk[(j)*2]) > (mk[(j + 1) * 2] - mk[(j)*2]) ? (mk[(j - 1) * 2] - mk[(j)*2]) : (mk[(j + 1) * 2] - mk[(j)*2]); //ȡ��ֵ��ģ������ֵ
                }
                // TODO: ��ֱ����Ĳ�ֵ�浽intensity��
                else if (mk[j * 2 + 1] - mk[(j - 1) * 2 + 1] > local_config["angle_pixel_dis"].as<int>() || mk[(j + 1) * 2 + 1] - mk[j * 2 + 1] > local_config["angle_pixel_dis"].as<int>())
                {
                    p.intensity = (mk[j * 2 + 1] - mk[(j - 1) * 2 + 1]) > (mk[(j + 1) * 2 + 1] - mk[j * 2 + 1]) ? (mk[j * 2 + 1] - mk[(j - 1) * 2 + 1]) : (mk[(j + 1) * 2 + 1] - mk[j * 2 + 1]); //ȡ��ֵ��ģ������ֵ
                }
                pc_feature->push_back(p); //delete horizontal features temply
            }
            else if (local_config["add_edge"].as<bool>())
            {
                if (i != 0 && i != rings - 1)
                {
                    if (pc_image[mk[j * 2 + 1]][i + 1] != -1 && pc_image[mk[j * 2 + 1]][i + 1] - pc_image[mk[j * 2 + 1]][i] > local_config["shuzhi_dis_th"].as<float>() * pc_image[mk[j * 2 + 1]][i])
                    {
                        pcl::PointXYZI p;
                        p.x = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180) * std::cos((mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180);
                        p.y = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180) * std::sin(-(mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180);
                        p.z = mk[(j)*2] * std::sin((i * factor_t + lowerBound) * M_PI / 180);
                        p.intensity = 0.5;
                        pc_feature->push_back(p);
                    }
                    else if (pc_image[mk[j * 2 + 1]][i - 1] != -1 && pc_image[mk[j * 2 + 1]][i - 1] - pc_image[mk[j * 2 + 1]][i] > local_config["shuzhi_dis_th"].as<float>() * pc_image[mk[j * 2 + 1]][i])
                    {
                        pcl::PointXYZI p;
                        p.x = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180) * std::cos((mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180);
                        p.y = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180) * std::sin(-(mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180);
                        p.z = mk[(j)*2] * std::sin((i * factor_t + lowerBound) * M_PI / 180);
                        p.intensity = 0.5;
                        pc_feature->push_back(p);
                    }
                }
            }
        }
    }

    //pc_image data structure  ˮƽ����
    //  **
    //  **   1000*64
    //  **
    for (int i = 0; i < pc_image.size(); i++) //i<1000
    {
        std::vector<float> mk;

        for (int j = 0; j < rings; j++) //j<64
        {
            if (pc_image[i][j] != -1) //������в�����??-1�ĵ�
            {
                mk.push_back(pc_image[i][j]); //�����??(index: 0 2 4 6 ...)
                mk.push_back(j);              //�����?? 0-64(index: 1 3 5 7 ...)
            }                                 //������
        }

        if (mk.size() < 2)
            continue;

#define DIS 0.1
        for (int j = 1; j < (mk.size() / 2) - 1; j++) //mk��size����һ��(64)���в�����-1�ĸ�����2
        {                                             //j=1;j<64-1;j++
            // if(mk[(j-1)*2]!=-1 && (
            if (
                // mk[(j-1)*2] - mk[(j)*2] > config["dis_threshold"].as<float>() || //ˮƽ��ߵ����˵����һ����ֵ(��������)
                // mk[(j+1)*2] - mk[(j)*2] > config["dis_threshold"].as<float>() || //ˮƽ�ұߵ����˵����һ�����?(��������)
                ((mk[(j - 1) * 2] - mk[(j)*2] > mk[(j)*2] / dis_threshold * 1.5)) || // && (std::abs(mk[(j-1)*2+1] - mk[(j)*2+1]) == 1 ))||//mk[(j)*2] / config["dis_threshold"].as<float>() || //ˮƽ��ߵ����˵����һ����ֵ(��������)
                ((mk[(j + 1) * 2] - mk[(j)*2] > mk[(j)*2] / dis_threshold * 1.5)) || //&& (std::abs(mk[(j+1)*2+1] - mk[(j)*2+1]) == 1 ))||//mk[(j)*2] / config["dis_threshold"].as<float>() || //ˮƽ�ұߵ����˵����һ�����?(��������)
                // mk[j*2+1] - mk[(j-1)*2+1] > local_config["angle_pixel_dis"].as<int>() || //ˮƽ�ǶȾ������һ�����?
                // mk[(j+1)*2+1] - mk[j*2+1] > local_config["angle_pixel_dis"].as<int>() ||
                local_config["show_all"].as<bool>())
            {

                // if(j == 0) // bottom
                // {
                //     // �м����� �� ���� ���� ������ֵ   ����    �м����� ���� ������ ������  ������ֵ
                //     // ȥ����Ⱥ�㣬��֤��ǰ����Χ��Ȧ(��ȥ������)������һ����
                //     //  **
                //     // #**
                //     //  **
                //     float cen = mk_rings[i][mk[2*j+1]];
                //     float up = mk_rings[i+1][mk[2*j+1]];
                //     float lu = mk_rings[i+1][mk[2*j-1]];
                //     float ru = mk_rings[i+1][mk[2*j+3]];
                //     float uu = mk_rings[i+2][mk[2*j+1]];
                //     float luu = mk_rings[i+2][mk[2*j-1]];
                //     float ruu = mk_rings[i+2][mk[2*j+3]];

                //     if( (abs(up-cen)>DIS&&abs(lu-cen)>DIS&&abs(ru-cen)>DIS)
                //         || (abs(uu-cen)>DIS&&abs(luu-cen)>DIS&&abs(ruu-cen)>DIS)
                //     )
                //     {
                //         continue;
                //     }
                // }

                //          j=[dis,j,dis,j...]
                //          ***
                // i=1000   ***
                //          ***
                //          ***

                if (j == 1 && i > 0 && i < pc_image.size() - 1)
                {
                    float cen = mk_rings[i][mk[2 * j + 1]];
                    float up = mk_rings[i][mk[2 * j + 3]];
                    float dw = mk_rings[i][mk[2 * j - 1]];
                    float lu = mk_rings[i - 1][mk[2 * j + 3]];
                    float lt = mk_rings[i - 1][mk[2 * j + 1]];
                    float ru = mk_rings[i + 1][mk[2 * j + 3]];
                    float rt = mk_rings[i + 1][mk[2 * j + 1]];
                    float ld = mk_rings[i - 1][mk[2 * j - 1]];
                    float rd = mk_rings[i + 1][mk[2 * j - 1]];
                    float uu = mk_rings[i][mk[2 * j + 5]];
                    float luu = mk_rings[i - 1][mk[2 * j + 5]];
                    float ruu = mk_rings[i + 1][mk[2 * j + 5]];
                    // �м����� �� �� ���� ���� ���� ���� ������ֵ   ����    �м����� ���� ������ ������  ������ֵ
                    // ȥ����Ⱥ�㣬��֤��ǰ����Χ��Ȧ(��ȥ������)������һ����
                    /*// ****
                    // *#**  ->��
                    // *****/

                    // ***
                    // *#*  ->��
                    // ***
                    if ((abs(up - cen) > DIS && abs(cen - dw) > DIS && abs(lu - cen) > DIS && abs(ru - cen) > DIS && abs(ld - cen) > DIS && abs(rd - cen) > DIS
                         // &&abs(lt-cen)>DIS&&abs(rt-cen)>DIS //��Ҫ���ҵĵ㣬��ΪԶ���򵽵���ĵ�ᱻ���㵽�����ߣ���Ϊ̫Զ�ˣ�����֮�����??����DIS
                         )
                        // || (abs(uu-cen)>DIS&&abs(luu-cen)>DIS&&abs(ruu-cen)>DIS)
                    )
                    {
                        continue;
                    }
                }
                // if(j == rings-1)
                // {
                //     float cen = mk_rings[i][mk[2*j+1]];
                //     float dw = mk_rings[i-1][mk[2*j+1]];
                //     float ld = mk_rings[i-1][mk[2*j-1]];
                //     float rd = mk_rings[i-1][mk[2*j+3]];
                //     float dd = mk_rings[i-2][mk[2*j+1]];
                //     float ldd = mk_rings[i-2][mk[2*j-1]];
                //     float rdd = mk_rings[i-2][mk[2*j+3]];

                //     // �м����� �� ���� ���� ������ֵ   ����    �м����� ���� ������ ������  ������ֵ
                //     // ȥ����Ⱥ�㣬��֤��ǰ����Χ��Ȧ(��ȥ������)������һ����
                //     // **
                //     // **#  ->��
                //     // **
                //     if( (abs(cen-dw)>DIS&&abs(ld-cen)>DIS&&abs(rd-cen)>DIS)
                //         || (abs(dd-cen)>DIS&&abs(ldd-cen)>DIS&&abs(rdd-cen)>DIS)
                //     )
                //     {
                //         continue;
                //     }
                // }
                if (j == rings - 2 && i > 0 && i < pc_image.size() - 1)
                {
                    float cen = mk_rings[i][mk[2 * j + 1]];
                    float up = mk_rings[i][mk[2 * j + 3]];
                    float dw = mk_rings[i][mk[2 * j - 1]];
                    float lu = mk_rings[i - 1][mk[2 * j + 3]];
                    float lt = mk_rings[i - 1][mk[2 * j + 1]];
                    float ru = mk_rings[i + 1][mk[2 * j + 3]];
                    float rt = mk_rings[i + 1][mk[2 * j + 1]];
                    float ld = mk_rings[i - 1][mk[2 * j - 1]];
                    float rd = mk_rings[i + 1][mk[2 * j - 1]];
                    float dd = mk_rings[i][mk[2 * j - 3]];
                    float ldd = mk_rings[i - 1][mk[2 * j - 3]];
                    float rdd = mk_rings[i + 1][mk[2 * j - 3]];
                    // �м����� �� �� ���� ���� ���� ���� ������ֵ   ����    �м����� ���� ������ ������  ������ֵ
                    // ȥ����Ⱥ�㣬��֤��ǰ����Χ��Ȧ(��ȥ������)������һ����
                    /* // ** *
                    // **#*  ->��
                    // ** *   */

                    // * *
                    // *#*  ->��
                    // * *
                    if ((abs(up - cen) > DIS && abs(cen - dw) > DIS && abs(lu - cen) > DIS && abs(ru - cen) > DIS && abs(ld - cen) > DIS && abs(rd - cen) > DIS
                         // &&abs(lt-cen)>DIS&&abs(rt-cen)>DIS
                         )
                        // || (abs(dd-cen)>DIS&&abs(ldd-cen)>DIS&&abs(rdd-cen)>DIS)
                    )
                    {
                        continue;
                    }
                }
                // std::cout<<"1"<<std::endl;
                if (j > 1 && j < rings - 2 && i > 0 && i < pc_image.size() - 1)
                {
                    // std::cout << "mk size = " << mk.size() << std::endl;
                    // std::cout << "mk[(j+1)*2+1]  = " << mk[(j+1)*2+1] << std::endl;
                    // std::cout << "rings size = " << mk_rings.size() << std::endl;
                    // std::cout<<"j i = " << j << " " << i << std::endl;
                    float cen = mk_rings[i][mk[2 * j + 1]];
                    float up = mk_rings[i][mk[2 * j + 3]];
                    float dw = mk_rings[i][mk[2 * j - 1]];
                    float lu = mk_rings[i - 1][mk[2 * j + 3]];
                    float lt = mk_rings[i - 1][mk[2 * j + 1]];
                    float ru = mk_rings[i + 1][mk[2 * j + 3]];
                    float rt = mk_rings[i + 1][mk[2 * j + 1]];
                    float ld = mk_rings[i - 1][mk[2 * j - 1]];
                    float rd = mk_rings[i + 1][mk[2 * j - 1]];
                    float uu = mk_rings[i][mk[2 * j + 5]];
                    float dd = mk_rings[i][mk[2 * j - 3]];
                    float luu = mk_rings[i - 1][mk[2 * j + 5]];
                    float ruu = mk_rings[i + 1][mk[2 * j + 5]];
                    float ldd = mk_rings[i - 1][mk[2 * j - 3]];
                    float rdd = mk_rings[i + 1][mk[2 * j - 3]];

                    // std::cout << "cen = " << cen<<" "<<up << " "<<dw<<" "<<lu<<" "<<ru<<" "<<ld<<" "<<rd<<std::endl;
                    // if(abs(mk_rings[mk[2*j+1]][i-1]-mk_rings[mk[2*j+1]][i])>0.2 || abs(mk_rings[mk[2*j+1]][i+1]-mk_rings[mk[2*j+1]][i])>0.2
                    // ||  abs(mk_rings[mk[2*j+1]][i-2]-mk_rings[mk[2*j+1]][i])>0.24 || abs(mk_rings[mk[2*j+1]][i+2]-mk_rings[mk[2*j+1]][i])>0.24
                    //    || abs(mk_rings[mk[2*j+1]][i+1]-mk_rings[mk[2*j+1]][i-1])>0.24 || abs(mk_rings[mk[2*j+1]][i+2]-mk_rings[mk[2*j+1]][i-2])>0.3
                    // �м����� �� �� ���� ���� ���� ���� ������ֵ   ����    �м����� ���� ������ ������ ���� ������ ������  ������ֵ
                    // ȥ����Ⱥ�㣬��֤��ǰ����Χ��Ȧ(��ȥ������)������һ����
                    /*  // ** **
                    // **#**  ->��
                    // ** **  */

                    // * *
                    // *#*  ->��
                    // * *
                    if ((abs(up - cen) > DIS && abs(cen - dw) > DIS && abs(lu - cen) > DIS && abs(ru - cen) > DIS && abs(ld - cen) > DIS && abs(rd - cen) > DIS
                         // &&abs(lt-cen)>DIS&&abs(rt-cen)>DIS
                         )
                        // || (abs(uu-cen)>DIS&&abs(luu-cen)>DIS&&abs(ruu-cen)>DIS && abs(dd-cen)>DIS&&abs(ldd-cen)>DIS&&abs(rdd-cen)>DIS)
                    )
                    {
                        continue;
                    }
                }
                // i->1000  j->64   mk[dis,j, dis,j, ...]
                // recover the point from distance, the layer, and angle of the point
                pcl::PointXYZI p; //mk[(j)*2]�ǵ�ľ������?
                p.x = mk[(j)*2] * std::cos((j * factor_t + lowerBound) * M_PI / 180) * std::cos((i * 0.18 - 90) * M_PI / 180);
                p.y = mk[(j)*2] * std::cos((j * factor_t + lowerBound) * M_PI / 180) * std::sin(-(i * 0.18 - 90) * M_PI / 180);
                p.z = mk[(j)*2] * std::sin((mk[j * 2 + 1] * factor_t + lowerBound) * M_PI / 180);

                //�ж�ˮƽ�����Ƿ����һ����ֵ��intensityȡ��ֵ�����һ������ֵԽ��Խ������һ�����?
                if (mk[(j - 1) * 2] - mk[(j)*2] > mk[(j)*2] / dis_threshold || mk[(j + 1) * 2] - mk[(j)*2] > mk[(j)*2] / dis_threshold)
                {
                    // if(mk[(j-1)*2] - mk[(j)*2] > config["dis_threshold"].as<float>() || mk[(j+1)*2] - mk[(j)*2] > config["dis_threshold"].as<float>()){
                    p.intensity = (mk[(j - 1) * 2] - mk[(j)*2]) > (mk[(j + 1) * 2] - mk[(j)*2]) ? (mk[(j - 1) * 2] - mk[(j)*2]) : (mk[(j + 1) * 2] - mk[(j)*2]); //ȡ�󣬴�����ֵ
                }
                //
                else if (mk[j * 2 + 1] - mk[(j - 1) * 2 + 1] > local_config["angle_pixel_dis"].as<int>() || mk[(j + 1) * 2 + 1] - mk[j * 2 + 1] > local_config["angle_pixel_dis"].as<int>())
                {
                    p.intensity = (mk[j * 2 + 1] - mk[(j - 1) * 2 + 1]) > (mk[(j + 1) * 2 + 1] - mk[j * 2 + 1]) ? (mk[j * 2 + 1] - mk[(j - 1) * 2 + 1]) : (mk[(j + 1) * 2 + 1] - mk[j * 2 + 1]); //ȡ�󣬴�����ֵ
                }
                // p.intensity = int(mk[j*2+1] * 30) % 255;
                p.intensity = 0.1;        // TEST: set 0.1 to label horizontal line features
                pc_feature->push_back(p); // not push back ˮƽ����
            }
            else if (local_config["add_edge"].as<bool>())
            {
                if (i != 0 && i != rings - 1)
                {
                    if (pc_image[mk[j * 2 + 1]][i + 1] != -1 && pc_image[mk[j * 2 + 1]][i + 1] - pc_image[mk[j * 2 + 1]][i] > local_config["shuzhi_dis_th"].as<float>() * pc_image[mk[j * 2 + 1]][i])
                    {
                        pcl::PointXYZI p;
                        p.x = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180) * std::cos((mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180);
                        p.y = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180) * std::sin(-(mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180);
                        p.z = mk[(j)*2] * std::sin((i * factor_t + lowerBound) * M_PI / 180);
                        p.intensity = 0.5;
                        pc_feature->push_back(p);
                    }
                    else if (pc_image[mk[j * 2 + 1]][i - 1] != -1 && pc_image[mk[j * 2 + 1]][i - 1] - pc_image[mk[j * 2 + 1]][i] > local_config["shuzhi_dis_th"].as<float>() * pc_image[mk[j * 2 + 1]][i])
                    {
                        pcl::PointXYZI p;
                        p.x = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180) * std::cos((mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180);
                        p.y = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180) * std::sin(-(mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180);
                        p.z = mk[(j)*2] * std::sin((i * factor_t + lowerBound) * M_PI / 180);
                        p.intensity = 0.5;
                        pc_feature->push_back(p);
                    }
                }
            }
        }
    }
}

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
        pcl::PointXYZI point;
        Eigen::Vector4f raw_point;
        Eigen::Vector3f trans_point3;
        point = pc_feature->points[j];

        raw_point(0, 0) = point.x;
        raw_point(1, 0) = point.y;
        raw_point(2, 0) = point.z;
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
        if (point.intensity < 0 || distance_image.at<uchar>(y, x) < 0)
        {
            std::cout << "\033[33mError: has intensity<0\033[0m" << std::endl;
            exit(0);
        }
    
        points_num++;

        double pt_dis = pow(point.x * point.x + point.y * point.y + point.z * point.z, double(1.0 / 2.0));
        //std::cout << point.x << "  " << point.y << "   " << point.z << "  " << pt_dis << std::endl;

        // add distance weight or not
        bool add_dis_weight = false;
        if (add_dis_weight)
        {
            if (pt_dis > 50)
            {
                one_score += distance_image.at<uchar>(y, x) * 0.5;
            }
            else if (pt_dis > 25)
            {
                one_score += distance_image.at<uchar>(y, x) * 0.8;
            }
            else
            {
                one_score += distance_image.at<uchar>(y, x);
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

float countScore_imgbased(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature, const cv::Mat distance_image,
                 Eigen::Matrix4f RT, Eigen::Matrix3f camera_param){
    // 拷贝一份图片供计算得分
    cv::Mat distance_image_copy = distance_image.clone();
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
        pcl::PointXYZI point;
        Eigen::Vector4f raw_point;
        Eigen::Vector3f trans_point3;
        point = pc_feature->points[j];

        raw_point(0, 0) = point.x;
        raw_point(1, 0) = point.y;
        raw_point(2, 0) = point.z;
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
        
        // Error
        if (point.intensity < 0 || distance_image.at<uchar>(y, x) < 0)
        {
            std::cout << "\033[33mError: has intensity<0\033[0m" << std::endl;
            exit(0);
        }
    
        points_num++;

        // 获取投影点灰度值，同时将该点置为0，防止重复计算
        one_score += distance_image_copy.at<uchar>(y, x);
        distance_image_copy.at<uchar>(y, x) = 0;
        // 该点周围的点也进行灰度衰减？

    }

    score = one_score / 255.0;
    return score;
}

Eigen::Matrix4f position2RT(double* p)
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

Eigen::Matrix4f particle2RT(const Particle &p)
{
    return position2RT(p.x);
}

void RT2position(const Eigen::Matrix4f &RT, double* p)
{
    Eigen::Matrix3f rotate_matrix3 = RT.block(0, 0, 3, 3);
    Eigen::Vector3f ea = rotate_matrix3.eulerAngles(0, 1, 2);
    p[0] = RT(0, 3);
    p[1] = RT(1, 3);
    p[2] = RT(2, 3);
    p[3] = ea[0];
    p[4] = ea[1];
    p[5] = ea[2];
}