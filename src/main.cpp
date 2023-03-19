#include "initCalib.h"
#include <fstream>
#include <vector>

bool getPointcloud(std::string filename, pcl::PointCloud<pcl::PointXYZI>::Ptr ptcloud)
{
    // Load the actual pointcloud.
    const size_t kMaxNumberOfPoints = 1e6; // From Readme for raw files.
    ptcloud->clear();
    ptcloud->reserve(kMaxNumberOfPoints);
    std::ifstream input(filename, std::ios::in | std::ios::binary);
    if (!input)
    {
        std::cout << "Could not open pointcloud file.\n";
        return false;
    }

    for (size_t i = 0; input.good() && !input.eof(); i++)
    {
        pcl::PointXYZI point;
        input.read((char *)&point.x, 3 * sizeof(float));
        input.read((char *)&point.intensity, sizeof(float));
        ptcloud->push_back(point);
    }
    input.close();
    return true;
}

void extract_pc_feature(pcl::PointCloud<pcl::PointXYZI>::Ptr &pc, pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_feature, YAML::Node config)
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
                ((mk[(j - 1) * 2] - mk[(j)*2] > mk[(j)*2] / config["dis_threshold"].as<float>())) || // && std::abs(mk[(j-1)*2+1]-mk[(j)*2+1])==1) || //ˮƽ��ߵ����˵����һ����ֵ(��������)
                ((mk[(j + 1) * 2] - mk[(j)*2] > mk[(j)*2] / config["dis_threshold"].as<float>())) || //&& std::abs(mk[(j+1)*2+1]-mk[(j)*2+1])==1) || //ˮƽ�ұߵ����˵����һ�����?(��������)
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
                if (mk[(j - 1) * 2] - mk[(j)*2] > mk[(j)*2] / config["dis_threshold"].as<float>() || mk[(j + 1) * 2] - mk[(j)*2] > mk[(j)*2] / config["dis_threshold"].as<float>())
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
                ((mk[(j - 1) * 2] - mk[(j)*2] > mk[(j)*2] / config["dis_threshold"].as<float>() * 1.5)) || // && (std::abs(mk[(j-1)*2+1] - mk[(j)*2+1]) == 1 ))||//mk[(j)*2] / config["dis_threshold"].as<float>() || //ˮƽ��ߵ����˵����һ����ֵ(��������)
                ((mk[(j + 1) * 2] - mk[(j)*2] > mk[(j)*2] / config["dis_threshold"].as<float>() * 1.5)) || //&& (std::abs(mk[(j+1)*2+1] - mk[(j)*2+1]) == 1 ))||//mk[(j)*2] / config["dis_threshold"].as<float>() || //ˮƽ�ұߵ����˵����һ�����?(��������)
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
                if (mk[(j - 1) * 2] - mk[(j)*2] > mk[(j)*2] / config["dis_threshold"].as<float>() || mk[(j + 1) * 2] - mk[(j)*2] > mk[(j)*2] / config["dis_threshold"].as<float>())
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


int main() {
    std::ofstream result_txt("results/result.txt", ios::app);
    result_txt << "The beginning of the PSO!!!!\n" << endl;

    YAML::Node config = YAML::LoadFile("configs/config0.yaml");
    
    // 读取相机内参
    Eigen::Matrix3f camera_param;
    camera_param << config["fx"].as<float>(), 0, config["cx"].as<float>(),
        0, config["fy"].as<float>(), config["cy"].as<float>(),
        0, 0, 1;

    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_source(new pcl::PointCloud<pcl::PointXYZI>);
    getPointcloud("./res/bin/0000000000.bin", pc_source);

    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_filtered(pc_source);
    pcl::PassThrough<pcl::PointXYZI> pass_filter;
    float x_max, x_min, y_min, y_max, z_min, z_max;
    x_max = config["x_max"].as<float>();
    x_min = config["x_min"].as<float>();
    y_min = config["y_min"].as<float>();
    y_max = config["y_max"].as<float>();
    z_min = config["z_min"].as<float>();
    z_max = config["z_max"].as<float>();
    pass_filter.setInputCloud(pc_filtered);
    pass_filter.setFilterFieldName("y");
    pass_filter.setFilterLimits(y_min, y_max);
    pass_filter.filter(*pc_filtered);
    pass_filter.setFilterFieldName("x");
    pass_filter.setFilterLimits(x_min, x_max); 
    pass_filter.filter(*pc_filtered);
    pass_filter.setFilterFieldName("z");
    pass_filter.setFilterLimits(z_min, z_max);
    pass_filter.filter(*pc_filtered);

    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature(new pcl::PointCloud<pcl::PointXYZI>);
    extract_pc_feature(pc_filtered, pc_feature, config);

    // // visualize the feature pointcloud
    // pcl::visualization::PCLVisualizer feature_viewer("pc_feature");
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> feature_color(pc_feature, 0, 255, 0);
    // feature_viewer.setBackgroundColor(0, 0, 0);
    // feature_viewer.addPointCloud(pc_feature, feature_color, "pc_feature");
    // feature_viewer.spin();

    cv::Mat img_source = cv::imread("res/semantic_imgs/0000000000.png", cv::IMREAD_GRAYSCALE);
    // cv::imshow("img_source", img_source);
    // cv::waitKey();
    
    // for( auto i = distance_img.begin<uchar>(); i != distance_img.end<uchar>(); ++i) {
    //     if(*i == 0) continue;
    //     *i = 255;
    // }
    cv::Mat img_thresh;
    cv::threshold(img_source, img_thresh, 0, 255, cv::THRESH_BINARY);
    cv::Mat distance_img = img_thresh.clone();
    cv::distanceTransform(distance_img, distance_img, 1, 3);
    cv::normalize(distance_img, distance_img, config["normalize_inside"].as<int>(), 0, cv::NORM_INF);
    distance_img.convertTo(distance_img, CV_8U);
    // cv::imshow("distance_img", distance_img);
    // cv::waitKey();
    for( auto i = distance_img.begin<uchar>(); i != distance_img.end<uchar>(); ++i) {
        if(*i == 0) continue;
        *i = 255-*i;
    }
    // cv::imshow("distance_img", distance_img);
    // cv::waitKey();

    cv::Mat notmap = ~img_thresh;
    cv::Mat notdistimg;
    cv::distanceTransform(notmap, notdistimg, 1, 3);
    cv::normalize(notdistimg, notdistimg, config["normalize_outside"].as<int>(), 0, cv::NORM_INF);
    notdistimg.convertTo(notdistimg, CV_8U);
    for( auto i = notdistimg.begin<uchar>(); i != notdistimg.end<uchar>(); ++i) {
        if(*i == 0) continue;
        *i = 255-*i;
    }
    // cv::imshow("distance_img", notdistimg);
    // cv::waitKey();
    add(distance_img, notdistimg, distance_img);
    // cv::imshow("distance_img", distance_img);
    // cv::waitKey();


    
    // cv::Mat img_before_pso;
    // cout << initcalib.particle2RT(initcalib.get_initial_partical()) << endl;
    // cout << initcalib.camera_param << endl;
    // cout << initcalib.get_initial_partical()->pbest_fitness << endl;
    // for(int i = 0; i < 6; ++i) cout << initcalib.get_initial_partical()->x[i] << " ";
    // cout << endl;

    // result_txt << "Initial_T:\n" << initcalib.particle2RT(initcalib.get_initial_partical()) << endl;
    // result_txt << "Camera_param:\n" << initcalib.camera_param << endl;
    // result_txt << "Initial_fitness:\n" << initcalib.get_initial_partical()->pbest_fitness << endl;

    // project2image(pc_feature, distance_img, img_before_pso, 
    //                     initcalib.particle2RT(initcalib.get_initial_partical()), initcalib.camera_param);
    // cv::imwrite("results/project_imgs/before_pso.jpg", img_before_pso);

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
    cout << "init_fitness: " << pso_calib->particle_swarm_->gbest_fitness << endl;
    cout << "init_position: ";
    for(int i = 0; i < 6; ++i) cout << pso_calib->particle_swarm_->gbest_position[i] << " ";
    
    // PSO算法迭代优化
    double temp_fitness = pso_calib->particle_swarm_->gbest_fitness; // 有隐患，可能存在position不同但fitness相同的情况
    for(int i = 0; i < config["search_loops"].as<int>(); ++i){
        pso_calib->step();
        cout << "\nROUND " << i << "!" << endl;
        if (pso_calib->particle_swarm_->gbest_fitness == temp_fitness) {
            cout << "result_fitness is not changed!" << endl;
            continue;
        }
        temp_fitness = pso_calib->particle_swarm_->gbest_fitness;
        cout << "搜索后结果" << endl;
        cout << "result_fitness:" << pso_calib->particle_swarm_->gbest_fitness << endl;
        cout << "result_position:";
        for(int i = 0; i < 6; ++i) cout << pso_calib->particle_swarm_->gbest_position[i] << " ";
        
        cv::Mat test1;
        project2image(pc_feature,distance_img, test1, 
                        position2RT(pso_calib->particle_swarm_->gbest_position), camera_param);
        cv::imwrite("results/project_imgs/" + std::to_string(i) + ".jpg", test1);
    }
    
    result_txt.close();
    

    return 0;
}