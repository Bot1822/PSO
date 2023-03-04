#include <iostream>
#include <boost/thread.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;

bool getPointcloud(std::string filename, pcl::PointCloud<pcl::PointXYZI>::Ptr ptcloud) //�����ļ��У��õ�����
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


int main() {
    cout << "start test pcl!####################################" << endl;

    pcl::PointCloud<pcl::PointXYZI>::Ptr testpcl(new pcl::PointCloud<pcl::PointXYZI>);
    // if (pcl::io::loadPCDFile<pcl::PointXYZI>("./res/pcd/1317013472.645279884.pcd", *testpcl) == -1) {
    //     PCL_ERROR("could not find pcd");
    // }

    getPointcloud("./res/bin/0000000000.bin", testpcl);

    pcl::visualization::PCLVisualizer viewer("test viewer");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> test_color(testpcl, 0, 255, 0);

    viewer.setBackgroundColor(0, 0, 0);
    viewer.addPointCloud(testpcl, test_color, "testpcd");

    viewer.spin();

    // getchar();

    return 0;

}