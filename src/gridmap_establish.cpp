#include "gridmap_establish/grid_map.h"

int main(int argc, char** argv){
    ros::init(argc, argv, "gridmap");
    ros::NodeHandle nh;
    
    GridMap map(&nh);

    ros::spin();
    return 0;
}