#include "gridmap_establish/raycaster.h"
#include "gridmap_establish/grid_map.h"

GridMap::GridMap(ros::NodeHandle* nh):node_(*nh)
{
  /* get parameter */
  double x_size, y_size, z_size;
  node_.param("resolution", mp_.resolution_, 0.05);
  node_.param("map_size_x", x_size, -1.0);
  node_.param("map_size_y", y_size, -1.0);
  node_.param("map_size_z", z_size, -1.0);
  node_.param("obstacles_inflation", mp_.obstacles_inflation_, -1.0);
  node_.param("threshold", mp_.threshold, 0.50);
  node_.param("visualization_truncate_height", mp_.visualization_truncate_height_, 999.0);
  node_.param("virtual_ceil_height", mp_.virtual_ceil_height_, 2.0);
  node_.param("point_cloud_topic", mp_.point_cloud_topic_, string("/rtabmap/cloud_map"));
  node_.param("global_frame_id", mp_.global_frame_id_, string("map"));
  node_.param("ground_height", mp_.ground_height_, 0.05);

  mp_.resolution_inv_ = 1 / mp_.resolution_;
  mp_.map_origin_ = Eigen::Vector3d(-x_size / 2.0, -y_size / 2.0, mp_.ground_height_);
  mp_.map_size_ = Eigen::Vector3d(x_size, y_size, z_size);

  for (int i = 0; i < 3; ++i)
    mp_.map_voxel_num_(i) = ceil(mp_.map_size_(i) / mp_.resolution_);

  mp_.map_min_boundary_ = mp_.map_origin_;
  mp_.map_max_boundary_ = mp_.map_origin_ + mp_.map_size_;

  // initialize data buffers
  int buffer_size = mp_.map_voxel_num_(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2);
  md_.occupancy_buffer_inflate_ = vector<double>(buffer_size, -1);
  md_.has_cloud = false;

  // use odometry and point cloud
  world_cloud_sub_ =
      node_.subscribe<sensor_msgs::PointCloud2>(mp_.point_cloud_topic_, 10, &GridMap::cloudCallback, this);

  md_.map_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
  md_.map_unknown = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
  downSizeFilter.setLeafSize(mp_.resolution_/2.0, mp_.resolution_/2.0, mp_.resolution_/2.0);

  vis_timer_ = node_.createTimer(ros::Duration(0.1), &GridMap::visMapCallback, this);

  grid_map_pub_ = node_.advertise<sensor_msgs::PointCloud2>("known_map", 10);
  unknown_map_pub_ = node_.advertise<sensor_msgs::PointCloud2>("unknown_map", 10);

  md_.map_update_time_ = 0.0;
}

void GridMap::clearMap()
{
  if(!md_.has_cloud) return;

  int buffer_size = md_.occupancy_buffer_inflate_.size();
  for (int i = 0; i < buffer_size; i++)
    md_.occupancy_buffer_inflate_[i] = -1;

  // add virtual ceiling to limit flight height
  if (mp_.virtual_ceil_height_ > 0)
  {
    int ceil_id = floor((mp_.virtual_ceil_height_ - mp_.map_origin_(2)) * mp_.resolution_inv_);
    for (int x = 0; x < mp_.map_voxel_num_(0); ++x)
      for (int y = 0; y < mp_.map_voxel_num_(1); ++y)
      {
        md_.occupancy_buffer_inflate_[toAddress(x, y, ceil_id)] = 1;
      }
  }
}

void GridMap::cloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
  ros::Time t_start = ros::Time::now();
  pcl::PointCloud<pcl::PointXYZ> raw_cloud, raw_free_cloud;
  pcl::fromROSMsg(*msg, raw_cloud);
  if (raw_cloud.points.size() == 0){
    md_.has_cloud = false;
    return;
  }
  else {
    md_.has_cloud = true;
  }

  RayCaster raycaster;
  Eigen::Vector3d half = Eigen::Vector3d(0.5, 0.5, 0.5);
  pcl::PointXYZ pt, ray_pt;
  Eigen::Vector3d p3d, ray_p3d;
  for (size_t i = 0; i < raw_cloud.points.size(); ++i)
  {
    pt = raw_cloud.points[i];
    p3d(0) = pt.x; p3d(1) = pt.y; p3d(2) = pt.z;
    raycaster.setInput(p3d / mp_.resolution_, Eigen::Vector3d::Zero());
    while (raycaster.step(ray_p3d))
    {
      Eigen::Vector3d tmp = (ray_p3d + half) * mp_.resolution_;
      if(isInMap(tmp))
      {
        ray_pt.x = tmp(0); ray_pt.y = tmp(1); ray_pt.z = tmp(2);
        raw_free_cloud.push_back(ray_pt);
      }
    }
  }

  md_.map_cloud->clear();
  downSizeFilter.setInputCloud(raw_cloud.makeShared());
  downSizeFilter.filter(*md_.map_cloud);
  md_.map_unknown->clear();
  downSizeFilter.setInputCloud(raw_free_cloud.makeShared());
  downSizeFilter.filter(*md_.map_unknown);

  Eigen::Vector3i pt_idx;
  for (size_t i = 0; i < md_.map_unknown->points.size(); ++i)
  {
    pt = md_.map_unknown->points[i];
    p3d(0) = pt.x; p3d(1) = pt.y; p3d(2) = pt.z;
    if (!isInMap(p3d))
      continue;
    posToIndex(p3d, pt_idx);
    md_.occupancy_buffer_inflate_[toAddress(pt_idx)] = 0;
  }

  clearMap();
  int inf_step = ceil(mp_.obstacles_inflation_ / mp_.resolution_);
  int inf_step_z = ceil(inf_step / 2);  // height inflation is not correct
  for (size_t i = 0; i < md_.map_cloud->points.size(); ++i)
  {
    pt = md_.map_cloud->points[i];
    p3d(0) = pt.x; p3d(1) = pt.y; p3d(2) = pt.z;
    if (!isInMap(p3d))
      continue;
    else{
      Eigen::Vector3d inf_pt;
      Eigen::Vector3i inf_pt_idx;
      for (int x = -inf_step; x <= inf_step; ++x)
        for (int y = -inf_step; y <= inf_step; ++y)
          for (int z = -inf_step_z; z <= inf_step_z; ++z)
          {
            inf_pt(0) = p3d(0) + x * mp_.resolution_;
            inf_pt(1) = p3d(1) + y * mp_.resolution_;
            inf_pt(2) = p3d(2) + z * mp_.resolution_;
            posToIndex(inf_pt, inf_pt_idx);
            if (!isInMap(inf_pt_idx))
              continue;
            md_.occupancy_buffer_inflate_[toAddress(inf_pt_idx)] = 1;
          }
    }
  }
  ros::Duration _dur = ros::Time::now() - t_start;
  md_.map_update_time_ = _dur.toSec();
}

void GridMap::visMapCallback(const ros::TimerEvent & /*event*/)
{
  if (grid_map_pub_.getNumSubscribers() <= 0)
    return;

  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;

  Eigen::Vector3i min_cut = Eigen::Vector3i::Zero();
  Eigen::Vector3i max_cut = mp_.map_voxel_num_ - Eigen::Vector3i::Ones();
  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z)
      {
        if (md_.occupancy_buffer_inflate_[toAddress(x, y, z)] < mp_.threshold)
          continue;
        Eigen::Vector3d pos;
        indexToPos(Eigen::Vector3i(x, y, z), pos);
        if (pos(2) > mp_.visualization_truncate_height_)
          continue;
        pt.x = pos(0);
        pt.y = pos(1);
        pt.z = pos(2);
        cloud.push_back(pt);
      }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.global_frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  grid_map_pub_.publish(cloud_msg);
}