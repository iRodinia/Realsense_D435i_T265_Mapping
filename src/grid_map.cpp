#include "gridmap_establish/grid_map.h"

GridMap::GridMap(ros::NodeHandle* nh):node_(*nh)
{
  /* get parameter */
  double x_size, y_size, z_size;
  node_.param("resolution", mp_.resolution_, 0.1);
  node_.param("map_size_x", x_size, -1.0);
  node_.param("map_size_y", y_size, -1.0);
  node_.param("map_size_z", z_size, -1.0);
  node_.param("local_update_range_x", mp_.local_update_range_(0), -1.0);
  node_.param("local_update_range_y", mp_.local_update_range_(1), -1.0);
  node_.param("local_update_range_z", mp_.local_update_range_(2), -1.0);
  node_.param("obstacles_inflation", mp_.obstacles_inflation_, -1.0);
  node_.param("p_occ", mp_.p_occ_, 0.50);
  node_.param("visualization_truncate_height", mp_.visualization_truncate_height_, 999.0);
  node_.param("virtual_ceil_height", mp_.virtual_ceil_height_, 2.0);
  node_.param("local_frame_id", mp_.local_frame_id_, string("camera"));
  node_.param("global_frame_id", mp_.global_frame_id_, string("world"));
  node_.param("ground_height", mp_.ground_height_, 0.1);

  mp_.resolution_inv_ = 1 / mp_.resolution_;
  mp_.map_origin_ = Eigen::Vector3d(-x_size / 2.0, -y_size / 2.0, mp_.ground_height_);
  mp_.map_size_ = Eigen::Vector3d(x_size, y_size, z_size);

  for (int i = 0; i < 3; ++i)
    mp_.map_voxel_num_(i) = ceil(mp_.map_size_(i) / mp_.resolution_);

  mp_.map_min_boundary_ = mp_.map_origin_;
  mp_.map_max_boundary_ = mp_.map_origin_ + mp_.map_size_;

  // initialize data buffers
  int buffer_size = mp_.map_voxel_num_(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2);
  md_.known_map_buffer_ = vector<double>(buffer_size, 0);
  md_.occupancy_buffer_inflate_ = vector<char>(buffer_size, -1);
  md_.cam2body_ << 0.0, 0.0, 1.0, 0.0,
      -1.0, 0.0, 0.0, 0.0,
      0.0, -1.0, 0.0, -0.02,
      0.0, 0.0, 0.0, 1.0;

  // use odometry and point cloud
  indep_cloud_sub_ =
      node_.subscribe<sensor_msgs::PointCloud2>("/grid_map/cloud", 10, &GridMap::cloudCallback, this);
  indep_odom_sub_ =
      node_.subscribe<nav_msgs::Odometry>("/grid_map/odom", 10, &GridMap::odomCallback, this);

  md_.orig_cloud_ptr = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
  md_.sampled_cloud_ptr = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
  md_.free_cloud_ptr = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
  downSizeFilter.setLeafSize(mp_.resolution_, mp_.resolution_, mp_.resolution_);

  vis_local_timer_ = node_.createTimer(ros::Duration(0.05), &GridMap::visLocalMapCallback, this);
  vis_global_timer_ = node_.createTimer(ros::Duration(0.2), &GridMap::visGlobalMapCallback, this);

  known_map_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/known_map", 10);
  local_map_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/local_map", 10);
  map_inf_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/occupancy_inflate", 10);
  unknown_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/unknown_map", 10);

  md_.has_odom_ = false;
  md_.has_cloud_ = false;
  md_.fuse_time_ = 0.0;
  md_.max_fuse_time_ = 0.0;
}

bool GridMap::odomValid() { return md_.has_odom_; }

Eigen::Vector3d GridMap::getOrigin() { return mp_.map_origin_; }

void GridMap::getRegion(Eigen::Vector3d &ori, Eigen::Vector3d &size)
{
  ori = mp_.map_origin_, size = mp_.map_size_;
}

void GridMap::resetBuffer()
{
  Eigen::Vector3d min_pos = mp_.map_min_boundary_;
  Eigen::Vector3d max_pos = mp_.map_max_boundary_;
  resetBuffer(min_pos, max_pos);

  md_.local_bound_min_ = Eigen::Vector3i::Zero();
  md_.local_bound_max_ = mp_.map_voxel_num_ - Eigen::Vector3i::Ones();
}

void GridMap::resetBuffer(Eigen::Vector3d min_pos, Eigen::Vector3d max_pos)
{

  Eigen::Vector3i min_id, max_id;
  posToIndex(min_pos, min_id);
  posToIndex(max_pos, max_id);

  boundIndex(min_id);
  boundIndex(max_id);
  /* reset occ and dist buffer */
  for (int x = min_id(0); x <= max_id(0); ++x)
    for (int y = min_id(1); y <= max_id(1); ++y)
      for (int z = min_id(2); z <= max_id(2); ++z)
      {
        md_.occupancy_buffer_inflate_[toAddress(x, y, z)] = -1;
      }
  
  md_.local_bound_min_ = min_id;
  md_.local_bound_max_ = max_id;
}

void GridMap::clearAndInflateLocalMap()
{
  // inflate occupied voxels to compensate robot size
  int inf_step = ceil(mp_.obstacles_inflation_ / mp_.resolution_);
  Eigen::Vector3i min_cut_m = md_.local_bound_min_ - Eigen::Vector3i(inf_step, inf_step, inf_step);
  Eigen::Vector3i max_cut_m = md_.local_bound_max_ + Eigen::Vector3i(inf_step, inf_step, inf_step);
  boundIndex(min_cut_m);
  boundIndex(max_cut_m);

  for (int x = min_cut_m(0); x <= max_cut_m(0); ++x)
    for (int y = min_cut_m(1); y <= max_cut_m(1); ++y)
      for (int z = min_cut_m(2); z <= max_cut_m(2); ++z)
      {
        md_.occupancy_buffer_inflate_[toAddress(x, y, z)] = 0;
      }

  // inflate obstacles
  vector<Eigen::Vector3i> inf_pts(pow(2 * inf_step + 1, 3));
  Eigen::Vector3i inf_pt_idx;
  for (int x = md_.local_bound_min_(0); x <= md_.local_bound_max_(0); ++x)
    for (int y = md_.local_bound_min_(1); y <= md_.local_bound_max_(1); ++y)
      for (int z = md_.local_bound_min_(2); z <= md_.local_bound_max_(2); ++z)
      {
        if (md_.known_map_buffer_[toAddress(x, y, z)] == 1)
        {
          inflatePoint(Eigen::Vector3i(x, y, z), inf_step, inf_pts);
          for (int k = 0; k < (int)inf_pts.size(); ++k)
          {
            inf_pt_idx = inf_pts[k];
            int idx_inf = toAddress(inf_pt_idx);
            if (idx_inf < 0 ||
                idx_inf >= mp_.map_voxel_num_(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2))
            {
              continue;
            }
            md_.occupancy_buffer_inflate_[idx_inf] = 1;
          }
        }
      }

  // add virtual ceiling to limit flight height
  if (mp_.virtual_ceil_height_ > -0.5)
  {
    int ceil_id = floor((mp_.virtual_ceil_height_ - mp_.map_origin_(2)) * mp_.resolution_inv_);
    for (int x = md_.local_bound_min_(0); x <= md_.local_bound_max_(0); ++x)
      for (int y = md_.local_bound_min_(1); y <= md_.local_bound_max_(1); ++y)
      {
        md_.occupancy_buffer_inflate_[toAddress(x, y, ceil_id)] = 1;
      }
  }
}

void GridMap::odomCallback(const nav_msgs::OdometryConstPtr &odom)
{
  md_.last_camera_pos_ = md_.camera_pos_;
  md_.last_camera_q_ = md_.camera_q_;

  md_.camera_pos_(0) = odom->pose.pose.position.x;
  md_.camera_pos_(1) = odom->pose.pose.position.y;
  md_.camera_pos_(2) = odom->pose.pose.position.z;
  md_.camera_q_.x() = odom->pose.pose.orientation.x;
  md_.camera_q_.y() = odom->pose.pose.orientation.y;
  md_.camera_q_.z() = odom->pose.pose.orientation.z;
  md_.camera_q_.w() = odom->pose.pose.orientation.w;

  try{
    listener.lookupTransform(mp_.global_frame_id_, mp_.local_frame_id_, ros::Time(0), md_.cam2world_tf_);
    md_.has_odom_ = true;
  }
  catch (tf::TransformException &ex) {
    ROS_ERROR("%s",ex.what());
    md_.has_odom_ = false;
  }
}

void GridMap::cloudCallback(const sensor_msgs::PointCloud2ConstPtr &img)
{
  md_.has_cloud_ = true;
  if (!md_.has_odom_)
  {
    ROS_INFO("No odom!");
    return;
  }
  pcl::fromROSMsg(*img, *md_.orig_cloud_ptr);

  pcl::PointCloud<pcl::PointXYZ> sampled_cloud_tf, free_cloud_tf;
  
  if (orig_cloud_ptr.points.size() == 0)
    return;
  if (isnan(md_.camera_pos_(0)) || isnan(md_.camera_pos_(1)) || isnan(md_.camera_pos_(2)))
    return;
  
  sampled_cloud_ptr.clear();
  downSizeFilter.setInputCloud(orig_cloud_ptr);

  pcl_ros::transformPointCloud(orig_cloud_ptr, free_cloud_ptr, md_.cam2world_tf_);

  // this->resetBuffer(md_.camera_pos_ - mp_.local_update_range_,
  //                   md_.camera_pos_ + mp_.local_update_range_);   // Reset region should be smaller than sensor detected area

  int inf_step = ceil(mp_.obstacles_inflation_ / mp_.resolution_);
  int inf_step_z = ceil(inf_step / 2);  // height inflation is not correct

  pcl::PointXYZ pt;
  Eigen::Vector3d p3d;
  for (size_t i = 0; i < free_cloud_ptr.points.size(); ++i)
  {
    pt = free_cloud_ptr.points[i];
    p3d(0) = pt.x;
    p3d(1) = pt.y;
    p3d(2) = pt.z;
    /* point inside update range */
    Eigen::Vector3d devi = p3d - md_.camera_pos_;
    Eigen::Vector3d inf_pt;
    Eigen::Vector3i inf_pt_idx;
    if (fabs(devi(0)) < mp_.local_update_range_(0) && fabs(devi(1)) < mp_.local_update_range_(1) &&
        fabs(devi(2)) < mp_.local_update_range_(2))
    {
      /* inflate the point */
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
}

void GridMap::visKnownMapCallback(const ros::TimerEvent & /*event*/)
{

}

void GridMap::visLocalMapCallback(const ros::TimerEvent & /*event*/)
{
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;

  Eigen::Vector3i min_cut = md_.local_bound_min_;
  Eigen::Vector3i max_cut = md_.local_bound_max_;
  boundIndex(min_cut);
  boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z)
      {
        if (md_.occupancy_buffer_inflate_[toAddress(x, y, z)] < mp_.p_occ_)
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
  cloud.header.frame_id = mp_.local_frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;

  pcl::toROSMsg(cloud, cloud_msg);
  local_map_pub_.publish(cloud_msg);
}

void GridMap::visGlobalMapCallback(const ros::TimerEvent & /*event*/)
{
  if (map_inf_pub_.getNumSubscribers() <= 0)
    return;

  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;

  Eigen::Vector3i min_cut = Eigen::Vector3i::Zero();
  Eigen::Vector3i max_cut = mp_.map_voxel_num_ - Eigen::Vector3i::Ones();
  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z)
      {
        if (md_.occupancy_buffer_inflate_[toAddress(x, y, z)] < mp_.p_occ_)
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
  map_inf_pub_.publish(cloud_msg);
}

void GridMap::visUnknownMapCallback(const ros::TimerEvent & /*event*/)
{
  if (unknown_pub_.getNumSubscribers() <= 0)
    return;

  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;
  Eigen::Vector3i min_cut = Eigen::Vector3i::Zero();
  Eigen::Vector3i max_cut = mp_.map_voxel_num_ - Eigen::Vector3i::Ones();

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z)
      {
        if (md_.occupancy_buffer_inflate_[toAddress(x, y, z)] < 0)
        {
          Eigen::Vector3d pos;
          indexToPos(Eigen::Vector3i(x, y, z), pos);
          if (pos(2) > mp_.visualization_truncate_height_)
            continue;

          pt.x = pos(0);
          pt.y = pos(1);
          pt.z = pos(2);
          cloud.push_back(pt);
        }
      }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.global_frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  unknown_pub_.publish(cloud_msg);
}