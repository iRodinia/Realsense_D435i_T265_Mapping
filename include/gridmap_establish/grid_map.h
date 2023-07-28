#ifndef _GRID_MAP_H
#define _GRID_MAP_H

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
// #include <pcl/filters/statistical_outlier_removal.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>

#include <tf/transform_listener.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>

using namespace std;

// voxel hashing
template <typename T>
struct matrix_hash : std::unary_function<T, size_t> {
  std::size_t operator()(T const& matrix) const {
    size_t seed = 0;
    for (size_t i = 0; i < matrix.size(); ++i) {
      auto elem = *(matrix.data() + i);
      seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

// mapping parameters
struct MappingParameters {
  /* map properties */
  Eigen::Vector3d map_origin_, map_size_;
  double virtual_ceil_height_, ground_height_;
  Eigen::Vector3d map_min_boundary_, map_max_boundary_;  // map range in pos
  Eigen::Vector3i map_voxel_num_;                        // map range in index
  double local_update_range_;
  double resolution_, resolution_inv_;
  double obstacles_inflation_;
  string local_frame_id_, global_frame_id_;
  int pose_type_;
  double p_occ_;                               // occupancy probability

  /* visualization and computation time display */
  double visualization_truncate_height_;
};

// mapping data
struct MappingData {
  // main map data, occupancy of each voxel and Euclidean distance
  std::vector<double> known_map_buffer_;
  std::vector<char> occupancy_buffer_inflate_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr orig_cloud_ptr, sampled_cloud_ptr, free_cloud_ptr, free_sampled_ptr;

  // camera position and pose data
  Eigen::Vector3d camera_pos_, last_camera_pos_;
  Eigen::Quaterniond camera_q_, last_camera_q_;
  Eigen::Matrix4d cam2body_;
  tf::StampedTransform cam2world_tf_;

  // flags of map state
  bool has_odom_, has_cloud_;

  // range of updating grid
  Eigen::Vector3i local_bound_min_, local_bound_max_;

  // computation time
  double fuse_time_, max_fuse_time_;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


class GridMap
{
public:
  GridMap(ros::NodeHandle* nh);
  ~GridMap(){}

  enum { POSE_STAMPED = 1, ODOMETRY = 2, INVALID_IDX = -10000 };

  // occupancy map management
  void resetBuffer();
  void resetBuffer(Eigen::Vector3d min, Eigen::Vector3d max);

  inline void posToIndex(const Eigen::Vector3d& pos, Eigen::Vector3i& id);
  inline void indexToPos(const Eigen::Vector3i& id, Eigen::Vector3d& pos);
  inline int toAddress(const Eigen::Vector3i& id);
  inline int toAddress(int& x, int& y, int& z);
  inline bool isInMap(const Eigen::Vector3d& pos);
  inline bool isInMap(const Eigen::Vector3i& idx);

  inline void setOccupancy(Eigen::Vector3d pos, double occ = 1);
  inline void setOccupied(Eigen::Vector3d pos);
  inline int getOccupancy(Eigen::Vector3d pos);
  inline int getOccupancy(Eigen::Vector3i id);
  inline int getInflateOccupancy(Eigen::Vector3d pos);

  inline void boundIndex(Eigen::Vector3i& id);
  inline bool isUnknown(const Eigen::Vector3i& id);
  inline bool isUnknown(const Eigen::Vector3d& pos);
  inline bool isKnownFree(const Eigen::Vector3i& id);
  inline bool isKnownOccupied(const Eigen::Vector3i& id);

  bool odomValid();
  void getRegion(Eigen::Vector3d& ori, Eigen::Vector3d& size);
  inline double getResolution();
  Eigen::Vector3d getOrigin();

  typedef std::shared_ptr<GridMap> Ptr;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  MappingParameters mp_;
  MappingData md_;

  ros::NodeHandle node_;
  ros::Subscriber indep_cloud_sub_, indep_odom_sub_;
  ros::Publisher known_map_pub_, local_map_pub_, map_inf_pub_, unknown_pub_;
  ros::Timer vis_local_timer_, vis_global_timer_;

  pcl::VoxelGrid<pcl::PointXYZ> downSizeFilter;

  tf::TransformListener listener;

  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& img);
  void odomCallback(const nav_msgs::OdometryConstPtr& odom);

  // visualization
  void visKnownMapCallback(const ros::TimerEvent& /*event*/);
  void visLocalMapCallback(const ros::TimerEvent& /*event*/);
  void visGlobalMapCallback(const ros::TimerEvent& /*event*/);
  void visUnknownMapCallback(const ros::TimerEvent& /*event*/);

  // main update process
  void clearAndInflateLocalMap();

  inline void inflatePoint(const Eigen::Vector3i& pt, int step, vector<Eigen::Vector3i>& pts);
};

/* ============================== definition of inline function
 * ============================== */

inline int GridMap::toAddress(const Eigen::Vector3i& id) {
  return id(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2) + id(1) * mp_.map_voxel_num_(2) + id(2);
}

inline int GridMap::toAddress(int& x, int& y, int& z) {
  return x * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2) + y * mp_.map_voxel_num_(2) + z;
}

inline void GridMap::boundIndex(Eigen::Vector3i& id) {
  Eigen::Vector3i id1;
  id1(0) = max(min(id(0), mp_.map_voxel_num_(0) - 1), 0);
  id1(1) = max(min(id(1), mp_.map_voxel_num_(1) - 1), 0);
  id1(2) = max(min(id(2), mp_.map_voxel_num_(2) - 1), 0);
  id = id1;
}

inline bool GridMap::isUnknown(const Eigen::Vector3i& id) {
  Eigen::Vector3i id1 = id;
  boundIndex(id1);
  int adr = toAddress(id1);
  return md_.known_map_buffer_[adr] == 0 && md_.occupancy_buffer_inflate_[adr] < 0;
}

inline bool GridMap::isUnknown(const Eigen::Vector3d& pos) {
  Eigen::Vector3i idc;
  posToIndex(pos, idc);
  return isUnknown(idc);
}

inline bool GridMap::isKnownFree(const Eigen::Vector3i& id) {
  Eigen::Vector3i id1 = id;
  boundIndex(id1);
  int adr = toAddress(id1);
  return md_.known_map_buffer_[adr] == 0 && md_.occupancy_buffer_inflate_[adr] >= 0 && md_.occupancy_buffer_inflate_[adr] < mp_.p_occ_;
}

inline bool GridMap::isKnownOccupied(const Eigen::Vector3i& id) {
  Eigen::Vector3i id1 = id;
  boundIndex(id1);
  int adr = toAddress(id1);
  return md_.known_map_buffer_[adr] == 1 || md_.occupancy_buffer_inflate_[adr] >= mp_.p_occ_;
}

inline void GridMap::setOccupied(Eigen::Vector3d pos) {
  if (!isInMap(pos)) return;
  Eigen::Vector3i id;
  posToIndex(pos, id);
  md_.occupancy_buffer_inflate_[id(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2) +
                                id(1) * mp_.map_voxel_num_(2) + id(2)] = 1;
}

inline void GridMap::setOccupancy(Eigen::Vector3d pos, double occ) {
  if (occ != 1 && occ != 0) {
    cout << "occ value error!" << endl;
    return;
  }

  if (!isInMap(pos)) return;
  Eigen::Vector3i id;
  posToIndex(pos, id);
  md_.known_map_buffer_[toAddress(id)] = occ;
}

inline int GridMap::getOccupancy(Eigen::Vector3d pos) {
  if (!isInMap(pos)) return -1;
  Eigen::Vector3i id;
  posToIndex(pos, id);
  return md_.known_map_buffer_[toAddress(id)] >= mp_.p_occ_ ? 1 : 0;
}

inline int GridMap::getInflateOccupancy(Eigen::Vector3d pos) {
  if (!isInMap(pos)) return -1;
  Eigen::Vector3i id;
  posToIndex(pos, id);
  return int(md_.occupancy_buffer_inflate_[toAddress(id)]);
}

inline int GridMap::getOccupancy(Eigen::Vector3i id) {
  if (id(0) < 0 || id(0) >= mp_.map_voxel_num_(0) || id(1) < 0 || id(1) >= mp_.map_voxel_num_(1) ||
      id(2) < 0 || id(2) >= mp_.map_voxel_num_(2))
    return -1;

  return md_.known_map_buffer_[toAddress(id)] >= mp_.p_occ_ ? 1 : 0;
}

inline bool GridMap::isInMap(const Eigen::Vector3d& pos) {
  if (pos(0) < mp_.map_min_boundary_(0) + 1e-4 || pos(1) < mp_.map_min_boundary_(1) + 1e-4 ||
      pos(2) < mp_.map_min_boundary_(2) + 1e-4) {
    // cout << "less than min range!" << endl;
    return false;
  }
  if (pos(0) > mp_.map_max_boundary_(0) - 1e-4 || pos(1) > mp_.map_max_boundary_(1) - 1e-4 ||
      pos(2) > mp_.map_max_boundary_(2) - 1e-4) {
    return false;
  }
  return true;
}

inline bool GridMap::isInMap(const Eigen::Vector3i& idx) {
  if (idx(0) < 0 || idx(1) < 0 || idx(2) < 0) {
    return false;
  }
  if (idx(0) > mp_.map_voxel_num_(0) - 1 || idx(1) > mp_.map_voxel_num_(1) - 1 ||
      idx(2) > mp_.map_voxel_num_(2) - 1) {
    return false;
  }
  return true;
}

inline void GridMap::posToIndex(const Eigen::Vector3d& pos, Eigen::Vector3i& id) {
  for (int i = 0; i < 3; ++i) id(i) = floor((pos(i) - mp_.map_origin_(i)) * mp_.resolution_inv_);
}

inline void GridMap::indexToPos(const Eigen::Vector3i& id, Eigen::Vector3d& pos) {
  for (int i = 0; i < 3; ++i) pos(i) = (id(i) + 0.5) * mp_.resolution_ + mp_.map_origin_(i);
}

inline void GridMap::inflatePoint(const Eigen::Vector3i& pt, int step, vector<Eigen::Vector3i>& pts) {
  int num = 0;
  for (int x = -step; x <= step; ++x)
    for (int y = -step; y <= step; ++y)
      for (int z = -step; z <= step; ++z) {
        pts[num++] = Eigen::Vector3i(pt(0) + x, pt(1) + y, pt(2) + z);
      }
}

inline double GridMap::getResolution() { return mp_.resolution_; }

#endif