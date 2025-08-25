
[safe-mbrl/utils/env_utils.py](https://github.com/liuzuxin/safe-mbrl/blob/bd18190f1995ae5122e054792ec46d967791100f/utils/env_utils.py)

We use the relative coordinates of the perceived objective instead of the pseudo-lidar.

논문에서 observation을 pseudo-lidar에서 상대 좌표계로 수정해서 사용했다고 했는데 이에 해당하는 코드

```python title:get_obs()
def get_obs(self):
	'''
	We will ingnore the z-axis coordinates in every poses.
	The returned obs coordinates are all in the robot coordinates.
	'''
	obs = {}
	robot_pos = self.env.robot_pos
	goal_pos = self.env.goal_pos
	vases_pos_list = self.env.vases_pos # list of shape (3,) ndarray
	hazards_pos_list = self.env.hazards_pos # list of shape (3,) ndarray
	gremlins_pos_list = self.env.gremlins_obj_pos # list of shape (3,) ndarray
	buttons_pos_list = self.env.buttons_pos # list of shape (3,) ndarray

	ego_goal_pos = self.recenter(goal_pos[:2])
	ego_vases_pos_list = [self.env.ego_xy(pos[:2]) for pos in vases_pos_list] # list of shape (2,) ndarray
	ego_hazards_pos_list = [self.env.ego_xy(pos[:2]) for pos in hazards_pos_list] # list of shape (2,) ndarray
	ego_gremlins_pos_list = [self.env.ego_xy(pos[:2]) for pos in gremlins_pos_list] # list of shape (2,) ndarray
	ego_buttons_pos_list = [self.env.ego_xy(pos[:2]) for pos in buttons_pos_list] # list of shape (2,) ndarray
	
	# append obs to the dict
	for sensor in self.xyz_sensors:  # Explicitly listed sensors
		if sensor=='accelerometer':
			obs[sensor] = self.env.world.get_sensor(sensor)[:1] # only x axis matters
		elif sensor=='ballquat_rear':
			obs[sensor] = self.env.world.get_sensor(sensor)
		else:
			obs[sensor] = self.env.world.get_sensor(sensor)[:2] # only x,y axis matters

	for sensor in self.angle_sensors:
		if sensor == 'gyro':
			obs[sensor] = self.env.world.get_sensor(sensor)[2:] #[2:] # only z axis matters
			#pass # gyro does not help
		else:
			obs[sensor] = self.env.world.get_sensor(sensor)

	obs["vases"] = np.array(ego_vases_pos_list) # (vase_num, 2)
	obs["hazards"] = np.array(ego_hazards_pos_list) # (hazard_num, 2)
	obs["goal"] = ego_goal_pos # (2,)
	obs["gremlins"] = np.array(ego_gremlins_pos_list) # (vase_num, 2)
	obs["buttons"] = np.array(ego_buttons_pos_list) # (hazard_num, 2)
	return obs
```
