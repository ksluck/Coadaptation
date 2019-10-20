from .robot_bases import XmlBasedRobot, MJCFBasedRobot, URDFBasedRobot
import numpy as np
import pybullet
import os, inspect
import pybullet_data
from .robot_bases import BodyPart
import tempfile
import atexit
import xmltodict

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def cleanup_func_for_tmp(filepath):
    os.remove(filepath)

class WalkerBase(MJCFBasedRobot):
  def __init__(self,  fn, robot_name, action_dim, obs_dim, power, self_collision=True):
    MJCFBasedRobot.__init__(self, fn, robot_name, action_dim, obs_dim, self_collision=True)
    self.power = power
    self.camera_x = 0
    self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
    self.walk_target_x = 1e3  # kilometer away
    self.walk_target_y = 0
    self.body_xyz=[0,0,0]

  def robot_specific_reset(self, bullet_client):
    self._p = bullet_client
    self._reset_position = []
    for j in self.ordered_joints:
      pos = self.np_random.uniform(low=-0.5, high=0.5)
      j.reset_current_position(pos, 0)
      self._reset_position.append(pos)

    self.feet = [self.parts[f] for f in self.foot_list]
    self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)
    self.scene.actor_introduce(self)
    self.initial_z = None

  def reset_position(self):
    for j, pos in zip(self.ordered_joints, self._reset_position):
      j.set_velocity(0)
      # j.disable_motor()

  def reset_position_final(self):
    for j, pos in zip(self.ordered_joints, self._reset_position):
      # j.reset_current_position(pos, 0)
      j.disable_motor()


  def apply_action(self, a):
    assert (np.isfinite(a).all())
    for n, j in enumerate(self.ordered_joints):
      j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))

  def calc_state(self):
    j = np.array([j.current_relative_position() for j in self.ordered_joints], dtype=np.float32).flatten()
    # even elements [0::2] position, scaled to -1..+1 between limits
    # odd elements  [1::2] angular speed, scaled to show -1..+1
    self.joint_speeds = j[1::2]
    self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

    body_pose = self.robot_body.pose()
    parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
    self.body_xyz = (
    parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2])  # torso z is more informative than mean z
    self.body_rpy = body_pose.rpy()
    z = self.body_xyz[2]
    if self.initial_z == None:
      self.initial_z = z
    r, p, yaw = self.body_rpy
    self.walk_target_theta = np.arctan2(self.walk_target_y - self.body_xyz[1],
                      self.walk_target_x - self.body_xyz[0])
    self.walk_target_dist = np.linalg.norm(
      [self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0]])
    angle_to_target = self.walk_target_theta - yaw

    rot_speed = np.array(
      [[np.cos(-yaw), -np.sin(-yaw), 0],
       [np.sin(-yaw), np.cos(-yaw), 0],
       [    0,       0, 1]]
    )
    vx, vy, vz = np.dot(rot_speed, self.robot_body.speed())  # rotate speed back to body point of view

    more = np.array([ z-self.initial_z,
      np.sin(angle_to_target), np.cos(angle_to_target),
      0.3* vx , 0.3* vy , 0.3* vz ,  # 0.3 is just scaling typical speed into -1..+1, no physical sense here
      r, p], dtype=np.float32)
    return np.clip( np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)

  def calc_potential(self):
    # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
    # all rewards have rew/frame units and close to 1.0
    debugmode=0
    if (debugmode):
      print("calc_potential: self.walk_target_dist")
      print(self.walk_target_dist)
      print("self.scene.dt")
      print(self.scene.dt)
      print("self.scene.frame_skip")
      print(self.scene.frame_skip)
      print("self.scene.timestep")
      print(self.scene.timestep)
    return - self.walk_target_dist / self.scene.dt

class HalfCheetah(WalkerBase):
  foot_list = ["ffoot", "fshin", "fthigh",  "bfoot", "bshin", "bthigh"]  # track these contacts with ground

  def __init__(self, design = None):
    self._adapted_xml_file = tempfile.NamedTemporaryFile(delete=False, prefix='halfcheetah_', suffix='.xml')
    self._adapted_xml_filepath = self._adapted_xml_file.name
    file = self._adapted_xml_filepath
    self._adapted_xml_file.close()
    atexit.register(cleanup_func_for_tmp, self._adapted_xml_filepath)
    self.adapt_xml(self._adapted_xml_filepath, design)
    #WalkerBase.__init__(self, self._adapted_xml_filepath, "torso", action_dim=6, obs_dim=26, power=0.90)
    WalkerBase.__init__(self, file, "torso", action_dim=6, obs_dim=26, power=0.90)

  def adapt_xml(self, file, design = None):
      with open(os.path.join(currentdir, 'half_cheetah.xml'), 'r') as fd:
            xml_string = fd.read()
      height = 1.0
      bth_r = 1.0
      bsh_r = 1.0
      bfo_r = 1.0
      fth_r = 1.0
      fsh_r = 1.0
      ffo_r = 1.0
      if design is None:
        bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r = np.random.uniform(low=0.5, high=1.5, size=6)
      else:
        bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r = design
      height = max(.145 * bth_r + .15 * bsh_r + .094 * bfo_r, .133 * fth_r + .106 * fsh_r +  .07 * ffo_r)
      height *= 2.0 + 0.01

      xml_dict = xmltodict.parse(xml_string)
      xml_dict['mujoco']['worldbody']['body']['@pos'] = "0 0 {}".format(height)

      ### Btigh

      xml_dict['mujoco']['worldbody']['body']['body'][0]['geom']['@pos'] = '.1 0 -.13'
      xml_dict['mujoco']['worldbody']['body']['body'][0]['geom']['@pos'] = '{} 0 {}'.format(.1 * bth_r, -.13 * bth_r)

      xml_dict['mujoco']['worldbody']['body']['body'][0]['geom']['@size'] = '0.046 .145'
      xml_dict['mujoco']['worldbody']['body']['body'][0]['geom']['@size'] = '0.046 {}'.format(.145 * bth_r)

      xml_dict['mujoco']['worldbody']['body']['body'][0]['body']['@pos'] = '.16 0 -.25'
      xml_dict['mujoco']['worldbody']['body']['body'][0]['body']['@pos'] = '{} 0 {}'.format(.16 * bth_r, -.25 * bth_r)

      ### bshin

      xml_dict['mujoco']['worldbody']['body']['body'][0]['body']['geom']['@pos'] = '-.14 0 -.07'
      xml_dict['mujoco']['worldbody']['body']['body'][0]['body']['geom']['@pos'] = '{} 0 {}'.format(-.14 * bsh_r, -.07 * bsh_r)

      xml_dict['mujoco']['worldbody']['body']['body'][0]['body']['geom']['@size'] = '0.046 .15'
      xml_dict['mujoco']['worldbody']['body']['body'][0]['body']['geom']['@size'] = '0.046 {}'.format(.15 * bsh_r)

      xml_dict['mujoco']['worldbody']['body']['body'][0]['body']['body']['@pos'] = '-.28 0 -.14'
      xml_dict['mujoco']['worldbody']['body']['body'][0]['body']['body']['@pos'] = '{} 0 {}'.format(-.28 * bsh_r, -.14 * bsh_r)

      ### bfoot

      xml_dict['mujoco']['worldbody']['body']['body'][0]['body']['body']['geom']['@pos'] = '.03 0 -.097'
      xml_dict['mujoco']['worldbody']['body']['body'][0]['body']['body']['geom']['@pos'] = '{} 0 {}'.format(.03 * bfo_r, -.097 * bfo_r)

      xml_dict['mujoco']['worldbody']['body']['body'][0]['body']['body']['geom']['@size'] = '0.046 .094'
      xml_dict['mujoco']['worldbody']['body']['body'][0]['body']['body']['geom']['@size'] = '0.046 {}'.format(.094 * bfo_r)

      ### fthigh

      xml_dict['mujoco']['worldbody']['body']['body'][1]['geom']['@pos'] = '-.07 0 -.12'
      xml_dict['mujoco']['worldbody']['body']['body'][1]['geom']['@pos'] = '{} 0 {}'.format(-.07 * fth_r, -.12 * fth_r)

      xml_dict['mujoco']['worldbody']['body']['body'][1]['geom']['@size'] = '0.046 .133'
      xml_dict['mujoco']['worldbody']['body']['body'][1]['geom']['@size'] = '0.046 {}'.format(.133 * fth_r)

      xml_dict['mujoco']['worldbody']['body']['body'][1]['body']['@pos'] = '-.14 0 -.24'
      xml_dict['mujoco']['worldbody']['body']['body'][1]['body']['@pos'] = '{} 0 {}'.format(-.14 *fth_r, -.24 * fth_r)

      ### fshin

      xml_dict['mujoco']['worldbody']['body']['body'][1]['body']['geom']['@pos'] = '.065 0 -.09'
      xml_dict['mujoco']['worldbody']['body']['body'][1]['body']['geom']['@pos'] = '{} 0 {}'.format(.065 * fsh_r, -.09 * fsh_r)

      xml_dict['mujoco']['worldbody']['body']['body'][1]['body']['geom']['@size'] = '0.046 .106'
      xml_dict['mujoco']['worldbody']['body']['body'][1]['body']['geom']['@size'] = '0.046 {}'.format(.106 * fsh_r)

      xml_dict['mujoco']['worldbody']['body']['body'][1]['body']['body']['@pos'] = '.13 0 -.18'
      xml_dict['mujoco']['worldbody']['body']['body'][1]['body']['body']['@pos'] = '{} 0 {}'.format(.13 * fsh_r, -.18 * fsh_r)

      ### ffoot

      xml_dict['mujoco']['worldbody']['body']['body'][1]['body']['body']['geom']['@pos'] = '.045 0 -.07'
      xml_dict['mujoco']['worldbody']['body']['body'][1]['body']['body']['geom']['@pos'] = '{} 0 {}'.format(.045 * ffo_r, -.07 * ffo_r)

      xml_dict['mujoco']['worldbody']['body']['body'][1]['body']['body']['geom']['@size'] = '0.046 .07'
      xml_dict['mujoco']['worldbody']['body']['body'][1]['body']['body']['geom']['@size'] = '0.046 {}'.format(.07 * ffo_r)

      xml_string = xmltodict.unparse(xml_dict, pretty=True)
      with open(file, 'w') as fd:
          fd.write(xml_string)

  def alive_bonus(self, z, pitch):
    # Use contact other than feet to terminate episode: due to a lot of strange walks using knees
    return 0 #+1 if np.abs(pitch) < 1.0 and not self.feet_contact[1] and not self.feet_contact[2] and not self.feet_contact[4] and not self.feet_contact[5] else -1

  def robot_specific_reset(self, bullet_client):
    WalkerBase.robot_specific_reset(self, bullet_client)
    self.jdict["bthigh"].power_coef = 120.0
    self.jdict["bshin"].power_coef  = 90.0
    self.jdict["bfoot"].power_coef  = 60.0
    self.jdict["fthigh"].power_coef = 90.0 #140?
    self.jdict["fshin"].power_coef  = 60.0
    self.jdict["ffoot"].power_coef  = 30.0
    body_pose = self.robot_body.pose()
    x, y, z = body_pose.xyz()
    self._initial_z = z

  def reset_design(self, bullet_client, design):
    self._adapted_xml_file = tempfile.NamedTemporaryFile(delete=False, prefix='halfcheetah_', suffix='.xml')
    self._adapted_xml_filepath = self._adapted_xml_file.name
    file = self._adapted_xml_filepath
    self._adapted_xml_file.close()
    atexit.register(cleanup_func_for_tmp, self._adapted_xml_filepath)

    self.adapt_xml(file, design)
    self.model_xml = file

    self.doneLoading = 0
    self.parts = None
    self.objects = []
    self.jdict = None
    self.ordered_joints = None
    self.robot_body = None
    self.robot_name = "torso"
    bullet_client.resetSimulation()
    self.reset(bullet_client)
    # self.reset(self._p)

  def calc_potential(self):
    return 0

  def calc_state(self):
    j = np.array([j.current_relative_position() for j in self.ordered_joints], dtype=np.float32).flatten()
    # even elements [0::2] position, scaled to -1..+1 between limits
    # odd elements  [1::2] angular speed, scaled to show -1..+1
    self.joint_pos = j[0::2]
    self.joint_speeds = j[1::2]
    body_pose = self.robot_body.pose()
    x, y, z = body_pose.xyz()
    z = z - self._initial_z
    r, p, yaw = self.body_rpy = body_pose.rpy()
    xv, yv, zv = self.robot_body.speed()
    vr, vp, vy = self.robot_body.speed_angular()
    state = np.array([xv, vp, zv, p, z])
    state = np.append(self.joint_speeds, state)
    state = np.append(self.joint_pos, state)
    return state
