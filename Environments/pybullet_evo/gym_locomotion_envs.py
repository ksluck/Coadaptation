from pybullet_envs.scene_stadium import SinglePlayerStadiumScene
from pybullet_envs.env_bases import MJCFBaseBulletEnv
import numpy as np
import pybullet
from .robot_locomotors import  HalfCheetah
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
from gym import spaces

class WalkerBaseBulletEnv(MJCFBaseBulletEnv):
  def __init__(self, robot, render=False):
    # print("WalkerBase::__init__ start")
    MJCFBaseBulletEnv.__init__(self, robot, render)

    self.camera_x = 0
    self.walk_target_x = 1e3  # kilometer away
    self.walk_target_y = 0
    self.stateId=-1
    self._projectM = None
    self._param_init_camera_width = 320
    self._param_init_camera_height = 200
    self._param_camera_distance = 2.0


  def create_single_player_scene(self, bullet_client):
    self.stadium_scene = SinglePlayerStadiumScene(bullet_client, gravity=9.8, timestep=0.0165/4, frame_skip=4)
    return self.stadium_scene

  def reset(self):
    if (self.stateId>=0):
      # print("restoreState self.stateId:",self.stateId)
      self._p.restoreState(self.stateId)

    r = MJCFBaseBulletEnv.reset(self)
    self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING,0)

    self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(self._p,
      self.stadium_scene.ground_plane_mjcf)
    # self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex], self.parts[f].bodyPartIndex) for f in
    #              self.foot_ground_object_names])
    self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING,1)
    if (self.stateId<0):
      self.stateId=self._p.saveState()
      #print("saving state self.stateId:",self.stateId)
    self._p.setGravity(0,0,-.5)
    for _ in range(200):
      self.robot.reset_position()
      self.scene.global_step()
    self.robot.reset_position_final()
    self._p.setGravity(0,0,-9.81)
    r = self.robot.calc_state()
    self.robot._initial_z = r[-1]
    # for _ in range(20):
    #   self.scene.global_step()
    #   self.robot.reset_position_final()
    #   time.sleep(0.1)
    # self.robot.reset_position()
    # self.scene.global_step()
    self.robot.initial_z = None
    r = self.robot.calc_state()

    return r

  def _isDone(self):
    return self._alive < 0

  def move_robot(self, init_x, init_y, init_z):
    "Used by multiplayer stadium to move sideways, to another running lane."
    self.cpp_robot.query_position()
    pose = self.cpp_robot.root_part.pose()
    pose.move_xyz(init_x, init_y, init_z)  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
    self.cpp_robot.set_pose(pose)

  electricity_cost   = -2.0  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
  stall_torque_cost  = -0.1  # cost for running electric current through a motor even at zero rotational speed, small
  foot_collision_cost  = -1.0  # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
  foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
  joints_at_limit_cost = -0.1  # discourage stuck joints

  def step(self, a):
    if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
      self.robot.apply_action(a)
      self.scene.global_step()

    state = self.robot.calc_state()  # also calculates self.joints_at_limit

    self._alive = float(self.robot.alive_bonus(state[0]+self.robot.initial_z, self.robot.body_rpy[1]))   # state[0] is body height above ground, body_rpy[1] is pitch
    done = self._isDone()
    if not np.isfinite(state).all():
      print("~INF~", state)
      done = True

    potential_old = self.potential
    self.potential = self.robot.calc_potential()
    progress = float(self.potential - potential_old)

    feet_collision_cost = 0.0
    for i,f in enumerate(self.robot.feet): # TODO: Maybe calculating feet contacts could be done within the robot code
      contact_ids = set((x[2], x[4]) for x in f.contact_list())
      #print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
      if (self.ground_ids & contact_ids):
                          #see Issue 63: https://github.com/openai/roboschool/issues/63
        #feet_collision_cost += self.foot_collision_cost
        self.robot.feet_contact[i] = 1.0
      else:
        self.robot.feet_contact[i] = 0.0


    electricity_cost  = self.electricity_cost  * float(np.abs(a*self.robot.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
    electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

    joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
    debugmode=0
    if(debugmode):
      print("alive=")
      print(self._alive)
      print("progress")
      print(progress)
      print("electricity_cost")
      print(electricity_cost)
      print("joints_at_limit_cost")
      print(joints_at_limit_cost)
      print("feet_collision_cost")
      print(feet_collision_cost)

    self.rewards = [
      self._alive,
      progress,
      electricity_cost,
      joints_at_limit_cost,
      feet_collision_cost
      ]
    if (debugmode):
      print("rewards=")
      print(self.rewards)
      print("sum rewards")
      print(sum(self.rewards))
    self.HUD(state, a, done)
    self.reward += sum(self.rewards)

    return state, sum(self.rewards), bool(done), {}

  # def camera_adjust(self):
  #   x, y, z = self.robot.body_xyz
  #   self.camera._p = self._p
  #   self.camera_x = 0.98*self.camera_x + (1-0.98)*x
  #   self.camera.move_and_look_at(self.camera_x, y-1.0, 1.4, x, y, 1.0)
  def camera_adjust(self):
        if self._p is None :
            return
        self.camera._p = self._p
        x, y, z = self.robot.body_xyz
        if self.camera_x is not None:
            self.camera_x = x # 0.98*self.camera_x + (1-0.98)*x
        else:
            self.camera_x = x
        # self.camera.move_and_look_at(self.camera_x, y-2.0, 1.4, x, y, 1.0)
        lookat = [self.camera_x, y, z]
        distance = self._param_camera_distance
        yaw = 10
        self._p.resetDebugVisualizerCamera(distance, yaw, -20, lookat)

  def render_camera_image(self, pixelWidthHeight = None):
    if pixelWidthHeight is not None or self._projectM is None:
        if self._projectM is None:
            self._pixelWidth = self._param_init_camera_width
            self._pixelHeight = self._param_init_camera_height
        else:
            self._pixelWidth = pixelWidthHeight[0]
            self._pixelHeight = pixelWidthHeight[1]
        nearPlane = 0.01
        farPlane = 10
        aspect = self._pixelWidth / self._pixelHeight
        fov = 60
        self._projectM = self._p.computeProjectionMatrixFOV(fov, aspect,
            nearPlane, farPlane)

    x, y, z = self.robot.robot_body.pose().xyz()
    lookat = [x, y, 0.5]
    distance = 2.0
    yaw = -20
    viewM = self._p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=lookat,
        distance=distance,
        yaw=10.,
        pitch=yaw,
        roll=0.0,
        upAxisIndex=2)

    # img_arr = pybullet.getCameraImage(self._pixelWidth, self._pixelHeight, viewM, self._projectM, shadow=1,lightDirection=[1,1,1],renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    img_arr = pybullet.getCameraImage(self._pixelWidth, self._pixelHeight, viewM, self._projectM, shadow=False, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL, flags=pybullet.ER_NO_SEGMENTATION_MASK)

    w=img_arr[0] #width of the image, in pixels
    h=img_arr[1] #height of the image, in pixels
    rgb=img_arr[2] #color data RGB

    image = np.reshape(rgb, (h, w, 4)) #Red, Green, Blue, Alpha
    image = image * (1./255.)
    image = image[:,:,0:3]
    return image

class HalfCheetahBulletEnv(WalkerBaseBulletEnv):
  def __init__(self, render=False, design = None):
    self.robot = HalfCheetah(design)
    WalkerBaseBulletEnv.__init__(self, self.robot, render)
    self.observation_space = spaces.Box(-np.inf, np.inf, shape=[17], dtype=np.float32)

  def _isDone(self):
    return False

  def disconnect(self):
      self._p.disconnect()

  def step(self, a):
    if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
      self.robot.apply_action(a)
      self.scene.global_step()

    state = self.robot.calc_state()  # also calculates self.joints_at_limit

    done = self._isDone()
    if not np.isfinite(state).all():
      print("~INF~", state)
      done = 0
    reward = max(state[-5]/10.0, 0.0)

    return state, reward, bool(done), {}

  def reset_design(self, design):
      self.stateId = -1
      self.scene = None
      self.robot.reset_design(self._p, design)
