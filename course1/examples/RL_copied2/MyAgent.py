import math
from typing import List
import numpy as np
import sys
from pathlib import Path
cur_path = str(Path(__file__).resolve().parent)
sys.path.append(cur_path)   # 解决相对路径
from baseAgent import * # 引入智能体
sys.path.remove(cur_path)   # 避免干扰

class MyAgent(BaseAgent):
    r"""智能体类，带有Optional的函数都是选择性实现，并不必要。
    当然你可以认为本类的其他函数也是不必要实现的

    Args:
        object (_type_): 基类object
    """

    def __init__(self,
                 tau: float = 0.1,
                 mass: float = 1,
                 max_step: int = 500,
                 max_capable: int = 1000,
                 energy_recover_rate: int = 200,
                 v_limits: float = 100.0,
                 f_limits: List[float] = [-100, 200],
                 theta_limits: List[float] = [-30, 30],
                 ) -> None:
        r"""类初始化函数

        Args:
            tau (float, optional): 时间步间隔. Defaults to 0.1.
            mass (float, optional): 质量. Defaults to 1.
            max_step (int, optional): 游戏最大时间步. Defaults to 500.
            max_capable (int, optional): 最大能量. Defaults to 1000.
            energy_recover_rate (int, optional): 能量恢复率. Defaults to 200.
            v_limits (float, optional): 最大速度. Defaults to 100.0.
            f_limits (List[float], optional): 驱动力的限制. Defaults to [-100,200].
            theta_limits (List[float], optional): 转角限制. Defaults to [-100,200].
        """
        super(BaseAgent, self).__init__()
        self.initFunc(
            tau=tau,
            mass=mass,
            max_step=max_step,
            max_capable=max_capable,
            v_limits=v_limits,
            energy_recover_rate=energy_recover_rate,
            f_limits=f_limits,
            theta_limits=theta_limits)

    def initFunc(self,
                 tau: float = 0.1,
                 mass: float = 1,
                 max_step: int = 500,
                 max_capable: int = 1000,
                 energy_recover_rate: int = 200,
                 v_limits: float = 100.0,
                 f_limits: List[float] = [-100, 200],
                 theta_limits: List[float] = [-30, 30],
                 ) -> None:
        r"""初始化函数的抽象，在__init__中被调用，方便重初始化，参数介绍请见__ini__"""
        self.max_step, self.v_limits = max_step, v_limits
        self.max_capable, self.init_capable = max_capable, max_capable
        self.mass = mass
        self.cur_step = 0
        self.cur_theta = 0
        self.cur_velocity = [0, 0]
        self.tau = 0.1                  # delta t 时间差
        self.gamma = 1                  # v衰减系数 速度衰减因子
        self.f_limits = f_limits        # 力的范围、加速减速范围
        self.theta_limits = theta_limits  # 角度旋转范围
        # 当前观察值
        # 用于碰撞后的速度/位置计算
        self.history_obs = []
        self.history_pred_v, self.history_pred_pos = [], []
        self.is_fatigue = False
        self.ang2rad, self.rad2ang = np.pi / 180, 180 / np.pi
        self.self_geo_center = (31.5, 19.5,)

        self.wzh_front_distance = 0  # 存储敌人的在视野内的纵坐标，用于判断敌人离自身的距离
        self.wzh_lose_start_wzh_time = 0  # 丢失敌人方位后，大幅度转向的次数
        self.wzh_is_last_pos_left = 0  # 敌人最后出现的位置是否在左侧
        self.wzh_is_last_pos_right = 0  # 敌人最后出现的位置是否在右侧
        self.wzh_time = 0  # 用于不断转向时的时间变量
        self.wzh_edge_choose = 0  # 正前方有边缘线时，根据边缘线的位置选择转弯的方向
        self.wzh_enemy_choose = 0  # 正前方有边缘线时，根据敌人的位置选择转弯的方向

    def act(self, obs: dict) -> List[float]:
        r"""执行计算的函数，包括信息预测(估算)、智能体决策过程与内部量计算
        根据观察值与内部信息决定策略，返回策略决定的动作，并估算修改内部信息

        Args:
            obs (dict): {'agent_obs':np.ndarray, 'id': str}

        Returns:
            List[float]: 动作列表 [f, theta]
        """

        # 事前预备
        self.cur_obs = obs['agent_obs']
        if self.cur_step == 0:
            agent_idx = int(obs['id'][-1])  # 获取智能体id，了解自己是第几个智能体
            self.getSelfAndOppElem(agent_idx)
        self.cur_step += 1

        ## 选做 -- 碰撞检测与能量修正
        ## FIXME: 通过观察值判断碰撞
        # is_collision_flag = self._isCollision()
        # if is_collision_flag: # 撞后可以通过后几个时隙修正，所以可以加一个cnt
        #     # FIXME: 通过观察值修正速度与能量以及位置，可持续多个回合
        #     self._fixCurVelocityPosEnergy()

        # 获取与特殊元素的距离
        obs_unique = list(np.unique(self.cur_obs))
        obs_special_elem, edge_flag, guide_flag, opp_flag = self._getInfoFromObs(
            obs_unique)
        flags = [edge_flag, guide_flag, opp_flag]
        distance_dict = self.calDisAndDir(obs_special_elem)

        # 执行算法计算，调用算法函数，获取驱动力与角度的值，算法的核心部分
        action_f, action_theta = self.algo(distance_dict=distance_dict,
                                           flags=flags)
        actions_list = [action_f, action_theta]     # 动作列表

        # 事后处理 -> 计算当前角度，估计当前速度(可选)、位置(可选)、能量(可选)
        if self.cur_step > self.max_step:
            print("game over? something goes wrong.")
        self._changeCurTheta(action_theta)
        ## 选做 -- 速度计算与位置计算
        # FIXME: 估计当前速度、位置、能量
        # self._calCurVelPosEgyAfterAccel(deepcopy(actions_list))
        return actions_list

    '''核心算法部分函数'''

    def algo(self,
             distance_dict: dict,
             flags: List[bool]) -> List[float]:
        r""" 算法核心，根据根据观察值与内部信息决定策略，返回策略决定的动作

        Returns:
            List[float]: 返回力量与角度动作值
        """
        action_f, action_theta = 0, 0

        if self.agent_idx == 0:                         # 情况/条件一
            action_f, action_theta = self.wzh_ALog()  # 执行对应算法
        else:                                       # 情况/条件二
            action_f, action_theta = self.wzh_ALog()   # 执行另外的算法
        # 限制在合法范围之内
        action_f = np.clip(action_f, *self.f_limits)
        action_theta = np.clip(action_theta, *self.theta_limits)
        return action_f, action_theta

    # 算法思路：通过不断检测边缘线来尽可能保证自己不出界的前提下，像一条疯狗一样一旦搜寻到敌人便冲上去咬死不松口，若丢失敌人位置则重新搜寻，不断重复以上过程。
    def wzh_ALog(self,) -> List[float]:
        action_f = 0  # 力的大小
        action_theta = 0  # 力的方向
        left = 0  # 记录左后方是否有边缘线
        right = 0  # 记录右后方是否有边缘线
        left_exist, right_exist, front_exist = 0, 0, 0  # 判断各个方位是否存在敌人，0代表未检测到，1代表检测到
        left_edge_exist, right_edge_exist, front_edge_exist, back_edge_exist = 0, 0, 0, 0  # 判断各个方位是否存在边缘线，0代表未检测到，1代表检测到
        non_zero_elem, edge_exist, guide_exist, opp_exist = self._getInfoFromObs(list(np.unique(self.cur_obs)))
        # 利用_getInfoFromObs函数检测视野内的特殊元素，包括是否检测到特殊元素、是否存在边缘线、是否存在中间引导线、是否存在敌人
        # 注：为避免代码过于复杂，non_zero_elem、guide_exist在本代码中未使用，可不考虑

        # 模块一：若检测到边缘线，我们需判断边缘线的方位
        if edge_exist:
            for i in range(0, 40):
                for j in range(40):
                    if self.cur_obs[i][j] == 1:   # 遍历视野中的代表边缘线的1，获取其坐标
                        if 5 < j and j < 20:
                            left_edge_exist = 1  # 若边缘线的坐标小于指定值，即此时边缘线在左侧距离过近，标记left_edge_exist为1，用于后续决策，下面三个变量同理
                        if 35 > j and j > 20:
                            right_edge_exist = 1  # 边缘线在右侧距离过近
                        if 15 < i < 32:
                            front_edge_exist = 1  # 边缘线在前侧距离过近
                        if 32 < i < 40 and 10 < j < 20:
                            left = 1  # 左后方有边缘线
                        elif 32 < i < 40 and 20 < j < 30:
                            right = 1  # 右后方有边缘线
                        if left and right:
                            back_edge_exist = 1  # 边缘线在后侧距离过近

        # 模块二：若检测到敌人，我们需判断敌人的方位
        if opp_exist:
            self.wzh_lose_start_wzh_time = 2
            for i in range(0, 40):
                for j in range(40):
                    if self.cur_obs[i][j] == self.opp_elem:  # 遍历视野中的代表敌人的8，获取其坐标
                        self.wzh_front_distance = i
                        if 0 < j and j < 20:
                            left_exist = 1  # 若敌人出现在视野左侧，标记left_exist为1，用于后续决策，下面两个变量同理
                            self.wzh_is_last_pos_left = 1  # 更新敌人最后出现的位置，用于丢失敌人位置后转向
                        if 40 > j and j > 20:
                            right_exist = 1  # 敌人出现在视野右侧
                            self.wzh_is_last_pos_right = 1  # 更新敌人最后出现的位置，用于丢失敌人位置后转向
                        if 0 < i < 32:
                            front_exist = 1  # 敌人出现在视野前侧

        # 模块三（核心函数）：若未检测到敌人，我们需调整视野尽快找到敌人
        if not opp_exist:
            self.wzh_time = self.wzh_time + 1  # 时间变量保持递增
            if self.wzh_time < 10:  # 在比赛开始的前十步，直线前进，抢占中心位置、尽快找到敌人，同时防止敌人开局的大力冲撞
                action_f = 110
                action_theta = 0
            elif self.wzh_time == 10:  # 在第十步向右调整视野，保证后续左右调整时视野的对称
                action_f = 20
                action_theta = 13
            elif self.wzh_time < 30:  # 利用时间变量不断左右切换视野，增大视野范围，用于开局第一次搜寻敌人，以防敌人贴着边缘线从侧边绕开
                if self.wzh_time % 2 == 0:
                    action_theta = 26
                else:
                    action_theta = -26
                action_f = 50
            else:  # 前面三个if情景判断主要用于开局的布置，后续不再使用；而这一情景通常用于后续在追击敌人过程中丢失视野的应对方法
                if self.wzh_is_last_pos_left and not self.wzh_is_last_pos_right:  # 若在丢失敌人方位的最后一刻敌人在左侧而非右侧，则立刻将视野向左侧调整
                    if self.wzh_time % 2 == 0:  # 若大幅度转向后仍未能找到敌人，则改为在小范围左右摇摆来搜寻
                        action_theta = -10
                    else:
                        action_theta = 10
                    action_f = 40
                    if self.wzh_lose_start_wzh_time:  # 丢失敌人方位后的前两步（self.wzh_lose_start_wzh_time在下一次找到敌人后会被重新置为0）进行大幅度转向左侧，以求能迅速找到敌人
                        action_theta = -30
                        self.wzh_lose_start_wzh_time = self.wzh_lose_start_wzh_time - 1
                        action_f = 100
                elif self.wzh_is_last_pos_right and not self.wzh_is_last_pos_left:  # 若在丢失敌人方位的最后一刻敌人在右侧而非左侧，则立刻将视野向右侧调整，其余部分与上同理
                    if self.wzh_time % 2 == 0:
                        action_theta = 10
                    else:
                        action_theta = -10
                    action_f = 40
                    if self.wzh_lose_start_wzh_time:
                        action_theta = 30
                        self.wzh_lose_start_wzh_time = self.wzh_lose_start_wzh_time - 1
                        action_f = 100
                else:  # 若在丢失敌人方位的最后一刻敌人在前侧，则保持中等速度前进；后侧不做讨论，实际测试中发现加入后侧的代码容易造成执行混论，故删去
                    action_theta = 0
                    action_f = 60

                # 视野丢失后，极易被偷袭，且由于缺少敌人的关键信息，难以做出合理判断，极容易自己撞上边缘线，因此选择转攻为守，当检测到边缘线靠近自身时，调整力的大小和方向，尽快远离边缘线，直到重新找到敌人
                if left_edge_exist and not right_edge_exist:  # 左前方有边缘线，迅速右转
                    action_f = 130
                    action_theta = 30
                    self.wzh_edge_choose = 1
                elif not left_edge_exist and right_edge_exist:  # 右前方有边缘线，迅速左转
                    action_f = 130
                    action_theta = -30
                    self.wzh_edge_choose = 2
                elif back_edge_exist:  # 边缘线在正后方，加速直行
                    action_f = 190
                    action_theta = 0
                elif left_edge_exist and front_edge_exist and right_edge_exist:  # 边缘线在正前方时，根据之前记录的边缘线位置及敌人位置的信息选择转向方向（实际测试中发现后退并不可行，一方面很难刹车，在发生了碰撞后更是如此；另一方面，刹车过程中速度很小，此时若遭到二次碰撞，十分危险，综合考虑选择迅速调头，调头方向依据此前记录的敌人位置和边缘线位置两个信息）
                    action_f = 170
                    if not self.wzh_edge_choose:  # 在比赛刚开始时，由于没有边缘线的位置的记录，暂时先利用敌人的位置进行转向
                        self.wzh_edge_choose = self.wzh_enemy_choose

                    if self.wzh_edge_choose == 1:  # 根据之前记录的边缘线位置进行转向
                        action_theta = 30
                    elif self.wzh_edge_choose == 2:
                        action_theta = -30
                    else:  # 无记录时默认右转调头
                        action_theta = 30
                        action_f = 170

        # 模块四：成功锁定敌人后，如何追击和进攻
        if opp_exist:
            self.wzh_is_last_pos_left = 0
            self.wzh_is_last_pos_right = 0
            if left_exist and front_exist and right_exist:  # 若敌人出现在正前方，则直线前进
                if self.wzh_front_distance < 15:  # self.wzh_front_distance代表敌人与自身的距离，梯度加速，若距离过近，则加强力量，硬碰硬，距离较远时，则缓慢加速，以确保前进方向正确，防止被甩开，且节省体力
                    action_f = 50
                elif self.wzh_front_distance < 20:
                    action_f = 100
                elif self.wzh_front_distance < 25:
                    action_f = 120
                elif self.wzh_front_distance < 29:
                    action_f = 170
                else:
                    action_f = 200
                action_theta = 0
            if left_exist and front_exist and not right_exist:  # 此时敌人在左前方，向左前方追击
                if self.wzh_front_distance < 15:  # 梯度加速
                    action_f = 60
                elif self.wzh_front_distance < 25:
                    action_f = 120
                elif self.wzh_front_distance < 32:
                    action_f = 165
                action_theta = -30
                self.wzh_is_last_pos_right = 0
                self.wzh_enemy_choose = 2  # 记录敌人位置，用于正前方有边缘线时的转向
            if not left_exist and front_exist and right_exist:  # 此时敌人在右前方，向右前方追击
                if self.wzh_front_distance < 15:  # 梯度加速
                    action_f = 60
                elif self.wzh_front_distance < 25:
                    action_f = 120
                elif self.wzh_front_distance < 32:
                    action_f = 165
                action_theta = 30
                self.wzh_is_last_pos_left = 0
                self.wzh_enemy_choose = 1  # 记录敌人位置，用于正前方有边缘线时的转向

        # 模块五：若在追击敌人的过程中发现自己在边缘线附近，优先躲避边缘线
        if left_edge_exist and not right_edge_exist:  # 边缘线在左前方，向右转
            action_f = 130
            action_theta = 30
            self.wzh_edge_choose = 1  # 记录边缘线位置，用于正前方有边缘线时的转向
        elif not left_edge_exist and right_edge_exist:  # 边缘线在右前方，向左转
            action_f = 130
            action_theta = -30
            self.wzh_edge_choose = 2  # 记录边缘线位置，用于正前方有边缘线时的转向
        elif back_edge_exist:  # 边缘线在正后方，加速直行
            action_f = 190
            action_theta = 0
        elif left_edge_exist and front_edge_exist and right_edge_exist:  # 边缘线在正前方时，根据之前记录的边缘线位置及敌人位置的信息选择转向方向
            action_f = 170
            if not self.wzh_edge_choose:  # 在比赛刚开始时，由于没有边缘线的位置的记录，暂时先利用敌人的位置进行转向
                self.wzh_edge_choose = self.wzh_enemy_choose

            if self.wzh_edge_choose == 1:  # 根据之前记录的边缘线位置进行转向
                action_theta = 30
            elif self.wzh_edge_choose == 2:
                action_theta = -30
            else:  # 无记录时默认右转调头
                action_theta = 30
                action_f = 170

        return action_f, action_theta

    def getSelfAndOppElem(self, agent_idx: int) -> None:
        r"""根据id和观察值获取自身代表元素与对手代表元素

        Args:
            agent_idx (int):   自身id
            obs_unique (list): 初始观察值包含的元素值
        """
        self.agent_idx = agent_idx
        '''
        self.agent_idx = 0 -> self.self_elem, self.opp_elem = 10, 8
        self.agent_idx = 1 -> self.self_elem, self.opp_elem = 8, 10
        '''
        self.self_elem = int(self.cur_obs[31, 20, ])   # 由几何中心自身元素
        # print(self.self_elem, 10 - 2 * self.agent_idx, np.unique(self.cur_obs))
        self.opp_elem = 18 - self.self_elem           # 对手元素
        # self.opp_elem = 8 if self.self_elem > 9 else 10


    def _getInfoFromObs(self, obs_unique: List[int]):
        r"""获知含有哪些元素，并返回特殊元素

        Args:
            obs_unique (List[int]): 视野矩阵中出现过的元素

        Returns:
            remove_self (List[int]): 特殊元素集合
            edge_exist, guide_exist, opp_exist (bool): 特殊元素标志位
        """
        non_zero_elem = obs_unique[1:]
        # non_zero_elem.remove(self.self_elem)#报错？
        edge_exist = 1 in non_zero_elem
        guide_exist = 4 in non_zero_elem
        opp_exist = self.opp_elem in non_zero_elem
        return non_zero_elem, edge_exist, guide_exist, opp_exist

    def calElemDisAndDirs(self, elem: int = 1,) -> List[float]:
        """ 距离计算函数

        Args:
            elem (int, optional): 元素. Defaults to 1.

        Returns:
            List[float]: 返回智能体几何中心与特殊元素的距离列表
        """
        dis_dir_list = []
        find_elem_pos1 = np.argwhere(self.cur_obs == elem)
        # FIXME: 你也可以修改为特殊元素之间的距离，让你的智能体获取更多的信息，
        # 令其更强大
        val2 = self.self_geo_center  # 几何中心
        for val1 in find_elem_pos1:
            L2_distance = np.linalg.norm(val1-val2) - 3.5
            direction = [self.rad2ang * math.atan2(
                (val2[1] - val1[1]), (val2[0] - val1[0]+1e-18))]
            dis_dir_list.append((val1, L2_distance, direction))

        return dis_dir_list

    def calDisAndDir(self, elems: List[int]) -> dict:
        r'''
        caluate the distances and directions to the elem
        return the distances between element object and
        agent, and also return it with this format
        [[(pos1),dis,dir],[(pos2),dis,dir]].
        elem:
            Edge        -> 1
            guideline   -> 4
            my_ai       -> 10/8 (agent_idx = 0/1)
            opponent    -> 8/10
        '''
        rt_val_dict = dict()
        for elem in elems:
            rt_val_dict[elem] = self.calElemDisAndDirs(elem=elem,)
        return rt_val_dict


    '''事后计算部分函数'''

    def _changeCurTheta(self, action_theta: float) -> None:
        r""" 改变当前智能体朝向角度，在加速之前(Optional函数)/act返回前调用

        Args:
            action_theta (float): 角度动作
        """
        self.cur_theta = (self.cur_theta + action_theta) % 360

    '''选做'''

    # def _fixCurVelocityPosEnergy(self, ) -> None:
    #     r"""(Optional) 计算/估算智能体当前速度

    #     Args:
    #         new_action (List[float]): _description_
    #     """
    #     # TODO： 此时函数被调用来修正碰撞后的速度
    #     # 可以通过多帧的观察值变化或智能体位置预测，
    #     pass

    # def _speed_limit(self):
    #     r"""(Optional)限制最大速度，copy from olympics_engine/core.py OlympicsBase
    #     有一个速度墙，撞速度墙会给各分量等量缩放至最大速度，但是施加的力量不返还(流氓！)
    #     """
    #     current_v = self.cur_velocity
    #     current_speed = math.sqrt(current_v[0]**2 + current_v[1]**2)

    #     if current_speed > self.v_limits:
    #         factor = self.v_limits/current_speed
    #         cap_v = [current_v[0] * factor, current_v[1] * factor]
    #         self.cur_velocity = cap_v

    # def _calCurVelPosEgyAfterAccel(self, new_action: List[float]) -> None:
    #     r"""(Optional) 根据动作计算/估算智能体当前速度、位置与能量

    #     Args:
    #         new_action (List[float]): _description_
    #     """
    #     # FIXME: 估算函数，用于计算智能体当前速度，没有考虑碰撞后的速度改变。但建议
    #     action = new_action
    #     last_vel = deepcopy(self.cur_velocity)
    #     if (action is None) or self.is_fatigue:
    #         accel = [0, 0]
    #     else:
    #         mass = self.mass
    #         force, theta = action[0] / mass, action[1]
    #         theta_old = self.cur_theta
    #         theta_new = theta_old + theta
    #         self.cur_theta = theta_new
    #         the2ang = theta_new * self.ang2rad
    #         accel_x = force * np.cos(the2ang)
    #         accel_y = force * np.sin(the2ang)
    #     accel = np.array([accel_x, accel_y])
    #     self.cur_velocity = self.gamma * self.cur_velocity + \
    #         accel * self.tau  # update v with acceleration
    #     self._speed_limit()
    #     # FIXME: 根据速度改变位置与能量
    #     # do somethings
    #     # 位置参考 olympics_engine/core.py stepPhysics
    #     # 能量参考 olympics_engine/core.py change_inner_state
    #     pass

    # def _isCollision(self,) -> bool:
    #     r"""(Optional) 根据观察值与内部信息(估计的速度/位置)估计是否发生了碰撞
    #     """
    #     collision_flag = False
    #     # TODO: 完成检测算法
    #     # hint: 根据先前位置速度预测下一时隙应该观测到的值，如果预测值与当前观测值不一样，那么发生了碰撞
    #     # do something to detect it. By using
    #     # self.history_pred_pos, self.history_obs,
    #     # self.history_pred_v, self.cur_obs
    #     return collision_flag
