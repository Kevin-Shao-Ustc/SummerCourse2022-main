import sys
from pathlib import Path
import numpy as np
cur_path = str(Path(__file__).resolve().parent)
sys.path.append(cur_path)   # 解决相对路径
from MyAgent import MyAgent # 引入智能体
sys.path.remove(cur_path)   # 避免干扰
del cur_path


agent = MyAgent()


def my_controller(observation, action_space, is_act_continuous=False):
    action_ = agent.act(observation['obs'],)
    # 智能体根据观察值做出动作
    agent_action = [np.array([val]) for val in action_]
    # 将数据转为 np.float64 数据类型，兼容接口所需数据类型
    return agent_action
