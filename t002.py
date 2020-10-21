from __future__ import absolute_import, division, print_function
from machin.frame.algorithms import DQN
from machin.utils.logging import default_logger as logger
from machin.model.nets import static_module_wrapper, dynamic_module_wrapper
import torch as t
import torch.nn as nn
import gym
from gym import spaces
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
import pandas as pd
import datetime
import sys
import re
import os
import copy

# configurations
#    observe_dim = 4
action_num = 2
#    max_episodes = 1000
max_steps = 200
solved_reward = 190
solved_repeat = 5

eval_interval = 1000  # @param {type:"integer"}
g_sDataDir="{}/data".format(sys.path[0])
g_fPip=0.0001
g_sTrainFileName = "eurusd_test"
g_sTestFileName = "eurusd_bb_02"
g_sEvalFileName = "eurusd_bb_03"
g_fAccBalance=1000.0
g_fLotSize=100000.0
g_fMaxLoss=0.3

m = re.search("([^\\\\\\/]+)\\.\\w+$", sys.argv[0])
g_sScriptFile=m.group(1)
g_sTrainFullFileName="{}/{}.csv".format(g_sDataDir, g_sTrainFileName)
g_sTestFullFileName="{}/{}.csv".format(g_sDataDir, g_sTestFileName)
g_sEvalFullFileName="{}/{}.csv".format(g_sDataDir, g_sEvalFileName)
g_sModel1="{}/model_{}".format(g_sDataDir, g_sScriptFile)
g_sTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
g_sLogDir1="{}/logs/{}/{}".format(sys.path[0], g_sScriptFile, g_sTime)
g_sCheckPointsDir1="{}/checkpoint/{}".format(sys.path[0], g_sScriptFile)

class CCsvQuotes:
    def fnClose(self):
        if (not self.m_oReader is None):
            del self.m_oReader
        self.m_oReader=None
        if (not self.m_asBuf is None):
            del self.m_asBuf
        self.m_asBuf=None
        self.m_iBufIndex=None

    def fnRead(self):
        try:
            if (self.m_asBuf is None or self.m_iBufIndex>=len(self.m_asBuf)):
                self.m_asBuf=next(self.m_oReader)
                self.m_iBufIndex=0
            asRes=(self.m_asBuf.iloc[self.m_iBufIndex]).to_numpy(dtype=str, copy=True)
            self.m_iBufIndex+=1
            return asRes
        except Exception as xCptn: # catch *all* exceptions
            iStopRightHere=-1
        self.fnClose()
        return None

    def fnOpen(self):
        self.m_hFile=open(self.m_sFileName)
        self.m_oReader=pd.read_csv(self.m_sFileName, header=None, chunksize=eval_interval)
        self.m_asBuf=None
        self.m_iBufIndex=None

    def fnReSet(self):
        self.fnClose()
        self.fnOpen()
        sHeader=self.fnRead()
        self.m_iColumnsNumber=len(sHeader)

    def fnCountLines(self):
        iLineCount=-1
        try:
            while (True):
                self.m_asBuf=next(self.m_oReader)
                iLineCount+=len(self.m_asBuf)
        except Exception as xCptn: # catch *all* exceptions
            iStopRightHere=-1
        self.fnReSet()
        return iLineCount

    def __init__(self, IN_sFileName):
        self.m_sFileName=IN_sFileName
        self.m_oReader=None
        self.m_asBuf=None
        self.m_iBufIndex=None
        self.fnOpen()
        self.m_iNumLines=self.fnCountLines()

#    https://github.com/openai/gym/blob/master/gym/envs/toy_text/hotter_colder.py
class CForex(gym.Env):

    def fnCalcDiff(self):
        return (self.m_afObservation[0]-self.m_fPositionPrice)*g_fLotSize

    def __init__(self, IN_sCsvFileName):
        self.m_asActions=["Long", "Short", "Close position"]
        self.m_fAccBalance = g_fAccBalance
        self.m_hQuotesFile=CCsvQuotes(IN_sCsvFileName)
        self._action_spec = spaces.Discrete(len(self.m_asActions))
        self._observation_spec = spaces.Box(low=np.array([-np.inf for x in range(self.m_hQuotesFile.m_iColumnsNumber)]),
                                            high=np.array([np.inf for x in range(self.m_hQuotesFile.m_iColumnsNumber)]),
                                            dtype=np.float32)
        self.m_afObservation = np.zeros(self.m_hQuotesFile.m_iColumnsNumber, dtype=np.float32)
        self.reset()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def close(self) -> None:
        self.m_fAccBalance = g_fAccBalance
        self.m_hQuotesFile.fnClose()
        self.m_iLastAction=2        #    Close position
        self.m_iPositionAction=2    #    Close position
        self.m_fPositionPrice=0
        self.m_afObservation.fill(0)

    def reset(self):
        self.close()
        self.m_hQuotesFile.fnReSet()
        return copy.deepcopy(self.m_afObservation)

    def step(self, action):
        assert isinstance(action, int)
        iAction=action
        assert 0<=iAction and len(self.m_asActions)>iAction

        asQuotes=self.m_hQuotesFile.fnRead()
        done=bool(True)
        fReward=0.0
        oInfo={"eof":True}
        if (asQuotes is None):
            self.m_afObservation.fill(0)
        else:
            done=bool(False)
            for i in range(len(asQuotes)):
                self.m_afObservation[i]=float(asQuotes[i])
            self.m_afObservation[1]=0   #    Kill the iAction from file

            fDiff = self.fnCalcDiff()
            oInfo={
                        "action":"no change",
                        "last action's price":self.m_fPositionPrice,
                        "current price":self.m_afObservation[0],
                        "diff":fDiff,
                        "account balance":self.m_fAccBalance
            }
            if (0==iAction):         #    Long
                if (1==self.m_iPositionAction):     #    Was short
                    oInfo={
                        "action":"close short, open long",
                        "last action's price":self.m_fPositionPrice,
                        "current price":self.m_afObservation[0],
                        "profit":-fDiff,
                        "old account balance":self.m_fAccBalance
                    }
                    fReward=-fDiff
                    self.m_fAccBalance+=fReward
                    self.m_iPositionAction=iAction    #    Long
                    self.m_fPositionPrice=self.m_afObservation[0]
                    done=bool(self.m_fAccBalance<(1.0-g_fMaxLoss)*g_fAccBalance)
                    oInfo["new account balance"]=self.m_fAccBalance
                    oInfo["done"]=done
                elif (2==self.m_iPositionAction):      #    Close position
                    oInfo={
                        "action":"open long position",
                        "at price":self.m_fPositionPrice,
                        "current price":self.m_afObservation[0],
                        "old account balance":self.m_fAccBalance
                    }
                    self.m_iPositionAction=iAction    #    Long
                    self.m_fPositionPrice=self.m_afObservation[0]
            elif (1==iAction):       #    Short
                if (0==self.m_iPositionAction):     #    Was Long
                    oInfo={
                        "action":"close long, open short",
                        "last action's price":self.m_fPositionPrice,
                        "current price":self.m_afObservation[0],
                        "profit":fDiff,
                        "old account balance":self.m_fAccBalance
                    }
                    fReward=fDiff
                    self.m_fAccBalance+=fReward
                    self.m_iPositionAction=iAction    #    Short
                    self.m_fPositionPrice=self.m_afObservation[0]
                    done=bool(self.m_fAccBalance<(1.0-g_fMaxLoss)*g_fAccBalance)
                    oInfo["new account balance"]=self.m_fAccBalance
                    oInfo["done"]=done
                elif (2==self.m_iPositionAction):      #    Close position
                    oInfo={
                        "action":"open short position",
                        "last action's price":self.m_fPositionPrice,
                        "current price":self.m_afObservation[0],
                        "old account balance":self.m_fAccBalance
                    }
                    self.m_iPositionAction=iAction    #    Short
                    self.m_fPositionPrice=self.m_afObservation[0]
            elif (2==iAction):       #    Close position
                if (0==self.m_iPositionAction):     #    Was Long
                    oInfo={
                        "action":"close long position",
                        "last action's price":self.m_fPositionPrice,
                        "current price":self.m_afObservation[0],
                        "profit":fDiff,
                        "old account balance":self.m_fAccBalance
                    }
                    fReward=fDiff
                    self.m_fAccBalance+=fReward
                    self.m_iPositionAction=iAction    #    Short
                    self.m_fPositionPrice=self.m_afObservation[0]
                    done=bool(self.m_fAccBalance<(1.0-g_fMaxLoss)*g_fAccBalance)
                    oInfo["new account balance"]=self.m_fAccBalance
                    oInfo["done"]=done
                elif (1==self.m_iPositionAction):     #    Was short
                    oInfo={
                        "action":"close short position",
                        "last action's price":self.m_fPositionPrice,
                        "current price":self.m_afObservation[0],
                        "profit":-fDiff,
                        "old account balance":self.m_fAccBalance
                    }
                    fReward=-fDiff
                    self.m_fAccBalance+=fReward
                    self.m_iPositionAction=iAction    #    Long
                    self.m_fPositionPrice=self.m_afObservation[0]
                    done=bool(self.m_fAccBalance<(1.0-g_fMaxLoss)*g_fAccBalance)
                    oInfo["new account balance"]=self.m_fAccBalance
                    oInfo["done"]=done

        self.m_iLastAction=iAction        #    Close position

        return copy.deepcopy(self.m_afObservation), np.float32(fReward), done, oInfo

    def render(self, mode: str = 'rgb_array'):
        """Renders the environment.

        Args:
            mode: Rendering mode. Only 'rgb_array' is supported.
            blocking: Whether to wait for the result.

        Returns:
            An ndarray of shape [width, height, 3] denoting an RGB image when
            blocking. Otherwise, callable that returns the rendered image.
        Raises:
            NotImplementedError: If the environment does not support rendering,
            or any other modes than `rgb_array` is given.
        """
        if ('rgb_array'==mode):
            #        self.m_fAccBalance=g_fAccBalance*2
            iArr=np.full((400,200,3), 255, dtype=np.uint8)
            iFraction=int(iArr.shape[1]/5)
            ProfitColour=[0, 255, 0]    #    Green
            iCurrBalance=iArr.shape[0]
            iInitBalance=int(iArr.shape[0]*g_fAccBalance/self.m_fAccBalance)
            if (self.m_fAccBalance<g_fAccBalance):
                ProfitColour=[255, 0, 0]    #    Red
                iCurrBalance=int(iArr.shape[0]*self.m_fAccBalance/g_fAccBalance)
                iInitBalance=iArr.shape[0]
            iArr[iArr.shape[0]-iInitBalance:iArr.shape[0], 1*iFraction:2*iFraction]=[0, 0, 0]        #    Black
            iArr[iArr.shape[0]-iCurrBalance:iArr.shape[0], 3*iFraction:4*iFraction]=ProfitColour
            return iArr
        elif ('ansi'==mode):
            sStr="Profit={}%".format(100.0*(self.m_fAccBalance-g_fAccBalance)/g_fAccBalance)
            return sStr 
        elif ('human'==mode):
            sStr="Profit={}%".format(100.0*(self.m_fAccBalance-g_fAccBalance)/g_fAccBalance)
            print(sStr)
            return None 
        else:
            raise NotImplementedError('Only rgb_array rendering mode is supported. '
                                'Got %s' % mode)

    def fnNumIterations(self):
        return self.m_hQuotesFile.m_iNumLines

env = CForex(g_sTrainFullFileName)
env.reset()
im=PIL.Image.fromarray(env.render())
#    ires=im.save("test.png")

#    env = gym.make("CartPole-v0")

# model definition
class QNet(nn.Module):
    def __init__(self, state_dim, action_num):
        super(QNet, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_num)

    def forward(self, some_state):
        a = t.relu(self.fc1(some_state))
        a = t.relu(self.fc2(a))
        return self.fc3(a)


if __name__ == "__main__":
    # let framework determine input/output device based on parameter location
    # a warning will be thrown.
    q_net = QNet(env.observation_spec().shape[0], action_num)
    q_net_t = QNet(env.observation_spec().shape[0], action_num)

    # to mark the input/output device Manually
    # will not work if you move your model to other devices
    # after wrapping

    # q_net = static_module_wrapper(q_net, "cpu", "cpu")
    # q_net_t = static_module_wrapper(q_net_t, "cpu", "cpu")

    # to mark the input/output device Automatically
    # will not work if you model locates on multiple devices

    # q_net = dynamic_module_wrapper(q_net)
    # q_net_t = dynamic_module_wrapper(q_net_t)

    dqn = DQN(q_net, q_net_t,
              t.optim.Adam,
              nn.MSELoss(reduction='sum'))

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    while episode < env.fnNumIterations():
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        state = t.tensor(env.reset(), dtype=t.float32).view(1, env.observation_spec().shape[0])

        while not terminal and step <= max_steps:
            step += 1
            with t.no_grad():
                old_state = state
                # agent model inference
                action = dqn.act_discrete_with_noise(
                    {"some_state": old_state}
                )
                state, reward, terminal, oInfo = env.step(action.item())
                state = t.tensor(state, dtype=t.float32).view(1, env.observation_spec().shape[0])
                total_reward += reward

                dqn.store_transition({
                    "state": {"some_state": old_state},
                    "action": {"action": action},
                    "next_state": {"some_state": state},
                    "reward": reward,
                    "terminal": terminal or step == max_steps
                })

        # update, update more if episode is longer, else less
        if episode > 100:
            for _ in range(step):
                dqn.update()

        # show reward
        smoothed_total_reward = (smoothed_total_reward * 0.9 +
                                 total_reward * 0.1)
        logger.info("Episode {} total reward={:.2f}"
                    .format(episode, smoothed_total_reward))

        if smoothed_total_reward > solved_reward:
            reward_fulfilled += 1
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")
                exit(0)
        else:
            reward_fulfilled = 0
