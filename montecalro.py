# coding: utf-8

import numpy as np
from math import floor
import matplotlib.pyplot as plt

class MonteCarloPolycyIteration:
    n_states = 3**9 # 状態数
    n_actions = 9   # 行動数
    T = 5           # 最大ステップ数

    visites = None
    states = None
    actions = None
    rewards = None
    drewards = None
    results = None
    rate = []

    # 盤面を回転させると同じ状態になる
    # その変換を行い状態数を減らす
    convert = [[0,1,2,3,4,5,6,7,8], # 元の状態
               [2,1,0,5,4,3,8,7,6], # 変換(2)
               [6,3,0,7,4,1,8,5,2], # 変換(3)
               [0,3,8,1,4,7,2,5,8], # 変換(4)
               [8,7,6,5,4,3,2,1,0], # 変換(5)
               [6,7,8,3,4,5,0,1,2], # 変換(6)
               [2,5,8,1,4,7,0,3,6], # 変換(7)
               [8,5,2,7,4,1,6,3,0]  # 変換(8)
               ]
    # 3進数から10進数への変換用ベクトル
    power = np.array([ 3**i for i in xrange(8,-1,-1) ], dtype=np.float64)

    def __init__(self,L,M,options):
        self.L = L # 政策反復回数
        self.M = M # エピソード数
        self.options = options
        self.Q =  self.init_Q()
        np.random.seed(555)

    def init_Q(self):
        """ Q関数の初期化(initialize look up table)  """
        return np.zeros((self.n_states,self.n_actions))

    def train(self):
        # policy iteration
        for l in xrange(self.L):
            self.visites = self.init_visites()
            self.states = self.init_matrix()
            self.actions = self.init_matrix()
            self.rewards = self.init_matrix()
            self.drewards = self.init_matrix()
            self.results = self.init_results()

            # episode
            for m in xrange(self.M):
                state3 = self.init_state3()

                # step
                for t in xrange(self.T):
                    state = self.encode(state3)
                    policy = self.generate_policy()
                    policy = self.policy_improvement(policy,state)

                    # 行動選択および実行
                    action, reward, state3, fin = self.action_train(policy, t, state3)

                    self.update(m, t, state, action, reward)

                    if self.isfinished(fin):
                        #print 'fin_state',state3
                        self.results[m] = fin

                        # 割引報酬和の計算
                        self.calculate_discounted_rewards(m, t)
                        break
            self.Q = self.calculate_state_action_value_function()
            self.output_results(l)
            self.rate.append( float(len(self.results[self.results==2]))/self.M )

    def init_visites(self):
        return np.ones((self.n_states, self.n_actions))

    def init_matrix(self):
        return np.ones((self.M, self.T))

    def init_results(self):
        return np.zeros(self.M)

    def init_state3(self):
        return np.zeros(self.n_actions)

    def generate_policy(self):
        return np.zeros(self.n_actions)

    def policy_improvement(self,policy,state):
        if self.options['pmode']==0:
            q = self.Q[state] # 現在の政策での状態価値
            v = max(q) # 現在の状態での最適な行動における価値
            a = np.where(q==v)[0][0] # 最適行動
            policy[a] = 1
        elif self.options['pmode']==1:
            q = self.Q[state] # 現在の政策での状態価値
            v = max(q) # 現在の状態での最適な行動における価値
            a = np.where(q==v)[0][0] # 最適行動
            policy = np.ones(self.n_actions)*self.options['epsilon']/self.n_actions
            policy[a] = 1-self.options['epsilon']+self.options['epsilon']/self.n_actions
        elif self.options['pmode']==2:
            policy = np.exp(self.Q[state]/self.options['tau'])/ \
                        sum(np.exp(self.Q[state]/self.options['tau']))
        return policy

    def enemy_action_train(self, policy, t, state3):
        enemy_action = self.select_enemy_action(t, state3)
        state3[enemy_action] = 1
        fin = self.check_game(state3)
        reward = self.calculate_reward(fin)

    def action_train(self, policy, t, state3):
        """
        enemy_action = self.select_enemy_action(t, state3)
        state3[enemy_action] = 1
        fin = self.check_game(state3)
        reward = self.calculate_reward(fin)
        if reward is not None:
            return self.npc_action, reward, state3, fin

        self.npc_action = self.select_npc_action(t, state3, policy)
        state3[self.npc_action] = 2
        fin = self.check_game(state3)
        reward = self.calculate_reward(fin)
        return self.npc_action, reward, state3, fin
    """

        npc_action = self.select_npc_action(t, state3, policy)
        state3[npc_action] = 2
        fin = self.check_game(state3)
        reward = self.calculate_reward(fin)
        if reward is not None:
            return npc_action, reward, state3, fin

        enemy_action = self.select_enemy_action(t, state3)
        state3[enemy_action] = 1
        fin = self.check_game(state3)
        reward = self.calculate_reward(fin)
        return npc_action, reward, state3, fin


    def select_npc_action(self, step, state3, policy):
        a = None
        if step==0:
            a = 0
        else:
            while 1:
                random = np.random.rand()
                cprob = 0
                for a in xrange(self.n_actions):
                    cprob += policy[a]
                    if random<cprob: break
                # 既にマスが埋まってないかどうか確認
                if state3[a]==0:break
        return a

    def select_enemy_action(self, step, state3):
        reach = 0
        pos = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[1,5,8],[0,4,8],[2,4,6]]
        a = None
        for i in xrange(len(pos)):
            state_i = state3[pos[i]]
            val = sum(state_i)
            num = len(state_i[state_i==0])
            if val==2 and num==1:
                idx = int( state_i[state_i==0][0] )
                a = pos[i][idx]
                reach = 1
                break
        if reach==0:
            while 1:
                a = floor(np.random.rand()*8)+1
                if state3[a]==0: break
        return a

    def check_game(self, state3):
        pos = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[1,5,8],[0,4,8],[2,4,6]]
        for i in xrange(len(pos)):
            state_i = state3[pos[i]]
            val_npc = sum(state_i==1)
            val_enemy = sum(state_i==2)
            if val_npc==3:
                return 1 # 勝ち
            elif val_enemy==3:
                return 2 # 負け
        if sum(state3==0)==0:
            return 3 # 引き分け
        return 0 # ゲーム続行

    def calculate_reward(self, fin):
        if fin==2:
            return 10
        elif fin==1:
            return -10
        elif fin==3:
            return 0
        elif fin==0:
            return None

    def update(self,m,t,state,action,reward):
        """ 状態、行動、報酬、出現回数の更新 """
        self.states[m,t] = state
        self.actions[m,t] = action
        self.rewards[m,t] = reward
        self.visites[state, action] += 1

    def isfinished(self,fin):
        if fin>0: return True
        else: return False

    def calculate_discounted_rewards(self, m, t):
        for pstep in xrange(t-1,-1,-1):
            self.drewards[m,t] = self.rewards[m,t]
            self.drewards[m,pstep] = \
                    self.options['gamma']*self.drewards[m,pstep+1]

    def calculate_state_action_value_function(self):
        Q = self.init_Q()
        for m in xrange(self.M):
            for t in xrange(self.T):
                s = self.states[m,t]
                if s==0: break # 9手目までに終わっている場合は処理終了
                a =self.actions[m,t]
                Q[s,a] += self.drewards[m,t]
        return Q/self.visites

    def output_results(self,l):
        print 'l=%d: Win=%d/%d, Draw=%d/%d, Lose=%d/%d\n' % (l, \
                        len(self.results[self.results==2]),self.M, \
                        len(self.results[self.results==3]),self.M, \
                        len(self.results[self.results==1]),self.M)

    def encode(self, state3):
        # stateに(2)～(8)の8種類の変換を加えた後、10進数へ変換
        cands = [ sum(state3[self.convert[i]]*self.power) # indexを入れ替えて、10進数に変換
                    for i in xrange(len(self.convert))]
        # 8個の候補のうち一番小さいものを選ぶ
        return min(cands)+1

if __name__=='__main__':
    options = {'pmode':1,'epsilon':0.05,'tau':2,'gamma':0.9}
    mcpi= MonteCarloPolycyIteration(6,100,options)
    mcpi.train()

    plt.plot(range(len(mcpi.rate)), mcpi.rate)
    plt.show()