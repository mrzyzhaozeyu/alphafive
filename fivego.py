#%%
import numpy as np
import itertools
import copy
import multiprocessing as mp
from collections import Counter
from math import sqrt, log
import random

playerA = 1  # 持子
playerB = -1


class Board:
    """
    记录棋盘的状态和可以走的点，并返回是否有人赢了。
    """

    def __init__(self, size):
        self.board_size = size
        self.board = np.zeros([size, size])
        self.available_points_list = [(x, y) for x in range(size) for y in range(size)]
        #  追踪路径和UCT
        self.move_recode = []  # 记录下来一条路径的move
        self.UCT = None  # 记录此时状态的UCT
        # 记录棋盘是否有获胜者，获胜者是谁
        self.winner = None

    def check_continus(self, stone_list, player_list):
        """
        检查连续性
        """

        return player_list in [list(g) for k, g in itertools.groupby(stone_list)]

    def check_for_win(self, player, continus_number):
        """
        检查是否获胜
        """
        m, n = self.board.shape
        player_list = [player] * continus_number

        for horizontal_index in range(n):
            check_ans = self.check_continus(list(self.board[horizontal_index, :]), player_list)

            if check_ans == True:
                self.winner=player
                return True

        for vertical_index in range(m):
            check_ans = self.check_continus(list(self.board[:, vertical_index]), player_list)

            if check_ans == True:
                self.winner = player
                return True

        for diagonal_index in range(-m // 2, m // 2):
            check_ans = self.check_continus(list(self.board.diagonal(diagonal_index)), player_list)

            if check_ans == True:
                self.winner = player
                return True

        for diagonal_index in range(-n // 2, n // 2):
            check_ans = self.check_continus(list(np.fliplr(self.board).diagonal(diagonal_index)), player_list)

            if check_ans == True:
                self.winner = player
                return True
        return False

    def upgrade_available_points(self):
        """
        更新空点信息
        """
        self.available_points_list = []
        index_list = np.where(self.board == 0)
        for x, y in zip(index_list[0], index_list[0]):
            self.available_points_list.append((x, y))

    def add_stone(self, x, y, player):
        """
        落子
        """
        if player in [-1, 0, 1] \
                and x < self.board_size \
                and y < self.board_size \
                and x >= 0 \
                and y >= 0 \
                and self.board[x][y] == 0:
            self.board[x][y] = player
            self.available_points_list.remove((x, y))
            self.move_recode.append((x, y))
        else:
            raise Exception('can not add stone here!!')

#%%
class Tree:
    """
    构建搜索树，可以返回任意一条路径的任意深度的结果和状态
    并对选定的节点进行统计采样
    """

    def __init__(self, initial_board, firstplayer=playerA, continus_number=3):
        self.firstplayer = firstplayer
        self.initial_board = initial_board
        self.continus_number = continus_number
        self.unvisited_board = {}
        self.visited_board = {}
        self.unvisited_path = []
        self.visited_path = []
        self.UCT_recode = {}
        self.tree_recode = {}
        self.visited_path = []

    def write_in_tree_recode(self, path, result):
        """
        将一个dict类型的path和result记录写入tree中{'playerA':3,'playerB':4}
        path是list类型，result是path结果
        [(1,2),(3,4),(5,6)] {'playerA':3,'playerB':4}，
        如果是该分支游戏结束了，result是{'playerA':1,'playerB':0} ,
        如果没结束result是{'playerA':0,'playerB':0}
        每次访问一次path，该path节点上的所有node   {'count': 1,'playerA':1,'playerB':0} path的最后一个节点不更新count参数
        """
        path_dict = {tuple(path): result}
        for k, v in path_dict.items():
            self.write_in_dict(k, v)

    def write_in_dict(self, k, v):
        """
        key:path truple
        value: 最后胜负结果
        利用dict的链表特性多重赋值
        把path编码写入dict tree
        """
        # 未结束路径的写入传播
        if v == 'unvisited':

            key_list = k
            curr_data = self.tree_recode
            for i in key_list[:-1]:
                if curr_data.__contains__(i):
                    curr_data[i]['count'] += 1  # 反向传播记录
                    curr_data = curr_data[i]
                    # curr_data['count'] += 1
                else:
                    curr_data[i] = {'count': 1, 'playerA': 0, 'playerB': 0, 'UCT': 0, 'state': 'unvisited'}
                    curr_data = curr_data[i]

            curr_data[key_list[-1]] = {'count': 0, 'playerA': 0, 'playerB': 0, 'UCT': 0, 'state': 'unvisited'}

        if v == 'playerA':

            key_list = k
            curr_data = self.tree_recode
            for i in key_list[:-1]:
                if curr_data.__contains__(i):
                    curr_data[i]['count'] += 1
                    curr_data[i]['playerA'] += 1
                    curr_data = curr_data[i]

                else:
                    curr_data[i] = {'count': 1, 'playerA': 1, 'playerB': 0, 'UCT': 0, 'state': 'unvisited'}
                    curr_data = curr_data[i]

            curr_data[key_list[-1]] = {'count': 0, 'playerA': 0, 'playerB': 0, 'UCT': 0, 'state': 'playerA'}

        if v == 'playerB':

            key_list = k
            curr_data = self.tree_recode
            for i in key_list[:-1]:
                if curr_data.__contains__(i):
                    curr_data[i]['count'] += 1
                    curr_data[i]['playerB'] += 1
                    curr_data = curr_data[i]

                else:
                    curr_data[i] = {'count': 1, 'playerA': 0, 'playerB': 1, 'UCT': 0, 'state': 'unvisited'}
                    curr_data = curr_data[i]

            curr_data[key_list[-1]] = {'count': 0, 'playerA': 0, 'playerB': 0, 'UCT': 0, 'state': 'playerB'}


        # key_list = k
        # curr_data = self.tree_recode
        # for i in key_list[:-1]:
        #     if curr_data.__contains__(i):
        #         curr_data = curr_data[i]
        #
        #     else:
        #         curr_data[i] = {}
        #         curr_data = curr_data[i]
        #
        # curr_data[key_list[-1]] = v

    def read_result_from_tree(self, path):
        """
        通过path来读取最后结果
        从树中读出路径下的结果
        """
        # tree = self.tree_recode
        order = "self.tree_recode"
        for point in path:
            order += "[{}]".format(point)
        result = eval(order)
        return result

    def winner_detection(self, board, path, player):
        """
        搜索给定的path的结果，中间换人落子，player优先落子
        """
        winner = None
        for point in path:
            board.add_stone(point[0], point[1], player=player)
            winner_exist = board.check_for_win(player, self.continus_number)
            if winner_exist:  # 如果有人获胜
                winner = player
                break
            player = -player  # 换人落子
        return winner, board

    def find_max_UCT_path(self, tree):
        """
        搜索每个deep最大的UCT节点，返回下一步要expand的节点
        """
        max_path = []
        while True:

            max_point, next_tree = self.find_layer_max_uct(tree)
            tree = next_tree
            if max_point is None:
                break
            else:
                max_path.append(max_point)
        return max_path

    def find_layer_max_uct(self, tree):
        """
        搜索每一层的最大UCT
        """
        max_uct_poin = None
        max_uct = None
        for k in tree:
            if isinstance(tree[k], dict):
                if tree[k].__contains__('UCT'):
                    if max_uct_poin is None or max_uct is None:
                        max_uct_poin = k
                        max_uct = tree[k]['UCT']
                    elif tree[k]['UCT'] > max_uct:
                        max_uct = tree[k]['UCT']
                        max_uct_poin = k
        if max_uct_poin is None or max_uct == 0:  # 最后的叶子节点UCT都为0 从上一级开始expand
            nest_tree = None
            max_uct_poin = None
        else:
            nest_tree = tree[max_uct_poin]

        return max_uct_poin, nest_tree

    def fully_expanded(self, para):
        """
        对从root开始指定深度的point进行随机抽样覆盖的搜索
        """
        unvisited_path = []
        visited_playerA_path = []
        visited_playerB_path = []
        next_player = -para['firstplayer']  # root后下一手棋手
        # board = copy.deepcopy(self.initial_board)  # 拷贝初始棋盘
        board = copy.deepcopy(para['board'])
        board.add_stone(para['point'][0], para['point'][1], para['firstplayer'])  # root点落子
        availabel_points_list = board.available_points_list  # 可落子的点
        # path_list = list(itertools.permutations(availabel_points_list, para['deep']))  # 得到深度为deep 这行太占内存了
        # 产生随机的拓展path
        path_list = []
        while len(path_list) < para['random_pick_num']:
            random_path = random.sample(availabel_points_list, para['deep'])
            if random_path not in path_list:
                path_list.append(random_path)


        winner_count_list = []

        for path in path_list:  # 开始遍历所有分支
            subboard = copy.deepcopy(board)
            winner, end_board = self.winner_detection(subboard, path, player=next_player)
            winner_count_list.append(winner)
            # self.visited_path.append(path)
            # 不要记录这些点的状态，内存太大
            if winner is None:
                # self.write_in_tree_recode(path=path, result={'playerA': 0, 'playerB': 0})
                unvisited_path.append(end_board.move_recode)  # 记录那些没有到头的搜索路径
            elif winner == playerA:
                # self.write_in_tree_recode(path=path, result={'winner': winner})
                visited_playerA_path.append(end_board.move_recode)  # 记录那些已经到头的搜索路径playerA 获胜
            elif winner == playerB:
                visited_playerB_path.append(end_board.move_recode)  # 记录那些已经到头的搜索路径playerB 获胜


        winner_recode = Counter(winner_count_list)  # 统计遍历结果得到胜率
        UCT = (winner_recode[self.firstplayer] / (len(winner_count_list))) + (sqrt(2) * sqrt(log(para['sum_root_path']) / len(winner_count_list)))  # 计算胜率 Q/All path

        return {'point': para['point'],  # 起始扩展点
                'UCT': UCT,  # 被扩展点的UCT
                'unvisited_path': unvisited_path,  # 未结束的路径
                'visited_playerA_path': visited_playerA_path,  # 已经结束playerA获胜path
                'visited_playerB_path': visited_playerB_path}  # 已经结束playerB获胜path

    def backpropagation(self, unvisited_path, visited_playerA_path, visited_playerB_path):
        """
        根据每条path的结果，更新tree的dict数据，并计算UCT和UCB
        """
        # TODO 反向传播
        for path in unvisited_path:
            result = 'unvisited'
            self.write_in_tree_recode(path, result)
        for path in visited_playerA_path:
            result = 'playerA'
            self.write_in_tree_recode(path, result)
        for path in visited_playerB_path:
            result = 'playerB'
            self.write_in_tree_recode(path, result)

    def UCT(self, winner, count, root_count, c):
        uct = (winner / (count + 1)) + sqrt(c * log(root_count) / (count + 1))
        return uct

    def updata_UCT(self, tree, root_count, player, c):
        for k in tree:
            if isinstance(tree[k], dict):
                tree[k]['UCT'] = self.UCT(tree[k][player], tree[k]['count'], root_count, c)
                self.updata_UCT(tree[k], tree[k]['count'], player, c)

    def traverse(self, board, deep, random_pick_num=100):
        """
        蒙特卡洛树中找到best_uct节点
        """
        # deep = 3  # 指定搜索深度
        best_point = None
        best_uct = None
        poll = mp.Pool(processes=8)  # 并行运算
        # root_width = len(self.initial_board.available_points_list)
        root_width = len(board.available_points_list)
        fully_expanded_para_list = []

        # 计算总的搜索数量
        sum_root_path = 1
        for i in range(deep):
            sum_root_path *= root_width-i

        # 构建多线程的传输参数
        for point in board.available_points_list:
            fully_expanded_para_list.append({'point': point,  # 开始搜索的root的第一手点位
                                             'sum_root_path': sum_root_path,   # root在深度为deep下的所有搜索路径
                                             'deep': deep,  # 搜索深度
                                             'firstplayer': self.firstplayer,
                                            'random_pick_num': random_pick_num,
                                             'board': board})  # 传入初始化的棋盘

        recode_collection = poll.map(self.fully_expanded, fully_expanded_para_list)
        #  求best uct和best point
        for recode in recode_collection:

            if best_point == None or best_uct == None:
                best_point = recode['point']
                best_uct = recode['UCT']

            elif recode['UCT'] > best_uct:
                best_uct = recode['UCT']
                best_point = recode['point']

            # self.unvisited_board[recode['point']] = recode['unvisited_board']
            # self.visited_board[recode['point']] = recode['visited_board']
            # self.UCT_recode[recode['point']] = recode['UCT']
            self.backpropagation(recode['unvisited_path'],
                                 recode['visited_playerA_path'],
                                 recode['visited_playerB_path'])

        return best_point, best_uct

    def initial_board(self):
        # TODO 生成某条准备探索的初始化棋盘，预先的path已经生成
        pass

    def monte_carlo_tree_search(self, deep, random_pick_num):
        start_time = time.time()
        # board = copy.deepcopy(self.initial_board)
        # while time.time() - start_time < 50:
        while time.time() - start_time < 60:
            # best_point, best_uct = self.traverse(deep=deep,
            #                                      random_pick_num=random_pick_num,
            #                                      board=self.initial_board)  # 第一手棋手开始搜索
            best_point, best_uct = self.traverse(deep=deep,
                                                 random_pick_num=random_pick_num,
                                                 board=self.initial_board)  # 第一手棋手开始搜索
            self.updata_UCT(self.tree_recode, 1, player='playerA', c=2)
            self.max_path = self.find_max_UCT_path(tree.tree_recode)

        # board = random.choice(self.unvisited_board[best_point])  # 选出一个没有完成的board来计算




        return best_point, best_uct



#%%

import time

board = Board(6)
# for point in [(0,0),(1,1), (2,2),(3,3),(4,4)]:
#     board.add_stone(point[0], point[1], 1)
# result = board.check_for_win(1,5)
# print(result)
# print(board.available_points_list)
# board.check_for_win(1, 3)
# board.add_stone(1, 1, -1)

tree = Tree(board, firstplayer=playerA, continus_number=4)
a = time.time()
best_point, best_uct = tree.monte_carlo_tree_search(deep=10, random_pick_num=2000)
print(best_point, best_uct)
b = time.time()
print(b - a)
#%%
tree_recode = tree.tree_recode

def UCT(winner, count, root_count, c):
    uct = (winner/(count+1)) + sqrt(c*log(root_count)/(count+1))
    return uct


def updata_UCT(tree, root_count, player, c):
    for k in tree:
        if isinstance(tree[k], dict):
            tree[k]['UCT'] = UCT(tree[k][player], tree[k]['count'], root_count,c)
            updata_UCT(tree[k], tree[k]['count'], player, c)

updata_UCT(tree_recode, 1, player='playerA', c=2)

#%%
# trees = {'a': {'UCT': 3, 'b': {'UCT': 8}}}

def find_max_UCT_path(tree):
    """
    搜索每个deep最大的UCT节点，返回下一步要expand的节点
    """
    max_path = []

    while True:

        max_point, next_tree = find_layer_max_uct(tree)
        tree = next_tree
        if max_point is None:
            break
        else:
            max_path.append(max_point)
    return max_path


def find_layer_max_uct(tree):
    max_uct_poin = None
    max_uct = None
    for k in tree:
        if isinstance(tree[k], dict):
            if tree[k].__contains__('UCT'):
                if max_uct_poin is None or max_uct is None:
                    max_uct_poin = k
                    max_uct = tree[k]['UCT']
                elif tree[k]['UCT'] > max_uct:
                    max_uct = tree[k]['UCT']
                    max_uct_poin = k
    if max_uct_poin is None:
        nest_tree = None
    else:
        nest_tree = tree[max_uct_poin]

    return max_uct_poin, nest_tree

max_path = find_max_UCT_path(tree.tree_recode)