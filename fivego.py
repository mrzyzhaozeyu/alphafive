# %%
import numpy as np
import itertools
import copy
import multiprocessing as mp
from collections import Counter
from math import sqrt, log
import random
from tkinter import *
from tkinter.messagebox import *
import pickle
import time

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
                self.winner = player
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


# %%
class Tree:
    """
    构建搜索树，可以返回任意一条路径的任意深度的结果和状态
    并对选定的节点进行统计采样
    """

    def __init__(self, initial_board, robotplayer=playerA, continus_number=3):
        self.firstplayer = robotplayer
        self.board = initial_board
        self.continus_number = continus_number
        self.unvisited_board = {}
        self.visited_board = {}
        self.unvisited_path = []
        self.visited_path = []
        self.UCT_recode = {}
        self.tree_recode = {}
        self.visited_path = []

    def save_tree(self):
        """
        保存预训练模型
        """
        output = open('pre_train_model.pkl', 'wb')
        pickle.dump(self.tree_recode, output)

    def read_tree(self):
        """
        读取预训练模型
        """
        pkl_file = open('pre_train_model.pkl', 'rb')
        self.tree_recode = pickle.load(pkl_file)

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
                    curr_data[i]['count'] += 1  # 反向传播记录总数
                    curr_data = curr_data[i]
                    # curr_data['count'] += 1
                else:
                    curr_data[i] = {'count': 1, 'playerA': 0, 'playerB': 0, 'UCT_A': 0, 'UCT_B': 0, 'state': 'unvisited'}
                    curr_data = curr_data[i]

            curr_data[key_list[-1]] = {'count': 0, 'playerA': 0, 'playerB': 0, 'UCT_A': 0, 'UCT_B': 0, 'state': 'unvisited'}

        if v == 'playerA':

            key_list = k
            curr_data = self.tree_recode
            for i in key_list[:-1]:
                if curr_data.__contains__(i):
                    curr_data[i]['count'] += 1
                    curr_data[i]['playerA'] += 1
                    curr_data = curr_data[i]

                else:
                    curr_data[i] = {'count': 1, 'playerA': 1, 'playerB': 0, 'UCT_A': 0, 'UCT_B': 0,'state': 'unvisited'}
                    curr_data = curr_data[i]

            curr_data[key_list[-1]] = {'count': 0, 'playerA': 0, 'playerB': 0, 'UCT_A': 0, 'UCT_B': 0,'state': 'playerA'}

        if v == 'playerB':

            key_list = k
            curr_data = self.tree_recode
            for i in key_list[:-1]:
                if curr_data.__contains__(i):
                    curr_data[i]['count'] += 1
                    curr_data[i]['playerB'] += 1
                    curr_data = curr_data[i]

                else:
                    curr_data[i] = {'count': 1, 'playerA': 0, 'playerB': 1, 'UCT_A': 0, 'UCT_B': 0, 'state': 'unvisited'}
                    curr_data = curr_data[i]

            curr_data[key_list[-1]] = {'count': 0, 'playerA': 0, 'playerB': 0, 'UCT_A': 0, 'UCT_B': 0, 'state': 'playerB'}

    def read_result_from_tree(self, path):
        """
        通过path来读取最后结果
        从树中读出路径下的结果
        """
        # tree = self.tree_recode
        order = "self.tree_recode"
        for point in path:
            order += "[{}]".format(point)
        print(order)
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

    def find_max_UCT_path(self, tree, player):
        """
        搜索每个deep最大的UCT节点，返回下一步要expand的节点
        """
        max_path = []
        while True:
            max_point, next_tree = self.find_layer_max_uct(tree, player)
            player = -player
            tree = next_tree
            if max_point is None:
                break
            else:
                max_path.append(max_point)
        return max_path

    def find_layer_max_uct(self, tree, player):
        """
        搜索每一层的最大UCT和最小UCT，交替player
        :param tree:
        :param player:
        :return:
        """
        max_uct_poin = None
        max_uct = None
        if player == 1:
            for k in tree:
                if isinstance(tree[k], dict):
                    if tree[k].__contains__('UCT_A'):
                        if tree[k]['state'] == 'unvisited':
                            if max_uct_poin is None or max_uct is None:
                                max_uct_poin = k
                                max_uct = tree[k]['UCT_A']
                            elif tree[k]['UCT_A'] > max_uct:
                                max_uct = tree[k]['UCT_A']
                                max_uct_poin = k
            if max_uct_poin is None:  # 最后的叶子节点UCT都为0 从上一级开始expand
                next_tree = None
                max_uct_poin = None
            else:
                next_tree = tree[max_uct_poin]
        elif player == -1:
            for k in tree:
                if isinstance(tree[k], dict):
                    if tree[k].__contains__('UCT_B'):
                        if tree[k]['state'] == 'unvisited':
                            if max_uct_poin is None or max_uct is None:
                                max_uct_poin = k
                                max_uct = tree[k]['UCT_B']
                            elif tree[k]['UCT_B'] > max_uct:
                                max_uct = tree[k]['UCT_B']
                                max_uct_poin = k
            if max_uct_poin is None:  # 最后的叶子节点UCT都为0 从上一级开始expand
                next_tree = None
                max_uct_poin = None
            else:
                next_tree = tree[max_uct_poin]

        return max_uct_poin, next_tree

    def fully_expanded(self, para):
        """
        对从root开始指定深度的point进行随机抽样覆盖的搜索
        """
        unvisited_path = []
        visited_playerA_path = []
        visited_playerB_path = []
        next_player = -para['firstplayer']  # root后下一手棋手
        board = copy.deepcopy(para['board'])
        board.add_stone(para['point'][0], para['point'][1], para['firstplayer'])  # root点落子
        # availabel_points_list = board.available_points_list  # 可落子的点
        availabel_points_list = self.strategy(board)
        print(availabel_points_list)
        print(len(availabel_points_list))
        if len(availabel_points_list) < para['deep']:
            para['deep'] = len(availabel_points_list)

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

        return {'point': para['point'],  # 起始扩展点
                # 'UCT': UCT,  # 被扩展点的UCT
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

    def updata_UCT(self, tree, root_count, c):
        for k in tree:
            if isinstance(tree[k], dict):
                tree[k]['UCT_A'] = self.UCT(tree[k]['playerA'], tree[k]['count'], root_count, c)
                tree[k]['UCT_B'] = self.UCT(tree[k]['playerB'], tree[k]['count'], root_count, c)
                self.updata_UCT(tree[k], tree[k]['count'], c)

    def traverse(self, board, deep, random_pick_num=100):
        """
        蒙特卡洛树中找到best_uct节点
        """
        poll = mp.Pool(processes=self.multiprocessing_num)  # 并行运算
        root_width = len(board.available_points_list)
        fully_expanded_para_list = []

        # 计算总的搜索数量
        sum_root_path = 1
        for i in range(deep):
            sum_root_path *= root_width - i

        # 构建多线程的传输参数
        for point in board.available_points_list:
            fully_expanded_para_list.append({'point': point,  # 开始搜索的root的第一手点位
                                             'sum_root_path': sum_root_path,  # root在深度为deep下的所有搜索路径
                                             'deep': deep,  # 搜索深度
                                             'firstplayer': self.firstplayer,
                                             'random_pick_num': random_pick_num,
                                             'board': board})  # 传入初始化的棋盘

        recode_collection = poll.map(self.fully_expanded, fully_expanded_para_list)
        #  求best uct和best point
        for recode in recode_collection:
            self.backpropagation(recode['unvisited_path'],
                                 recode['visited_playerA_path'],
                                 recode['visited_playerB_path'])

    def generate_initial_board(self, pre_path, board, player=playerA):
        """
        生成某条准备探索的初始化棋盘，预先的path已经生成
        :param pre_path:
        :param board:
        :param player: 初始化开始的棋手默认是PlayerA=1
        :return:
        """
        for point in pre_path:
            try:
                board.add_stone(point[0], point[1], player)
            except:
                pass
            player = -player
        return board

    def strategy(self, board):
        last_stone = board.move_recode[-1]
        select_points = []
        x_list = []
        y_list = []
        for point in board.move_recode:
            x_list.append(point[0])
            y_list.append(point[1])
        x_centre = int(np.mean(x_list))
        y_centre = int(np.mean(y_list))
        for x in range(x_centre-2, x_centre+2):
            for y in range(y_centre-2, y_centre+2):
                if x>=0 and x<board.board_size and y>=0 and y<board.board_size:
                    if (x, y) in board.available_points_list:
                        select_points.append((x, y))
        # if len(select_points) < 10:
        for x in range(last_stone[0] - 2, last_stone[0] + 2):
            for y in range(last_stone[1] - 2, last_stone[1] + 2):
                if x >= 0 and x < board.board_size and y >= 0 and y < board.board_size:
                    if (x, y) in board.available_points_list and (x, y) not in select_points:
                        select_points.append((x, y))

        return select_points

    def monte_carlo_tree_search(self, deep, random_pick_num, multiprocessing_num, robot_player_order='playerB'):
        """
        main function
        :param deep:
        :param random_pick_num:
        :param multiprocessing_num:
        :param robot_player_order: 默认机器后下，如果机器先下，应该是playerB
        :return:
        """
        try:
            self.read_tree()
        except:
            pass
        start_time = time.time()
        self.multiprocessing_num = multiprocessing_num
        pre_path = self.board.move_recode
        print('pre_path:', pre_path)
        while time.time() - start_time < 180:
            next_board = self.generate_initial_board(pre_path, copy.deepcopy(self.board))
            self.traverse(deep=deep,
                          random_pick_num=random_pick_num,
                          board=self.generate_initial_board(pre_path, copy.deepcopy(self.board),
                                                            player=playerA))  # 从一个选定的叶子开始扩展并更新tree 此处的棋盘board要深度拷贝不要污染初始化棋盘
            self.updata_UCT(self.tree_recode, 1, c=1.75)  # 更新UCT
            # self.read_result_from_tree(pre_path)
            self.max_path = self.find_max_UCT_path(self.tree_recode, playerB)  # 贪心算法找到UCT最大路径
            # print(self.tree_recode)
            # self.max_path = self.find_max_UCT_path(self.read_result_from_tree(pre_path), playerB)  # 贪心算法找到UCT最大路径, 更新最佳路径
            print('-------self.max_path--------:', self.max_path)
            pre_path = self.max_path
            # pre_path = self.max_path[len(self.board.move_recode):]
        self.save_tree()

        return self.max_path[len(self.board.move_recode)]





# %%

class ChessBoard():

    def __init__(self):
        # self.recode = []
        self.size = 8  # 棋盘横纵的数量
        self.Matrix = [[0 for y in range(self.size)] for y in range(self.size)]
        self.mesh = 25  # 棋盘网格间隙
        self.ratio = 0.8  # 棋子占网格的比例
        self.board_color = "#F1C40F"
        self.step = self.mesh / 2
        self.chess_r = self.step * self.ratio
        self.is_start = False  # 是否开始
        self.score = [0, 0]  # 比分
        self.last_stone = []

        #########################################

        self.root = Tk()
        self.root.title("Five-in-a-Row By Group13")
        self.root.resizable(width=False, height=False)

        self.f_header = Frame(self.root, highlightthickness=0, bg="#CDC0B0")
        self.f_header.pack(fill=BOTH, ipadx=10)

        self.b_start = Button(self.f_header, text="Start", command=self.bt_start, font=("Fixdsys", 18, "bold"))
        self.b_start.pack(side=LEFT, padx=10)

        self.l_info = Label(self.f_header, text=str(self.score[0]) + ":" + str(self.score[1]), bg="#CDC0B0",
                            font=("Fixdsys", 18, "bold"),
                            fg="white")
        self.l_info.pack(side=LEFT, expand=YES, fill=BOTH, padx=10)

        self.board = Canvas(self.root, bg=self.board_color, width=(self.size + 1) * self.mesh,
                            height=(self.size + 1) * self.mesh, highlightthickness=0)
        self.draw_board()
        self.board.bind("<Button-1>", self.update_by_man)
        self.board.pack()

        self.root.mainloop()

    def bt_start(self):
        self.is_start = True
        self.Matrix = [[0 for y in range(self.size)] for x in range(self.size)]
        self.draw_board()
        self.l_info.config(text=str(self.score[0]) + ":" + str(self.score[1]))

    def update_by_man(self, e):
        tag = 1
        x, y = int((e.y - self.step) / self.mesh), int((e.x - self.step) / self.mesh)
        center_x, center_y = self.mesh * (x + 1), self.mesh * (y + 1)
        distance = ((center_x - e.y) ** 2 + (center_y - e.x) ** 2) ** 0.5
        if distance > self.step * 0.95 or self.Matrix[x][y] != 0 or not self.is_start:
            return
        self.last_stone.append(tuple([x, y]))
        color = "#000000"
        self.draw_stone(x, y, color)
        self.Matrix[x][y] = tag

        if self.check_for_done(tag):
            self.score[0] += 1
            text = "Man wins!"
            self.board.create_text(int(int(self.board['width']) / 2), int(int(self.board['height']) / 2), text=text,
                                   font=("Fixdsys", 18, "bold"), fill="red")
            return
        #
        self.update_by_pc()

    def update_by_pc(self):
        board = Board(8)
        start_player = playerA
        for stone in self.last_stone:
            board.add_stone(stone[0], stone[1], start_player)
            start_player = -start_player
        # print(self.last_stone)
        time.sleep(5)
        # print(self.last_stone)
        tree = Tree(board, robotplayer=playerB, continus_number=5)
        best_point = tree.monte_carlo_tree_search(deep=10, random_pick_num=3000, multiprocessing_num=3)
        x = best_point[0]
        y = best_point[1]

        ##############################需要算出x，y###############################
        #         x,y=10,10
        self.last_stone.append(tuple([x, y]))
        tag = -1
        self.Matrix[x][y] = tag
        color = "#FFFFFF"
        self.draw_stone(x, y, color)
        if self.check_for_done(tag):
            self.score[1] += 1
            text = "PC wins!"
            self.board.create_text(int(int(self.board['width']) / 2), int(int(self.board['height']) / 2), text=text,
                                   font=("Fixdsys", 18, "bold"), fill="red")
            return

    def draw_board(self):
        for x in range(self.size):
            for y in range(self.size):
                center_x, center_y = self.mesh * (x + 1), self.mesh * (y + 1)
                self.board.create_rectangle(center_y - self.step, center_x - self.step,
                                            center_y + self.step, center_x + self.step,
                                            fill=self.board_color, outline=self.board_color)
                a, b = [0, 1] if y == 0 else [-1, 0] if y == self.size - 1 else [-1, 1]
                c, d = [0, 1] if x == 0 else [-1, 0] if x == self.size - 1 else [-1, 1]
                self.board.create_line(center_y + a * self.step, center_x, center_y + b * self.step, center_x)
                self.board.create_line(center_y, center_x + c * self.step, center_y, center_x + d * self.step)

    # 画x行y列处的棋子，color指定棋子颜色
    def draw_stone(self, x, y, color):
        center_x, center_y = self.mesh * (x + 1), self.mesh * (y + 1)
        self.board.create_oval(center_y - self.chess_r,
                               center_x - self.chess_r,
                               center_y + self.chess_r,
                               center_x + self.chess_r,
                               fill=color)

    def check_for_done(self, player):
        flag = False
        for i in range(self.size):
            for j in range(self.size):
                if self.Matrix[i][j] == player:
                    if i + 4 < self.size and self.Matrix[i + 1][j] == player and self.Matrix[i + 2][j] == player and \
                            self.Matrix[i + 3][j] == player and self.Matrix[i + 4][j] == player:  # horizontal
                        flag = True
                        self.is_start = False
                    if j + 4 < self.size and self.Matrix[i][j + 1] == player and self.Matrix[i][j + 2] == player and \
                            self.Matrix[i][j + 3] == player and self.Matrix[i][j + 4] == player:  # vertical
                        flag = True
                        self.is_start = False
                    if i + 4 < self.size and j + 4 < self.size and self.Matrix[i + 1][j + 1] == player and \
                            self.Matrix[i + 2][j + 2] == player and self.Matrix[i + 3][j + 3] == player and \
                            self.Matrix[i + 4][j + 4] == player:  # diagonal
                        flag = True
                        self.is_start = False
                    if i - 4 > 0 and j + 4 < self.size and self.Matrix[i - 1][j + 1] == player and self.Matrix[i - 2][
                        j + 2] == player and self.Matrix[i - 3][j + 3] == player and self.Matrix[i - 4][
                        j + 4] == player:  # anti-diagonal
                        flag = True
                        self.is_start = False
        return flag


# %%
if __name__ == "__main__":
    # board = Board(8)
    # # board.add_stone(5, 4, playerA)
    # # board.add_stone(5, 5, playerB)
    # # board.add_stone(6, 5, playerA)
    # # board.add_stone(6, 6, playerB)
    # # board.add_stone(4, 3, playerA)
    # # board.add_stone(7, 6, playerB)
    # # board.add_stone(5, 4, playerA)
    # # board.add_stone(5, 5, playerB)
    # # board.add_stone(5, 4, playerA)
    # # board.add_stone(5, 5, playerB)
    #
    # board.add_stone(3, 3, playerA)
    # board.add_stone(2, 2, playerB)
    # board.add_stone(3, 2, playerA)
    # board.add_stone(3, 1, playerB)
    # board.add_stone(2, 3, playerA)
    # board.add_stone(0, 4, playerB)
    # board.add_stone(4, 3, playerA)
    # tree = Tree(board, robotplayer=playerB, continus_number=5)
    # a = time.time()
    # best_point = tree.monte_carlo_tree_search(deep=10, random_pick_num=10000, multiprocessing_num=3)
    # print(best_point)
    # print(tree.board.board)
    # # print(tree.tree_recode)
    # # best_point = tree.monte_carlo_tree_search(deep=12, random_pick_num=5000, multiprocessing_num=3)
    # # print(best_point)
    # b = time.time()
    # print(b - a)

    ChessBoard()
