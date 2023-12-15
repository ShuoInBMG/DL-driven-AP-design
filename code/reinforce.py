import random
import math
from tqdm.auto import tqdm
import numpy as np

import torch
from .models import TimeSeriesTransformer

class score_model:
    def __init__(self, model_path):
        self.model = TimeSeriesTransformer(input_size=21, output_size=1)
        self.model.load_state_dict(torch.load(model_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
    @torch.no_grad()
    def predict(self, sequence):
        sequence == np.array(sequence)
        sequence = np.expand_dims(sequence, axis = 0)

        num, length = sequence.shape
        padding = length - 10
        if padding > 0:
            padding_array = np.zeros(shape = (num, padding))
            sequence = np.concatenate([sequence, padding_array], axis = 1)
        input_sequence = torch.from_numpy(sequence).long()
        input_sequence = input_sequence.to(self.device)

        mask = torch.tril(torch.ones(1, 1)).to(self.device) == 0
        AP_value = self.model(input_sequence, mask)
        return AP_value.cpu().numpy().item()
    
class MonteCarloTreeSearch:
    def __init__(self, sequence_length:int, num_choices:int, predictor, differ:int, mask):
        self.sequence_length = sequence_length
        self.num_choices = num_choices
        self.root = None
        self.predictor = predictor
        self.differ = differ
        self.mask = mask
        if self.mask != None:
            self.position_box = [x for x in range(self.sequence_length) if self.mask[x] == True]

    def evaluate_sequence(self, sequence):
        ap_value = self.predictor.predict(sequence)
        return ap_value

    class Node:
        def __init__(self, sequence, parent=None):
            self.sequence = sequence
            self.visits = 0
            self.score = 0
            self.parent = parent
            self.level = 0
            self.children = []
        def __repr__(self):
            sequence = f"sequence: {self.sequence}\n"
            visits = f"visits: {self.visits}\n"
            score = f"score: {self.score}\n"
            if self.parent == None:
                parent = f"parent: None\n"
            else:
                parent = f"parent: {self.parent.sequence}\n"
            children = f"children: {len(self.children)}\n"
            return sequence + visits + score + parent + children
    
    def find_max_index(self, lst:list):
        max_value = max(lst)
        max_indices = [i for i, value in enumerate(lst) if value == max_value]
        random_index = random.choice(max_indices)
        return random_index
        
    def select_child(self, node:Node):
        exploration_constant = 1 / math.sqrt(2)
        total_visits = sum(child.visits for child in node.children)
        ucb_values = []
        for child in node.children:
            if child.visits != 0:
                ucb_value = child.score / child.visits + exploration_constant * math.sqrt(2 * math.log(total_visits) / child.visits)
            else:
                if total_visits == 0:
                    ucb_value = child.score / 0.5 + exploration_constant * math.sqrt(4) # total_visits = 1
                else:
                    ucb_value = child.score / 0.5 + exploration_constant * math.sqrt(2 * math.log(total_visits) / 0.5)
            ucb_values.append(ucb_value)
        #max_ucb_value = max(ucb_values)
        #max_ucb_index = ucb_values.index(max_ucb_value)
        max_ucb_index = self.find_max_index(ucb_values)
        return node.children[max_ucb_index]
    
    def expand(self, node):
        if self.mask is None:
            for new_amino_acid in range(1, self.num_choices+1): 
                for position in range(self.sequence_length):                                  # 选择有20种
                    new_sequence = node.sequence[:]
                    new_sequence[position] = new_amino_acid
                    if self.is_valid_child(new_sequence):
                        new_node = self.Node(new_sequence, parent=node)
                        new_node.visits = node.visits
                        new_node.score = node.score
                        new_node.level = node.level + 1
                        node.children.append(new_node)
        else:
            for new_amino_acid in range(1, self.num_choices+1): 
                for position in self.position_box:                                  # 选择有20种
                    new_sequence = node.sequence[:]
                    new_sequence[position] = new_amino_acid
                    if self.is_valid_child(new_sequence):
                        new_node = self.Node(new_sequence, parent=node)
                        new_node.visits = node.visits
                        new_node.score = node.score
                        new_node.level = node.level + 1
                        node.children.append(new_node)
        random.shuffle(node.children)
    def is_valid_child(self, new_sequence:list):
        count = 0
        for i,j in zip(self.start, new_sequence):
            if i != j:
                count += 1
        if count == 0:   # 如果全对，说明和父序列相同，跳过
            return False
        else:
            if count <= self.differ:        # 如果对不上的数量小于容许值则ok
                return True
            else:
                return False

    def simulate(self, sequence):
        return self.evaluate_sequence(sequence)

    def backpropagate(self, node, score):
        while node:
            node.visits += 1                # 增加节点的访问次数
            node.score += score             # 累加节点的得分
            node = node.parent              # 将当前节点更新为父节点，向上回溯

    def search(self, iterations, init: list):
        self.start = init                   # 设置初始序列
        loop = tqdm(total=iterations)       # 创建进度条
        self.history = []                   # 初始化历史记录列表
        self.x_history = []                 # 压缩历史记录
        self.root = self.Node(init)         # 初始化初始的self.root节点

        self.highest = 0                         # 最高分
        self.highest_sequence = None               # 最高分序列
        for i in range(iterations):
            node = self.root                # 将当前节点设置为根节点
            loop.update(1)                  # 更新进度条
            best_now = self.get_best_sequence(node)     # 获取当前最佳序列
            best_score = self.simulate(best_now)
            self.history.append(best_score)        #收集历史得分

            if best_score > self.highest:
                self.highest = best_score
                self.highest_sequence = best_now
                self.x_history.append([i, self.highest])


            node = self.search_branch(node)     # 更新节点

            if not node.visits and not node.score:      # 如果节点没有被访问过，且得分为0，表示该节点是刚刚创建的，没有模拟过
                node.score = self.evaluate_sequence(node.sequence)  # 给节点评分
                node.visits += 1            # 节点访问次数+1
                continue
            score = self.simulate(node.sequence)
            self.backpropagate(node, score) # 向回爬，parent路上的每个点都会更新这个得分
            self.root = node  # 更新self.root节点为当前搜索结束的节点
        loop.close()
        self.x_history = np.array(self.x_history)
    
    def search_branch(self, node:Node):
        done = False
        while not done:
            if not node.visits: # 如果当前节点没有被访问过
                done = True
                break
            if not node.children:   # 如果当前节点被访问过，但是没有子节点
                self.expand(node)
                done = True
                break
            node = self.select_child(node)  # 从探索过的枝杈中选择一个子节点
        return node

    def get_best_sequence(self, node=None):
        if node is None:
            node = self.root
        if len(node.children) == 0:
            return node.sequence
        else:
            best_child = None
            done = False
            while not done:
                if not node.children:
                    done = True
                    break
                for child in node.children:
                    if best_child is None or child.visits > best_child.visits:
                        best_child = child
                node = best_child
            return best_child.sequence
class explain:
    def __init__(self):
        self.aa_table = self._init_aa_table()
        self.aa_table_r = self._init_aa_table_r()
    def _init_aa_table_r(self):
        aa_table = {}
        aa_table[1] = "A"
        aa_table[2] = "C"
        aa_table[3] = "D"
        aa_table[4] = "E"
        aa_table[5] = "F"
        aa_table[6] = "G"
        aa_table[7] = "H"
        aa_table[8] = "I"
        aa_table[9] = "K"
        aa_table[10] = "L"
        aa_table[11] = "M"
        aa_table[12] = "N"
        aa_table[13] = "P"
        aa_table[14] = "Q"
        aa_table[15] = "R"
        aa_table[16] = "S"
        aa_table[17] = "T"
        aa_table[18] = "V"
        aa_table[19] = "W"
        aa_table[20] = "Y"
        return aa_table
    def _init_aa_table(self):
        aa_table = {}
        aa_table["A"] = 1
        aa_table["C"] = 2
        aa_table["D"] = 3
        aa_table["E"] = 4
        aa_table["F"] = 5
        aa_table["G"] = 6
        aa_table["H"] = 7
        aa_table["I"] = 8
        aa_table["K"] = 9
        aa_table["L"] = 10
        aa_table["M"] = 11
        aa_table["N"] = 12
        aa_table["P"] = 13
        aa_table["Q"] = 14
        aa_table["R"] = 15
        aa_table["S"] = 16
        aa_table["T"] = 17
        aa_table["V"] = 18
        aa_table["W"] = 19
        aa_table["Y"] = 20
        return aa_table
    def index2symbol(self, sequence:list):
        symbol = [self.aa_table_r[x] for x in sequence]
        return "".join(symbol)
    def symbol2index(self, sequence:str):
        index = [self.aa_table[x] for x in sequence]
        return index