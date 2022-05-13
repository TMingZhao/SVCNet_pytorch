

'''测试过程参数'''
iteration = 3
repeatability = 3

dim = 128


class Denoise():
    def __init__(self):
        self.log = ""
        self.dim = dim
        self.mini_point = 32
        self.up_ratio = 1

        self.nhead = 4
        self.dropout = 0.1
        self.num_encoder = 1
        self.num_decoder = 1

        self.num_parallel = 2

        # 边缘约束超参
        self.edge_loss_weight = 0.20
        self.edge_thres = 0.15
    
    def set_values(self, log="", num_encoder=1, num_decoder=1, num_parallel=2, mini_point=32, edge_loss_weight=0.20, edge_thres=0.15):
        self.log = log
        self.num_encoder = num_encoder
        self.num_decoder = num_decoder
        self.num_parallel = num_parallel
        self.mini_point = mini_point
        self.edge_loss_weight = edge_loss_weight
        self.edge_thres = edge_thres

    def __str__(self) -> str:
        str_ = ""
        for key, value in self.__dict__.items():
            if key == "log":
                continue
            str_ = str_ + "{}={}, ".format(key, value)
        return str_

DENOISE = Denoise()


class Ablation():
    def __init__(self):
        # 加入边缘约束
        self.edge_constraint = True

    def set_flags(self, edge_constraint = False):
        self.edge_constraint = edge_constraint


    def __str__(self) -> str:
        str_ = ""
        for key, value in self.__dict__.items():
            if value == True:
                str_ = str_ + "{}={}, ".format(key, value)
        return str_

ABLATION = Ablation()
