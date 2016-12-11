
import numpy
import pygame
from pygame.locals import *
import sys
import copy
from carbm import *
import math
from cadbn import *
pygame.init()

numpy.random.seed(0)

WIDTH, HEIGHT = 640, 640
SIZE = (WIDTH, HEIGHT)
screen = pygame.display.set_mode(SIZE)

CELL_N = (3, 3)
CELL_SIZE = (WIDTH/CELL_N[0], HEIGHT/CELL_N[1])


class Player(object):
    def __init__(self):
        pass

    def get_strategy(self, board):
        pass

    def win(self, pos, board):
        pass

    def lose(self, pos, board):
        pass


class CPU(Player):
    def get_strategy(self, board):
        return board.get_empty_random()


class AI(Player):
    def __init__(self):
        self._good_set = []
        self._good_teach_set = []
        self._bad_set = []
        self._bad_teach_set = []
        self.network = None
        self.pre_epochs = 1000
        self.rng = numpy.random.RandomState(1234)
        self.good_bookmark = 0
        self.bad_bookmark = 0
        self.threshold = -float("inf")

        super(AI, self).__init__()

    def field2data(self, field):
        data = []
        for f in field:
            if f == 0:
                data.extend([0, 0])
            elif f == 1:
                data.extend([0, 1])
            elif f == 10:
                data.extend([1, 0])
            else:
                print "error field!!"
        return data

    def act2data(self, act):
        data = [0] * 9
        data[act] = 1
        return data

    def revolution(self, field):
        result = [0] * 9
        # moved_list = [6, 3, 0, 7, 4, 1, 8, 5, 2]
        moved_list = [3, 0, 1, 6, 4, 2, 7, 8, 5]
        for pos in xrange(9):
            result[moved_list[pos]] = field[pos]
        return result

    def _train_network(self, network=None, _good_set=None, _good_teach=None, _bad_set=None, _bad_teach=None, bad_learning=True):
        if network is None:
            network = self.network
        if _good_set is None:
            _good_set = self._good_set
        if _good_teach is None:
            _good_teach = self._good_teach_set
        if _bad_set is None:
            _bad_set = self._bad_set
        if _bad_teach is None:
            _bad_teach = self._bad_teach_set

        mini_epoch = 100
        for epoch in xrange(1000/mini_epoch):
            network.reset_data(_good_set, _good_teach)
            network.finetune(lr=0.1, epochs=mini_epoch)
            if bad_learning:
                network.reset_data(_bad_set, _bad_teach)
                network.finetune(lr=-0.01, epochs=mini_epoch)

    def make_network(self):
        self.network = DDBN(input=numpy.array(self._good_set + self._bad_set),
                            label=numpy.array(self._good_teach_set + self._bad_teach_set),
                            n_ins=len(self._good_set[0]), hidden_layer_sizes=[1],
                            n_outs=len(self._good_teach_set[0]), numpy_rng=self.rng)
        self.network.pretrain(k=1, epochs=self.pre_epochs)
        self._train_network()
        self.threshold = self.network.rbm_layers[0].get_max_energy(self._good_set)
        print "threshold = ", self.threshold

    def remake_network(self):
        new_dataset, new_teachset = self.network.rbm_layers[0].observe_by_threshold(self._good_set, self._good_teach_set, self.threshold)
        if len(new_dataset) != 0:

            print "make network!"
            new_network = DDBN(input=numpy.array(new_dataset + self._bad_set),
                            label=numpy.array(new_teachset + self._bad_teach_set),
                            n_ins=len(new_dataset[0]), hidden_layer_sizes=[1],
                            n_outs=len(new_teachset[0]), numpy_rng=self.rng)

            new_network.pretrain(k=1, epochs=self.pre_epochs)
            # self._train_network(new_network, _good_set=new_dataset, _good_teach=new_teachset)
            self.network.synthesis(new_network)

        self._train_network(bad_learning=False)

        self.threshold = self.network.rbm_layers[0].get_max_energy(self._good_set)
        print "threshold = ", self.threshold


    def win(self, pos, board):
        set_buf = board.empty_list_history[-1]
        teach_buf = self.act2data(pos)
        for i in xrange(8):
            fdata = self.field2data(set_buf)
            if (fdata not in self._good_set[self.good_bookmark:]) \
                    and (teach_buf not in self._good_teach_set[self.good_bookmark:]):
                self._good_set.append(fdata)
                self._good_teach_set.append(teach_buf)
            set_buf = self.revolution(set_buf)
            teach_buf = self.revolution(teach_buf)

        # print "win"
        # print set_buf
        # print teach_buf


    def lose(self, pos, board):
        set_buf = board.empty_list_history[-2]
        teach_buf = self.act2data(pos)
        for i in xrange(8):
            fdata = self.field2data(set_buf)
            if (fdata not in self._bad_set[self.good_bookmark:]) \
                    and (teach_buf not in self._bad_teach_set[self.good_bookmark:]):
                self._bad_set.append(fdata)
                self._bad_teach_set.append(teach_buf)
            set_buf = self.revolution(set_buf)
            teach_buf = self.revolution(teach_buf)

        # print "lose"
        # print set_buf
        # print teach_buf

    def get_strategy(self, board):
        if self.network is None:
            return board.get_empty_random()

        eplist = []
        predict_result = self.network.predict(self.field2data(board.empty_list))
        result = -1
        """ probability """
        r_num = random.random()
        print r_num
        for i in xrange(len(predict_result)):
            if r_num < sum(predict_result[:i+1]):
                result = i
                break
        if board.empty_list[result] != 0:

            """ necessarry """
            for p, b in zip(self.network.predict(self.field2data(board.empty_list)), board.empty_list):
                eplist.append(p-b)
            result = numpy.argmax(eplist)

        if board.empty_list[result] == 0:
            return result
        print "error"
        return None



class MAN(Player):
    def get_strategy(self, board):
        input_line = raw_input()
        return int(input_line)


class Board(object):
    def __init__(self):
        self.empty_list = [0] * 9
        self.empty_list_history = []

    def get_empty_random(self):
        point = numpy.random.randint(0, 9)
        for i in xrange(9):
            if self.empty_list[point] == 0:
                return point
            point = (point + 1) % 9
        return None

    def get_empty_list(self):
        return copy.copy(self.empty_list_history[-1])

    def put(self, pos, var):
        if self.empty_list[pos] != 0:
            print "this position is not empty"
            return -1
        self.empty_list_history.append(copy.copy(self.empty_list))
        self.empty_list[pos] = var
        if (sum(self.empty_list[:3]) == 3*var or
            sum(self.empty_list[3:6]) == 3*var or
            sum(self.empty_list[6:]) == 3*var or
            sum([self.empty_list[n] for n in xrange(9) if (n % 3 == 0)]) == 3*var or
            sum([self.empty_list[n] for n in xrange(9) if (n % 3 == 1)]) == 3*var or
            sum([self.empty_list[n] for n in xrange(9) if (n % 3 == 2)]) == 3*var or
            (self.empty_list[0] + self.empty_list[4] + self.empty_list[8] == 3*var) or
            (self.empty_list[2] + self.empty_list[4] + self.empty_list[6] == 3*var)):
            return var
        return None


def clear_map():
    pygame.draw.rect(screen, (255, 255, 255), Rect(0, 0, WIDTH, HEIGHT))
    for i in xrange(CELL_N[0]):
        for j in xrange(CELL_N[1]):
            pygame.draw.rect(screen, (0, 0, 0), Rect(CELL_SIZE[0]*i, CELL_SIZE[1]*j,
                                                     CELL_SIZE[0], CELL_SIZE[1]), 1)
    pygame.display.update()


def game_init():
    game_reset()


def game_reset():
    clear_map()
    return Board()


def game():

    game_init()
    wait_time = 0
    scoreline = [0, 0, 0]
    players = [AI(), CPU()]
    phase = 0

    while 1:
        board = game_reset()
        end_flag = 0

        pos_hist = []
        for turn in xrange(9):

            for no, player in enumerate(players):
                if turn % 2 == no:
                    continue
                pos = player.get_strategy(board)
                result = board.put(pos, 10**no)
                while result == -1:
                    pos = player.get_strategy(board)
                    result = board.put(pos, 10**no)
                pos_hist.append(pos)

                color = [0, 0, 0]
                color[no] = 255
                pygame.draw.rect(screen, color, Rect(CELL_SIZE[0] * (pos % 3), CELL_SIZE[1] * (pos / 3),
                                                         CELL_SIZE[0], CELL_SIZE[1]))
                pygame.display.update()

                current_field = players[0].field2data(board.empty_list_history[-1])
                if result is not None:
                    # print "Player %d win" % no
                    scoreline[no] += 1
                    players[no].win(pos, board)
                    players[(no+1) % 2].lose(pos_hist[-2], board)
                    end_flag = 1
                elif current_field in players[0]._good_set:
                    #print "A",
                    players[0].win(pos, board)
                    players[1].lose(pos_hist[-2], board)
                elif current_field in players[0]._bad_set:
                    #print "B",
                    if len(pos_hist) >= 2:
                        players[1].win(pos, board)
                        players[0].lose(pos_hist[-2], board)

            for event in pygame.event.get():
                if event.type == QUIT:
                    print len(players[0]._good_set), len(players[0]._bad_set)
                    print "good ", players[0]._good_set
                    print "good teach", players[0]._good_teach_set
                    print "bad  ", players[0]._bad_set
                    print "bad  teach", players[0]._bad_teach_set
                    sys.exit()

            if event.type == MOUSEBUTTONDOWN and event.button == 1:
                wait_time *= 2
                if wait_time < 1:
                    wait_time = 1
            if event.type == MOUSEBUTTONDOWN and event.button == 3:
                wait_time /= 2
                if wait_time < 1:
                    wait_time = 0

            pygame.time.wait(wait_time)
            if end_flag == 1:
                break

        if (end_flag == 0):
            # print "Draw..."
            scoreline[-1] += 1



        # print "turn : ", phase
        phase += 1
        if phase % 10 == 0:
            print "Scoreline : ", scoreline
            players[0].good_bookmark += scoreline[0]
            players[0].bad_bookmark += scoreline[1]
            scoreline = [0, 0, 0]
            if players[0].network is None:
                print "make network!"
                players[0].make_network()
            else:
                players[0].remake_network()

            pygame.time.wait(3000)


        #end of while






if __name__ == "__main__":
    game()