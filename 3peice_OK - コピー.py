
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

    def make_network(self):
        self.network = DDBN(input=numpy.array(self._good_set + self._bad_set),
                            label=numpy.array(self._good_teach_set + self._bad_teach_set),
                            n_ins=len(self._good_set[0]), hidden_layer_sizes=[1],
                            n_outs=len(self._good_teach_set[0]), numpy_rng=self.rng)
        self.network.pretrain(k=1, epochs=self.pre_epochs)
        mini_epoch = 1
        for epoch in xrange(1000/mini_epoch):
            self.network.x = self._good_set
            self.network.y = self._good_teach_set
            self.network.finetune(lr=0.1, epochs=mini_epoch)
            self.network.x = self._bad_set
            self.network.y = self._bad_teach_set
            self.network.finetune(lr=-0.1, epochs=mini_epoch)

    def win(self, pos, board):
        set_buf = board.get_empty_list()
        teach_buf = self.act2data(pos)
        for i in xrange(8):
            fdata = self.field2data(set_buf)
            self._good_set.append(fdata)
            self._good_teach_set.append(teach_buf)
            set_buf = self.revolution(set_buf)
            teach_buf = self.revolution(teach_buf)

    def lose(self, pos, board):
        self._bad_set.append(self.field2data(board.get_empty_list()))
        self._bad_teach_set.append(self.act2data(pos))

    def get_strategy(self, board):
        if self.network is None:
            return board.get_empty_random()

        eplist = []
        for p, b in zip(self.network.predict(self.field2data(board.empty_list)), board.empty_list):
            eplist.append(p-b)
        result = numpy.argmax(eplist)
        # print self.network.predict(self.field2data(board.empty_list))
        # print board.empty_list
        # print eplist
        # print result
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

        for turn in xrange(9):

            for no, player in enumerate(players):
                if (turn + 1) % 2 == no:
                    continue
                pos = player.get_strategy(board)
                result = board.put(pos, 10**no)
                while result == -1:
                    pos = player.get_strategy(board)
                    result = board.put(pos, 10**no)

                color = [0, 0, 0]
                color[no] = 255
                pygame.draw.rect(screen, color, Rect(CELL_SIZE[0] * (pos % 3), CELL_SIZE[1] * (pos / 3),
                                                         CELL_SIZE[0], CELL_SIZE[1]))
                pygame.display.update()

                if result is not None:
                    print "Player %d win" % no
                    scoreline[no] += 1
                    players[no].win(pos, board)
                    players[(no+1) % 2].lose(pos, board)
                    end_flag = 1

            for event in pygame.event.get():
                if event.type == QUIT:
                    print len(players[0]._good_set), len(players[0]._bad_set)
                    print "good ", len(players[0]._good_set)
                    print "good teach", len(players[0]._good_teach_set)
                    print "bad  ", len(players[0]._bad_set)
                    print "bad  teach", len(players[0]._bad_teach_set)
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
            print "Draw..."
            scoreline[-1] += 1

        print "Scoreline : ", scoreline

        print "turn : ", phase
        phase += 1
        if phase % 1000 == 0:
            print "make network!"
            scoreline = [0, 0, 0]
            players[0].make_network()



        #end of while






if __name__ == "__main__":
    game()