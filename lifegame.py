
import numpy
from bp import *
import pygame
from pygame.locals import *
import sys
import copy
pygame.init()

STAT_WALL = -1
STAT_EMP = 0
STAT_RED = 1
STAT_GREEN = 2



width, height, parm_h = 640, 480, 480
size = width, (height + parm_h)
screen = pygame.display.set_mode(size)

class Field(object):

    direct_sets = [(a, b) for a in xrange(-1, 2) for b in xrange(-1, 2)]
    direct_sets.remove((0, 0))
    direct_sets = numpy.array(direct_sets)

    def __init__(self, view_distance_max=100):
        self.field = numpy.zeros((width, height))
        self.view_distance_max = view_distance_max

    def get_view(self, pos):
        res_view = numpy.zeros(len(Field.direct_sets) * 3)
        for direct_no, direct in enumerate(Field.direct_sets):
            for distance in xrange(self.view_distance_max):
                n_pos = pos +(distance * direct)
                try:
                    pos_stat = self.field[n_pos[0]][n_pos[1]]
                except:
                    print "exception!"
                    pos_stat = STAT_WALL
                if pos_stat == STAT_WALL:
                    res_view[direct_no * 3 + 0] = (1. - (float(distance) / self.view_distance_max))
                    break
                elif pos_stat == STAT_RED:
                    res_view[direct_no * 3 + 1] = (1. - (float(distance) / self.view_distance_max))
                    break
                elif pos_stat == STAT_GREEN:
                    res_view[direct_no * 3 + 2] = (1. - (float(distance) / self.view_distance_max))
                    break

        return res_view


class Life(object):
    def __init__(self, name="red.png", field=None, nn=None):
        self.name = name
        self.fig = pygame.image.load(name)
        self.char_rect = self.fig.get_rect()
        self.speed = [2, 2]
        initial_pos = numpy.random.rand(2) * [width, height]
        self.char_rect = self.char_rect.move(initial_pos)
        self.field = field
        self.field.field[initial_pos[0]][initial_pos[1]] = (STAT_RED if self.name == "red.png" else STAT_GREEN)
        self.nn = copy.deepcopy(nn)

    def get_pos(self):
        pos = ((self.char_rect[0] + self.char_rect[1]) / 2, (self.char_rect[2] + self.char_rect[3]) / 2)
        return pos

    def move(self):

        size_of_collision = 10

        # before check
        n_pos = self.get_pos()
        pygame.draw.rect(screen, (0, 0, 0), self.char_rect)
        for dist in xrange(size_of_collision):
            for direct in Field.direct_sets:
                self.field.field[n_pos[0]+direct[0]*dist][n_pos[1]+direct[1]*dist] = STAT_EMP
        if self.name == "green.png" and self.field.field[n_pos[0]][n_pos[1]] == STAT_RED:
            del(self)
            return -1

        self.char_rect = self.char_rect.move(self.speed)
        if self.char_rect.left < 0 or self.char_rect.right > width:
            del(self)
            return -1
        if self.char_rect.top < 0 or self.char_rect.bottom > height:
            del(self)
            return -1

        n_pos = self.get_pos()

        # after check
        if self.field.field[n_pos[0]][n_pos[1]] != STAT_EMP:
            if self.field.field[n_pos[0]][n_pos[1]] == STAT_WALL:
                del(self)
                return -1
            if self.name == "red.png":
                if self.field.field[n_pos[0]][n_pos[1]] == STAT_RED:
                    self.char_rect = self.char_rect.move((-self.speed[0], -self.speed[1]))
                elif self.field.field[n_pos[0]][n_pos[1]] == STAT_GREEN:
                    pass
            elif self.name == "green.png":
                if self.field.field[n_pos[0]][n_pos[1]] == STAT_RED:
                    del(self)
                    return -1
                elif self.field.field[n_pos[0]][n_pos[1]] == STAT_GREEN:
                    self.char_rect = self.char_rect.move((-self.speed[0], -self.speed[1]))

        screen.blit(self.fig, self.char_rect)
        n_pos = self.get_pos()
        for dist in xrange(size_of_collision):
            for direct in Field.direct_sets:
                self.field.field[n_pos[0]+direct[0]*dist][n_pos[1]+direct[1]*dist] = STAT_RED if self.name == "red.png" else STAT_GREEN

    def strategy(self):

        if self.field is None: return

        self.nn.propagate(self.field.get_view(self.get_pos()))

        if numpy.random.random() < 0.3:
            self.speed[0] = -self.speed[0]
        if numpy.random.random() < 0.3:
            self.speed[1] = -self.speed[1]


def game():

    field = Field()
    red_nn = Network(24, 5, 1, eta=0.05)
    green_nn = Network(24, 5, 1, eta=0.05)

    chars = []
    chars.extend([Life("red.png", field, red_nn) for i in range(10)])
    chars.extend([Life("green.png", field, green_nn) for i in range(10)])

    while 1:

        for char in chars:
            char.strategy()
            if char.move() == -1:
                for epoch in xrange(10):
                    green_nn.train([field.get_view(char.get_pos())], [-1])
                t_name = char.name
                chars[chars.index(char)] = (Life(t_name, field, green_nn))


        # print graph
        for char_n, char in enumerate(chars[:10]):
            pygame.draw.line(screen, (0, 0, 0),
                             (20+char_n*60, height+10), (20+char_n*60, height+100*2))
            pygame.draw.line(screen, (255, 255, 255),
                             (20+char_n*60, height+10), (20+char_n*60, height+100*(char.nn.output+1)))
            for line_n, line in enumerate(char.nn.hidden_layer[:-1]):
                pygame.draw.line(screen, (0, 0, 0),
                                 (20+char_n*60 + (line_n+1)*3, height+10), (20+char_n*60 + (line_n+1)*3, height+100*2))
                pygame.draw.line(screen, (0, 255, 0),
                                 (20+char_n*60 + (line_n+1)*3, height+10), (20+char_n*60 + (line_n+1)*3, height+100*(line+1)))
        for char_n, char in enumerate(chars[10:]):
            pygame.draw.line(screen, (0, 0, 0),
                             (20+char_n*60, height+parm_h/2+10), (20+char_n*60, height+parm_h/2+100*2))
            pygame.draw.line(screen, (255, 255, 255),
                             (20+char_n*60, height+parm_h/2+10), (20+char_n*60, height+parm_h/2+100*(char.nn.output+1)))
            for line_n, line in enumerate(char.nn.hidden_layer[:-1]):
                pygame.draw.line(screen, (0, 0, 0),
                                 (20+char_n*60 + (line_n+1)*3, height+parm_h/2+10),
                                 (20+char_n*60 + (line_n+1)*3, height+parm_h/2+100*2))
                pygame.draw.line(screen, (255, 0, 0),
                                 (20+char_n*60 + (line_n+1)*3, height+parm_h/2+10),
                                 (20+char_n*60 + (line_n+1)*3, height+parm_h/2+100*(line+1)))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == QUIT:
                exit()

        pygame.time.wait(50)

if __name__ == "__main__":
    game()