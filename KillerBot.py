#!/usr/bin/env python3
# Python 3.6

# Import the Halite SDK, which will let you interact with the game.
import hlt
import sys

# This library contains constant values.
from hlt import constants

# This library contains direction metadata to better interface with the game.
from hlt.positionals import Direction, Position
# heap
from heapq import heappush, heappop
# This library allows you to generate random numbers.
import random

# Logging allows you to save messages for yourself. This is required because the regular STDOUT
#   (print statements) are reserved for the engine-bot communication.
import logging

""" <<<Game Begin>>> """

# This game object contains the initial game state.
game = hlt.Game()
# At this point "game" variable is populated with initial map data.
# This is a good place to do computationally expensive start-up pre-processing.
# As soon as you call "ready" function below, the 2 second per turn timer will start.
game.ready("Sea-Whackers")

# Now that your bot is initialized, save a message to yourself in the log file with some important information.
#   Here, you log here your id, which you can always fetch from the game object by using my_id.
logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))
""" <<<Game Loop>>> """

enemy_shipyard = Position(0, 0)

i = 0
while True:
    game.update_frame()
    me = game.me
    game_map = game.game_map

    if i == 0:
        logging.info("OG Type {}".format(type(hlt.entity.Shipyard)))
        for i in range(game.game_map.height):
            for j in range(game.game_map.width):
                logging.info("Type {}".format(game.game_map[Position(j, i)].structure))
                if type(game.game_map[Position(j, i)].structure) is hlt.entity.Shipyard and \
                        game.me.shipyard.position != Position(j, i):
                    enemy_shipyard = Position(j, i)
        i+=1
        logging.info("Shipyard pos {} ".format(enemy_shipyard))

    command_queue = []

    for s in me.get_ships():
        if s.position == enemy_shipyard:
            command_queue.append(s.move(Direction.Still))
        else:
            dir = game_map.get_target_direction(s.position, enemy_shipyard)
            new_dir = dir[0] if dir[0] is not None else dir[1]
            command_queue.append(s.move(new_dir))

    if len(me.get_ships()) < 1:
        command_queue.append(me.shipyard.spawn())

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)
