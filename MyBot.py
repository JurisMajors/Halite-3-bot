#!/usr/bin/env python3
# Python 3.6

# Import the Halite SDK, which will let you interact with the game.
import hlt

# This library contains constant values.
from hlt import constants

# This library contains direction metadata to better interface with the game.
from hlt.positionals import Direction, Position
#heap
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
game.ready("Juris-Bot-v2")

# Now that your bot is initialized, save a message to yourself in the log file with some important information.
#   Here, you log here your id, which you can always fetch from the game object by using my_id.
logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))
""" <<<Game Loop>>> """
ship_state = {}
ship_dest = {} # destination -> ship.id
halite_positions = {} # halite -> position
previous_position = {} # ship.id-> previous pos
#search area for halite relative to shipyard
size = 30

def f(h_amount, h_distance): # function for determining patch priority
    return h_amount/(2*h_distance + 1)

while True:
    game.update_frame()    
    me = game.me
    game_map = game.game_map

    command_queue = []
    h = [] # stores halite amount * -1 with its position in a minheap
    top_left = Position(-15 , -15) + me.shipyard.position # top left of scan area
    for y in range(size):
        for x in range(size):
            p = Position((top_left.x + x) % game_map.width, (top_left.y + y) % game_map.height) # position of patch
            factor = f(game_map[p].halite_amount * -1, game_map.calculate_distance(p, me.shipyard.position)) # f(negative halite amount,  distance from shipyard to patch)
            halite_positions[factor] = p
            heappush(h, factor) # add negative halite amounts so that would act as maxheap
    ships = [] # ship priority queue
    for s in me.get_ships(): 
        if s.id in ship_state:
            # importance, the lower the number, bigger importance
            if ship_state[s.id] == "returning":
                importance = game_map.calculate_distance(s.position, me.shipyard.position) / (game_map.width * 2) # 0,1 range 
            elif ship_state[s.id] == "exploring":
                importance = game_map.calculate_distance(s.position, me.shipyard.position) # normal distance
            else:# collecting
                importance  = game_map.calculate_distance(s.position, me.shipyard.position) * game_map.width * 2 # normal distance * X since processing last
        else:
            importance = 0 # newly spawned ships max importance
        heappush(ships, (importance, s))


    while not len(ships) == 0:
        ship = heappop(ships)[1]
        if ship.id not in previous_position:
            previous_position[ship.id] = me.shipyard.position
        find_new_dest = False
        possible_moves = []

        # setup state
        if ship.id not in ship_dest: # if ship hasnt received a destination yet
            biggest_halite = heappop(h) # get biggest halite
            while halite_positions[biggest_halite] in ship_dest.values(): # get biggest halite while its a position no other ship goes to
                biggest_halite = heappop(h)
            ship_dest[ship.id] = halite_positions[biggest_halite] # set the destination
            ship_state[ship.id] = "exploring" # explore
        

        # transition
        if ship_state[ship.id] == "exploring" and (ship.position == ship_dest[ship.id] or game_map[ship.position].halite_amount > 300) :
            # collect if reached destination or on medium sized patch
            ship_state[ship.id] = "collecting"          
        elif ship_state[ship.id] == "exploring" and ship.halite_amount >= constants.MAX_HALITE*0.7:
            # return if ship is 70+% full
            ship_state[ship.id] = "returning" 
            ship_dest[ship.id] = me.shipyard.position
        elif ship_state[ship.id] == "collecting" and (game_map[ship.position].halite_amount < 10 or ship.halite_amount >= constants.MAX_HALITE*0.7): # return to shipyard if enough halite
            # return if patch has little halite or ship is 70% full
            ship_state[ship.id] = "returning"
            ship_dest[ship.id] = me.shipyard.position 
            
        elif ship_state[ship.id] == "returning" and ship.position == ship_dest[ship.id]:
            # explore again when back in shipyard
            ship_state[ship.id] = "exploring"
            find_new_dest = True


        # find new destination for exploring shop
        if find_new_dest:
            biggest_halite = heappop(h) # get biggest halite
            while halite_positions[biggest_halite] in ship_dest.values(): # get biggest halite while its a position no other ship goes to
                biggest_halite = heappop(h)
            find_new_dest = True
            ship_dest[ship.id] = halite_positions[biggest_halite] # set the destination


        logging.info("ship:{} , state:{} ".format(ship.id, ship_state[ship.id]))
        logging.info("destination: {}, {} ".format(ship_dest[ship.id].x, ship_dest[ship.id].y))

        # clear dictionaries of crushed ships
        for ship_id in list(ship_dest.keys()):
            if not me.has_ship(ship_id):
                del ship_dest[ship_id]
                del ship_state[ship_id]


        # make move
        if ship_state[ship.id] == "exploring": # if exploring move to its destinition in ship_dest dictionary
            move = game_map.smart_navigate(previous_position[ship.id], ship, ship_dest[ship.id])
            command_queue.append(ship.move(move))
            
        elif ship_state[ship.id] == "returning": # if returning
            move = game_map.smart_navigate(previous_position[ship.id], ship, ship_dest[ship.id])
            command_queue.append(ship.move(move))
            
        elif ship_state[ship.id] == "collecting": 
            move = Direction.Still # collect
            command_queue.append(ship.move(move))

        previous_position[ship.id] = ship.position
        
    # check if shipyard is surrounded by ships
    shipyard_surrounded = True
    for direction in Direction.get_all_cardinals():
        position = me.shipyard.position.directional_offset(direction)
        if not game_map[me.shipyard.position.directional_offset(direction)].is_occupied:
            shipyard_surrounded = False
            break

    if game.turn_number <= 150 and me.halite_amount >= constants.SHIP_COST and not (game_map[me.shipyard].is_occupied or shipyard_surrounded):
        command_queue.append(me.shipyard.spawn())

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)


