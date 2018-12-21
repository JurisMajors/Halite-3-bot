class DestinationProcessor():
    """
    Contains functions:

    Needs access to dicts:
        ship_dest
        ship_state

    Needs access to functions:
        get_shipyard
        amount_of_enemies
        halite_priority_q
        get_dropoff_positions
        prcntg_ships_returning_to_doff()

    """
    def __init__(self, game):
        self.game = game
        self.game_map = game.game_map


    def find_new_destination(self, h, ship):
        ''' h: priority queue of halite factors,
                                        halite_pos: dictionary of halite factor -> patch position '''
        ship_id = ship.id
        biggest_halite, position = heappop(h)  # get biggest halite
        destination = self.game_map.normalize(position)
        not_first_dest = ship_id in ship_dest
        # get biggest halite while its a position no other ship goes to
        while not self.dest_viable(destination, ship) or GF.amount_of_enemies(destination, 4) >= 4 \
                or self.too_many_near_dropoff(ship, destination) \
                or (not_first_dest and destination == ship_dest[ship_id]):
            if len(h) == 0:
                logging.info("ran out of options")
                return
            biggest_halite, position = heappop(h)
            destination = self.game_map.normalize(position)

        ship_dest[ship_id] = destination  # set the destination
        # if another ship had the same destination
        s = self.get_ship_w_destination(destination, ship_id)
        if s is not None:  # find a new destination for it
            self.process_new_destination(s)


    def process_new_destination(self, ship):
        ship_path[ship.id] = []
        if 0 < self.game_map.calculate_distance(ship.position, ship_dest[ship.id]) <= GC.SHIP_SCAN_AREA:
            source = ship.position
        else:
            source = ship_dest[ship.id]

        ship_h = GF.halite_priority_q(source, GC.SHIP_SCAN_AREA)
        self.find_new_destination(ship_h, ship)


    def dest_viable(self, position, ship):
        if position in ship_dest.values():
            inspectable_ship = self.get_ship_w_destination(position, ship.id)
            if inspectable_ship is None:
                # if this ship doesnt exist for some reason
                return True

            my_dist = self.game_map.calculate_distance(position, ship.position)
            their_dist = self.game_map.calculate_distance(
                position, inspectable_ship.position)

            return my_dist < their_dist
        else:
            return True  # nobody has the best patch, all good


    def too_many_near_dropoff(self, ship, destination):
        if GF.get_shipyard(ship.position) == GF.get_shipyard(destination):
            return False
        else:
            return GF.prcntg_ships_returning_to_doff(GF.get_shipyard(destination)) > (1 / len(GF.get_dropoff_positions()))


    @staticmethod
    def get_ship_w_destination(dest, this_id):
        if dest in ship_dest.values():
            for s in ship_dest.keys():  # get ship with the same destination
                if not s == this_id and ship_state[s] == "exploring" and ship_dest[s] == dest and me.has_ship(s):
                    return me.get_ship(s)
        return None








