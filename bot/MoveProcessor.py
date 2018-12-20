class MoveProcessor():
    """
    needs functions:
        make_returning_move()
        state_switch()
        interim_exploring_dest()
        get_step()
        get_shipyard()

    Needs variables:
        has_moved
        command_queue
    """
    def __init__(self, game, ship_obj, ship_dest, previous_state, ship_path):
        self.game = game
        self.game_map = game.game_map
        self.me = game.me
        self.ship_obj = ship_obj
        self.ship_dest = ship_dest
        self.previous_state = previous_state
        self.ship_path = ship_path



    def produce_move(self, ship):
        if ship.id not in ship_obj:
            self.ship_obj[ship.id] = ship
        state = ship_state[ship.id]
        destination = self.ship_dest[ship.id]
        ''' produces move for ship '''

        if ship.halite_amount < self.game_map[ship.position].halite_amount / 10:
            return Direction.Still


        mover = {
            "collecting": self.collecting,
            "returning": self.returning,
            "harakiri": self.harakiri,
            "assassinate": self.assassinate,
            "exploring": self.exploring,
            "build": self.exploring,
            "fleet": self.exploring,
            "backup": self.exploring,
        }

        return mover[state](ship, destination)


    def collecting(self, ship, destination):
        return Direction.Still


    def returning(self, ship, destination):
        return GF.make_returning_move(ship, has_moved, command_queue)


    def harakiri(self, ship, destination):
        shipyard = GF.get_shipyard(ship.position)
        ship_pos = self.game_map.normalize(ship.position)
        if ship.position == shipyard:  # if at shipyard
            return Direction.Still  # let other ships crash in to you
        else:  # otherwise move to the shipyard
            target_dir = self.game_map.get_target_direction(
                ship.position, shipyard)
            return target_dir[0] if target_dir[0] is not None else target_dir[1]


    def assassinate(self, ship, destination):
        GF.state_switch(ship.id, self.previous_state[ship.id])
        if self.game_map.calculate_distance(ship.position, destination) == 1:
            target_direction = self.game_map.get_target_direction(
                ship.position, destination)
            return target_direction[0] if target_direction[0] is not None else target_direction[1]
        else:
            return self.exploring(destination)


    def exploring(self, destination):
        # next direction occupied, recalculate
        if ship.id not in self.ship_path or not self.ship_path[ship.id]:
            self.ship_path[ship.id] = self.game_map.explore(ship, destination)
        else:
            direction = self.ship_path[ship.id][0][0]
            if self.game_map[ship.position.directional_offset(direction)].is_occupied and not direction == Direction.Still:
                if self.game_map.calculate_distance(destination, ship.position) > 10:
                    new_dest = GF.interim_exploring_dest(
                        ship.position, self.ship_path[ship.id])
                    # use intermediate unoccpied position instead of actual
                    # destination
                    self.ship_path[ship.id] = self.game_map.explore(
                        ship, new_dest) + self.ship_path[ship.id]
                else:
                    self.ship_path[ship.id] = self.game_map.explore(ship, destination)
        # move in calculated direction
        return GF.get_step(self.ship_path[ship.id])
