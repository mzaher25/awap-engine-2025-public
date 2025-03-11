from src.player import Player
from src.map import Map
from src.robot_controller import RobotController
from src.game_constants import Team, Tile, GameConstants, Direction, BuildingType, UnitType

from src.units import Unit
from src.buildings import Building
import numpy as np 
import random 

class BotPlayer(Player):
    def __init__(self, map: Map):
        self.map = map
        self.alpha = 0.4
        self.gamma = 0.7
        self.train = True
        self.epsilon = 0.1 if self.train else 0
        self.old_feat = np.array([0,0,0,0,0,0,0,0,0])
        self.feat = np.array([0,0,0,0,0,0,0,0,0])
        self.weights = np.array([1,1,1,1,1,1,1,1,1])
        self.turn = 0
        

    def play_turn(self, rc: RobotController):
        for i in range(len(self.weights)):
            if abs(self.weights[i]) < 0.001: self.weights[i]=0
        print("TURN",self.turn)
        # print("1")
        self.turn+=1
        #BUILDINGS/UNITS/etc etc etc .....
        place_x, place_y = None, None
        placeable_map = rc.get_building_placeable_map()
        for x in range(len(placeable_map)):
            for y in range(len(placeable_map[0])):
                if placeable_map[x][y] and rc.can_build_building(BuildingType.FARM_1,x,y): 
                    place_x, place_y = x, y
                    break
            if (place_x != None): break
        
        # places building
        bol = None
        spawnT = False
        if place_x != None and place_y != None:
            bol = rc.build_building(BuildingType.FARM_1, place_x, place_y)
            spawnT = True
            
        i = rc.get_building_ids(rc.get_ally_team())[0]
        

        team = rc.get_ally_team()
        ally_castle_id = -1

        ally_buildings = rc.get_buildings(team)
        for building in ally_buildings:
            if building.type == BuildingType.MAIN_CASTLE:
                ally_castle_id = rc.get_id_from_building(building)[1]
                break


         # if can spawn knight, spawn knight
        if rc.can_spawn_unit(UnitType.KNIGHT, ally_castle_id):
            rc.spawn_unit(UnitType.KNIGHT, ally_castle_id)
            rc.spawn_unit(UnitType.KNIGHT, ally_castle_id)
            rc.spawn_unit(UnitType.KNIGHT, ally_castle_id)
            rc.spawn_unit(UnitType.KNIGHT, ally_castle_id)
            rc.spawn_unit(UnitType.KNIGHT, ally_castle_id)

        # print("2")
        self.old_feat = self.feat
        
        #Knights:
        if len(rc.get_units(rc.get_ally_team())) != 0:
            b = False
            for i in rc.get_units(rc.get_ally_team()):
                best_action = self.act(rc,i)
                old_ul = len(rc.get_units(rc.get_enemy_team()))
                if old_ul != 0 and best_action!=None:
                    old_health = (best_action[0]).health
                    
                    un = best_action[0]

                    if best_action[1] == "A" :
                        rc.unit_attack_unit(rc.get_id_from_unit(best_action[0])[1],rc.get_id_from_unit(best_action[2])[1])
                        print("OOPS")
                    
                    elif best_action[1] == "B":
                        #ATTACK BUIDING ADD HERE 
                        rc.unit_attack_building(rc.get_id_from_unit(best_action[0])[1], rc.get_id_from_building(best_action[2])[1])
                        b = True


                    else:
                        rc.move_unit_in_direction(rc.get_id_from_unit(best_action[0])[1],best_action[3])
                    
                    #takes newly gained info and uses it to update the weights 
                    if self.train:
                        reward =(old_ul-len(rc.get_units(rc.get_enemy_team())))
                        
                        
                       
                        self.feat = self.get_feat(rc,best_action)
                        print("SELFNEW",self.feat)
                        print("SELFOLD",self.old_feat)
                        self.weight_update(reward,rc,un)
                    
        # print("3")
        return
        

    #evaluation fn 
    def get_feat(self,rc: RobotController, action):
        # takes in an action (unit id (that is doing the action), "M" (move)/"A" (attack unit)/"B" (attack building), unit we're attacking, direction (of move; None if attack))
        # simulates the next feauture based on the action 
        # NOTE: we're not acutally running the action, we're just simulating it!

        #TODO:  need to simulate action for **EVERYTHNIG**!!
        #MARY:
            #TODO:  predict a win/loss 
            #TODO: add a # of enemies feature (and alter that based on attack)
        # print("4")
        unit, choice, unit_of_attack, direction = action
        # print("UNIT", unit)
        if unit_of_attack != None:
            x = unit_of_attack.x
            y = unit_of_attack.y
        unit_team, unit_id = rc.get_id_from_unit(unit)
        
        if direction != None:
            loc_x,loc_y = rc.new_location(unit.x,unit.y,direction)

        # extrats game state information
        ally_team = rc.get_ally_team()
        ally_units = rc.get_units(ally_team) 
        # print("LIST OF UNITS", ally_units)

        ally_castles = len(rc.get_buildings(ally_team))
        ally_gold = rc.get_balance(ally_team)

        enemy_team = rc.get_enemy_team()
        enemy_buildings = rc.get_buildings(enemy_team)
        enemy_castles = len(rc.get_buildings(enemy_team))

        # print("5")
        # ally units stored in a dict w/ a list of tuples (unit id, unit health, unit position)
        ally_unit_dict = {unit.type: [] for unit in ally_units}
        for unit in ally_units:
            ally_unit_dict[unit.type].append((unit.id, unit.health, (unit.x, unit.y)))
        
        ally_num_units = len(rc.get_units(ally_team))

        # enemy units stored in a dict w/ a list of tuples (unit id, unit health, unit position)
        enemy_units = rc.get_units(enemy_team)
        enemy_unit_dict = {unit.type: [] for unit in enemy_units}
        for unit in enemy_units:
            enemy_unit_dict[unit.type].append((unit.id, unit.health, (unit.x, unit.y)))
        
        enemy_num_units = len(rc.get_units(enemy_team))

        # Max health values for unit types
        max_health_values = {
            "KNIGHT": 10,
            "WARRIOR": 10,
            "SWORDSMAN": 10,
            "DEFENDER": 15,
            "RAT": 5,
            "LAND_HEALER_1": 10,
            "LAND_HEALER_2": 10,
            "LAND_HEALER_3": 10
        }
        #average of health of units 
        ally_unit_health_normalized = [unit.health / max_health_values.get(unit.type, 1) for unit in ally_units]
        avg_ally_health = int(sum(ally_unit_health_normalized) / len(ally_unit_health_normalized)) if ally_unit_health_normalized else 0
        
        enemy_unit_health_normalized = [unit.health / max_health_values.get(unit.type, 1) for unit in enemy_units]
        avg_enemy_health = int(sum(enemy_unit_health_normalized) / len(enemy_unit_health_normalized)) if enemy_unit_health_normalized else 0

        # print("6")
        # predict win/loss:
        if choice == "A":
            if unit_of_attack.level > unit.level:
                health = unit.health 
                #changes the ally_unit_health depending on the damages dealt 
                avg_ally_health += -unit.health/max_health_values.get(unit.type,1) + (unit.health-unit_of_attack.damage)/max_health_values.get(unit.type,1)
                
                if unit.health - unit_of_attack.damage <=0:
                    ally_num_units -= 1
            else:
                enemy_health = unit_of_attack.health
                #changes the enemy_unit_health depending on the damages dealt 
                avg_enemy_health += -(unit_of_attack.health/max_health_values.get(unit_of_attack.type,1)) + (unit_of_attack.health-unit.damage)/max_health_values.get(unit_of_attack.type,1)
                if  enemy_health <=0:
                    enemy_num_units -= 1
        
        num_of_enemy_buildings = len(enemy_buildings)
        if choice == "B":
            #rewards them for attacking the building 
            num_of_enemy_buildings -= 1
        
        distance_to_closest_target = None
        new_loc_x = None
        new_loc_y = None
        #distance from knighft in question to target (for the knight in action)
        if len(ally_units)!=0 and len(enemy_units)!=0:
            if choice == "M":
                #new loc is : loc_x,loc_y
                #want to calculate distance b/t loc_x, loc_y and closest enemy
                #to find closest enemy, loop over all enemy units, find distance, then find the min
                new_loc_x = loc_x
                new_loc_y = loc_y

            else: #action == A or B
                new_loc_x = x
                new_loc_y = y
                
            distance_list = []
            for enemy in enemy_units:
                distance = rc.get_chebyshev_distance(enemy.x, enemy.y, new_loc_x, new_loc_y)
                distance_list.append(distance)
            distance_to_closest_target = min(distance_list)
            # closest_target = enemy_units[distance_list.index(min_dist)]

        elif len(enemy_units)== 0:
            # if no enemies, then calculate distane to closest building
            if choice == "M":
                #new loc after move : loc_x,loc_y
                #want to calculate distance b/t loc_x, loc_y and closest building
                new_loc_x = loc_x
                new_loc_y = loc_y
            else: #action == A or B
                new_loc_x = x
                new_loc_y = y
                
            distance_list = []
            for building in enemy_buildings:
                distance = rc.get_chebyshev_distance(building.x, building.y, new_loc_x, new_loc_y)
                distance_list.append(distance)
            distance_to_closest_target = min(distance_list)
        # print("7")

        #distance of selected unti to enemy castle
        if choice == "M":
            #new loc after move : loc_x,loc_y
            #want to calculate distance b/t loc_x, loc_y and closest building
            new_loc_x = loc_x
            new_loc_y = loc_y
        else: #action == A or B
            new_loc_x = x
            new_loc_y = y
        
        enemy_castle = None
        for eb in enemy_buildings:
            if eb.type == BuildingType.MAIN_CASTLE:
                enemy_castle = eb

        distance_to_enemy_castle = rc.get_chebyshev_distance(enemy_castle.x, enemy_castle.y, new_loc_x, new_loc_y)

        # of enemies near knight (when radius = 3)
        if (choice == "M"):
            nearbyUnits = len(rc.sense_units_within_radius(ally_team, loc_x, loc_y, 3))

        else:
            # unit = rc.get_unit_from_id(unit_id)
            nearbyUnits = len(rc.sense_buildings_within_radius(ally_team, unit.x, unit.y, 3))
        # print("8")
        # RETURN NUMPY ARRAY
        if (ally_castles == None):
            print("ally_castles")
        if (enemy_castles == None):
            print("enemy_castles")
        if (ally_gold == None):
            print("ally_gold")
        if (ally_num_units == None):
            print("ally_num_units")
        if (enemy_num_units == None):
            print("enemy_num_units")

        if (avg_ally_health == None):
            print("avg_ally_health")
        if (avg_enemy_health == None):
            print("avg_enemy_health")

        if (nearbyUnits == None):
            print("nearbyUnits")
        if (distance_to_closest_target == None):
            print("distance_to_closest_target")
        L = [ally_castles, -enemy_castles, ally_gold, ally_num_units, -enemy_num_units*(10), round(avg_ally_health*(0.2)), -(avg_enemy_health), nearbyUnits, -100*distance_to_closest_target] #num_of_enemy_buildings, distance_to_enemy_castle
        L_test = []
        for i in L:
            L_test.append(-1*i)
        print("L_TEST: ",L_test)
        return np.array(L_test)

    def q_value(self, features):
        return np.dot(self.weights, features)        
    
    # def rand_act(self,rc):
    #     for i in rc.get_units(rc.get_ally_team()):
    #         if i.type == UnitType.KNIGHT:
    #             r = len(rc.unit_possible_move_directions(rc.get_id_from_unit(i)))
    #             ind = random.randint(0,r)
    #             return (i,"M",rc.new_location((i.x,i.y),r[ind]))

        
    #Decides how/which unit moves 
    def act(self, rc, i):
       # if random.uniform(0,1) < self.epsilon:
        #   
        #  return self.rand_act(rc)
        if False: return
        else:
            # print("9")
            max_q = None
            best_action = None

        
            #finds any attack-able buildings and units, caculates q-vals 
            for u in rc.get_units(rc.get_enemy_team()):
                if rc.can_unit_attack_unit(rc.get_id_from_unit(i)[1], rc.get_id_from_unit(u)[1]):
                    print("BOOP")
                    q_val = self.q_value(self.get_feat(rc,(i,"A",u,None)))
                    if max_q == None or q_val > max_q: 
                            max_q = q_val
                            best_action = (i,"A",u,None)
            
            for u in rc.get_buildings(rc.get_enemy_team()):
                if rc.can_unit_attack_building(rc.get_id_from_unit(i)[1], rc.get_id_from_building(u)[1]):
                    print("OIEWIG")
                    q_val = self.q_value(self.get_feat(rc,(i,"B",u,None)))
                    if max_q == None or q_val > max_q: 
                            max_q = q_val
                            best_action = (i,"B",u,None)
            for j in rc.unit_possible_move_directions(rc.get_id_from_unit(i)[1]):
                x = i.x
                y = i.y
                if j == Direction.UP:
                    x = i.x 
                    y = i.y+1
                if j == Direction.DOWN:
                    x = i.x
                    y = i.y-1
                if j == Direction.LEFT:
                    x = i.x -1
                    y = i.y
                if j == Direction.RIGHT:
                    x = i.x + 1
                    y = i.y
                if j == Direction.UP_LEFT:
                    x = i.x - 1
                    y = i.y+1
                if j == Direction.UP_RIGHT:
                    x = i.x + 1
                    y = i.y+1
                if j == Direction.DOWN_LEFT:
                    x = i.x - 1
                    y = i.y-1
                if j == Direction.DOWN_RIGHT:
                    x = i.x + 1
                    y = i.y-1
                
                if rc.can_move_unit_in_direction(rc.get_id_from_unit(i)[1], j):
                    q_val = self.q_value((self.get_feat(rc,(i,"M",None,j))))
                    # print("HERET",q_val)
                    if max_q == None or q_val > max_q: 
                        max_q = q_val
                        best_action = (i,"M",None,j)

                # if rc.can_unit_attack_location(rc.get_id_from_unit(i)[1], x, y):
                #     q_val = self.q_value(self.get_feat(rc,(i,"A",(x,y),None)))
                #     print("HERE",q_val)
                #     if max_q == None or q_val > max_q: 
                #         max_q = q_val
                #         best_action = (i,"A",(x,y),None)
            # print("10")

            # print(best_action)
            return best_action

       
    #updates the weights after an action has been performed 
    def weight_update(self, reward,rc,unit):
        # print("11")
         #TODO: STORE CURRENT FEATURES SOMEPLACE AS WELL AS OLD FEATURES 
        #Calculates the Q-value prior to making the move
        old_q = self.q_value(self.old_feat)
        

        #Compares it with the Q-value after making the move 
        max_future_q = self.q_value(self.feat)#,(unit,None,None,None)))
        print("W",max_future_q,old_q)
      
        td_error = reward + self.gamma * (max_future_q - old_q)
        print(td_error)
        self.weights =self.weights + self.alpha*td_error*self.feat
        print(self.weights)
        # print("12")
