class Player:
    def __init__(self, player_id):
        self.player_id = player_id

        # Initialize player's abilities
        self.keys = [1, 2, 2, 3, 4]
        self.privilegium = ['White', 'Orange', 'Pink', 'Black']
        self.max_pieces_movable = [2, 3, 4, 5]  # represents the maximum total number of pieces a player can move
        self.actions = [2, 3, 3, 4, 4, 5]
        self.bags = [3, 5, 7, 'E']

        # Initialize the index for each ability
        self.keys_index = 0
        self.privilegium_index = 0
        self.max_pieces_movable_index = 0
        self.actions_index = 0
        self.bags_index = 0

        # Initialize player's score
        self.score = 0

        # Initialize player's supply
        self.personal_supply = {'cubes': 5 + player_id, 'circles': 1}
        self.general_stock = {'cubes': 7 - player_id}
        
    def draw_from_general_stock(self, num_cubes=0, num_circles=0):
        if num_cubes + num_circles > self.bags[self.bags_index]:
            print('Trying to draw more items than allowed by bag capacity.')
            return

        actual_num_cubes = min(num_cubes, self.general_stock.get('cubes', 0))
        actual_num_circles = min(num_circles, self.general_stock.get('circles', 0))

        if actual_num_cubes > 0:
            self.general_stock['cubes'] -= actual_num_cubes
            self.personal_supply['cubes'] += actual_num_cubes

        if actual_num_circles > 0:
            self.general_stock['circles'] -= actual_num_circles
            self.personal_supply['circles'] += actual_num_circles
            
    # method to upgrade an ability
    def upgrade_ability(self, ability):
        if ability == "keys":
            if self.keys_index < len(self.keys) - 1:
                self.keys_index += 1
                self.personal_supply['cubes'] += 1  # Player gets another cube when ability is upgraded
        elif ability == "privilegium":
            if self.privilegium_index < len(self.privilegium) - 1:
                self.privilegium_index += 1
                self.personal_supply['cubes'] += 1  # Player gets another cube when ability is upgraded
        elif ability == "max_pieces_movable":
            if self.max_pieces_movable_index < len(self.max_pieces_movable) - 1:
                self.max_pieces_movable_index += 1
                self.personal_supply['circles'] += 1  # Player gets another circle when circles ability is upgraded
        elif ability == "actions":
            if self.actions_index < len(self.actions) - 1:
                self.actions_index += 1
                self.personal_supply['cubes'] += 1  # Player gets another cube when ability is upgraded
        elif ability == "bags":
            if self.bags_index < len(self.bags) - 1:
                self.bags_index += 1
                self.personal_supply['cubes'] += 1 # Player gets another cube when ability is upgraded
                