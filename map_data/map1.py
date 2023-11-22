# map1.py
from map_data.map_attributes import Map, City, Upgrade, Office, Route
from map_data.constants import BLACKISH_BROWN, CIRCLE_RADIUS, SPACING, DARK_RED

class Map1(Map):
    def __init__(self, num_players):
        super().__init__()  # Call the parent class constructor
        self.cities = []
        self.routes = []
        self.upgrade_cities = []
        self.east_west_cities = ['Stendal', 'Arnheim']
        self.specialprestigepoints = None
        self.max_full_cities = 10
        self.max_full_cities_x_pos = 77
        self.max_full_cities_y_pos = 52

        self.create_cities_and_routes(num_players)  # Populate cities, offices, and routes etc., specifically for Map1
        self.assign_starting_bonus_markers()

        self.map_width = 1800
        self.map_height = 1255

        #keep the cities in alphabetical order - helps when searching
        self.bonus_marker_positions = {
            ('Emden', 'Groningen'): (339, 159),
            ('Emden', 'Osnabruck'): (454, 356),
            ('Kampen', 'Osnabruck'): (336, 439),
            ('Arnheim', 'Kampen'): (208, 558),
            ('Arnheim', 'Duisburg'): (110, 832),
            ('Arnheim', 'Munster'): (339, 758),
            ('Dortmund', 'Duisburg'): (338, 880),
            ('Bremen', 'Osnabruck'): (665, 382),
            ('Minden', 'Munster'): (657, 626),
            ('Bremen', 'Minden'): (802, 483),
            ('Minden', 'Paderborn'): (774, 789),
            ('Dortmund', 'Paderborn'): (632, 862),
            ('Coellen', 'Warburg'): (420, 1113),
            ('Paderborn', 'Warburg'): (678, 1052),
            ('Hamburg', 'Stade'): (1125, 70),
            ('Bremen', 'Hamburg'): (1125, 244),
            ('Hannover', 'Minden'): (974, 530),
            ('Bremen', 'Hannover'): (1047, 377),
            ('Hildesheim', 'Paderborn'): (988, 907),
            ('Hannover', 'Luneburg'): (1225, 401),
            ('Brunswick', 'Minden'): (1041, 694),
            ('Hamburg', 'Lubeck'): (1484, 78),
            ('Hamburg', 'Luneburg'): (1404, 272),
            ('Luneburg', 'Perleberg'): (1507, 430),
            ('Perleberg', 'Stendal'): (1717, 499),
            ('Brunswick', 'Stendal'): (1476, 605),
            ('Magdeburg', 'Stendal'): (1719, 821),
            ('Goslar', 'Magdeburg'): (1537, 830),
            ('Goslar', 'Quedlinburg'): (1335, 996),
            ('Goslar', 'Hildesheim'): (1243, 861),
            ('Halle', 'Quedlinburg'): (1519, 1083),
            ('Gottingen', 'Quedlinburg'): (1194, 1066),
        }
        if num_players != 3:
            self.bonus_marker_positions[('Emden', 'Stade')] = (723, 127)
            self.bonus_marker_positions[('Gottingen', 'Warburg')] = (869, 1084)

    def create_cities_and_routes(self, num_players):
        # Define Map1-specific cities and offices
        # Define cities
        Groningen = City('Groningen', 115, 210, BLACKISH_BROWN)
        Groningen.add_office(Office("square", "WHITE", 1))
        Groningen.add_office(Office("circle", "ORANGE", 0))
        Groningen.assign_upgrade_type('Book')
        self.cities.append(Groningen)

        Emden = City('Emden', 476, 182, BLACKISH_BROWN)
        Emden.add_office(Office("circle", "WHITE", 0))
        Emden.add_office(Office("square", "PINK", 0))
        self.cities.append(Emden)

        Osnabruck = City('Osnabruck', 434, 537, BLACKISH_BROWN)
        Osnabruck.add_office(Office("square", "WHITE", 0))
        Osnabruck.add_office(Office("square", "ORANGE", 0))
        Osnabruck.add_office(Office("square", "BLACK", 0))
        self.cities.append(Osnabruck)

        Kampen = City('Kampen', 107, 385, BLACKISH_BROWN)
        if num_players == 3:
            Kampen.add_office(Office("square", "ORANGE", 0))
            Kampen.add_office(Office("square", "BLACK", 0))
        else:
            Kampen.add_office(Office("circle", "ORANGE", 0))
            Kampen.add_office(Office("square", "BLACK", 0))
        self.cities.append(Kampen)

        Arnheim = City('Arnheim', 65, 655, DARK_RED)
        Arnheim.add_office(Office("square", "WHITE", 0))
        Arnheim.add_office(Office("circle", "WHITE", 0))
        Arnheim.add_office(Office("square", "ORANGE", 0))
        Arnheim.add_office(Office("square", "PINK", 0))
        self.cities.append(Arnheim)

        Duisburg = City('Duisburg', 179, 911, BLACKISH_BROWN)
        Duisburg.add_office(Office("square", "WHITE", 0))
        self.cities.append(Duisburg)

        Dortmund = City('Dortmund', 432, 894, BLACKISH_BROWN)
        if num_players == 3:
            Dortmund.add_office(Office("circle", "WHITE", 0))
            Dortmund.add_office(Office("square", "ORANGE", 0))
        else:
            Dortmund.add_office(Office("circle", "WHITE", 0))
            Dortmund.add_office(Office("square", "ORANGE", 0))
            Dortmund.add_office(Office("square", "PINK", 0))
        self.cities.append(Dortmund)

        Munster = City('Munster', 464, 697, BLACKISH_BROWN)
        Munster.add_office(Office("circle", "WHITE", 0))
        Munster.add_office(Office("square", "ORANGE", 0))
        self.cities.append(Munster)

        Coellen = City('Coellen', 105, 1117, BLACKISH_BROWN)
        Coellen.add_office(Office("square", "WHITE", 1))
        Coellen.add_office(Office("square", "PINK", 0))
        Coellen.assign_upgrade_type('SpecialPrestigePoints')
        self.cities.append(Coellen)

        Warburg = City('Warburg', 625, 1139, BLACKISH_BROWN)
        if num_players == 3:
            Warburg.add_office(Office("square", "ORANGE", 1))
            Warburg.add_office(Office("square", "PINK", 0))
        else:
            Warburg.add_office(Office("square", "ORANGE", 0))
            Warburg.add_office(Office("square", "PINK", 0))
        self.cities.append(Warburg)

        Paderborn = City('Paderborn', 766, 893, BLACKISH_BROWN)
        Paderborn.add_office(Office("square", "WHITE", 0))
        Paderborn.add_office(Office("circle", "BLACK", 0))
        self.cities.append(Paderborn)

        Minden = City('Minden', 757, 621, BLACKISH_BROWN)
        Minden.add_office(Office("square", "WHITE", 0))
        Minden.add_office(Office("square", "ORANGE", 0))
        Minden.add_office(Office("square", "PINK", 0))
        Minden.add_office(Office("square", "BLACK", 0))
        self.cities.append(Minden)

        Bremen = City('Bremen', 865, 283, BLACKISH_BROWN)
        if num_players == 3:
            Bremen.add_office(Office("square", "PINK", 0))
        else:
            Bremen.add_office(Office("circle", "WHITE", 0))
            Bremen.add_office(Office("square", "PINK", 0))
        self.cities.append(Bremen)

        Stade = City('Stade', 914, 118, BLACKISH_BROWN)
        if num_players == 3:
            Stade.add_office(Office("circle", "WHITE", 1))
        else:
            Stade.add_office(Office("circle", "WHITE", 0))
        Stade.assign_upgrade_type('Privilege')
        self.cities.append(Stade)

        Hannover = City('Hannover', 1090, 490, BLACKISH_BROWN)
        Hannover.add_office(Office("square", "WHITE", 0))
        Hannover.add_office(Office("square", "PINK", 0))
        self.cities.append(Hannover)

        Hildesheim = City('Hildesheim', 1089, 755, BLACKISH_BROWN)
        Hildesheim.add_office(Office("square", "WHITE", 0))
        Hildesheim.add_office(Office("square", "BLACK", 0))
        self.cities.append(Hildesheim)

        Gottingen = City('Gottingen', 989, 1075, BLACKISH_BROWN)
        if num_players == 3:
            Gottingen.add_office(Office("square", "WHITE", 0))
            Gottingen.add_office(Office("square", "ORANGE", 0))
        else:
            Gottingen.add_office(Office("square", "WHITE", 0))
            Gottingen.add_office(Office("circle", "WHITE", 0))
            Gottingen.add_office(Office("square", "PINK", 0))
        Gottingen.assign_upgrade_type('Actions')
        self.cities.append(Gottingen)

        Quedlinburg = City('Quedlinburg', 1315, 1107, BLACKISH_BROWN)
        Quedlinburg.add_office(Office("circle", "ORANGE", 0))
        Quedlinburg.add_office(Office("circle", "PINK", 0))
        self.cities.append(Quedlinburg)

        Goslar = City('Goslar', 1387, 793, BLACKISH_BROWN)
        if num_players == 3:
            Goslar.add_office(Office("square", "WHITE", 0))
        else:
            Goslar.add_office(Office("square", "WHITE", 0))
            Goslar.add_office(Office("square", "BLACK", 0))
        self.cities.append(Goslar)

        Brunswick = City('Brunswick', 1253, 622, BLACKISH_BROWN)
        Brunswick.add_office(Office("square", "ORANGE", 0))
        self.cities.append(Brunswick)

        Luneburg = City('Luneburg', 1359, 390, BLACKISH_BROWN)
        if num_players == 3:
            Luneburg.add_office(Office("circle", "WHITE", 0))
        else:
            Luneburg.add_office(Office("circle", "ORANGE", 0))
            Luneburg.add_office(Office("square", "BLACK", 0))
        self.cities.append(Luneburg)

        Hamburg = City('Hamburg', 1274, 80, BLACKISH_BROWN)
        Hamburg.add_office(Office("square", "WHITE", 0))
        Hamburg.add_office(Office("square", "ORANGE", 0))
        Hamburg.add_office(Office("square", "BLACK", 0))
        self.cities.append(Hamburg)

        Lubeck = City('Lubeck', 1574, 122, BLACKISH_BROWN)
        Lubeck.add_office(Office("square", "WHITE", 1))
        Lubeck.add_office(Office("square", "PINK", 0))
        Lubeck.assign_upgrade_type('Bank')
        self.cities.append(Lubeck)

        Perleberg = City('Perleberg', 1600, 308, BLACKISH_BROWN)
        Perleberg.add_office(Office("square", "WHITE", 0))
        Perleberg.add_office(Office("circle", "BLACK", 0))
        self.cities.append(Perleberg)

        Stendal = City('Stendal', 1595, 645, DARK_RED)
        Stendal.add_office(Office("square", "WHITE", 0))
        Stendal.add_office(Office("circle", "WHITE", 0))
        Stendal.add_office(Office("square", "ORANGE", 0))
        Stendal.add_office(Office("square", "PINK", 0))
        self.cities.append(Stendal)

        Magdeburg = City('Magdeburg', 1598, 929, BLACKISH_BROWN)
        Magdeburg.add_office(Office("circle", "WHITE", 0))
        Magdeburg.add_office(Office("square", "ORANGE", 0))
        self.cities.append(Magdeburg)

        Halle = City('Halle', 1630, 1117, BLACKISH_BROWN)
        Halle.add_office(Office("square", "WHITE", 1))
        Halle.add_office(Office("square", "ORANGE", 0))
        Halle.assign_upgrade_type('Keys')
        
        self.cities.append(Halle)

        for city in self.cities:
            city.update_city_size_based_on_offices()

        #Upgrades 
        UPGRADE_Y_AXIS_OFFSET = 29
        self.upgrade_cities.append(Upgrade('Halle', 'Keys', Halle.x_pos, Halle.y_pos-UPGRADE_Y_AXIS_OFFSET, width=Halle.width, height=Halle.height-30))
        self.upgrade_cities.append(Upgrade('Stade', 'Privilege', Stade.x_pos, Stade.y_pos-UPGRADE_Y_AXIS_OFFSET, width=Stade.width, height=Stade.height-30))
        self.upgrade_cities.append(Upgrade('Groningen', 'Book', Groningen.x_pos, Groningen.y_pos-UPGRADE_Y_AXIS_OFFSET, width=Groningen.width, height=Groningen.height-30))
        self.upgrade_cities.append(Upgrade('Gottingen', 'Actions', Gottingen.x_pos, Gottingen.y_pos-UPGRADE_Y_AXIS_OFFSET, width=Gottingen.width, height=Gottingen.height-30))
        self.upgrade_cities.append(Upgrade('Lubeck', 'Bank', Lubeck.x_pos, Lubeck.y_pos-UPGRADE_Y_AXIS_OFFSET, width=Lubeck.width, height=Lubeck.height-30))
        
        #Upgrade - Coellen which has SpecialPrestigePoints
        circle_size = CIRCLE_RADIUS*2

        # Calculate the total width of the SpecialPrestigePoints box
        x_specialprestigepoints_box_size = circle_size * 4 + SPACING*4  # Using SPACING*3 because there's no spacing after the last circle

        # Determine the starting x-position for the rectangle, ensuring it's centered
        x_start_pos = (Coellen.x_pos + (Coellen.width / 2)) - (x_specialprestigepoints_box_size / 2)

        # Determine the starting y-position for the rectangle so it touches Coellen
        y_start_pos = Coellen.y_pos - circle_size - (SPACING*2)

        # Use the calculated positions for the Upgrade
        self.specialprestigepoints = Upgrade('Coellen', 'SpecialPrestigePoints', x_start_pos, y_start_pos, width=x_specialprestigepoints_box_size, height=circle_size + SPACING*2)

        # Routes
        self.routes.append(Route([Groningen, Emden], 3))
        self.routes.append(Route([Emden, Osnabruck], 4))
        self.routes.append(Route([Kampen, Osnabruck], 2))
        self.routes.append(Route([Kampen, Arnheim], 3))
        self.routes.append(Route([Arnheim, Duisburg], 3))
        self.routes.append(Route([Arnheim, Munster], 3))
        self.routes.append(Route([Duisburg, Dortmund], 2))
        self.routes.append(Route([Osnabruck, Bremen], 3, has_bonus_marker=True))
        self.routes.append(Route([Munster, Minden], 3))
        self.routes.append(Route([Bremen, Minden], 3))
        self.routes.append(Route([Minden, Paderborn], 3))
        self.routes.append(Route([Dortmund, Paderborn], 3))
        self.routes.append(Route([Coellen, Warburg], 4))
        self.routes.append(Route([Paderborn, Warburg], 3))
        self.routes.append(Route([Stade, Hamburg], 3))
        self.routes.append(Route([Bremen, Hamburg], 4))
        self.routes.append(Route([Minden, Hannover], 3))
        self.routes.append(Route([Bremen, Hannover], 3))
        self.routes.append(Route([Paderborn, Hildesheim], 3))
        self.routes.append(Route([Hannover, Luneburg], 3))
        self.routes.append(Route([Minden, Brunswick], 4))
        self.routes.append(Route([Lubeck, Hamburg], 3))
        self.routes.append(Route([Luneburg, Hamburg], 4))
        self.routes.append(Route([Luneburg, Perleberg], 3, has_bonus_marker=True))
        self.routes.append(Route([Stendal, Perleberg], 3))
        self.routes.append(Route([Stendal, Brunswick], 4))
        self.routes.append(Route([Stendal, Magdeburg], 3))
        self.routes.append(Route([Goslar, Magdeburg], 2))
        self.routes.append(Route([Goslar, Quedlinburg], 4))
        self.routes.append(Route([Goslar, Hildesheim], 3, has_bonus_marker=True))
        self.routes.append(Route([Halle, Quedlinburg], 4))
        self.routes.append(Route([Gottingen, Quedlinburg], 3))

        if num_players != 3:
            self.routes.append(Route([Emden, Stade], 3))
            self.routes.append(Route([Gottingen, Warburg], 3))
        #3p - 101 total posts, 32 routes, 27 cities
        #4-5p - 107 total posts, 34 routes, 27 cities