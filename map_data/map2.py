# map2.py
from map_data.map_attributes import Map, City, Upgrade, Office, Route
from map_data.constants import BLACKISH_BROWN, CIRCLE_RADIUS, SPACING, DARK_GREEN, DARK_RED

class Map2(Map):
    def __init__(self):
        super().__init__()  # Call the parent class constructor
        self.cities = []
        self.routes = []
        self.upgrade_cities = []
        self.east_west_cities = ['Lubeck', 'Danzig']
        self.specialprestigepoints = None
        self.max_full_cities = 10
        self.max_full_cities_x_pos = 77
        self.max_full_cities_y_pos = 52

        self.create_cities_and_routes()  # Populate cities, offices, and routes etc., specifically for Map1
        self.assign_starting_bonus_markers()

        self.map_width = 1800
        self.map_height = 1255

        #keep the cities in alphabetical order - helps when searching
        self.bonus_marker_positions = {
            ('Lubck', 'Mismar'): (198, 244),
            ('Mismar', 'Waren'): (435, 363),
            ('Havelberg', 'Waren'): (271, 647),
            ('Havelberg', 'Perleberg'): (135, 529),
            ('Stettin', 'Waren'): (622, 530),
            ('Anklam', 'Stettin'): (680, 440),
            ('Anklam', 'Munster'): (875, 344),
            ('Danzig', 'Munster'): (1163, 365),
            ('Danzig', 'Konigsberg'): (1569, 265),
            ('Braunsberg', 'Danzig'): (1568, 394),
            ('Braunsberg', 'Elbing'): (1524, 600),
            ('Elbing', 'Kulm'): (1321, 571),
            ('Kulm', 'Munster'): (1180, 568),
            ('Kulm', 'Stettin'): (991, 612),
            ('Brandenburg', 'Stettin'): (592, 669),
            ('Brandenburg', 'Tangermunde'): (293, 758),
            ('BerlinColin', 'Brandenburg'): (601, 836),
            ('Brandenburg', 'Halle'): (399, 944),
            ('Halle', 'Magedeburg'): (308, 1036),
            ('Halle', 'Mittenberg'): (560, 952),
            ('Dresden', 'Halle'): (681, 1159),
            ('Dresden', 'Krackau'): (1372, 1188),
            ('Breslau', 'Krackau'): (1553, 1033),
            ('Breslau', 'Frankfurt'): (1148, 970),
            ('Frankfurt', 'Stettin'): (865, 755),
            ('Krackau', 'Thorn'): (1689, 992),
            ('Allenstein', 'Thorn'): (1641, 772),
            ('Allenstein', 'Braunsberg'): (1660, 670),
            #permanent bonus market locations:
            ('Mismar', 'Stralsund'): (474, 153),
            ('Malmo', 'Stralsund'): (773, 90),
            ('Malmo', 'Visby'): (1196, 72),
            ('Danzig', 'Malmo'): (1106, 253),
        }

    def create_cities_and_routes(self):
        # Define Map2-specific cities and offices
        # Define cities
        Lubeck = City('Lubeck', 30, 120, BLACKISH_BROWN)
        Lubeck.add_office(Office("square", "WHITE", 0))
        Lubeck.add_office(Office("square", "ORANGE", 0))
        Lubeck.add_office(Office("square", "BLACK", 0))
        Lubeck.change_color(DARK_RED)
        self.cities.append(Lubeck)

        Mismar = City('Mismar', 296, 210, BLACKISH_BROWN)
        Mismar.add_office(Office("square", "ORANGE", 0))
        Mismar.add_office(Office("circle", "PINK", 0))
        self.cities.append(Mismar)

        Stralsund = City('Stralsund', 586, 150, BLACKISH_BROWN)
        Stralsund.add_office(Office("square", "ORANGE", 0))
        Stralsund.add_office(Office("circle", "ORANGE", 0))
        self.cities.append(Stralsund)

        Malmo = City('Malmo', 900, 75, BLACKISH_BROWN)
        Malmo.add_office(Office("circle", "WHITE", 0))
        Malmo.add_office(Office("square", "ORANGE", 0))
        Malmo.add_office(Office("circle", "PINK", 0))
        self.cities.append(Malmo)

        Visby = City('Visby', 1348, 128, BLACKISH_BROWN)
        Visby.add_office(Office("circle", "WHITE", 0))
        self.cities.append(Visby)

        Danzig = City('Danzig', 1290, 310, BLACKISH_BROWN)
        Danzig.add_office(Office("square", "WHITE", 0))
        Danzig.add_office(Office("square", "ORANGE", 0))
        Danzig.add_office(Office("square", "PINK", 0))
        Danzig.add_office(Office("square", "BLACK", 0))
        Danzig.change_color(DARK_RED)
        self.cities.append(Danzig)

        Konigsberg = City('Konigsberg', 1627, 100, BLACKISH_BROWN)
        Konigsberg.add_office(Office("circle", "WHITE", 0))
        Konigsberg.add_office(Office("square", "ORANGE", 0))
        Konigsberg.add_office(Office("square", "PINK", 0))
        Konigsberg.assign_upgrade_type('SpecialPrestigePoints')
        self.cities.append(Konigsberg)

        Belgard = City('Munster', 909, 450, BLACKISH_BROWN)
        Belgard.add_office(Office("square", "WHITE", 0))
        Belgard.add_office(Office("square", "WHITE", 0))
        Belgard.add_office(Office("square", "WHITE", 0))
        Belgard.add_office(Office("square", "WHITE", 0))
        Belgard.add_office(Office("square", "WHITE", 0))
        Belgard.add_office(Office("square", "WHITE", 0))
        Belgard.change_color(DARK_GREEN)
        self.cities.append(Belgard)

        Anklam = City('Anklam', 630, 270, BLACKISH_BROWN)
        Anklam.add_office(Office("square", "WHITE", 0))
        Anklam.add_office(Office("square", "PINK", 0))
        self.cities.append(Anklam)

        Waren = City('Waren', 300, 485, BLACKISH_BROWN)
        Waren.assign_upgrade_type('Bank')
        Waren.assign_upgrade_type('Actions')
        Waren.add_office(Office("square", "WHITE", 0))
        Waren.add_office(Office("square", "WHITE", 0))
        Waren.add_office(Office("square", "WHITE", 0))
        Waren.add_office(Office("square", "WHITE", 0))
        Waren.add_office(Office("square", "WHITE", 0))
        Waren.add_office(Office("square", "WHITE", 0))
        Waren.change_color(DARK_GREEN)
        self.cities.append(Waren)

        Perleberg = City('Perleberg', 50, 366, BLACKISH_BROWN)
        Perleberg.add_office(Office("square", "ORANGE", 0))
        self.cities.append(Perleberg)

        Havelberg = City('Havelberg', 25, 660, BLACKISH_BROWN)
        Havelberg.add_office(Office("square", "WHITE", 1))
        Havelberg.add_office(Office("circle", "WHITE", 0))
        Havelberg.add_office(Office("square", "PINK", 0))
        Havelberg.add_office(Office("square", "BLACK", 0))
        self.cities.append(Havelberg)

        Stettin = City('Stettin', 775, 605, BLACKISH_BROWN)
        Stettin.add_office(Office("square", "ORANGE", 0))
        Stettin.add_office(Office("square", "BLACK", 0))
        self.cities.append(Stettin)

        Kulm = City('Kulm', 1200, 670, BLACKISH_BROWN)
        Kulm.add_office(Office("square", "WHITE", 0))
        Kulm.add_office(Office("square", "PINK", 0))
        Kulm.assign_upgrade_type('Privilege')
        self.cities.append(Kulm)

        Elbing = City('Elbing', 1391, 526, BLACKISH_BROWN)
        Elbing.add_office(Office("square", "ORANGE", 0))
        Elbing.add_office(Office("square", "BLACK", 0))
        self.cities.append(Elbing)

        Braunsberg = City('Braunsberg', 1675, 500, BLACKISH_BROWN)
        Braunsberg.add_office(Office("square", "WHITE", 0))
        Braunsberg.add_office(Office("circle", "ORANGE", 0))
        self.cities.append(Braunsberg)

        Allenstein = City('Allenstein', 1435, 750, BLACKISH_BROWN)
        Allenstein.add_office(Office("square", "WHITE", 0))
        Allenstein.add_office(Office("square", "PINK", 0))
        self.cities.append(Allenstein)

        Frankfurt = City('Frankfurt', 950, 791, BLACKISH_BROWN)
        Frankfurt.add_office(Office("square", "WHITE", 0))
        Frankfurt.add_office(Office("square", "ORANGE", 0))
        Frankfurt.add_office(Office("square", "PINK", 0))
        self.cities.append(Frankfurt)

        BerlinColln = City('BerlinColln', 750, 756, BLACKISH_BROWN)
        BerlinColln.add_office(Office("square", "ORANGE", 0))
        self.cities.append(BerlinColln)

        Brandenburg = City('Brandenburg', 403, 750, BLACKISH_BROWN)
        Brandenburg.add_office(Office("circle", "WHITE", 0))
        Brandenburg.add_office(Office("square", "PINK", 0))
        self.cities.append(Brandenburg)

        Tangermunde = City('Tangermunde', 136, 800, BLACKISH_BROWN)
        Tangermunde.add_office(Office("square", "ORANGE", 0))
        self.cities.append(Tangermunde)

        Magdeburg = City('Magdeburg', 132, 1065, BLACKISH_BROWN)
        Magdeburg.add_office(Office("square", "WHITE", 1))
        Magdeburg.add_office(Office("square", "ORANGE", 0))
        Magdeburg.assign_upgrade_type('Privilege')
        self.cities.append(Magdeburg)

        Halle = City('Halle', 414, 1065, BLACKISH_BROWN)
        Halle.add_office(Office("square", "WHITE", 0))
        Halle.add_office(Office("circle", "ORANGE", 0))
        self.cities.append(Halle)

        Mittenberg = City('Mittenberg', 655, 900, BLACKISH_BROWN)
        Mittenberg.add_office(Office("square", "PINK", 0))
        self.cities.append(Mittenberg)

        Dresden = City('Dresden', 900, 1100, BLACKISH_BROWN)
        Dresden.add_office(Office("square", "WHITE", 0))
        Dresden.add_office(Office("square", "WHITE", 0))
        Dresden.add_office(Office("square", "WHITE", 0))
        Dresden.add_office(Office("square", "WHITE", 0))
        Dresden.add_office(Office("square", "WHITE", 0))
        Dresden.add_office(Office("square", "WHITE", 0))
        Dresden.assign_upgrade_type('Keys')
        Dresden.change_color(DARK_GREEN)
        self.cities.append(Dresden)

        Breslau = City('Breslau', 1290, 1000, BLACKISH_BROWN)
        Breslau.add_office(Office("square", "WHITE", 0))
        Breslau.add_office(Office("circle", "ORANGE", 0))
        Breslau.add_office(Office("square", "PINK", 0))
        Breslau.assign_upgrade_type('Book')
        self.cities.append(Breslau)

        Thorn = City('Thorn', 1675, 830, BLACKISH_BROWN)
        Thorn.add_office(Office("square", "WHITE", 1))
        Thorn.add_office(Office("square", "ORANGE", 0))
        Thorn.add_office(Office("square", "BLACK", 0))
        self.cities.append(Thorn)

        Krackau = City('Krackau', 1675, 1127, BLACKISH_BROWN)
        Krackau.add_office(Office("square", "WHITE", 0))
        Krackau.add_office(Office("square", "PINK", 0))
        Krackau.add_office(Office("square", "BLACK", 0))
        self.cities.append(Krackau)

        for city in self.cities:
            city.update_city_size_based_on_offices()

        #Upgrades 
        UPGRADE_Y_AXIS_OFFSET = 29
        self.upgrade_cities.append(Upgrade('Dresden', 'Keys', Dresden.x_pos, Dresden.y_pos-UPGRADE_Y_AXIS_OFFSET, width=Dresden.width, height=Dresden.height-30))
        self.upgrade_cities.append(Upgrade('Magdeburg', 'Privilege', Magdeburg.x_pos, Magdeburg.y_pos-UPGRADE_Y_AXIS_OFFSET, width=Magdeburg.width, height=Magdeburg.height-30))
        self.upgrade_cities.append(Upgrade('Breslau', 'Book', Breslau.x_pos, Breslau.y_pos-UPGRADE_Y_AXIS_OFFSET, width=Breslau.width, height=Breslau.height-30))
        self.upgrade_cities.append(Upgrade('Waren', 'Actions', Waren.x_pos, Waren.y_pos-UPGRADE_Y_AXIS_OFFSET, width=Waren.width/2, height=Waren.height-30))
        self.upgrade_cities.append(Upgrade('Waren', 'Bank', Waren.x_pos+Waren.width/2, Waren.y_pos-UPGRADE_Y_AXIS_OFFSET, width=Waren.width/2, height=Waren.height-30))
        
        #Upgrade - Coellen which has SpecialPrestigePoints
        circle_size = CIRCLE_RADIUS*2

        # Calculate the total width of the SpecialPrestigePoints box
        x_specialprestigepoints_box_size = circle_size * 4 + SPACING*4  # Using SPACING*3 because there's no spacing after the last circle

        # Determine the starting x-position for the rectangle, ensuring it's centered
        x_start_pos = (Konigsberg.x_pos + (Konigsberg.width / 2)) - (x_specialprestigepoints_box_size / 2)

        # Determine the starting y-position for the rectangle so it touches Coellen
        y_start_pos = Konigsberg.y_pos - circle_size - (SPACING*2)

        # Use the calculated positions for the Upgrade
        self.specialprestigepoints = Upgrade('Konigsberg', 'SpecialPrestigePoints', x_start_pos, y_start_pos, width=x_specialprestigepoints_box_size, height=circle_size + SPACING*2)

        # Routes
        self.routes.append(Route([Lubeck, Mismar], 3))
        self.routes.append(Route([Mismar, Stralsund], 3, permanent_bm_type="MoveAny2", required_circles=2))
        self.routes.append(Route([Stralsund, Malmo], 3, permanent_bm_type="+1Priv", required_circles=2))
        self.routes.append(Route([Malmo, Visby], 3, permanent_bm_type="ClaimGreenCity", required_circles=1))
        self.routes.append(Route([Malmo, Danzig], 4, permanent_bm_type="Place2TradesmenFromRoute", required_circles=1))
        self.routes.append(Route([Danzig, Konigsberg], 4))
        self.routes.append(Route([Danzig, Belgard], 4))
        self.routes.append(Route([Belgard, Anklam], 3))
        self.routes.append(Route([Belgard, Kulm], 3))
        self.routes.append(Route([Anklam, Stettin], 3))
        self.routes.append(Route([Stettin, Waren], 4))
        self.routes.append(Route([Waren, Mismar], 3))
        self.routes.append(Route([Waren, Havelberg], 3))
        self.routes.append(Route([Havelberg, Perleberg], 2))
        self.routes.append(Route([Stettin, Kulm], 3, has_bonus_marker=True))
        self.routes.append(Route([Kulm, Elbing], 2))
        self.routes.append(Route([Elbing, Braunsberg], 3))
        self.routes.append(Route([Danzig, Braunsberg], 4))
        self.routes.append(Route([Braunsberg, Allenstein], 3, has_bonus_marker=True))
        self.routes.append(Route([Allenstein, Thorn], 3))
        self.routes.append(Route([Thorn, Krackau], 3))
        self.routes.append(Route([Krackau, Breslau], 3))
        self.routes.append(Route([Breslau, Frankfurt], 3))
        self.routes.append(Route([Frankfurt, Stettin], 2))
        self.routes.append(Route([Stettin, Brandenburg], 3))
        self.routes.append(Route([Brandenburg, Tangermunde], 2, has_bonus_marker=True))
        self.routes.append(Route([Brandenburg, BerlinColln], 3))
        self.routes.append(Route([Brandenburg, Halle], 3))
        self.routes.append(Route([Halle, Magdeburg], 3))
        self.routes.append(Route([Halle, Mittenberg], 2))
        self.routes.append(Route([Halle, Dresden], 4))
        self.routes.append(Route([Dresden, Krackau], 4))