# map3.py
from map_data.map_attributes import Map, City, Upgrade, Office, Route
from map_data.constants import GREY, CIRCLE_RADIUS, SPACING, BLACKISH_BROWN, DARK_RED, DARK_BLUE

class Map3(Map):
    def __init__(self, num_players):
        super().__init__()  # Call the parent class constructor
        self.cities = []
        self.routes = []
        self.upgrade_cities = []
        self.east_west_cities = ['York', 'Oxford']
        self.max_full_cities = 8
        self.max_full_cities_x_pos = 1000
        self.max_full_cities_y_pos = 52

        self.create_cities_and_routes(num_players)  # Populate cities, offices, and routes etc., specifically for Map1
        self.assign_starting_bonus_markers()

        self.map_width = 1800
        self.map_height = 1370

        #keep the cities in alphabetical order - helps when searching
        self.bonus_marker_positions = {
            ('Lubck', 'Mismar'): (198, 244),

            #permanent bonus market locations:
            ('Mismar', 'Stralsund'): (474, 153),
            ('Malmo', 'Stralsund'): (773, 90),
            ('Malmo', 'Visby'): (1196, 72),
            ('Danzig', 'Malmo'): (1106, 253),
        }

    def create_cities_and_routes(self, num_players):
        # Define Map3-specific cities and offices
        # Define cities
        if num_players > 3:
            Glasgom = City('Glasgom', 100, 105, DARK_BLUE)
            Glasgom.add_office(Office("square", "ORANGE", 0))
            self.cities.append(Glasgom)

            Edinbaurgh = City('Edinbaurgh', 400, 20, DARK_BLUE)
            Edinbaurgh.add_office(Office("square", "WHITE", 0))
            Edinbaurgh.add_office(Office("circle", "PINK", 0))
            self.cities.append(Edinbaurgh)

            Dunbar = City('Dunbar', 750, 40, DARK_BLUE)
            Dunbar.add_office(Office("square", "BLACK", 0))
            self.cities.append(Dunbar)

            Falkirk = City('Falkirk', 415, 130, DARK_BLUE)
            Falkirk.add_office(Office("square", "WHITE", 0))
            Falkirk.add_office(Office("square", "PINK", 0))
            Falkirk.add_office(Office("square", "BLACK", 0))
            self.cities.append(Falkirk)

            Carlisle = City('Carlisle', 350, 320, DARK_BLUE)
            Carlisle.add_office(Office("square", "WHITE", 0))
            Carlisle.add_office(Office("square", "ORANGE", 0))
            Carlisle.add_office(Office("square", "PINK", 0))
            Carlisle.add_office(Office("square", "BLACK", 0))
            Carlisle.add_office(Office("circle", "BLACK", 0)) #BLUE CAPITAL CITY
            self.cities.append(Carlisle)

            Newcastle = City('Newcastle', 1000, 190, DARK_BLUE)
            Newcastle.add_office(Office("square", "WHITE", 0))
            Newcastle.add_office(Office("square", "ORANGE", 0))
            Newcastle.add_office(Office("circle", "PINK", 0))
            Newcastle.assign_upgrade_type('Bank')
            self.cities.append(Newcastle)

            IsleOfMan = City('IsleOfMan', 60, 500, (65, 103, 114)) ##BLUE BROWN CITY
            IsleOfMan.add_office(Office("circle", "WHITE", 0))
            IsleOfMan.add_office(Office("circle", "PINK", 0))
            self.cities.append(IsleOfMan)
        else:
            Carlisle = City('Carlisle', 350, 320, GREY)
            Carlisle.add_office(Office("square", "WHITE", 0))
            Carlisle.add_office(Office("square", "ORANGE", 0))
            Carlisle.add_office(Office("square", "BLACK", 0))
            self.cities.append(Carlisle)

            Newcastle = City('Newcastle', 1000, 190, GREY)
            Newcastle.add_office(Office("square", "WHITE", 0))
            Newcastle.add_office(Office("square", "ORANGE", 0))
            Newcastle.add_office(Office("circle", "PINK", 0))
            Newcastle.assign_upgrade_type('Bank')
            self.cities.append(Newcastle)

            IsleOfMan = City('IsleOfMan', 60, 500, BLACKISH_BROWN) ## BROWN CITY
            IsleOfMan.add_office(Office("square", "WHITE", 0))
            IsleOfMan.add_office(Office("square", "PINK", 0))
            self.cities.append(IsleOfMan)

        Conway = City('Conway', 75, 740, BLACKISH_BROWN)
        Conway.add_office(Office("square", "WHITE", 0))
        Conway.add_office(Office("square", "ORANGE", 0))
        self.cities.append(Conway)

        Chester = City('Chester', 380, 620, BLACKISH_BROWN)
        Chester.add_office(Office("square", "WHITE", 0))
        Chester.add_office(Office("circle", "ORANGE", 0))
        Chester.add_office(Office("square", "BLACK", 0))
        Chester.assign_upgrade_type('Actions')
        self.cities.append(Chester)

        Montgomery = City('Montgomery', 350, 900, BLACKISH_BROWN)
        Montgomery.add_office(Office("square", "ORANGE", 0))
        Montgomery.add_office(Office("square", "PINK", 0))
        self.cities.append(Montgomery)

        Pembroke = City('Pembroke', 45, 1080, BLACKISH_BROWN)
        Pembroke.add_office(Office("square", "WHITE", 0))
        Pembroke.add_office(Office("square", "ORANGE", 0))
        Pembroke.add_office(Office("circle", "BLACK", 0))
        self.cities.append(Pembroke)

        Cardiff = City('Cardiff', 200, 1290, BLACKISH_BROWN)
        Cardiff.add_office(Office("square", "WHITE", 0))
        Cardiff.add_office(Office("square", "ORANGE", 0))
        Cardiff.add_office(Office("square", "PINK", 0))
        Cardiff.add_office(Office("circle", "PINK", 0))
        if num_players > 3:
            Cardiff.add_office(Office("circle", "BLACK", 0))
        self.cities.append(Cardiff)

        # GREY
        Richmond = City('Richmond', 644, 500, GREY)
        Richmond.add_office(Office("square", "WHITE", 0))
        self.cities.append(Richmond)

        Durham = City('Durham', 1175, 435, GREY)
        Durham.add_office(Office("square", "ORANGE", 0))
        self.cities.append(Durham)

        Lancaster = City('Lancaster', 790, 675, GREY)
        Lancaster.add_office(Office("square", "PINK", 0))
        self.cities.append(Lancaster)

        York = City('York', 1240, 690, DARK_RED)
        York.add_office(Office("square", "WHITE", 0))
        York.add_office(Office("circle", "ORANGE", 0))
        York.assign_upgrade_type('Keys')
        self.cities.append(York)

        Hereford = City('Hereford', 630, 865, GREY)
        Hereford.add_office(Office("square", "WHITE", 0))
        Hereford.add_office(Office("square", "PINK", 0))
        Hereford.add_office(Office("square", "BLACK", 0))
        self.cities.append(Hereford)

        Coventry = City('Coventry', 670, 1050, GREY)
        Coventry.add_office(Office("square", "WHITE", 0))
        Coventry.add_office(Office("square", "PINK", 0))
        self.cities.append(Coventry)

        Nottingham = City('Nottingham', 920, 800, GREY)
        Nottingham.add_office(Office("square", "ORANGE", 0))
        Nottingham.add_office(Office("square", "PINK", 0))
        self.cities.append(Nottingham)

        Norwich = City('Norwich', 1250, 865, GREY)
        Norwich.add_office(Office("square", "WHITE", 0))
        Norwich.add_office(Office("square", "PINK", 0))
        Norwich.assign_upgrade_type('Book')
        self.cities.append(Norwich)

        Cambridge = City('Cambridge', 920, 1005, GREY)
        Cambridge.add_office(Office("circle", "WHITE", 0))
        Cambridge.add_office(Office("square", "BLACK", 0))
        self.cities.append(Cambridge)

        Ipswich = City('Ipswich', 1300, 1100, GREY)
        Ipswich.add_office(Office("square", "ORANGE", 2))
        self.cities.append(Ipswich)

        Oxford = City('Oxford', 688, 1170, DARK_RED)
        Oxford.add_office(Office("circle", "WHITE", 0))
        Oxford.add_office(Office("square", "ORANGE", 0))
        self.cities.append(Oxford)

        London = City('London', 1200, 1300, GREY)
        London.add_office(Office("square", "WHITE", 0))
        London.add_office(Office("square", "ORANGE", 0))
        if num_players > 3:
            London.add_office(Office("square", "ORANGE", 0))
            London.add_office(Office("square", "PINK", 0))
        London.add_office(Office("square", "PINK", 0))
        London.add_office(Office("circle", "PINK", 0))
        London.add_office(Office("square", "BLACK", 0))
        London.add_office(Office("circle", "BLACK", 0))
        London.assign_upgrade_type('Privilege')
        self.cities.append(London)

        Canterbury = City('Canterbury', 953, 1473, GREY)
        Canterbury.add_office(Office("square", "WHITE", 0))
        Canterbury.add_office(Office("circle", "PINK", 0))
        self.cities.append(Canterbury)

        Calais = City('Calais', 1300, 1560, GREY)
        Calais.add_office(Office("square", "WHITE", 0))
        Calais.add_office(Office("circle", "ORANGE", 0))
        Calais.add_office(Office("square", "PINK", 0))
        self.cities.append(Calais)

        Southhampton = City('Southhampton', 850, 1560, GREY)
        Southhampton.add_office(Office("square", "WHITE", 0))
        Southhampton.add_office(Office("square", "ORANGE", 0))
        Southhampton.add_office(Office("square", "BLACK", 0))
        self.cities.append(Southhampton)

        Salisbury = City('Salisbury', 640, 1500, GREY)
        Salisbury.add_office(Office("square", "ORANGE", 0))
        Salisbury.add_office(Office("square", "PINK", 0))
        self.cities.append(Salisbury)

        Plymouth = City('Plymouth', 160, 1470, GREY)
        Plymouth.add_office(Office("square", "WHITE", 0))
        Plymouth.add_office(Office("circle", "ORANGE", 2))
        Plymouth.assign_upgrade_type('SpecialPrestigePoints')
        self.cities.append(Plymouth)

        Bristol = City('Bristol', 500, 1300, GREY)
        Bristol.add_office(Office("square", "ORANGE", 2))
        self.cities.append(Bristol)

        for city in self.cities:
            city.update_city_size_based_on_offices()

        #Upgrades 
        UPGRADE_Y_AXIS_OFFSET = 29
        self.upgrade_cities.append(Upgrade('York', 'Keys', York.x_pos, York.y_pos-UPGRADE_Y_AXIS_OFFSET, width=York.width, height=York.height-30))
        self.upgrade_cities.append(Upgrade('London', 'Privilege', London.x_pos, London.y_pos-UPGRADE_Y_AXIS_OFFSET, width=London.width, height=London.height-30))
        self.upgrade_cities.append(Upgrade('Norwich', 'Book', Norwich.x_pos, Norwich.y_pos-UPGRADE_Y_AXIS_OFFSET, width=Norwich.width, height=Norwich.height-30))
        self.upgrade_cities.append(Upgrade('Chester', 'Actions', Chester.x_pos, Chester.y_pos-UPGRADE_Y_AXIS_OFFSET, width=Chester.width, height=Chester.height-30))
        self.upgrade_cities.append(Upgrade('Newcastle', 'Bank', Newcastle.x_pos, Newcastle.y_pos-UPGRADE_Y_AXIS_OFFSET, width=Newcastle.width, height=Newcastle.height-30))
        
        #Upgrade - Coellen which has SpecialPrestigePoints
        circle_size = CIRCLE_RADIUS*2

        # Calculate the total width of the SpecialPrestigePoints box
        x_specialprestigepoints_box_size = circle_size * 4 + SPACING*4  # Using SPACING*3 because there's no spacing after the last circle

        # Determine the starting x-position for the rectangle, ensuring it's centered
        x_start_pos = (Plymouth.x_pos + (Plymouth.width / 2)) - (x_specialprestigepoints_box_size / 2)

        # Determine the starting y-position for the rectangle so it touches Coellen
        y_start_pos = Plymouth.y_pos - circle_size - (SPACING*2)

        # Use the calculated positions for the Upgrade
        self.specialprestigepoints = Upgrade('Plymouth', 'SpecialPrestigePoints', x_start_pos, y_start_pos, width=x_specialprestigepoints_box_size, height=circle_size + SPACING*2)

        # Routes
        #  BLUE
        if num_players > 3:
            self.routes.append(Route([Glasgom, Edinbaurgh], 2, color=DARK_BLUE, region="Scotland"))
            self.routes.append(Route([Dunbar, Edinbaurgh], 2, color=DARK_BLUE, region="Scotland"))
            self.routes.append(Route([Dunbar, Newcastle], 3, color=DARK_BLUE, region="Scotland"))
            self.routes.append(Route([Glasgom, Falkirk], 2, color=DARK_BLUE, region="Scotland"))
            self.routes.append(Route([Carlisle, Falkirk], 2, color=DARK_BLUE, region="Scotland"))
            self.routes.append(Route([Carlisle, IsleOfMan], 3, required_circles=2, color=DARK_BLUE, region="Scotland"))
        else:
            self.routes.append(Route([Carlisle, IsleOfMan], 3, required_circles=2, permanent_bm_type="MoveAny2"))
            self.routes.append(Route([Carlisle, Newcastle], 4))

        #  BROWN
        self.routes.append(Route([Conway, IsleOfMan], 3, required_circles=2, color=BLACKISH_BROWN, region="Wales"))
        self.routes.append(Route([Conway, Chester], 3, color=BLACKISH_BROWN, region="Wales"))
        self.routes.append(Route([Conway, Montgomery], 3, color=BLACKISH_BROWN, region="Wales"))
        self.routes.append(Route([Hereford, Montgomery], 3, color=BLACKISH_BROWN, region="Wales"))
        self.routes.append(Route([Pembroke, Montgomery], 3, color=BLACKISH_BROWN, region="Wales"))
        self.routes.append(Route([Pembroke, Cardiff], 3, color=BLACKISH_BROWN, region="Wales"))
        self.routes.append(Route([Cardiff, Montgomery], 3, color=BLACKISH_BROWN, region="Wales"))

        #  WHITE
        if num_players > 3:
            self.routes.append(Route([Carlisle, Chester], 3, required_circles=2))
        self.routes.append(Route([Carlisle, Richmond], 3))
        self.routes.append(Route([Carlisle, Durham], 4))
        self.routes.append(Route([Lancaster, Richmond], 2, has_bonus_marker=True))
        self.routes.append(Route([Lancaster, Durham], 3))
        self.routes.append(Route([Newcastle, Durham], 3))
        self.routes.append(Route([Chester, Hereford], 4))
        self.routes.append(Route([Lancaster, Hereford], 2))
        self.routes.append(Route([Coventry, Hereford], 3))
        self.routes.append(Route([Coventry, Nottingham], 3, has_bonus_marker=True))
        self.routes.append(Route([York, Nottingham], 3))
        self.routes.append(Route([York, Durham], 3))
        self.routes.append(Route([Norwich, Nottingham], 3))
        self.routes.append(Route([Norwich, Cambridge], 3))
        self.routes.append(Route([Ipswich, Cambridge], 4))
        self.routes.append(Route([Coventry, Cambridge], 3))
        self.routes.append(Route([London, Cambridge], 3))
        self.routes.append(Route([Coventry, Cardiff], 3))
        self.routes.append(Route([Oxford, Cardiff], 4))
        self.routes.append(Route([Oxford, London], 4))
        self.routes.append(Route([Oxford, Salisbury], 4))
        self.routes.append(Route([Bristol, Salisbury], 4))
        self.routes.append(Route([Plymouth, Salisbury], 4))
        self.routes.append(Route([Southhampton, Salisbury], 2, has_bonus_marker=True))
        self.routes.append(Route([Southhampton, Calais], 3, permanent_bm_type="Place2ScotlandOrWales", required_circles=2))
        self.routes.append(Route([Canterbury, Calais], 3, permanent_bm_type="MoveAny2", required_circles=1))
        self.routes.append(Route([Canterbury, London], 3))

        #3P - 111 posts, 35 routes, 26 cities
        #4-5P - 121 posts, 40 routes, 30 cities