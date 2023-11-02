# map1.py
import pygame
from map_data.map_attributes import Map, City, Upgrade, Office, Route
from map_data.constants import BLACKISH_BROWN

class Map1(Map):
    def __init__(self):
        super().__init__()  # Call the parent class constructor
        self.cities = []
        self.routes = []
        self.upgrades = []
        self.create_cities_and_routes()  # Populate cities, offices, and routes etc., specifically for Map1

        self.map_width = 1800
        self.map_height = 1255

    def create_cities_and_routes(self):
        # Define Map1-specific cities and offices
        # Define cities
        Groningen = City('Groningen', (115, 210), BLACKISH_BROWN)
        Groningen.add_office(Office("square", "white", True))
        Groningen.add_office(Office("circle", "orange", False))
        Groningen.assign_upgrade_type('Book')
        self.cities.append(Groningen)

        Emden = City('Emden', (476, 207), BLACKISH_BROWN)
        Emden.add_office(Office("circle", "white", False))
        Emden.add_office(Office("square", "pink", False))
        self.cities.append(Emden)

        Osnabruck = City('Osnabruck', (434, 537), BLACKISH_BROWN)
        Osnabruck.add_office(Office("square", "white", False))
        Osnabruck.add_office(Office("square", "orange", False))
        Osnabruck.add_office(Office("square", "black", False))
        self.cities.append(Osnabruck)

        Kampen = City('Kampen', (107, 410), BLACKISH_BROWN)
        Kampen.add_office(Office("square", "orange", False))
        Kampen.add_office(Office("square", "black", False))
        self.cities.append(Kampen)

        Arnheim = City('Arnheim', (88, 655), BLACKISH_BROWN)
        Arnheim.add_office(Office("square", "white", False))
        Arnheim.add_office(Office("circle", "white", False))
        Arnheim.add_office(Office("square", "orange", False))
        Arnheim.add_office(Office("square", "pink", False))
        self.cities.append(Arnheim)

        Duisburg = City('Duisburg', (179, 911), BLACKISH_BROWN)
        Duisburg.add_office(Office("square", "white", False))
        self.cities.append(Duisburg)

        Dortmund = City('Dortmund', (432, 894), BLACKISH_BROWN)
        Dortmund.add_office(Office("circle", "white", False))
        Dortmund.add_office(Office("square", "orange", False))
        self.cities.append(Dortmund)

        Munster = City('Munster', (474, 702), BLACKISH_BROWN)
        Munster.add_office(Office("circle", "white", False))
        Munster.add_office(Office("square", "orange", False))
        self.cities.append(Munster)

        Coellen = City('Coellen', (105, 1062), BLACKISH_BROWN)
        Coellen.add_office(Office("square", "white", True))
        Coellen.add_office(Office("square", "pink", False))
        self.cities.append(Coellen)

        Warburg = City('Warburg', (672, 1117), BLACKISH_BROWN)
        Warburg.add_office(Office("square", "orange", True))
        Warburg.add_office(Office("square", "pink", False))
        self.cities.append(Warburg)

        Paderborn = City('Paderborn', (786, 925), BLACKISH_BROWN)
        Paderborn.add_office(Office("square", "white", False))
        Paderborn.add_office(Office("circle", "black", False))
        self.cities.append(Paderborn)

        Minden = City('Minden', (757, 621), BLACKISH_BROWN)
        Minden.add_office(Office("square", "white", False))
        Minden.add_office(Office("square", "orange", False))
        Minden.add_office(Office("square", "pink", False))
        Minden.add_office(Office("square", "black", False))
        self.cities.append(Minden)

        Bremen = City('Bremen', (865, 283), BLACKISH_BROWN)
        Bremen.add_office(Office("square", "pink", False))
        self.cities.append(Bremen)

        Stade = City('Stade', (934, 138), BLACKISH_BROWN)
        Stade.add_office(Office("circle", "white", True))
        Stade.assign_upgrade_type('Privilege')
        self.cities.append(Stade)

        Hannover = City('Hannover', (1049, 499), BLACKISH_BROWN)
        Hannover.add_office(Office("square", "white", False))
        Hannover.add_office(Office("square", "pink", False))
        self.cities.append(Hannover)

        Hildesheim = City('Hildesheim', (1089, 755), BLACKISH_BROWN)
        Hildesheim.add_office(Office("square", "white", False))
        Hildesheim.add_office(Office("square", "black", False))
        self.cities.append(Hildesheim)

        Gottingen = City('Gottingen', (1014, 1075), BLACKISH_BROWN)
        Gottingen.add_office(Office("square", "white", False))
        Gottingen.add_office(Office("square", "orange", False))
        Gottingen.assign_upgrade_type('Actions')
        self.cities.append(Gottingen)

        Quedlinburg = City('Quedlinburg', (1405, 983), BLACKISH_BROWN)
        Quedlinburg.add_office(Office("circle", "orange", False))
        Quedlinburg.add_office(Office("circle", "pink", False))
        self.cities.append(Quedlinburg)

        Goslar = City('Goslar', (1387, 839), BLACKISH_BROWN)
        Goslar.add_office(Office("square", "white", True))
        self.cities.append(Goslar)

        Brunswick = City('Brunswick', (1253, 582), BLACKISH_BROWN)
        Brunswick.add_office(Office("square", "orange", True))
        self.cities.append(Brunswick)

        Luneburg = City('Luneburg', (1359, 358), BLACKISH_BROWN)
        Luneburg.add_office(Office("circle", "white", True))
        self.cities.append(Luneburg)

        Hamburg = City('Hamburg', (1274, 105), BLACKISH_BROWN)
        Hamburg.add_office(Office("square", "white", False))
        Hamburg.add_office(Office("square", "orange", False))
        Hamburg.add_office(Office("square", "black", False))
        self.cities.append(Hamburg)

        Lubeck = City('Lubeck', (1540, 122), BLACKISH_BROWN)
        Lubeck.add_office(Office("square", "white", True))
        Lubeck.add_office(Office("square", "pink", False))
        Lubeck.assign_upgrade_type('Bank')
        self.cities.append(Lubeck)

        Perleberg = City('Perleberg', (1574, 308), BLACKISH_BROWN)
        Perleberg.add_office(Office("square", "white", True))
        Perleberg.add_office(Office("circle", "black", False))
        self.cities.append(Perleberg)

        Stendal = City('Stendal', (1538, 633), BLACKISH_BROWN)
        Stendal.add_office(Office("square", "white", False))
        Stendal.add_office(Office("circle", "white", False))
        Stendal.add_office(Office("square", "orange", False))
        Stendal.add_office(Office("square", "pink", False))
        self.cities.append(Stendal)

        Magdeburg = City('Magdeburg', (1598, 929), BLACKISH_BROWN)
        Magdeburg.add_office(Office("circle", "white", False))
        Magdeburg.add_office(Office("square", "orange", False))
        self.cities.append(Magdeburg)

        Halle = City('Halle', (1544, 1063), BLACKISH_BROWN)
        Halle.add_office(Office("square", "white", True))
        Halle.add_office(Office("square", "orange", False))
        Halle.assign_upgrade_type('Keys')
        
        self.cities.append(Halle)

        for city in self.cities:
            city.update_city_size_based_on_offices()

        #Upgrades 
        UPGRADE_Y_AXIS_OFFSET = 50
        self.upgrades.append(Upgrade('Halle', 'Keys', Halle.pos[0], Halle.pos[1]-UPGRADE_Y_AXIS_OFFSET, width=Halle.width, height=Halle.height))
        self.upgrades.append(Upgrade('Stade', 'Privilege', Stade.pos[0], Stade.pos[1]-UPGRADE_Y_AXIS_OFFSET, width=Stade.width, height=Stade.height))
        self.upgrades.append(Upgrade('Groningen', 'Book', Groningen.pos[0], Groningen.pos[1]-UPGRADE_Y_AXIS_OFFSET, width=Groningen.width, height=Groningen.height))
        self.upgrades.append(Upgrade('Gottingen', 'Actions', Gottingen.pos[0], Gottingen.pos[1]-UPGRADE_Y_AXIS_OFFSET, width=Gottingen.width, height=Gottingen.height))
        self.upgrades.append(Upgrade('Lubeck', 'Bank', Lubeck.pos[0], Lubeck.pos[1]-UPGRADE_Y_AXIS_OFFSET, width=Lubeck.width, height=Lubeck.height))
        
        # Routes
        self.routes.append(Route([Groningen, Emden], 3))
        self.routes.append(Route([Emden, Osnabruck], 4))
        self.routes.append(Route([Kampen, Osnabruck], 2))
        self.routes.append(Route([Kampen, Arnheim], 3))
        self.routes.append(Route([Arnheim, Duisburg], 3))
        self.routes.append(Route([Arnheim, Munster], 3))
        self.routes.append(Route([Duisburg, Dortmund], 2))
        self.routes.append(Route([Osnabruck, Bremen], 3, True))
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
        self.routes.append(Route([Luneburg, Perleberg], 3, True))
        self.routes.append(Route([Stendal, Perleberg], 3))
        self.routes.append(Route([Stendal, Brunswick], 4))
        self.routes.append(Route([Stendal, Magdeburg], 3))
        self.routes.append(Route([Goslar, Magdeburg], 2))
        self.routes.append(Route([Goslar, Quedlinburg], 4))
        self.routes.append(Route([Goslar, Hildesheim], 3, True))
        self.routes.append(Route([Halle, Quedlinburg], 4))
        self.routes.append(Route([Gottingen, Quedlinburg], 3))
