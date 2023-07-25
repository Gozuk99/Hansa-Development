from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Office:
    color: int
    point: int = False
    merch: bool = False

@dataclass
class City:
    name: str
    offices: List[Office]
    upgrade: str = ''
    color: str = ''
    position: List[int] = field(default_factory=list)
    neighbors: List[str] = field(default_factory=list)

@dataclass
class Route:
    from_city: str
    to_city: str
    posts: int
    tavern: bool = False

@dataclass
class GameMap:
    cities: Dict[str, City]
    routes: List[Route]

    def add_neighbors(self):
        for city in self.cities.values():
            city.neighbors = []
            for route in self.routes:
                if route.from_city == city.name:
                    city.neighbors.append(route.to_city)
                if route.to_city == city.name:
                    city.neighbors.append(route.from_city)
        return self

BaseMap = GameMap(
    cities = {
        "Groningen": City(
            name="Groningen",
            offices=[Office(color=0, point=1), Office(color=1, merch=True)],
            upgrade="book",
            color="yellow",
            position=[113, 178],
        ),
        "Kampen": City(
            name="Kampen",
            offices=[Office(color=1), Office(color=3)],
            position=[102, 286],
        ),
        "Arnheim": City(
            name="Arnheim",
            offices=[Office(color=0), Office(color=0, merch=True), Office(color=1), Office(color=2)],
            color="red",
            position=[111, 438],
        ),
        "Duisburg": City(
            name="Duisburg",
            offices=[Office(color=0)],
            position=[119,589],
        ),
        "Coellen": City(
            name="Coellen",
            offices=[Office(color=0, point=True), Office(color=2)],
            color="yellow",
            position=[84,687],
        ),
        "Emden": City(
            name="Emden",
            offices=[Office(color=0, merch=True), Office(color=2)],
            position=[329,177],
        ),
        "Osnabruck": City(
            name="Osnabruck",
            offices=[Office(color=0), Office(color=1), Office(color=3)],
            position=[308,365],
        ),
        "Munster": City(
            name="Munster",
            offices=[Office(color=0, merch=True), Office(color=1)],
            position=[320,471],
        ), 
        "Warburg": City(
            name="Warburg",
            offices=[Office(color=1), Office(color=2)],
            position=[401,720],
        ),
        "Dortmund": City(
            name="Dortmund",
            offices=[Office(color=0, merch=True), Office(color=1)],
            position=[289,583],
        ),
        "Stade": City(
            name="Stade",
            offices=[Office(color=0, point=1, merch=True)],
            upgrade="privilege",
            color="yellow",
            position=[585,142],
        ),
        "Bremen": City(
            name="Bremen",
            offices=[Office(color=0), Office(color=1), Office(color=1, point=1, merch=True)],
            position=[542, 227],
        ),
        "Hamburg": City(
            name="Hamburg",
            offices=[Office(color=0), Office(color=1), Office(color=3)],
            upgrade="countinghouse",
            color="red",
            position=[798, 129],
        ),
        "Luneburg": City(
            name="Luneburg",
            offices=[Office(color=1, merch=True), Office(color=3)],
            position=[828, 281],
        ),
        "Brunswick": City(
            name="Brunswick",
            offices=[Office(color=0), Office(color=1), Office(color=2), Office(color=3)],
            position=[769, 403],
        ),
        "Magdeburg": City(
            name="Magdeburg",
            offices=[Office(color=0), Office(color=1), Office(color=2, point=1, merch=True)],
            position=[981, 612],
        ),        
        "Lubeck": City(
            name="Lubeck",
            offices=[Office(color=0, point=1), Office(color=2)],
            upgrade="bags",
            color="yellow",
            position=[935,140],
        ),
        "Hildesheim": City(
            name="Hildesheim",
            offices=[Office(color=0), Office(color=1)],
            position=[679, 500],
        ),        
        "Gottingen": City(
            name="Gottingen",
            offices=[Office(color=0), Office(color=0, merch=True), Office(color=2)],
            position=[636,692],
        ),
        "Minden": City(
            name="Minden",
            offices=[Office(color=0), Office(color=1), Office(color=2), Office(color=3)],
            position=[511, 426],
        ),        
        "Paderborn": City(
            name="Paderborn",
            offices=[Office(color=0), Office(color=3, merch=True)],
            position=[501,600],
        ),
        "Quedlinburg": City(
            name="Quedlinburg",
            offices=[Office(color=1, merch=True), Office(color=2, merch=True)],
            position=[869,639],
        ),
        "Perleberg": City(
            name="Regensburg",
            offices=[Office(color=0), Office(color=2), Office(color=3, merch=True)],
            position=[971, 249],
        ),
        "Goslar": City(
            name="Goslar",
            offices=[Office(color=0), Office(color=3)],
            position=[843,551],
        ),        
        "Hannover": City(
            name="Hannover",
            offices=[Office(color=0), Office(color=2)],
            position=[656,350],
        ),
        "Halle": City(
            name="Halle",
            offices=[Office(color=0, point=1), Office(color=2)],
            upgrade="keys",
            color="yellow",
            position=[947, 689],
        ),
        "Stendal": City(
            name="Konigsberg",
            offices=[Office(color=0), Office(color=0, merch=True), Office(color=2), Office(color=3)],
            upgrade="countinghouse",
            color="red",
            position=[971, 437],
        ),
    },
    routes=[
        Route(from_city="Groningen", to_city="Emden", posts=3),
        Route(from_city="Emden", to_city="Osnabruck", posts=4),
        Route(from_city="Kampen", to_city="Osnabruck", posts=2),
        Route(from_city="Kampen", to_city="Arnheim", posts=3),
        Route(from_city="Arnheim", to_city="Duisburg", posts=3),
        Route(from_city="Arnheim", to_city="Munster", posts=3),
        Route(from_city="Duisburg", to_city="Dortmund", posts=2),
        Route(from_city="Osnabruck", to_city="Bremen", posts=3, tavern=True),
        Route(from_city="Munster", to_city="Minden", posts=3),
        Route(from_city="Bremen", to_city="Minden", posts=3),
        Route(from_city="Minden", to_city="Paderborn", posts=3),
        Route(from_city="Dortmund", to_city="Paderborn", posts=3),
        Route(from_city="Coellen", to_city="Warburg", posts=4),
        Route(from_city="Paderborn", to_city="Warburg", posts=3),
        Route(from_city="Stade", to_city="Hamburg", posts=3),
        Route(from_city="Bremen", to_city="Hamburg", posts=4),
        Route(from_city="Minden", to_city="Hannover", posts=3),
        Route(from_city="Bremen", to_city="Hannover", posts=3),
        Route(from_city="Paderborn", to_city="Hildesheim", posts=3),
        Route(from_city="Hannover", to_city="Luneburg", posts=3),
        Route(from_city="Minden", to_city="Brunswick", posts=4),
        Route(from_city="Lubeck", to_city="Hamburg", posts=3),
        Route(from_city="Luneburg", to_city="Hamburg", posts=4),
        Route(from_city="Luneburg", to_city="Perleberg", posts=3, tavern=True),
        Route(from_city="Stendal", to_city="Perleberg", posts=3),
        Route(from_city="Stendal", to_city="Brunswick", posts=4),
        Route(from_city="Stendal", to_city="Magdeburg", posts=3),
        Route(from_city="Goslar", to_city="Magdeburg", posts=2),
        Route(from_city="Goslar", to_city="Quedlinburg", posts=4),
        Route(from_city="Goslar", to_city="Hildesheim", posts=3, tavern=True),
        Route(from_city="Halle", to_city="Quedlinburg", posts=4),
        Route(from_city="Gottingen", to_city="Quedlinburg", posts=3),
    ],
)

# Standard3P = BaseMap.add_neighbors()
Standard5P = BaseMap.add_neighbors()
Standard5P.routes.extend(
    [
        Route(from_city="Warburg", to_city="Gottingen", posts=3),
        Route(from_city="Emden", to_city="Stade", posts=3),
    ]
)
