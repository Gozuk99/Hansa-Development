import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import TheHanseaticLeagueMap

class StartScreen:
    def __init__(self, root):
        self.root = root
        self.root.title("Hansa Teutonica Game")
        self.map_photo = None  # Store the map photo as an attribute

        # Dropdown for number of players
        self.players_var = tk.StringVar()
        self.players_choices = ['3', '4', '5']
        self.players_dropdown = ttk.Combobox(root, textvariable=self.players_var, values=self.players_choices)
        self.players_dropdown.grid(column=0, row=0)
        self.players_dropdown.current(0)
        # self.players_dropdown.bind("<<ComboboxSelected>>", self.update_map_choices)

        # Dropdown for map choice
        self.maps_var = tk.StringVar()
        self.maps_choices = ['Map1 - The Hanseatic League', 'Map2 - The Easter Hanseatic League', 'Map3 - Britannia']
        self.maps_dropdown = ttk.Combobox(root, textvariable=self.maps_var, values=self.maps_choices)
        self.maps_dropdown.grid(column=0, row=1)
        self.maps_dropdown.current(0)
        
        # Button to proceed with player and map selection
        self.button = tk.Button(root, text="Start Game", command=self.open_map_image)
        self.button.grid(column=0, row=2)

    # def update_map_choices(self, event):
        # selected_players = self.players_var.get()

        # if selected_players == '3':
            # self.maps_choices = ['Map1 - The Hanseatic League (3 player version)', 'Map3 - Britannia (3 player version)', 'Map2 - The Easter Hanseatic League']
        # else:
            # self.maps_choices = ['Map1 - The Hanseatic League (4-5 player version)', 'Map3 - Britannia (4-5 player version)']
        
        # self.maps_dropdown['values'] = self.maps_choices
        # self.maps_dropdown.current(0)
    def open_map_image(self):
        selected_players = self.players_var.get()
        selected_map = self.maps_var.get()

        # Load the appropriate map image based on selections
        if selected_players == '3' and selected_map.startswith('Map1'):
            map_image = Image.open("Map1_3player.jpg")
        elif selected_players == '3' and selected_map.startswith('Map3'):
            map_image = Image.open("Map3_3player.jpg")
        elif selected_players != '3' and selected_map.startswith('Map1'):
            map_image = Image.open("Map1_4-5p.jpg")
        elif selected_players != '3' and selected_map.startswith('Map3'):
            map_image = Image.open("Map3_4-5player.jpg")
        elif selected_map.startswith('Map2'):
            map_image = Image.open("Map2.jpg")
        else:
            # Handle invalid selections
            return

        # Destroy the root window
        self.root.destroy()

        # Create a new window to display the map image
        map_window = tk.Tk()
        map_window.title("Game Map")

        # Resize the map image if necessary
        if map_image.width > 1080 or map_image.height > 813:
            map_image = map_image.resize((1080, 813))

        # Convert the map image to PhotoImage
        self.map_photo = ImageTk.PhotoImage(map_image)

        # Create a canvas to draw the game board
        canvas = tk.Canvas(map_window, width=1080, height=813)
        canvas.pack()

        # Draw the map image as the background
        canvas.create_image(0, 0, anchor=tk.NW, image=self.map_photo)
        
        # Define color mapping for offices
        color_mapping = {0: "white", 1: "orange", 2: "pink", 3: "black"}

        for city_name, city_data in TheHanseaticLeagueMap.Standard5P.cities.items():
            city_position = city_data.position
            city_color = "yellow" if city_data.color == "yellow" else "grey"
            
            office_size = 15  # Increased office size
            merchant_size = 18  # Bigger size for merchants
            offset = office_size + 6  # Adding less space between offices

            city_width = office_size + offset * (len(city_data.offices) - 1)  # Calculate the city width based on number of offices
            city_height = 20

            start_x = city_position[0] - city_width // 2  # Calculate the initial position of the first office based on city width
            start_y = city_position[1] - office_size // 2

            # Draw a square or rectangle representing the city on the canvas
            canvas.create_rectangle(city_position[0]-city_width//2, city_position[1]-city_height//2, city_position[0]+city_width//2, city_position[1]+city_height//2, fill=city_color)

            for i, office in enumerate(city_data.offices):
                office_x = start_x + i * offset
                office_y = start_y
                office_color = color_mapping[office.color]

                if office.merch:
                    # Draw a circle for merchant offices, with a bigger size and adjusted y-position for alignment
                    canvas.create_oval(office_x, office_y - (merchant_size - office_size) // 2, office_x + merchant_size, office_y + merchant_size - (merchant_size - office_size) // 2, fill=office_color)
                else:
                    # Draw a square for normal offices
                    canvas.create_rectangle(office_x, office_y, office_x + office_size, office_y + office_size, fill=office_color)
        
        # Here, we will draw the routes and posts after drawing all the cities and offices
        post_size = 10
        for route in TheHanseaticLeagueMap.Standard5P.routes:
            start_city_name = route.from_city
            end_city_name = route.to_city

            start_city_pos = TheHanseaticLeagueMap.Standard5P.cities[start_city_name].position
            end_city_pos = TheHanseaticLeagueMap.Standard5P.cities[end_city_name].position

            # Calculate the vector for the route
            dx = end_city_pos[0] - start_city_pos[0]
            dy = end_city_pos[1] - start_city_pos[1]

            # Calculate the length (distance) of the route
            distance = math.sqrt(dx**2 + dy**2)

            # Normalize the vector
            dx /= distance
            dy /= distance

            # Calculate the position of the first post
            first_post_x = start_city_pos[0] + city_width/2 + dx*post_distance  # city_width/2 ensures the post isn't drawn over the city
            first_post_y = start_city_pos[1] + dy*post_distance

            # Draw the posts
            for i in range(route.posts):
                post_x = first_post_x + i*dx*post_distance
                post_y = first_post_y + i*dy*post_distance

                if i%2 == 0:  # Square post
                    canvas.create_rectangle(post_x-post_size/2, post_y-post_size/2, post_x+post_size/2, post_y+post_size/2, outline="black")
                else:  # Circular post
                    canvas.create_oval(post_x-post_size/2, post_y-post_size/2, post_x+post_size/2, post_y+post_size/2, outline="black")

            # Draw the route line
            canvas.create_line(start_city_pos[0]+city_width/2, start_city_pos[1], end_city_pos[0]-city_width/2, end_city_pos[1], fill="black")
            
        # Run the event loop for the new window
        map_window.mainloop()


# root = tk.Tk()
# start_screen = StartScreen(root)
# # Load the map image
# map_image = Image.open("Hansa-Teutonica-Hansa-Teutonica.jpg")
# map_photo = ImageTk.PhotoImage(map_image)

# # Create a canvas to draw the game board
# canvas = tk.Canvas(root, width=1080, height=813)
# canvas.pack()

# # Draw the map image as the background
# canvas.create_image(0, 0, anchor=tk.NW, image=map_photo)

root = tk.Tk()
start_screen = StartScreen(root)
root.mainloop()
