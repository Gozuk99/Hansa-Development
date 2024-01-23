Project created to play Hansa Teutonica against an AI/Computer.

Long Term goal is to have an evaluation bar, and the ability to load game states, and make suggestions.

Use and tested on:
Python 3.11.4
import pygame
import sys
import random
import torch
import gc

python sample_hansa_game.py

Left Click to claim with square.
Right Click to claim with circle.

Shift-Click an Upgrade (yellow city), to auto upgrade an ability. Used for testing/training, calling it "Admin Mode".
AI still needs to be trained, models not uploaded yet.

Long term goals:
Mini Expansion Bonus Markers
Reward structure for more advanced scenarios (ex: blocking East/West Connection completion)
