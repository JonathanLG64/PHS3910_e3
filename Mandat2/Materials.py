class Material:
    def __init__(self, sound_speed, density) -> None:
        sound_speed = sound_speed
        density = density

matériaux = { 'aluminium': Material(sound_speed=5000, density=2700),
              'plexiglass' : Material(sound_speed=2690, density=1180),
              'copper' : Material(sound_speed=4760, density=8960),
              'acrylic' : Material(sound_speed=1430, density=1180), #https://www.rshydro.co.uk/sound-speeds/
              'steel' : Material(sound_speed=5100, density=7850)
}

class Note:
    def __init__(self) -> None:
        pass

