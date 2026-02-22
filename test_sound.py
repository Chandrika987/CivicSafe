import winsound

winsound.PlaySound("alert.wav",
                   winsound.SND_FILENAME | winsound.SND_ASYNC)

input("Press Enter to exit...")