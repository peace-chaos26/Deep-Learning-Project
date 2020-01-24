# Add necessary imports
import threading
import time
import tkinter.messagebox
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from ttkthemes import themed_tk as tk
from mutagen.mp3 import MP3
from pygame import mixer
import glob
import face_recognition as face_expression_recognizer
import os
root = tk.ThemedTk()
root.get_themes()
root.set_theme("radiance")


statusbar = ttk.Label(root, text="Welcome to Music Player", relief=SUNKEN, anchor=W, font='Times 10')
statusbar.pack(side=BOTTOM, fill=X)

# Create the menubar
menubar = Menu(root)
root.config(menu=menubar)

# Create the submenu
subMenu = Menu(menubar, tearoff=0)
playlist = []
emotion_dict = {
    "Angry" : "angry",
    "Disgust": "angry",
    "Fear": "sad",
    "Happy": "happy",
    "Sad": "sad",
    "Surprise": "happy",
    "Neutral":"relaxed"
    }

def merge_emotion_dict(emotion):
    return emotion_dict[emotion]

#function to browse audio files
def browse_file():
    refresh_playlist()
    emotion = face_expression_recognizer.get_emotion()
    print("Emotion retrieved : "+emotion)
    emotion = merge_emotion_dict(emotion)
    print("Retrieving songs for emotion: "+emotion)
    preprocess_emotion(emotion)
    stop_song()

#function to clear playlist when new emotion is detected
def refresh_playlist():
    playlistbox.delete(0,'end')
    playlist = []

#function to iterate over songs of particular emotion and add to playlist
def preprocess_emotion(emotion):
    #Process emotions
    path = "/home/sahil/G:/minor/songs/"+emotion+"/"
    audio_files = os.listdir(path)
    print("Audio files for emotion:"+emotion+" are:")
    index = 0
    for file in audio_files:
        print(str(index)+": "+file)
        index = index + 1

    for file in audio_files:
        global filename_path
        filename_path = path+file
    try:
        mixer.music.queue(filename_path)
    except:
        print("Skipped: "+file)
        add_to_playlist(filename_path)

# function to add a song file to playlist
def add_to_playlist(filename):
    print("Adding :"+filename)
    filename = os.path.basename(filename)
    index = 0
    playlistbox.insert(index, filename)
    playlist.insert(index, filename_path)
    index += 1

#add menu items
menubar.add_cascade(label="File", menu=subMenu)
subMenu.add_command(label="Open", command=browse_file)
subMenu.add_command(label="Exit", command=root.destroy)
subMenu = Menu(menubar, tearoff=0)
root.title("Player")
mixer.init()


leftframe = Frame(root)
leftframe.pack(side=LEFT, padx=30, pady=30)

playlistbox = Listbox(leftframe)
playlistbox.pack()

addBtn = ttk.Button(leftframe, text="+ Add to playlist", command=browse_file)
addBtn.pack(side=LEFT)

#function to delete a song from playlist
def del_song():
    selected_song = playlistbox.curselection()
    selected_song = int(selected_song[0])
    playlistbox.delete(selected_song)
    playlist.pop(selected_song)


delBtn = ttk.Button(leftframe, text="- Del", command=del_song)
delBtn.pack(side=LEFT)

rightframe = Frame(root)
rightframe.pack(pady=30)

topframe = Frame(rightframe)
topframe.pack()

#function to play music
def play_song():
    global paused

    if paused:
        mixer.music.unpause()
        statusbar['text'] = "Music Resumed"
        paused = FALSE
    else:
        try:
            stop_song()
            time.sleep(1)
            selected_song = playlistbox.curselection()
            selected_song = int(selected_song[0])
            play_it = playlist[selected_song]
            print(play_it)
            mixer.music.load(play_it)
            mixer.music.play()
            statusbar['text'] = "Playing music" + ' - ' + os.path.basename(play_it)

        except:
            tkinter.messagebox.showerror('File not found', 'Melody could not find the file. Please check again.')

#function to stop music
def stop_song():
    mixer.music.stop()
    statusbar['text'] = "Music Stopped"
#set paused state to false
paused = FALSE

# Pause music
def pause_music():
    global paused
    paused = TRUE
    mixer.music.pause()
    statusbar['text'] = "Music Paused"


def rewind_music():
    play_song()
    statusbar['text'] = "Music Rewinded"


def set_vol(val):
    volume = float(val) / 100
    mixer.music.set_volume(volume)

muted = FALSE

#define mute music
def mute_music():
    global muted
    if muted:
        mixer.music.set_volume(0.7)
        volumeBtn.configure(image=volumePhoto)
        scale.set(70)
        muted = FALSE
    else:
        mixer.music.set_volume(0)
        volumeBtn.configure(image=mutePhoto)
        scale.set(0)
        muted = TRUE


middleframe = Frame(rightframe)
playPhoto = PhotoImage(file='images/play1.png')
playBtn = ttk.Button(middleframe, image=playPhoto, command=play_song)
stopPhoto = PhotoImage(file='images/stop1.png')
stopBtn = ttk.Button(middleframe, image=stopPhoto, command=stop_song)
pausePhoto = PhotoImage(file='images/pause1.png')
pauseBtn = ttk.Button(middleframe, image=pausePhoto, command=pause_music)
bottomframe = Frame(rightframe)
rewindPhoto = PhotoImage(file='images/rewind1.png')
rewindBtn = ttk.Button(bottomframe, image=rewindPhoto, command=rewind_music)
mutePhoto = PhotoImage(file='images/mute1.png')
volumePhoto = PhotoImage(file='images/volume1.png')
volumeBtn = ttk.Button(bottomframe, image=volumePhoto, command=mute_music)
scale = ttk.Scale(bottomframe, from_=0, to=100, orient=HORIZONTAL, command=set_vol)

#Position various elements
def set_properties():
    middleframe.pack(pady=30, padx=30)
    playBtn.grid(row=1, column=1, padx=10)
    stopBtn.grid(row=1, column=2, padx=10)
    pauseBtn.grid(row=1, column=3, padx=10)
    bottomframe.pack()
    rewindBtn.grid(row=0, column=0)
    volumeBtn.grid(row=0, column=1)
    scale.set(70)
    mixer.music.set_volume(0.7)
    scale.grid(row=0, column=2, pady=15, padx=30)

set_properties()
def on_closing():
    stop_song()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
