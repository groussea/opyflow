This test case aim to download any video and perform opyFlow on it.

#command lines To save the online youtube video of the land slide (or any other video) (you need the pytube package)
# 'pip install pytube' on the cmd prompt

from pytube import YouTube
#You may run this script for any video link from youtube
yt=YouTube("https://www.youtube.com/watch?time_continue=2&v=5-nyAz484WA")
stre=yt.streams.first()
stre.download(folder_main)
