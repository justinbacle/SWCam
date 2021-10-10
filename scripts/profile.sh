python3.8 -m cProfile -o profile.prof SWcam.py
python3.8 -m snakeviz profile.prof
