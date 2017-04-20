import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

def show_video_subtitle(frames, subtitle):
    fig, ax = plt.subplots()
    fig.show()

    text = plt.text(0.5, 0.1, "", 
        ha='center', va='center', transform=ax.transAxes, 
        fontdict={'fontsize': 15, 'color':'white', 'fontweight': 500})
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),
        path_effects.Normal()])

    subs = subtitle.split()
    inc = max(len(frames)/(len(subs)+1), 0.01)

    i = 0
    img = None
    for frame in frames:
        sub = " ".join(subs[:int(i/inc)])

        text.set_text(sub)

        if img is None:
            img = plt.imshow(frame)
        else:
            img.set_data(frame)
        fig.canvas.draw()
        i += 1