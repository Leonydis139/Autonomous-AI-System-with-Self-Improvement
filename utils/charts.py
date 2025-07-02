
import matplotlib.pyplot as plt, io, base64
CLR='#1B47A7';BG='#1A1A1A';FG='#EDEDED';GRID='#6D6D6D'

def lengths_chart(texts):
    vals=[len(t) for t in texts]
    plt.subplots(figsize=(9,6));plt.subplots_adjust(left=0.15,right=0.85,top=0.85,bottom=0.15)
    plt.bar(range(len(vals)), vals, color=CLR)
    plt.title('Result title lengths', pad=15, color=FG)
    plt.xlabel('Result', labelpad=10, color=FG);plt.ylabel('Chars', labelpad=10, color=FG)
    ax=plt.gca();ax.set_axisbelow(True)
    ax.set_facecolor(BG);plt.gcf().patch.set_facecolor(BG)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.grid(True,color=GRID,linestyle='--')
    buf=io.BytesIO();plt.savefig(buf,format='png',facecolor=BG);plt.close();buf.seek(0)
    return base64.b64encode(buf.read()).decode()
