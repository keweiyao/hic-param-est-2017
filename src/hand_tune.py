from . import cachedir, lazydict, model, expt, systems, nPDFs
from .design import Design
from .emulator import Emulator
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import tkinter as tk

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.createWidgets()
        self.emu = {nPDF: Emulator('PbPb5020', nPDF) for nPDF in nPDFs}
        self.A = 1.0
        self.B = 1.0
        self.M = 1.0
        self.nPDF = 'nCTEQ' # default

    def toggle(self):
        if self.change_nPDF.config('text')[-1] == 'nCTEQ, click to switch':
            print('1')
            self.change_nPDF.config(text='EPS09, click to switch')
            self.nPDF = 'EPS09'
        else:
            print('2')
            self.change_nPDF.config(text='nCTEQ, click to switch')
            self.nPDF = 'nCTEQ'

    def createWidgets(self):
        fig=plt.figure(figsize=(8,6))
        gs = GridSpec(2, 6)
        ax1 = plt.subplot(gs[0, :3])
        ax2 = plt.subplot(gs[0, 3:])
        ax3 = plt.subplot(gs[1, :2])
        ax4 = plt.subplot(gs[1, 2:4])
        ax5 = plt.subplot(gs[1, 4:])
        ax=[[ax1, ax2], [ax3, ax4, ax5]]

        canvas=FigureCanvasTkAgg(fig,master=root)
        canvas.get_tk_widget().grid(row=0,column=1)
        plt.tight_layout(True)
        plt.subplots_adjust(top=0.9)
        canvas.show()

        # plot
        self.plotbutton=tk.Button(master=root, text="plot", 
                           command=lambda: self.plot(canvas,ax))
        self.plotbutton.grid(row=1,column=0)

        # switching nPDF
        self.change_nPDF = tk.Button(text="nCTEQ, click to switch", command=lambda: [self.toggle(), self.plot(canvas,ax)])
        self.change_nPDF.grid(row=2, column=1)

        # tune A
        label = tk.Label(master=root, text="A")
        label.grid(row=1, column=2)
        self.tuneA = tk.Scale(master=root, from_=0.0, to=10.0,
            resolution=0.1, length=500, orient="horizontal",tickinterval=1)
        self.tuneA.set(5.0)
        self.tuneA.grid(row=1, column=1, columnspan=2)
        self.tuneA.bind("<ButtonRelease>",
            lambda event: self.changeA(canvas, ax, final_update=True))
        self.tuneA.bind("<Motion>",
            lambda event: self.changeA(canvas, ax, final_update=False))

        # tune B
        label = tk.Label(master=root, text="B [GeV^2]")
        label.grid(row=2, column=2)
        self.tuneB = tk.Scale(master=root, from_=0.0, to=10.0,
            resolution=0.1, length=500, orient="horizontal",tickinterval=1)
        self.tuneB.set(5.0)
        self.tuneB.grid(row=2, column=1, columnspan=2)
        self.tuneB.bind("<ButtonRelease>",
            lambda event: self.changeB(canvas, ax, final_update=True))
        self.tuneB.bind("<Motion>",
            lambda event: self.changeB(canvas, ax, final_update=False))
 
        # tune M
        label = tk.Label(master=root, text="log(\mu)")
        label.grid(row=3, column=2)
        self.tuneM = tk.Scale(master=root, from_=-0.5, to=1.5,
            resolution=0.02, length=500, orient="horizontal",tickinterval=.5)
        self.tuneM.set(0.0)
        self.tuneM.grid(row=3, column=1, columnspan=2)
        self.tuneM.bind("<ButtonRelease>",
            lambda event: self.changeM(canvas, ax, final_update=True))
        self.tuneM.bind("<Motion>",
            lambda event: self.changeM(canvas, ax, final_update=False))


        # quit
        self.quit = tk.Button(master=root, text="QUIT", fg="red",
                              command=root.destroy)
        self.quit.grid(row=3, column=0)

    def changeA(self, canvas, ax, final_update):
        if final_update:
            self.A = np.log(1.+self.tuneA.get())
            self.plot(canvas, ax)
    def changeB(self, canvas, ax, final_update):
        if final_update:
            self.B = np.log(1.+self.tuneB.get())
            self.plot(canvas, ax)
    def changeM(self, canvas, ax, final_update):
        if final_update:
            self.M = self.tuneM.get()
            self.plot(canvas, ax)


    def plot(self, canvas, ax):
        cenlist = {'RAA':['0-10','0-100'], 'V2':['0-10','10-30','30-50']}
        ylim = {'RAA': [0,1], 'V2': [-0.05, 0.30]}
        for obs, axrow in zip(['RAA', 'V2'], ax):
            for cen, axi in zip(cenlist[obs], axrow):
                axi.clear()
                dset = expt.data['PbPb5020'][obs]['D0'][cen]
                x = dset['x']
                y = dset['y']
                xerr = [(ph-pl)/2. for (pl, ph) in dset['pT']]
                yerr = np.sqrt(sum(
                    e**2 for e in dset['yerr'].values()
                ))
                axi.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='kD')
                pred = self.emu[self.nPDF].predict(np.array([[self.M, self.A, self.B]]))
                axi.plot(x, pred[obs]['D0'][cen][0], 'r')
                axi.set_ylim(ylim[obs])
                axi.set_ylabel(obs + '@'+ cen + '%')
                axi.set_xlabel(r'$p_T$ [GeV]')
        plt.suptitle(r"{} $\mu = {:1.2f}, A = {:1.2f}, B = {:1.2f}$ ".format(self.nPDF, np.exp(self.M), self.A, self.B)+r"[GeV${}^2$]")
        plt.tight_layout(True)
        plt.subplots_adjust(top=0.9)
        canvas.draw()

       




if __name__ == '__main__':
    root = tk.Tk()
    app = Application(master=root)
    app.master.title('Hand tuning your parameters')
    app.mainloop()
