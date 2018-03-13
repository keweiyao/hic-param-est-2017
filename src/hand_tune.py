from . import cachedir, lazydict, model, expt, systems, nPDFs
from .design import Design
from .emulator import Emulator
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import \
				FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import tkinter as tk

# parameter transformation
phy_to_design = { 
'A': lambda x: np.log(1.+x),
'B': lambda x: np.log(1.+x),
'mu': lambda x: np.log(x)
}

design_to_phy = { 
'A': lambda x: np.exp(x)-1.,
'B': lambda x: np.exp(x)-1.,
'mu': lambda x: np.exp(x)
} 


class Application(tk.Frame):
    def __init__(self, Names, system='PbPb5020', master=None):
        self.obs = ['RAA', 'V2']
        self.cenlist = {'RAA':['0-10','0-100'], 'V2':['0-10','10-30','30-50']}
        self.ylim = {'RAA': [0,1], 'V2': [-0.05, 0.30]}
        self.system = system
        self.Names = Names
        super().__init__(master)
        self.createWidgets()
        self.emu = {nPDF: Emulator.from_cache(system, nPDF) for nPDF in nPDFs}
        self.nPDF = 'nCTEQ' # default
       
        self.plot_exp()

    def toggle(self):
        if self.change_nPDF.config('text')[-1] == 'nCTEQ, click to switch':
            self.change_nPDF.config(text='EPS09, click to switch')
            self.nPDF = 'EPS09'
        else:
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
        self.ax=[[ax1, ax2], [ax3, ax4, ax5]]

        self.canvas=FigureCanvasTkAgg(fig, master=root)
        self.canvas.get_tk_widget().grid(row=0,column=1)
        plt.tight_layout(True)
        plt.subplots_adjust(top=0.9)
        self.canvas.show()

        # plot
        self.plotbutton=tk.Button(master=root, text="plot", 
                           command=lambda: self.plot_emu())
        self.plotbutton.grid(row=1,column=0)

        # switching nPDF
        self.change_nPDF = tk.Button(text="nCTEQ, click to switch", command=lambda: [self.toggle(), self.plot_emu()])
        self.change_nPDF.grid(row=2, column=0)

        # quit
        self.quit = tk.Button(master=root, text="QUIT", fg="red",
                              command=root.destroy)
        self.quit.grid(row=3, column=0)

        ranges = Design(self.system).range
        for name, row, limits in zip(self.Names, range(1,len(self.Names)+1), ranges):
            # Get the range of the parameters
            l, h = limits
            L, H = design_to_phy[name](l), design_to_phy[name](h)
            # add the variable holder, start from the midpoint of design
            setattr(self, name, l)
            # add the slide scale for this variable
            setattr(self, 'tune'+name, tk.Scale(master=root, 
                from_=L, to=H, resolution=(H-L)/50., length=500, 
                orient="horizontal", tickinterval=1))

            # labelling the slide scale
            setattr(self, 'label'+name, tk.Label(master=root, text=name))
            getattr(self, 'label'+name).grid(row=row, column=2)

            getattr(self, 'tune'+name).set(1.0)
            getattr(self, 'tune'+name).grid(row=row, column=1, columnspan=2)
            getattr(self, 'tune'+name).bind("<ButtonRelease>",
                lambda event: self.plot_emu() )  

    def plot_setting(f):
        def df(self):
            for var_name in self.Names:
                phy_var = getattr(self, "tune"+var_name).get()
                design_var = phy_to_design[var_name](phy_var)
                setattr(self, var_name, design_var)
            f(self)
            plt.tight_layout(True)
            plt.subplots_adjust(top=0.9)
            self.canvas.draw()
        return df

    @plot_setting
    def plot_exp(self):
        for obs, axrow in zip(self.obs, self.ax):
            for cen, axi in zip(self.cenlist[obs], axrow):
                dset = expt.data[self.system][obs]['D0'][cen]
                x = dset['x']
                y = dset['y']
                xerr = [(ph-pl)/2. for (pl, ph) in dset['pT']]
                yerr = np.sqrt(sum(
                    e**2 for e in dset['yerr'].values()
                ))
                axi.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='kD')
                axi.set_ylim(self.ylim[obs])
                axi.set_ylabel(obs + '@'+ cen + '%')
                axi.set_xlabel(r'$p_T$ [GeV]')

    @plot_setting
    def plot_emu(self):
        pred = self.emu[self.nPDF].predict(
                np.array([[self.__dict__[name] for name in self.Names]]))
        for obs, axrow in zip(self.obs, self.ax):
            for cen, axi in zip(self.cenlist[obs], axrow):
                x = expt.data[self.system][obs]['D0'][cen]['x']
                if len(axi.lines) == 2:
                    del axi.lines[-1]
                axi.plot(x, pred[obs]['D0'][cen][0], 'r')
        plt.suptitle(r"{} $\mu = {:1.2f}, A = {:1.2f}, B = {:1.2f}$ "\
                    .format(self.nPDF, *[design_to_phy[name](self.__dict__[name]) 
                            for name in self.Names])+r"[GeV${}^2$]")

if __name__ == '__main__':
    root = tk.Tk()
    app = Application(['mu', 'A', 'B'], master=root)
    app.master.title('Hand tuning your parameters')
    app.mainloop()
