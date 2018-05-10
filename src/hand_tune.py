from . import cachedir, lazydict, model, expt, systems, nPDFs
from .design import Design
from .plots import observables_at
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
'tau_0': lambda x: x,
'qhat_A': lambda x: np.log(1.+x),
'qhat_B': lambda x: np.log(1.+x),
'mu': lambda x: np.log(x)
}

design_to_phy = { 
'tau_0': lambda x: x,
'qhat_A': lambda x: np.exp(x)-1.,
'qhat_B': lambda x: np.exp(x)-1.,
'mu': lambda x: np.exp(x)
} 


class Application(tk.Frame):
    def __init__(self, Names, master=None):
        self.Names = Names
        super().__init__(master)
        self.nPDF = 'EPPS' # default
        self.system = 'PbPb5020'
        self.createWidgets()

    def toggle(self):
        if self.change_nPDF.config('text')[-1] == 'nCTEQ, click to switch':
            self.change_nPDF.config(text='EPPS, click to switch')
            self.nPDF = 'EPPS'
        else:
            self.change_nPDF.config(text='nCTEQ, click to switch')
            self.nPDF = 'nCTEQ'

    def createWidgets(self):
        fig=plt.figure(figsize=(8,6))
        gs = GridSpec(4, 3)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[0, 2])
        ax4 = plt.subplot(gs[1, 0])
        ax5 = plt.subplot(gs[1, 1])
        ax6 = plt.subplot(gs[1, 2])
        ax7 = plt.subplot(gs[2, 0])
        ax8 = plt.subplot(gs[2, 1])
        ax9 = plt.subplot(gs[2, 2])
        ax10 = plt.subplot(gs[3, 0])
        ax11 = plt.subplot(gs[3, 1])
        ax12 = plt.subplot(gs[3, 2])
        system = 'PbPb5020'
        self.axes = [[ax1, ax2, ax3], [ax4, ax5, ax6], [ax7, ax8, ax9], [ax10, ax11], [ax12]]

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
    def plot_emu(self):
        val = [self.__dict__[name] for name in self.Names]
        print(val)
        observables_at(val, self.nPDF, self.axes)


if __name__ == '__main__':
    root = tk.Tk()
    app = Application(['tau_0', 'mu', 'qhat_A', 'qhat_B'], master=root)
    app.master.title('Hand tuning your parameters')
    app.mainloop()
