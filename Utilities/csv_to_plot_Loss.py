import numpy as np
import matplotlib.pyplot as plt

### Parameters #################################################################
font_size = 10

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': font_size})
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amssymb}']
plt.rcParams['text.latex.preamble'] = r'\nonstopmode'
###

data_1 = np.loadtxt('DCGAN_256x256_bs128_lr0001_ks4_e75_con_noweights_scans1and2_fin_loss_discriminator.csv',  delimiter=',')
data_2 = np.loadtxt('DCGAN_256x256_bs128_lr0001_ks4_e75_con_noweights_scans1and2_fin_loss_generator.csv',  delimiter=',')
#data_3 = np.loadtxt('DCGAN_2D_bs128_lr0001_FID.csv',  delimiter=',')


print('mean_discriminator:', np.mean(data_1[-300:]))

epochs=75
num_batches = 228

##################################################################################

cm = 1/2.54  # centimeters in inches
fig, ax1 = plt.subplots(figsize=(15.5*cm, 5*cm))
ax1.plot(range(1, len(data_2)+1, 1), data_2, label="Generator" , color="green", linewidth=.5) 
ax1.plot(range(1, len(data_1)+1, 1), data_1, label="Discriminator" , color="blue", linewidth=.5) 

ax1.set_title('DCGAN')
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Loss")

ax1.set_ylim(0,40)
ax1.set_xlim(0,len(data_2)+1)

ax2 = ax1.twiny()                       # Add second axis
ax2.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
ax2.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
ax2.spines['bottom'].set_position(('outward', 40))

ax2.set_xlabel('Epochs')
ax2.set_xlim(ax1.get_xlim())


x_ticks = np.arange(0, epochs+1, 15)
#newpos   = [t for t in newlabel]   # position of the xticklabels in the old x-axis
ax2.set_xticks(x_ticks*num_batches)
ax2.set_xticklabels(x_ticks)

legend = ax1.legend()
legend.get_frame().set_linewidth(.5)
ax1.legend(loc="upper left", prop={'size': font_size-2})


#fig.legend(loc="upper right",bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

props = dict(boxstyle='round', color = 'lightgray', alpha = 0.5, linewidth=.5)
plt.text(.27, .945, r'Image size: 256\,px \texttimes\ 256\,px''\n'r'Batch size: 128''\n'r'Learning rate: 0.0001''\n'r'Kernel size: 4\,px \texttimes\ 4\,px', fontsize=font_size-2, transform = ax1.transAxes, verticalalignment='top', horizontalalignment='left', bbox=props)

#\begin{flushleft}Image size: 256\,px \texttimes\ 256\,px\\Batch size: 128\\Learning rate: 0.0001\\Kernel size: 4\,px \texttimes\ 4\,px\end{flushleft}
# r'\begin{flushleft}Image size: 256\,px \texttimes\ 256\,px''\n'r'Batch size: 128''\n'r'Learning rate: 0.0001''\n'r'Kernel size: 4\,px \texttimes\ 4\,px\end{flushleft}'

plt.plot()
#plt.show()

# print('Saving images...')
print('[0/2]')
plt.savefig('./DCGAN_256x256_bs128_lr0001_ks4_e75_con_noweights_scans1and2_fin_loss_medium_2.png',bbox_inches='tight', dpi=250)
print('[1/2]')
plt.savefig('./DCGAN_256x256_bs128_lr0001_ks4_e75_con_noweights_scans1and2_fin_loss_medium_2.eps',bbox_inches='tight', format='eps', dpi=250)
print('[2/2]')