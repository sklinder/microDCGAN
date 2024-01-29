import numpy as np
import matplotlib.pyplot as plt

####
font_size = 10

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': font_size})

epochs=75
num_batches = 228
####

data_1 = np.loadtxt('DCGAN_256x256_bs128_lr0001_ks4_e75_con_noweights_scans1and2_fin_FID.csv',  delimiter=',')
#data_2 = np.loadtxt('GAN_2D_bs128_lr0001_FID.csv',  delimiter=',')
#data_3 = np.loadtxt('DCGAN_2D_bs128_lr0001_FID.csv',  delimiter=',')

cm = 1/2.54  # centimeters in inches
fig, ax1 = plt.subplots(figsize=(15.5*cm, 5*cm))
ax1.plot(range(1, len(data_1)+1, 1), data_1, label="DCGAN" , color="black", linewidth=.5) 

ax1.set_ylim(0,450)
ax1.set_xlim(0,len(data_1)+1)

ax1.set_title('DCGAN')
ax1.set_xlabel("Epochs")
ax1.set_ylabel("FID")

x_ticks = np.arange(0, epochs+1, 25)
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_ticks)
#ax1.legend(loc="upper right")

# ax2 = ax1.twiny()                       # Add second axis
# ax2.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
# ax2.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
# ax2.spines['bottom'].set_position(('outward', 70))
# ax2.set_xlabel('Epochs')
# ax2.set_xlim(ax1.get_xlim())


# x_ticks = np.arange(0, epochs+1, 10)
# #newpos   = [t for t in newlabel]   # position of the xticklabels in the old x-axis
# ax2.set_xticks(x_ticks*num_batches)
# ax2.set_xticklabels(x_ticks)
# ax1.legend(loc="upper right")
# #ax1.legend(loc="upper left")

#ax2 = ax1.twinx() 
#ax2.plot(range(1, len(data_2)+1, 1), data_2, label="DCGAN (kernel size 4x4)" , color="green")   
#ax2.set_ylabel("FID (DCGAN)")
#fig.legend(loc="upper right",bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

props = dict(boxstyle='round', color = 'lightgray', alpha = 0.5, linewidth=.5)
plt.text(.68, .945, r'Image size: 256\,px \texttimes\ 256\,px''\n'r'Batch size: 128''\n'r'Learning rate: 0.0001''\n'r'Kernel size: 4\,px \texttimes\ 4\,px', fontsize=font_size-2, transform = ax1.transAxes, verticalalignment='top', horizontalalignment='left', bbox=props)



plt.plot()
#plt.show()

print('Saving images...')
print('[0/2]')
plt.savefig('./DCGAN_256x256_bs128_lr0001_ks4_e75_con_noweights_scans1and2_fin_FID_medium_2.png', dpi=250, bbox_inches='tight')
print('[1/2]')
plt.savefig('./DCGAN_256x256_bs128_lr0001_ks4_e75_con_noweights_scans1and2_fin_FID_medium_2.eps', format='eps',bbox_inches='tight', dpi=250)
print('[2/2]')