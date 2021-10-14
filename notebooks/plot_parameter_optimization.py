# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 17:45:21 2021

@author: ab6801

analyze and plot results of experiments
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import yaml


#E2: How large can I make my output domainn without loosing skill?
E2_results = pd.read_csv('E2_results_t2m_34_t2m.csv',sep =';')

#E1:
E1_results = pd.read_csv('E1_results_t2m_34_t2m.csv',sep =';')

#E1 label smoothing
E1_smooth_results = pd.read_csv('E1_label_smoothing_results_t2m_34_t2m.csv',sep =';')

#%%
###############################################################################
###################################E2
#E2 sharpness of forecasts
E2_sharpness = E2_results[['fold_no','radius_basis_func','epochs', 'accuracy', 'loss', 'RPSS_year1', 'RPSS_year2', 'min_percentage', 'max_percentage']]
E2_sharpness = E2_sharpness.melt(id_vars = ['fold_no','radius_basis_func','epochs', 'accuracy', 'loss',  'RPSS_year1', 'RPSS_year2'], 
                                           value_vars = ['min_percentage', 'max_percentage'], var_name = 'minmax', value_name = 'probability')
sb.lineplot(x="epochs", y="probability",hue="radius_basis_func", style="minmax", data=E2_sharpness)
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)#ncol = 2)
plt.xticks([5,10])
plt.tight_layout()
plt.savefig('plots/E2_sharpness.png')

#%%
#E2 achieved RPSS
results_epoch_end = E2_results[['fold_no','radius_basis_func','epochs', 'accuracy', 'loss', 'RPSS_year1', 'RPSS_year2', 'min_percentage', 'max_percentage']]
results_epoch_end = results_epoch_end.melt(id_vars = ['fold_no','radius_basis_func','epochs', 'accuracy', 'loss',  'min_percentage', 'max_percentage'], 
                                           value_vars = ['RPSS_year1', 'RPSS_year2'], var_name = 'year', value_name = 'RPSS')

sb.lineplot( data = results_epoch_end, x = 'epochs', y = 'RPSS', hue = 'radius_basis_func')#, style = 'year')
plt.xticks([5,10])
plt.hlines(y = 0, xmin= results_epoch_end.epochs.min(), xmax = results_epoch_end.epochs.max(), color = 'black')
plt.tight_layout()
plt.savefig('plots/E2_RPSS.png')

#%%
#E2 history
def extract_history(results):
    df_list = []
    for i in range(0,results.shape[0],2):
        accuracy = yaml.load(results.history[i])['accuracy'] + yaml.load(results.history[i + 1])['accuracy']
        val_accuracy = yaml.load(results.history[i])['val_accuracy'] + yaml.load(results.history[i + 1])['val_accuracy']
        loss = yaml.load(results.history[i])['loss'] + yaml.load(results.history[i + 1])['loss']
        val_loss = yaml.load(results.history[i])['val_loss'] + yaml.load(results.history[i + 1])['val_loss']
        
        df_tr = pd.DataFrame(accuracy, columns = ['accuracy'])
        df_tr['loss'] = loss
        df_tr['dataset'] = 'train'
        df_tr['epochs'] = np.array([1,2,3,4,5,6,7,8,9,10])
        
        df_val = pd.DataFrame(val_accuracy, columns = ['accuracy'])
        df_val['loss'] = val_loss
        df_val['dataset'] = 'validation'
        df_val['epochs'] = np.array([1,2,3,4,5,6,7,8,9,10])
        
        df = pd.concat([df_tr, df_val])
       # df['val_accuracy'] = val_accuracy
        #df['loss'] = loss 
       # df['val_loss'] = val_loss
        df['radius'] = results.radius_basis_func[i]
        df['fold'] = results.fold_no[i]
        
        
        df_list.append(df)
    history = pd.concat(df_list)
    history.reset_index(inplace = True)
    return history
E2_history = extract_history(E2_results)
#%%
#E2 train and val accuracy and loss from the model history 
sb.lineplot(x = 'epochs', y = 'accuracy', hue = 'radius', style = 'dataset',  data = E2_history)
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.tight_layout()
plt.savefig('plots/E2_accuracy.png')
plt.figure()
sb.lineplot(x = 'epochs', y = 'loss', hue = 'radius', style = 'dataset',  data = E2_history)
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.tight_layout()
plt.savefig('plots/E2_loss.png')
#%%
###############################################################################
##########################E1
#E1 sharpness
E1_sharpness = E1_results[['fold_no','radius_basis_func','input_lats','input_lons','epochs', 'accuracy', 'loss', 'RPSS_year1', 'RPSS_year2', 'min_percentage', 'max_percentage']]
E1_sharpness = E1_sharpness.melt(id_vars = ['fold_no','radius_basis_func','input_lats','input_lons','epochs', 'accuracy', 'loss',  'RPSS_year1', 'RPSS_year2'], 
                                           value_vars = ['min_percentage', 'max_percentage'], var_name = 'minmax', value_name = 'probability')
sb.lineplot(x="epochs", y="probability",hue="input_lats", style="minmax", data=E1_sharpness)
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)#ncol = 2)
plt.xticks([5,10])
plt.tight_layout()
plt.savefig('plots/E1_sharpness.png')
#%%
#E1 achieved RPSS
results_epoch_end_E1 = E1_results[['fold_no','radius_basis_func','input_lats','input_lons','epochs', 'accuracy', 'loss', 'RPSS_year1', 'RPSS_year2', 'min_percentage', 'max_percentage']]
results_epoch_end_E1 = results_epoch_end_E1.melt(id_vars = ['fold_no','radius_basis_func','input_lats','input_lons','epochs', 'accuracy', 'loss',  'min_percentage', 'max_percentage'], 
                                           value_vars = ['RPSS_year1', 'RPSS_year2'], var_name = 'year', value_name = 'RPSS')

sb.lineplot( data = results_epoch_end_E1, x = 'epochs', y = 'RPSS', hue = 'input_lats')#, style = 'year')
plt.xticks([5,10])
plt.hlines(y = 0, xmin= results_epoch_end_E1.epochs.min(), xmax = results_epoch_end_E1.epochs.max(), color = 'black')
#plt.axhline(y = 0)
#plt.show()
plt.tight_layout()
plt.savefig('plots/E1_RPSS.png')
#%%
#E1 history
def extract_history_E1(results):
    df_list = []
    for i in range(0,results.shape[0],2):
        accuracy = yaml.load(results.history[i])['accuracy'] + yaml.load(results.history[i + 1])['accuracy']
        val_accuracy = yaml.load(results.history[i])['val_accuracy'] + yaml.load(results.history[i + 1])['val_accuracy']
        loss = yaml.load(results.history[i])['loss'] + yaml.load(results.history[i + 1])['loss']
        val_loss = yaml.load(results.history[i])['val_loss'] + yaml.load(results.history[i + 1])['val_loss']
        
        df_tr = pd.DataFrame(accuracy, columns = ['accuracy'])
        df_tr['loss'] = loss
        df_tr['dataset'] = 'train'
        df_tr['epochs'] = np.array([1,2,3,4,5,6,7,8,9,10])
        
        df_val = pd.DataFrame(val_accuracy, columns = ['accuracy'])
        df_val['loss'] = val_loss
        df_val['dataset'] = 'validation'
        df_val['epochs'] = np.array([1,2,3,4,5,6,7,8,9,10])
        
        df = pd.concat([df_tr, df_val])
       # df['val_accuracy'] = val_accuracy
        #df['loss'] = loss 
       # df['val_loss'] = val_loss
        df['radius'] = results.radius_basis_func[i]
        df['input_lats'] = results.input_lats[i]
        df['input_lons'] = results.input_lons[i]
        df['fold'] = results.fold_no[i]
        
        
        df_list.append(df)
    history = pd.concat(df_list)
    history.reset_index(inplace = True)
    return history

E1_history = extract_history_E1(E1_results)

sb.lineplot(x = 'epochs', y = 'accuracy', hue = 'input_lats', style = 'dataset',  data = E1_history)
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.tight_layout()
plt.savefig('plots/E1_accuracy.png')
plt.figure()
sb.lineplot(x = 'epochs', y = 'loss', hue = 'input_lats', style = 'dataset',  data = E1_history)
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.tight_layout()
plt.savefig('plots/E1_loss.png')
#%%
###############################################################################
##########################E1
#E1 sharpness
E1_sharpness = E1_smooth_results[['fold_no','radius_basis_func','input_lats','input_lons','label_smoothing','epochs', 'accuracy', 'loss', 'RPSS_year1', 'RPSS_year2', 'min_percentage', 'max_percentage']]
E1_sharpness = E1_sharpness.melt(id_vars = ['fold_no','radius_basis_func','input_lats','input_lons','label_smoothing','epochs', 'accuracy', 'loss',  'RPSS_year1', 'RPSS_year2'], 
                                           value_vars = ['min_percentage', 'max_percentage'], var_name = 'minmax', value_name = 'probability')
sb.lineplot(x="epochs", y="probability",hue="label_smoothing", style="minmax", data=E1_sharpness)
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)#ncol = 2)
plt.xticks([5,10,15,20])
plt.tight_layout()
plt.savefig('plots/E1_sharpness.png')
#%%
#E1 achieved RPSS
results_epoch_end_E1 = E1_smooth_results[['fold_no','radius_basis_func','input_lats','input_lons','label_smoothing','epochs', 'accuracy', 'loss', 'RPSS_year1', 'RPSS_year2', 'min_percentage', 'max_percentage']]
results_epoch_end_E1 = results_epoch_end_E1.melt(id_vars = ['fold_no','radius_basis_func','input_lats','input_lons','label_smoothing','epochs', 'accuracy', 'loss',  'min_percentage', 'max_percentage'], 
                                           value_vars = ['RPSS_year1', 'RPSS_year2'], var_name = 'year', value_name = 'RPSS')

sb.lineplot( data = results_epoch_end_E1, x = 'epochs', y = 'RPSS', hue = 'label_smoothing')#, style = 'year')
plt.xticks([5,10,15,20])
plt.hlines(y = 0, xmin= results_epoch_end_E1.epochs.min(), xmax = results_epoch_end_E1.epochs.max(), color = 'black')
#plt.axhline(y = 0)
#plt.show()
plt.tight_layout()
plt.savefig('plots/E1_RPSS.png')
#%%
#E1 history
def extract_history_E1(results):
    df_list = []
    for i in range(0,results.shape[0],4):
        accuracy = yaml.load(results.history[i])['accuracy'] + yaml.load(results.history[i + 1])['accuracy']+ yaml.load(results.history[i + 2])['accuracy']+ yaml.load(results.history[i + 3])['accuracy']
        val_accuracy = yaml.load(results.history[i])['val_accuracy'] + yaml.load(results.history[i + 1])['val_accuracy']+ yaml.load(results.history[i + 2])['val_accuracy']+ yaml.load(results.history[i + 3])['val_accuracy']
        loss = yaml.load(results.history[i])['loss'] + yaml.load(results.history[i + 1])['loss'] + yaml.load(results.history[i + 2])['loss'] + yaml.load(results.history[i + 3])['loss']
        val_loss = yaml.load(results.history[i])['val_loss'] + yaml.load(results.history[i + 1])['val_loss'] + yaml.load(results.history[i + 2])['val_loss'] + yaml.load(results.history[i + 3])['val_loss']
        
        df_tr = pd.DataFrame(accuracy, columns = ['accuracy'])
        df_tr['loss'] = loss
        df_tr['dataset'] = 'train'
        df_tr['epochs'] = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
        
        df_val = pd.DataFrame(val_accuracy, columns = ['accuracy'])
        df_val['loss'] = val_loss
        df_val['dataset'] = 'validation'
        df_val['epochs'] = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
        
        df = pd.concat([df_tr, df_val])
       # df['val_accuracy'] = val_accuracy
        #df['loss'] = loss 
       # df['val_loss'] = val_loss
        df['radius'] = results.radius_basis_func[i]
        df['input_lats'] = results.input_lats[i]
        df['input_lons'] = results.input_lons[i]
        df['label_smoothing'] = results.label_smoothing[i]
        df['fold'] = results.fold_no[i]
        
        
        df_list.append(df)
    history = pd.concat(df_list)
    history.reset_index(inplace = True)
    return history

E1_history = extract_history_E1(E1_smooth_results)
#%%
sb.lineplot(x = 'epochs', y = 'accuracy', hue = 'label_smoothing', style = 'dataset',  data = E1_history)
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.tight_layout()
plt.savefig('plots/E1_smooth_accuracy.png')
plt.figure()
sb.lineplot(x = 'epochs', y = 'loss', hue = 'label_smoothing', style = 'dataset',  data = E1_history)
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.tight_layout()
plt.savefig('plots/E1_smooth_loss.png')
#%%
